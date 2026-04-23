#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import os
import tqdm
import shutil

import cv2
import numpy as np
import pycolmap
import open3d as o3d


def load_reconstruction(model_dir: Path) -> pycolmap.Reconstruction:
    recon = pycolmap.Reconstruction(model_dir)
    if len(recon.cameras) == 0:
        raise RuntimeError(f"No cameras found in {model_dir}")
    if len(recon.images) == 0:
        raise RuntimeError(f"No images found in {model_dir}")
    return recon


def load_shared_camera(recon: pycolmap.Reconstruction) -> pycolmap.Camera:
    cameras = list(recon.cameras.values())
    return cameras


def colmap_cameras_to_opencv(cameras, scale: float):
    out_cameras = {}
    for camera in cameras:
        key = camera.camera_id
 
        model = camera.model
        p = np.asarray(camera.params, dtype=np.float64)
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)

        if model in (pycolmap.CameraModelId.PINHOLE, pycolmap.CameraModelId.SIMPLE_PINHOLE):
            dist = np.zeros((4, 1), dtype=np.float64)
            use_fisheye = False

        elif model == pycolmap.CameraModelId.SIMPLE_RADIAL:
            dist = np.array([p[3], 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
            use_fisheye = False

        elif model == pycolmap.CameraModelId.RADIAL:
            dist = np.array([p[3], p[4], 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
            use_fisheye = False

        elif model == pycolmap.CameraModelId.OPENCV:
            dist = np.array([p[4], p[5], p[6], p[7]], dtype=np.float64).reshape(-1, 1)
            use_fisheye = False

        elif model == pycolmap.CameraModelId.FULL_OPENCV:
            dist = np.array(p[4:12], dtype=np.float64).reshape(-1, 1)
            use_fisheye = False

        elif model == pycolmap.CameraModelId.OPENCV_FISHEYE:
            dist = np.array(p[4:8], dtype=np.float64).reshape(-1, 1)
            use_fisheye = True

        else:
            raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model_name}")

        H_new = int(round(camera.height * scale))
        W_new = int(round(camera.width * scale))

        if use_fisheye:
            K_new = K.copy()
            K_new[0, :] *= scale
            K_new[1, :] *= scale
            K_new[2, 2] = 1.0
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, dist, np.eye(3), K_new, (W_new, H_new), cv2.CV_32FC1
            )
        elif model in (pycolmap.CameraModelId.PINHOLE, pycolmap.CameraModelId.SIMPLE_PINHOLE) :
            K_new = K
            map1, map2 = (None, None)
        else:
            K_new, _ = cv2.getOptimalNewCameraMatrix(
                K, dist, (camera.width, camera.height), 0.0, (W_new, H_new), centerPrincipalPoint=True
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, None, K_new, (W_new, H_new), cv2.CV_32FC1
            )

        out_cameras[key] = {
            "K": K,
            "K_new": K_new,
            "D": dist,
            "W": camera.width,
            "H": camera.height,
            "W_new": W_new,
            "H_new": H_new,
            "map1": map1,
            "map2": map2,
        }
    return out_cameras


def undistort(img: np.ndarray, map1, map2, interpolation: int) -> np.ndarray:
    if map1 is None and map2 is None: return img
    return cv2.remap(img, map1, map2, interpolation=interpolation)


def iter_files(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in [".png", ".jpeg", ".bmp", ".jpg"]:
            yield p


def process_folder(src_dir: Path, dst_dir: Path,recon, cv_cameras, interpolation: int):
    if not src_dir.exists():
        print(f"Skipping missing folder: {src_dir}")
        return

    print(f"Processing {src_dir}")
    image_map = {Path(img.name).stem: img for img in recon.images.values()}

    for src_path in tqdm.tqdm(list(iter_files(src_dir))):
        if src_path.name.endswith(".json"): continue
        name = src_path.stem
        if name.endswith(".jpeg"): name = name.removesuffix(".jpeg")
        if name not in image_map:
            print(f"\tSkipping (not in reconstruction): {name}")
            continue

        img_colmap = image_map[name]
        cam = cv_cameras[img_colmap.camera_id]

        map1, map2 = cam["map1"], cam["map2"] 
        
        rel = src_path.relative_to(src_dir)
        stem = rel.name.split('.')[0]
        dst_path = dst_dir / rel.parent / f"{stem}.png"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        

        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"\tSkipping unreadable file: {src_path}")
            continue

        out = undistort(img, map1, map2, interpolation)
        if not cv2.imwrite(str(dst_path), out):
            raise IOError(f"Failed to write: {dst_path}")


def write_cameras_json(path: Path, recon: pycolmap.Reconstruction, cv_cameras: dict, images_out_dir: Path):
    frames = []
    cameras = []
    for _, img in sorted(recon.images.items(), key=lambda kv: kv[1].name):
        key = img.camera_id
        if key not in cv_cameras:
            raise RuntimeError(f"Missing camera for image {img.name}") 
        cv_camera = cv_cameras[key]
        image_name = Path(img.name).with_suffix(".png").name

        # world -> camera in COLMAP
        R_wc = img.cam_from_world().rotation.matrix()
        t_wc = np.asarray(img.cam_from_world().translation, dtype=np.float64)
        
        # derive from filename: "00001.png" -> 0
        stem = Path(img.name).stem
        ID = int(stem) - 1 
        assert ID >= 0

        frames.append({
            "image": image_name,
            "image_path": f"images/{image_name}",
            "mask_path": f"masks/{image_name}",
            "id": int(ID),
            "R": R_wc.tolist(),
            "T": t_wc.tolist(),
            "camera_id": key
        })
        
        cameras.append({
            "camera_id": key,
            "K": cv_camera["K_new"].tolist(),
            "width": int(cv_camera["W_new"]),
            "height": int(cv_camera["H_new"]), 
        })

    data = {
        "cameras": cameras,
        "frames": frames,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_points3D_ply(path: Path, recon: pycolmap.Reconstruction):
    points = []
    colors = []

    for _, p in recon.points3D.items():
        points.append(np.asarray(p.xyz, dtype=np.float64))
        colors.append(np.asarray(p.color, dtype=np.float64) / 255.0)

    if len(points) == 0:
        print("Warning: no 3D points found, skipping points3D.ply")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    o3d.io.write_point_cloud(str(path), pcd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-m", "--model_dir", type=Path, required=True)
    parser.add_argument("-s", "--scale", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir

    images_in = input_dir / "images"
    masks_in = input_dir / "masks"

    images_out = output_dir / "images"
    masks_out = output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    recon = load_reconstruction(model_dir)
    cameras = load_shared_camera(recon)
    cv_cameras = colmap_cameras_to_opencv(cameras, args.scale)

    shutil.copytree(model_dir, output_dir / "sparse" / "0", dirs_exist_ok=True)

    process_folder(images_in, images_out, recon,cv_cameras, interpolation=cv2.INTER_LINEAR)
    process_folder(masks_in, masks_out, recon,cv_cameras, interpolation=cv2.INTER_NEAREST)

    write_cameras_json(output_dir / "cameras.json", recon, cv_cameras, images_out)
    write_points3D_ply(output_dir / "points3D.ply", recon)

    print("Done.")
    print(f"Images:  {images_out}")
    print(f"Masks:   {masks_out}")
    print(f"Cameras: {output_dir / 'cameras.json'}")
    print(f"Points:  {output_dir / 'points3D.ply'}")


if __name__ == "__main__":
    main()