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

from utils import camera_filtering

def filter_small_blobs(mask: np.ndarray, area_threshold_ratio: float = 0.05) -> np.ndarray:
    """
    Removes connected components (blobs) that are smaller than a percentage of the total image area.
    """
    if not np.any(mask):
        return mask

    # Calculate total image area and pixel threshold
    total_area = mask.shape[0] * mask.shape[1]
    pixel_threshold = total_area * area_threshold_ratio

    # Ensure mask is 8-bit single channel for OpenCV
    mask_8u = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)

    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)
    
    # Create a clean output mask
    filtered_mask = np.zeros_like(mask, dtype=bool)

    # Component 0 is always the background, so start at 1
    for label in range(1, num_labels):
        blob_area = stats[label, cv2.CC_STAT_AREA]
        if blob_area >= pixel_threshold:
            filtered_mask |= (labels == label)

    return filtered_mask

def load_reconstruction(model_dir: Path) -> pycolmap.Reconstruction:
    recon = pycolmap.Reconstruction(model_dir)
    if len(recon.cameras) == 0:
        raise RuntimeError(f"No cameras found in {model_dir}")
    if len(recon.images) == 0:
        raise RuntimeError(f"No images found in {model_dir}")
    return recon


def load_shared_camera(recon: pycolmap.Reconstruction) -> list:
    return list(recon.cameras.values())


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


def find_symmetric_crop_params(valid_mask: np.ndarray, cx: float, cy: float, padding: int, mul_of: int):
    """
    Computes bounds centered precisely at (cx, cy) to enclose the padded mask profile.
    Output dimensions are pushed up to satisfy the MUL_OF constraint.
    """
    h, w = valid_mask.shape[:2]
    ys, xs = np.where(valid_mask)
    
    if len(ys) == 0:
        half_w = int(max(cx, w - cx))
        half_h = int(max(cy, h - cy))
    else:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        
        xmin = max(0, xmin - padding)
        xmax = min(w - 1, xmax + padding)
        ymin = max(0, ymin - padding)
        ymax = min(h - 1, ymax + padding)
        
        dist_x = max(abs(cx - xmin), abs(cx - (xmax + 1)))
        dist_y = max(abs(cy - ymin), abs(cy - (ymax + 1)))
        
        half_w = int(np.ceil(dist_x))
        half_h = int(np.ceil(dist_y))
        
    width = int(np.ceil((half_w * 2) / mul_of) * mul_of)
    height = int(np.ceil((half_h * 2) / mul_of) * mul_of)
    
    half_w = width // 2
    half_h = height // 2
    
    center_x = int(np.round(cx))
    center_y = int(np.round(cy))
    
    minx = center_x - half_w
    maxx = center_x + half_w
    miny = center_y - half_h
    maxy = center_y + half_h
    
    return minx, maxx, miny, maxy


def crop_and_pad_image(img: np.ndarray, minx: int, maxx: int, miny: int, maxy: int) -> np.ndarray:
    """
    Slices an image according to specified targets. If target coordinates exceed
    the actual image dimensions, it clips safely and pads the missing space with zeros.
    """
    h, w = img.shape[:2]
    
    pad_top = max(0, -miny)
    pad_bottom = max(0, maxy - h)
    pad_left = max(0, -minx)
    pad_right = max(0, maxx - w)
    
    c_miny = max(0, miny)
    c_maxy = min(h, maxy)
    c_minx = max(0, minx)
    c_maxx = min(w, maxx)
    
    cropped = img[c_miny:c_maxy, c_minx:c_maxx]
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if cropped.ndim == 3:
            cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            
    return cropped


def crop_and_pad_maps(map1, map2, minx: int, maxx: int, miny: int, maxy: int, w_orig: int, h_orig: int):
    """
    Slices the remap matrix maps based on the ROI parameters. If coordinates fall outside 
    borders, fills map targets with a constant fallback coordinate out of frame (-1) to pad gracefully.
    """
    target_h = maxy - miny
    target_w = maxx - minx
    
    c_miny = max(0, miny)
    c_maxy = min(h_orig, maxy)
    c_minx = max(0, minx)
    c_maxx = min(w_orig, maxx)
    
    out_map1 = np.full((target_h, target_w), -1, dtype=np.float32)
    out_map2 = np.full((target_h, target_w), -1, dtype=np.float32)
    
    t_miny = max(0, -miny)
    t_maxy = t_miny + (c_maxy - c_miny)
    t_minx = max(0, -minx)
    t_maxx = t_minx + (c_maxx - c_minx)
    
    if (c_maxy > c_miny) and (c_maxx > c_minx):
        out_map1[t_miny:t_maxy, t_minx:t_maxx] = map1[c_miny:c_maxy, c_minx:c_maxx]
        out_map2[t_miny:t_maxy, t_minx:t_maxx] = map2[c_miny:c_maxy, c_minx:c_maxx]
        
    return out_map1, out_map2


def process_dataset(input_dir: Path, output_dir: Path, recon: pycolmap.Reconstruction, 
                    cv_cameras: dict, has_segmentation: bool, padding: int, mul_of: int, filter_redundant:bool):
    images_in = input_dir / "images"
    masks_in = input_dir / "masks"
    segmentation_in = input_dir / "segmentation"
    
    images_out = output_dir / "images"
    masks_out = output_dir / "masks"
    segmentation_out = output_dir / "segmentation"
    
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    if has_segmentation:
        segmentation_out.mkdir(parents=True, exist_ok=True)
        
    image_map = {Path(img.name).stem: img for img in recon.images.values()}
    
    frames = []
    cameras_json = []
    
    print("Processing datasets with symmetric crop and zoom-to-resolution pipeline...")
    
    camera_pose_filtering_data = {}
    
    seen_cameras = set()
    for src_path in tqdm.tqdm(list(iter_files(images_in))):
        if src_path.name.endswith(".json"): continue
        stem = src_path.stem
        if stem.endswith(".jpeg"): stem = stem.removesuffix(".jpeg")
        
        if stem not in image_map:
            continue
            
        img_colmap = image_map[stem]
        cam = cv_cameras[img_colmap.camera_id]
        map1, map2 = cam["map1"], cam["map2"]
        K_new = cam["K_new"]
        W_new, H_new = cam["W_new"], cam["H_new"]
        
        rel = src_path.relative_to(images_in)
        rel_stem = rel.with_suffix(".png")
        
        mask_file = None
        cand = masks_in / (rel.name+".png")
        if cand.exists():
            mask_file = cand
                
        seg_file = None
        if has_segmentation:
            cand = segmentation_in / (rel.name+".png")
            if cand.exists():
                seg_file = cand
                    
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        mask_undist = None
        valid_mask = np.zeros((H_new, W_new), dtype=bool)
        
        if mask_file:
            m = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            m_filtered = filter_small_blobs(m>0)
            m = m * m_filtered
            if m is not None:
                mask_undist = undistort(m, map1, map2, cv2.INTER_NEAREST)
                if mask_undist.ndim == 3:
                    valid_mask |= (mask_undist.sum(axis=2) > 0)
                else:
                    valid_mask |= (mask_undist > 0)
                    
        seg_undist = None
        if seg_file:
            s = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)
            if s is not None:
                seg_undist = undistort(s, map1, map2, cv2.INTER_NEAREST)
                seg_undist_filtered = filter_small_blobs(seg_undist > 0)
                seg_undist = seg_undist * seg_undist_filtered
                if seg_undist.ndim == 3:
                    valid_mask |= (seg_undist.sum(axis=2) > 0)
                else:
                    valid_mask |= (seg_undist > 0)

        # 1. Compute the symmetrical crop bounds around the original camera center
        cx, cy = K_new[0, 2], K_new[1, 2]
        minx, maxx, miny, maxy = find_symmetric_crop_params(valid_mask, cx, cy, padding, mul_of)
        
        cw = maxx - minx
        ch = maxy - miny
        
        # 2. Extract the cropped space using the map-slicing ROI optimization
        if map1 is not None and map2 is not None:
            map1_roi, map2_roi = crop_and_pad_maps(map1, map2, minx, maxx, miny, maxy, W_new, H_new)
            img_crop = cv2.remap(img, map1_roi, map2_roi, interpolation=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img_crop = crop_and_pad_image(img, minx, maxx, miny, maxy)
            
        # 3. Compute uniform zoom scale factor
        # Using max ensures the entire crop area is scaled up uniformly to fill the target resolution frame
        scale_uniform = max(W_new / float(cw), H_new / float(ch))
        
        # Determine the intermediate size before forcing back to exact W_new, H_new
        inter_w = int(round(cw * scale_uniform))
        inter_h = int(round(ch * scale_uniform))
        
        # Resize images to intermediate uniform scale
        img_scaled = cv2.resize(img_crop, (inter_w, inter_h), interpolation=cv2.INTER_CUBIC)
        
        # Center-crop or pad the uniformly scaled image to exactly match the target output dimensions
        # This keeps the image perfectly centered with a clean uniform aspect ratio
        img_zoomed = crop_and_pad_image(img_scaled, 
                                        minx=(inter_w - W_new) // 2, maxx=(inter_w - W_new) // 2 + W_new,
                                        miny=(inter_h - H_new) // 2, maxy=(inter_h - H_new) // 2 + H_new)
        
        img_out_path = images_out / rel_stem
        img_out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out_path), img_zoomed)
        
        img_for_camera_filtering = img_zoomed
        img_mask_for_camera_filtering = None
        
        if mask_undist is not None:
            mask_scaled = cv2.resize(crop_and_pad_image(mask_undist, minx, maxx, miny, maxy), (inter_w, inter_h), interpolation=cv2.INTER_NEAREST)
            mask_zoomed = crop_and_pad_image(mask_scaled, 
                                            minx=(inter_w - W_new) // 2, maxx=(inter_w - W_new) // 2 + W_new,
                                            miny=(inter_h - H_new) // 2, maxy=(inter_h - H_new) // 2 + H_new)
            mask_out_path = masks_out / rel_stem
            mask_out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_out_path), mask_zoomed)
            
            img_mask_for_camera_filtering = mask_zoomed > 0
            
        if seg_undist is not None:
            seg_scaled = cv2.resize(crop_and_pad_image(seg_undist, minx, maxx, miny, maxy), (inter_w, inter_h), interpolation=cv2.INTER_NEAREST)
            seg_zoomed = crop_and_pad_image(seg_scaled, 
                                           minx=(inter_w - W_new) // 2, maxx=(inter_w - W_new) // 2 + W_new,
                                           miny=(inter_h - H_new) // 2, maxy=(inter_h - H_new) // 2 + H_new)
            seg_out_path = segmentation_out / rel_stem
            seg_out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(seg_out_path), seg_zoomed)
            
            if img_mask_for_camera_filtering is None:
                img_mask_for_camera_filtering = seg_zoomed > 0
                
            
            
            
        # 4. Scale focal lengths uniformly to adjust for the geometric zoom factor
        K_zoom = K_new.copy()
        K_zoom[0, 0] *= scale_uniform  # fx
        K_zoom[1, 1] *= scale_uniform  # fy
        K_zoom[0, 2] = W_new / 2.0     # cx (perfectly centered)
        K_zoom[1, 2] = H_new / 2.0     # cy (perfectly centered)
        
        
        
        cam_id = f"cam_{img_colmap.image_id}"
        R_wc = img_colmap.cam_from_world().rotation.matrix()
        t_wc = np.asarray(img_colmap.cam_from_world().translation, dtype=np.float64)
        
        
        
        
        try:
            ID = int(stem) - 1
        except ValueError:
            ID = img_colmap.image_id
            
            
        camera_pose_filtering_data[ID] = {
            "image_score": camera_filtering.compute_sharpness(img_for_camera_filtering, img_mask_for_camera_filtering),
            "R_w2c": R_wc,
            "t_w2c": t_wc
        }
            
        frames.append({
            "image": rel_stem.name,
            "image_path": f"images/{rel_stem.as_posix()}",
            "mask_path": f"masks/{rel_stem.as_posix()}" if mask_undist is not None else "",
            "segmentation_path": f"segmentation/{rel_stem.as_posix()}" if seg_undist is not None else "",
            "id": int(ID),
            "R": R_wc.tolist(),
            "T": t_wc.tolist(),
            "camera_id": cam_id,
            "redundant": False
        })
        
        if cam_id not in seen_cameras:
            cameras_json.append({
                "camera_id": cam_id,
                "K": K_zoom.tolist(),
                "width": int(W_new),
                "height": int(H_new),
            })
            seen_cameras.add(cam_id)
        
        
    print("Filtering based on camera pose and image score")
    if filter_redundant:
        filtered_list = camera_filtering.filter_and_visualize_poses_adaptive(camera_pose_filtering_data, pos_threshold=None, K=4.5, angle_threshold_deg=5) 
        for i in range(len(frames)): frames[i]["redundant"] = frames[i]["id"] not in filtered_list
        
    return frames, cameras_json


def write_cameras_json(path: Path, frames: list, cameras: list):
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
    parser.add_argument("--padding", type=int, default=10, help="Padding pixel boundary size")
    parser.add_argument("--mul_of", type=int, default=8, help="Output grid size factor requirement")
    parser.add_argument("--filter_redundant", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir

    segmentation_in = input_dir / "segmentation"
    has_segmentation = os.path.exists(segmentation_in)

    recon = load_reconstruction(model_dir)
    cameras = load_shared_camera(recon)
    cv_cameras = colmap_cameras_to_opencv(cameras, args.scale)

    shutil.copytree(model_dir, output_dir / "sparse" / "0", dirs_exist_ok=True)

    frames, cameras_json = process_dataset(
        input_dir, output_dir, recon, cv_cameras, has_segmentation, args.padding, args.mul_of, args.filter_redundant
    )
    
    
    
    

    write_cameras_json(output_dir / "cameras.json", frames, cameras_json)
    write_points3D_ply(output_dir / "points3D.ply", recon)

    print("Done.")
    print(f"Cameras: {output_dir / 'cameras.json'}")


if __name__ == "__main__":
    main()