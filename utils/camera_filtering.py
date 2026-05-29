import cv2
import numpy as np
import open3d as o3d

def compute_sharpness(image_path_or_array, mask):
    """Computes the Laplacian variance of an image as a proxy for sharpness."""
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = image_path_or_array
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return cv2.Laplacian(img, cv2.CV_64F)[mask.squeeze()].var()


def filter_and_visualize_poses_adaptive(data_dict, pos_threshold=None, K=2.5, angle_threshold_deg=5.0):
    """Filters near cameras keeping the sharpest one, with automatic threshold estimation.

    Args:
        data_dict: dict structured as {
            img_id: {
                'w2c': np.ndarray (4x4),
                'image': np.ndarray or str (path to image)
            }
        }
        pos_threshold: Max distance to consider positions 'close'. If None, computed dynamically.
        K: Multiplier for the median consecutive position shift if pos_threshold is None.
        angle_threshold_deg: Max angle difference (in degrees) to consider orientations 'close'
    """
    img_ids = list(data_dict.keys())
    camera_centers = []
    view_dirs = []
    sharpness_scores = []
    c2w_mats = []

    # 1. Parse data and extract camera extrinsic properties
    for img_id in img_ids:
        R_w2c = data_dict[img_id]["R_w2c"]
        t_w2c = data_dict[img_id]["t_w2c"]
        w2c = np.concat((R_w2c, t_w2c[:,None]),axis=-1)
        w2c = np.concat((w2c, np.array([[0,0,0,1]],dtype=w2c.dtype)),0)
        c2w = np.linalg.inv(w2c)
        c2w_mats.append(c2w)

        # Camera center is the translation vector of c2w
        center = c2w[:3, 3]
        camera_centers.append(center)

        # Viewing direction is the Z-axis of the camera (third column of c2w)
        view_dir = c2w[:3, 2]
        view_dirs.append(view_dir / np.linalg.norm(view_dir))

        # Sharpness calculation
        score = data_dict[img_id]["image_score"]
        sharpness_scores.append(score)

    camera_centers = np.array(camera_centers)
    num_cams = len(img_ids)

    # 2. Dynamic Threshold Estimation
    if pos_threshold is None:
        if num_cams > 1:
            # Compute pairwise distance matrix between all camera centers: shape (N, N)
            # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diffs = camera_centers[:, np.newaxis, :] - camera_centers[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diffs, axis=2)
            
            # Fill the diagonal with infinity so a camera doesn't select itself as its own nearest neighbor
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Find the distance to the single closest neighbor for each camera
            nearest_distances = np.min(dist_matrix, axis=1)
            
            # Compute the median spatial density step across the unstructured cloud
            median_nn_dist = np.median(nearest_distances)
            pos_threshold = median_nn_dist * K
        else:
            pos_threshold = 0.0  # Fallback for single image

    kept_indices = set()
    discarded_indices = set()

    # Sort indices by sharpness descending so we anchor groups with the highest quality frames
    sorted_indices = np.argsort(sharpness_scores)[::-1]

    # 3. Greedy Clustering Loop
    for idx in sorted_indices:
        if idx in kept_indices or idx in discarded_indices:
            continue

        kept_indices.add(idx)

        for other_idx in range(num_cams):
            if other_idx == idx or other_idx in kept_indices or other_idx in discarded_indices:
                continue

            # Distance check
            dist = np.linalg.norm(camera_centers[idx] - camera_centers[other_idx])

            # Angular alignment check
            dot_product = np.clip(np.dot(view_dirs[idx], view_dirs[other_idx]), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))

            if dist < pos_threshold and angle < angle_threshold_deg:
                discarded_indices.add(other_idx)

    print(f"Total input frames: {num_cams}")
    print(f"Kept (Sharp & Unique): {len(kept_indices)} | Discarded (Redundant): {len(discarded_indices)}")

    # 4. Open3D Visualization Setup
    geometries = []

    def create_camera_frustum(c2w, color, size=0.05):
        points = np.array([
            [0, 0, 0],
            [-size, -size, size * 2],
            [size, -size, size * 2],
            [size, size, size * 2],
            [-size, size, size * 2]
        ])
        points_homo = np.hstack((points, np.ones((5, 1))))
        points_world = (c2w @ points_homo.T).T[:, :3]

        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    for i, c2w in enumerate(c2w_mats):
        color = [0.0, 1.0, 0.0] if i in kept_indices else [1.0, 0.0, 0.0]
        geometries.append(create_camera_frustum(c2w, color, size=0.04))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries, window_name="Adaptive Camera Filtering")

    filtered_dict = {img_ids[i]: data_dict[img_ids[i]] for i in kept_indices}
    return filtered_dict
