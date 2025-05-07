import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, NamedTuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class CameraExtrinsic(NamedTuple):
    rotation: np.ndarray
    translation: np.ndarray


class CameraIntrinsic(NamedTuple):
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]


def load_optimization_results(input_dir: str, extrinsic_file_name: str = None, intrinsic_file_name: str = None,
                              points_file_name: str = None, format_type: str = 'json'):
    """
    Load optimized camera extrinsic, intrinsic parameters and 3D point cloud data

    Args:
        input_dir: Input directory path
        extrinsic_file_name: Extrinsic parameter filename (without extension)
        intrinsic_file_name: Intrinsic parameter filename (without extension)
        points_file_name: 3D points filename (without extension)
        format_type: File format, options: 'pickle' or 'json'

    Returns:
        tuple: (camera extrinsic dictionary, camera intrinsic dictionary, 3D point cloud dictionary)
    """
    # Check filenames and build complete paths
    extrinsics_filename = os.path.join(input_dir, f"{extrinsic_file_name}.{format_type}")
    intrinsics_filename = os.path.join(input_dir, f"{intrinsic_file_name}.{format_type}")
    points3d_filename = os.path.join(input_dir, f"{points_file_name}.{format_type}")

    # Check if files exist
    if not os.path.exists(extrinsics_filename):
        raise FileNotFoundError(f"no extrinsic: {extrinsics_filename}")
    if not os.path.exists(intrinsics_filename):
        raise FileNotFoundError(f"no intrinsic: {intrinsics_filename}")
    if not os.path.exists(points3d_filename):
        raise FileNotFoundError(f"no 3D points: {points3d_filename}")

    if format_type.lower() == 'pickle':
        # Load Pickle format files
        with open(extrinsics_filename, 'rb') as f:
            camera_extrinsics = pickle.load(f)

        with open(intrinsics_filename, 'rb') as f:
            camera_intrinsics = pickle.load(f)

        with open(points3d_filename, 'rb') as f:
            points_3d = pickle.load(f)

    elif format_type.lower() == 'json':
        # Load JSON format files
        with open(extrinsics_filename, 'r') as f:
            json_extrinsics = json.load(f)

        with open(intrinsics_filename, 'r') as f:
            json_intrinsics = json.load(f)

        with open(points3d_filename, 'r') as f:
            json_points3d = json.load(f)

        camera_extrinsics = {}
        for cam_id, data in json_extrinsics.items():
            camera_extrinsics[cam_id] = CameraExtrinsic(
                rotation=np.array(data['rotation']),
                translation=np.array(data['translation'])
            )

        camera_intrinsics = {}
        for cam_id, data in json_intrinsics.items():
            camera_intrinsics[cam_id] = CameraIntrinsic(
                camera_matrix=np.array(data['camera_matrix']),
                dist_coeffs=np.array(data['dist_coeffs']),
                image_size=tuple(data['image_size'])
            )

        points_3d = {}
        for frame, point in json_points3d.items():
            points_3d[int(frame)] = np.array(point)

    else:
        raise ValueError(f"format not support: {format_type}")

    print(f"successfully loaded data:")
    print(f"  - camera extrinsic: {len(camera_extrinsics)} cameras")
    print(f"  - camera intrinsic: {len(camera_intrinsics)} cameras")
    print(f"  - 3D points: {len(points_3d)} points")

    return camera_extrinsics, camera_intrinsics, points_3d


def load_2d_data(cam_info_path: str, data2d_path: str, use_clustering: bool = True,
                 min_points_for_clustering: int = 10):
    """
    Load camera information and 2D data points - Enhanced version, supporting point clustering

    Args:
        cam_info_path: Camera information CSV file path
        data2d_path: 2D point data CSV file path
        use_clustering: Whether to use clustering to divide points into two groups
        min_points_for_clustering: Minimum number of points required for clustering

    Returns:
        tuple: (cam_id mapping dictionary, 2D point data grouped by camera ID, including cluster labels)
    """
    # Load camera information
    cam_info = pd.read_csv(cam_info_path)
    cam_id_mapping = dict(zip(cam_info['camn'], cam_info['cam_id']))

    # Load 2D point data
    data2d = pd.read_csv(data2d_path)

    # Replace camn with cam_id
    data2d['cam_id'] = data2d['camn'].map(cam_id_mapping)

    # Check if each camera has two points per frame
    point_counts = data2d.groupby(['cam_id', 'frame']).size().reset_index(name='count')
    valid_frames = point_counts[point_counts['count'] == 2][['cam_id', 'frame']]

    # Create a joint key for filtering
    data2d['key'] = data2d['cam_id'].astype(str) + '_' + data2d['frame'].astype(str)
    valid_frames['key'] = valid_frames['cam_id'].astype(str) + '_' + valid_frames['frame'].astype(str)
    valid_keys = set(valid_frames['key'])

    # Filter data, keep only frames with two points
    filtered_data = data2d[data2d['key'].isin(valid_keys)]

    print(f"Original data has {len(data2d)} rows of 2D data")
    print(f"After filtering, kept {len(filtered_data)} rows of 2D data (only keeping frames with two points)")

    # Check if data_id or obj_id is available to ensure consistent point order
    has_obj_id = 'obj_id' in filtered_data.columns
    has_data_id = 'data_id' in filtered_data.columns

    if has_obj_id:
        print("Detected obj_id column, will use it to ensure consistent point order")
    elif has_data_id:
        print("Detected data_id column, will use it to ensure consistent point order")
    else:
        print("Warning: No obj_id or data_id column detected, will use original data order")

    # Group by camera ID
    grouped_data = {}

    # Store all points by camera for subsequent clustering
    all_points_by_camera = defaultdict(list)
    all_frames_by_camera = defaultdict(list)
    all_point_indices_by_camera = defaultdict(list)  # Store the index of each point in original order

    # Construct two point coordinates for each frame of each camera - strictly maintain original order
    for cam_id, cam_group in filtered_data.groupby('cam_id'):
        grouped_data[cam_id] = {}
        frame_points = {}
        point_index = 0

        for frame, frame_group in cam_group.groupby('frame'):
            # Ensure this frame really has two points
            if len(frame_group) == 2:
                # Get point coordinates in original order - absolutely no sorting!

                # If obj_id exists, sort by obj_id to ensure consistent point order across cameras
                if has_obj_id:
                    frame_group = frame_group.sort_values('obj_id')
                # If data_id exists, sort by data_id
                elif has_data_id:
                    frame_group = frame_group.sort_values('data_id')
                # Otherwise, maintain original order

                # Note: Maintaining the original order from the data file is very important
                points = frame_group[['x', 'y']].values

                frame_points[frame] = {
                    'point1': points[0],  # First point
                    'point2': points[1],  # Second point
                    'group1': None,  # Cluster group 1
                    'group2': None  # Cluster group 2
                }

                # Collect all points for subsequent clustering
                all_points_by_camera[cam_id].append(points[0])
                all_frames_by_camera[cam_id].append(frame)
                all_point_indices_by_camera[cam_id].append((point_index, 0))  # (point index, point index within frame)

                all_points_by_camera[cam_id].append(points[1])
                all_frames_by_camera[cam_id].append(frame)
                all_point_indices_by_camera[cam_id].append((point_index, 1))  # (point index, point index within frame)

                point_index += 1

        # Only add this camera if there is at least one valid frame of data
        if frame_points:
            grouped_data[cam_id] = frame_points

    print(f"Loaded valid data for {len(grouped_data)} cameras")
    valid_frames_count = sum(len(frames) for frames in grouped_data.values())
    print(f"Total {valid_frames_count} valid frames (each frame contains two points)")

    all_cluster_labels = {}

    # If clustering is enabled, perform point clustering for each camera
    if use_clustering:
        for cam_id, points in all_points_by_camera.items():
            if len(points) >= min_points_for_clustering:
                print(f"Clustering {len(points)} points for camera {cam_id}...")

                # Convert points to numpy array
                points_array = np.array(points)

                # Use KMeans clustering
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(points_array)

                # Save all cluster labels for this camera
                all_cluster_labels[cam_id] = clusters.tolist()

                # Apply cluster labels back to original data
                for i, (frame_idx, point_idx) in enumerate(all_point_indices_by_camera[cam_id]):
                    frame = all_frames_by_camera[cam_id][i]
                    cluster_label = clusters[i]

                    # Update the cluster label for the corresponding frame and point
                    if point_idx == 0:  # point1
                        grouped_data[cam_id][frame]['group1'] = cluster_label
                    else:  # point2
                        grouped_data[cam_id][frame]['group2'] = cluster_label

                # Print the count statistics for each cluster
                cluster_counts = np.bincount(clusters)
                print(f"Camera {cam_id} clustering results: Group 0: {cluster_counts[0]} points, Group 1: {cluster_counts[1]} points")
            else:
                print(f"Camera {cam_id} has only {len(points)} points, not enough for reliable clustering")

    return cam_id_mapping, grouped_data, all_points_by_camera, all_cluster_labels


def project_3d_to_2d(point_3d: np.ndarray, camera_extrinsic: CameraExtrinsic,
                     camera_intrinsic: CameraIntrinsic) -> np.ndarray:
    """
    Project 3D point onto camera plane to get 2D point

    Args:
        point_3d: 3D point coordinates (3,)
        camera_extrinsic: Camera extrinsic parameters
        camera_intrinsic: Camera intrinsic parameters

    Returns:
        np.ndarray: Projected 2D point coordinates (2,)
    """
    # Convert 3D point from world coordinate system to camera coordinate system
    R = camera_extrinsic.rotation
    t = camera_extrinsic.translation

    # Point coordinates in camera system
    point_cam = R @ point_3d + t

    # Project to image plane
    fx = camera_intrinsic.camera_matrix[0, 0]
    fy = camera_intrinsic.camera_matrix[1, 1]
    cx = camera_intrinsic.camera_matrix[0, 2]
    cy = camera_intrinsic.camera_matrix[1, 2]

    # Simplified processing, not considering distortion
    x = fx * point_cam[0] / point_cam[2] + cx
    y = fy * point_cam[1] / point_cam[2] + cy

    return np.array([x, y])


def calculate_reprojection_error(point_3d: np.ndarray, point_2d: np.ndarray,
                                 camera_extrinsic: CameraExtrinsic,
                                 camera_intrinsic: CameraIntrinsic) -> float:
    """
    Calculate 3D point reprojection error

    Args:
        point_3d: 3D point coordinates
        point_2d: Actual observed 2D point coordinates
        camera_extrinsic: Camera extrinsic parameters
        camera_intrinsic: Camera intrinsic parameters

    Returns:
        float: Reprojection error (Euclidean distance)
    """
    projected_2d = project_3d_to_2d(point_3d, camera_extrinsic, camera_intrinsic)
    error = np.linalg.norm(projected_2d - point_2d)
    return error


def triangulate_point(points_2d: Dict[str, np.ndarray],
                      camera_extrinsics: Dict[str, CameraExtrinsic],
                      camera_intrinsics: Dict[str, CameraIntrinsic]) -> Tuple[np.ndarray, float]:
    """
    Triangulate 3D point using 2D points from multiple cameras - Optimized version

    Args:
        points_2d: Dictionary of 2D points observed by cameras {cam_id: [x, y]}
        camera_extrinsics: Dictionary of camera extrinsic parameters
        camera_intrinsics: Dictionary of camera intrinsic parameters

    Returns:
        tuple: (Triangulated 3D point, average reprojection error)
    """
    # Construct coefficient matrix A for triangulation
    A = []
    valid_cameras = []

    for cam_id, point_2d in points_2d.items():
        # Check if camera and point are valid
        if cam_id not in camera_extrinsics or cam_id not in camera_intrinsics:
            continue

        if np.isnan(point_2d).any():
            continue

        # Get camera parameters
        R = camera_extrinsics[cam_id].rotation
        t = camera_extrinsics[cam_id].translation
        K = camera_intrinsics[cam_id].camera_matrix

        # Inverse of intrinsic matrix
        K_inv = np.linalg.inv(K)

        # Convert 2D point to normalized coordinates
        uv1 = np.array([point_2d[0], point_2d[1], 1.0])
        xy1 = K_inv @ uv1

        # Construct two rows of coefficient matrix
        P = np.hstack((R, t.reshape(3, 1)))

        A.append(xy1[0] * P[2, :] - P[0, :])
        A.append(xy1[1] * P[2, :] - P[1, :])
        valid_cameras.append(cam_id)

    # Check if there are at least 2 cameras' data (at least 4 constraint equations)
    if len(A) < 4:
        return np.array([np.nan, np.nan, np.nan]), np.inf

    # Solve overdetermined system of equations
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    point_3d_homo = Vh[-1, :]

    # Convert homogeneous coordinates to 3D coordinates
    point_3d = point_3d_homo[:3] / point_3d_homo[3]

    # Calculate reprojection error
    errors = []
    for cam_id in valid_cameras:
        point_2d = points_2d[cam_id]
        error = calculate_reprojection_error(
            point_3d, point_2d,
            camera_extrinsics[cam_id],
            camera_intrinsics[cam_id]
        )
        errors.append(error)

    mean_error = np.mean(errors) if errors else np.inf
    return point_3d, mean_error


def find_best_cluster_correspondence(
        frames_data: Dict[int, Dict[str, Dict[str, np.ndarray]]],
        camera_extrinsics: Dict[str, CameraExtrinsic],
        camera_intrinsics: Dict[str, CameraIntrinsic],
        max_sample_frames: int = 20,
        reproj_error_threshold: float = 5.0
) -> Dict[str, Dict[int, int]]:
    """
    Determine the best point correspondence based on cluster labels

    Args:
        frames_data: Data organized by frame, including cluster labels
        camera_extrinsics: Dictionary of camera extrinsic parameters
        camera_intrinsics: Dictionary of camera intrinsic parameters
        max_sample_frames: Maximum number of sample frames used to determine correspondence
        reproj_error_threshold: Reprojection error threshold

    Returns:
        Dict: Mapping of which 3D point group each cluster group of each camera should correspond to
    """
    print("Using cluster labels to determine point correspondence...")

    # Sort frames by number of shared cameras in descending order
    frames_with_camera_counts = []
    for frame, cam_data in frames_data.items():
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]
        if len(valid_cameras) >= 2:
            frames_with_camera_counts.append((frame, len(valid_cameras)))

    # Sort by camera count in descending order
    frames_with_camera_counts.sort(key=lambda x: x[1], reverse=True)

    # Select the top N frames as samples
    sample_frames = [frame for frame, _ in frames_with_camera_counts[:max_sample_frames]]

    if not sample_frames:
        print("Warning: Not enough sample frames found")
        return {}  # Default mapping is empty

    print(f"Using {len(sample_frames)} sample frames to determine cluster correspondence")

    # Create analysis structure for cluster correspondence between cameras
    cluster_correspondence = {}

    # Select the first valid camera as the reference camera
    reference_cameras = set()
    for frame in sample_frames:
        for cam_id, cam_data in frames_data[frame].items():
            if ('group1' in cam_data and cam_data['group1'] is not None and
                    'group2' in cam_data and cam_data['group2'] is not None):
                reference_cameras.add(cam_id)

    if not reference_cameras:
        print("Warning: No valid clustering data found")
        return {}

    ref_cam_id = sorted(list(reference_cameras))[0]
    print(f"Selected camera {ref_cam_id} as reference camera")

    # Initialize cluster correspondence between cameras
    for cam_id in reference_cameras:
        if cam_id != ref_cam_id:
            # Initialize four possible matching combinations
            cluster_correspondence[cam_id] = {
                # Combination 1: Reference camera 0 matches this camera 0, reference camera 1 matches this camera 1
                'combination1': {'error': 0, 'count': 0, 'mapping': {0: 0, 1: 1}},
                # Combination 2: Reference camera 0 matches this camera 1, reference camera 1 matches this camera 0
                'combination2': {'error': 0, 'count': 0, 'mapping': {0: 1, 1: 0}},
            }

    # Use sample frames to test cluster correspondence for each camera
    for frame in sample_frames:
        # Check if reference camera has valid data in current frame
        if ref_cam_id not in frames_data[frame]:
            continue

        ref_cam_data = frames_data[frame][ref_cam_id]

        # Check if reference camera has valid cluster grouping
        if ref_cam_data['group1'] is None or ref_cam_data['group2'] is None:
            continue

        # Cluster groups of reference camera
        ref_group1 = ref_cam_data['group1']
        ref_group2 = ref_cam_data['group2']

        # Point coordinates of reference camera
        ref_point1 = ref_cam_data['point1']
        ref_point2 = ref_cam_data['point2']

        # Test cluster correspondence for each other camera
        for cam_id, correspondence in cluster_correspondence.items():
            if cam_id not in frames_data[frame]:
                continue

            cam_data = frames_data[frame][cam_id]

            # Check if current camera has valid cluster grouping
            if cam_data['group1'] is None or cam_data['group2'] is None:
                continue

            # Cluster groups of current camera
            cam_group1 = cam_data['group1']
            cam_group2 = cam_data['group2']

            # Point coordinates of current camera
            cam_point1 = cam_data['point1']
            cam_point2 = cam_data['point2']

            # Test combination 1: Reference camera group 0 matches current camera group 0, reference camera group 1 matches current camera group 1
            # Build correspondence
            points_combo1_1 = {
                ref_cam_id: ref_point1 if ref_group1 == 0 else ref_point2,
                cam_id: cam_point1 if cam_group1 == 0 else cam_point2
            }
            points_combo1_2 = {
                ref_cam_id: ref_point2 if ref_group1 == 0 else ref_point1,
                cam_id: cam_point2 if cam_group1 == 0 else cam_point1
            }

            # Triangulation
            _, error_combo1_1 = triangulate_point(points_combo1_1, camera_extrinsics, camera_intrinsics)
            _, error_combo1_2 = triangulate_point(points_combo1_2, camera_extrinsics, camera_intrinsics)

            # Test combination 2: Reference camera group 0 matches current camera group 1, reference camera group 1 matches current camera group 0
            # Build correspondence
            points_combo2_1 = {
                ref_cam_id: ref_point1 if ref_group1 == 0 else ref_point2,
                cam_id: cam_point2 if cam_group1 == 0 else cam_point1
            }
            points_combo2_2 = {
                ref_cam_id: ref_point2 if ref_group1 == 0 else ref_point1,
                cam_id: cam_point1 if cam_group1 == 0 else cam_point2
            }

            # Triangulation
            _, error_combo2_1 = triangulate_point(points_combo2_1, camera_extrinsics, camera_intrinsics)
            _, error_combo2_2 = triangulate_point(points_combo2_2, camera_extrinsics, camera_intrinsics)

            # Calculate total error
            total_error_combo1 = error_combo1_1 + error_combo1_2
            total_error_combo2 = error_combo2_1 + error_combo2_2

            # Update correspondence statistics
            if total_error_combo1 < reproj_error_threshold * 2:
                correspondence['combination1']['error'] += total_error_combo1
                correspondence['combination1']['count'] += 1

            if total_error_combo2 < reproj_error_threshold * 2:
                correspondence['combination2']['error'] += total_error_combo2
                correspondence['combination2']['count'] += 1

    # Determine the best cluster correspondence for each camera
    final_cluster_mapping = {}

    # Reference camera mapping is fixed
    final_cluster_mapping[ref_cam_id] = {0: 0, 1: 1}  # self-to-self mapping

    for cam_id, correspondence in cluster_correspondence.items():
        combo1 = correspondence['combination1']
        combo2 = correspondence['combination2']

        # Print statistics
        print(f"\nCluster correspondence statistics for camera {cam_id}:")
        print(f"  Combination 1 (0->0, 1->1): Total error {combo1['error']:.2f}, Valid frames {combo1['count']}")
        print(f"  Combination 2 (0->1, 1->0): Total error {combo2['error']:.2f}, Valid frames {combo2['count']}")

        # Choose the best combination based on valid frame count and average error
        if combo1['count'] == 0 and combo2['count'] == 0:
            print(f"  Warning: Camera {cam_id} did not produce valid correspondence")
            final_cluster_mapping[cam_id] = {0: 0, 1: 1}  # Default mapping
        elif combo1['count'] == 0:
            print(f"  Selecting combination 2 as best correspondence (combination 1 invalid)")
            final_cluster_mapping[cam_id] = combo2['mapping']
        elif combo2['count'] == 0:
            print(f"  Selecting combination 1 as best correspondence (combination 2 invalid)")
            final_cluster_mapping[cam_id] = combo1['mapping']
        else:
            # Calculate average error
            avg_error1 = combo1['error'] / combo1['count']
            avg_error2 = combo2['error'] / combo2['count']

            if avg_error1 <= avg_error2:
                print(f"  Selecting combination 1 as best correspondence (average error: {avg_error1:.2f} vs {avg_error2:.2f})")
                final_cluster_mapping[cam_id] = combo1['mapping']
            else:
                print(f"  Selecting combination 2 as best correspondence (average error: {avg_error2:.2f} vs {avg_error1:.2f})")
                final_cluster_mapping[cam_id] = combo2['mapping']

    return final_cluster_mapping


def remove_outliers(points, threshold=2.0):
    """
    Remove outliers - Enhanced version

    Args:
        points: Set of points
        threshold: Z-score threshold

    Returns:
        Filtered set of points
    """
    if len(points) < 4:  # If too few points, do not filter
        return points

    # Calculate distance of each point from the center point
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)

    # Calculate z-score and remove outliers
    z_scores = zscore(distances)
    mask = np.abs(z_scores) < threshold

    # Ensure at least half of the points are retained
    if np.sum(mask) < len(points) / 2:
        # If too many points would be filtered out, only filter out the farthest few
        keep_count = max(2, len(points) // 2)
        indices = np.argsort(distances)
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[indices[:keep_count]] = True
        mask = new_mask

    return points[mask]


def process_data(
        optimization_dir: str,
        cam_info_path: str,
        data2d_path: str,
        real_distance: float,
        output_dir: str,
        extrinsic_file_name: str = None,
        intrinsic_file_name: str = None,
        points_file_name: str = None,
        format_type: str = 'json',
        reproj_error_threshold: float = 11.0,
        outlier_threshold: float = 2.0,
        use_clustering: bool = True,
        vis_2D: bool = False,
        vis_3D: bool = True
):
    """
    Main processing function - Using clustering grouping strategy for point correspondence
    """
    print("===== Start Processing Data =====")

    # 1. Load optimization results
    camera_extrinsics, camera_intrinsics, original_points_3d  = load_optimization_results(
        optimization_dir,
        format_type=format_type,
        extrinsic_file_name=extrinsic_file_name,
        intrinsic_file_name=intrinsic_file_name,
        points_file_name=points_file_name
    )

    # 2. Load 2D data and perform clustering
    cam_id_mapping, grouped_2d_data, all_points_by_camera, all_cluster_labels = load_2d_data(
        cam_info_path,
        data2d_path,
        use_clustering=use_clustering
    )

    # 2.1 New: Generate cluster visualization (if clustering is enabled)
    if use_clustering and all_cluster_labels and vis_2D:
        # Get camera image size for setting visualization boundaries
        visualize_2d_clusters(all_points_by_camera, all_cluster_labels, camera_intrinsics)
        print("Generated 2D cluster visualization images for all cameras")

    # 3. Organize data by frame, prepare for triangulation
    frames_data = defaultdict(dict)

    for cam_id, cam_data in grouped_2d_data.items():
        for frame, points in cam_data.items():
            if frame not in frames_data:
                frames_data[frame] = {}
            frames_data[frame][cam_id] = points

    print(f"Prepared to process {len(frames_data)} frames of data")

    # 4. Determine best point correspondence
    if use_clustering:
        # Use clustering information to determine point correspondence
        cluster_mapping = find_best_cluster_correspondence(
            frames_data,
            camera_extrinsics,
            camera_intrinsics,
            max_sample_frames=20,
            reproj_error_threshold=reproj_error_threshold
        )

        if not cluster_mapping:
            print("Warning: Failed to determine valid cluster mapping, will use default point order matching")
            swap_points = find_best_point_correspondence(
                frames_data, camera_extrinsics, camera_intrinsics,
                max_sample_frames=20, reproj_error_threshold=reproj_error_threshold
            )
            use_clustering = False  # Fall back to regular matching
    else:
        # Use traditional method to determine point correspondence
        swap_points = find_best_point_correspondence(
            frames_data, camera_extrinsics, camera_intrinsics,
            max_sample_frames=20, reproj_error_threshold=reproj_error_threshold
        )

    # 5. Triangulate all frames using the determined point correspondence strategy
    points_3d = {
        'point1': {},
        'point2': {}
    }
    residuals = {
        'point1': {},
        'point2': {}
    }

    total_processed = 0
    successful_triangulations = 0

    for frame, cam_data in frames_data.items():
        # Get valid cameras
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]

        if len(valid_cameras) < 2:
            continue

        total_processed += 1

        # Prepare point data according to the determined strategy
        if use_clustering and cluster_mapping:
            # Prepare points based on cluster mapping
            points_for_3d1 = {}
            points_for_3d2 = {}

            for cam_id in valid_cameras:
                # Get cluster mapping for current camera
                if cam_id not in cluster_mapping:
                    print(f"Warning: Camera {cam_id} has no cluster mapping, will skip")
                    continue

                # Get point and group information for this camera in current frame
                point1 = cam_data[cam_id]['point1']
                point2 = cam_data[cam_id]['point2']
                group1 = cam_data[cam_id]['group1']
                group2 = cam_data[cam_id]['group2']

                # Skip this camera if no clustering information
                if group1 is None or group2 is None:
                    print(f"Warning: Frame {frame} camera {cam_id} has no cluster labels")
                    continue

                # Assign points according to cluster mapping
                cam_mapping = cluster_mapping[cam_id]

                # Map points to the correct 3D point group
                if cam_mapping[group1] == 0:
                    points_for_3d1[cam_id] = point1
                    points_for_3d2[cam_id] = point2
                else:
                    points_for_3d1[cam_id] = point2
                    points_for_3d2[cam_id] = point1
        else:
            # Use original order or swapped order
            if swap_points:
                # Use swapped order
                points_for_3d1 = {cam_id: cam_data[cam_id]['point2'] for cam_id in valid_cameras}
                points_for_3d2 = {cam_id: cam_data[cam_id]['point1'] for cam_id in valid_cameras}
            else:
                # Use original order
                points_for_3d1 = {cam_id: cam_data[cam_id]['point1'] for cam_id in valid_cameras}
                points_for_3d2 = {cam_id: cam_data[cam_id]['point2'] for cam_id in valid_cameras}

        # Ensure at least two cameras' data for triangulation
        if len(points_for_3d1) < 2 or len(points_for_3d2) < 2:
            continue

        # Triangulation
        point_3d1, error1 = triangulate_point(points_for_3d1, camera_extrinsics, camera_intrinsics)
        point_3d2, error2 = triangulate_point(points_for_3d2, camera_extrinsics, camera_intrinsics)

        # Store valid triangulation results
        valid_result = True

        if np.isnan(point_3d1).any() or error1 > reproj_error_threshold:
            valid_result = False

        if np.isnan(point_3d2).any() or error2 > reproj_error_threshold:
            valid_result = False

        if valid_result:
            points_3d['point1'][frame] = point_3d1
            points_3d['point2'][frame] = point_3d2
            residuals['point1'][frame] = error1
            residuals['point2'][frame] = error2
            successful_triangulations += 1

    print(f"Processed {total_processed} frames, successfully triangulated {successful_triangulations} frames")
    print(f"point1 has {len(points_3d['point1'])} 3D points")
    print(f"point2 has {len(points_3d['point2'])} 3D points")

    if not points_3d['point1'] or not points_3d['point2']:
        print("Error: Triangulation failed, no valid 3D points generated")
        return

    if residuals['point1'] and residuals['point2']:
        print(f"point1 average reprojection error: {np.mean(list(residuals['point1'].values())):.4f}")
        print(f"point2 average reprojection error: {np.mean(list(residuals['point2'].values())):.4f}")

    # 6. Process point cloud data
    point1_array = np.array(list(points_3d['point1'].values()))
    point2_array = np.array(list(points_3d['point2'].values()))

    # Remove outliers
    point1_filtered = remove_outliers(point1_array, outlier_threshold)
    point2_filtered = remove_outliers(point2_array, outlier_threshold)

    print(f"After outlier removal: point1 has {len(point1_filtered)} points, point2 has {len(point2_filtered)} points")

    # 7. Calculate center position of the two points
    center1 = np.mean(point1_filtered, axis=0)
    center2 = np.mean(point2_filtered, axis=0)

    # 8. Calculate distance between two center points
    current_distance = np.linalg.norm(center1 - center2)
    print(f"Current estimated distance between points: {current_distance:.4f}")
    print(f"Actual distance between points: {real_distance:.4f}")

    # Calculate scaling factor
    scale_factor = real_distance / current_distance
    print(f"Scaling factor: {scale_factor:.4f}")

    # 9.1 Show 3D visualization
    if vis_3D:
        # Visualize 3D point cloud and reference points
        print("Generating 3D point cloud visualization...")
        visualize_3d_clusters_with_centers(
            point1_filtered, point2_filtered,
            center1, center2,
            real_distance, current_distance
        )

    # 10. Scale camera extrinsic parameters
    scaled_extrinsics = {}
    for cam_id, extrinsic in camera_extrinsics.items():
        scaled_translation = extrinsic.translation * scale_factor
        scaled_extrinsics[cam_id] = CameraExtrinsic(
            rotation=extrinsic.rotation,  # Keep rotation unchanged
            translation=scaled_translation
        )

    # 11. Scale originally loaded 3D points, not the points obtained from triangulation
    scaled_points_3d = {}
    for frame, point in original_points_3d.items():
        scaled_points_3d[frame] = point * scale_factor

    # 12. Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    save_optimization_results(
        output_dir,
        scaled_extrinsics,
        camera_intrinsics,  # Intrinsic parameters remain unchanged
        scaled_points_3d,
        format_type,
        extrinsic_file_name=extrinsic_file_name,
        intrinsic_file_name=intrinsic_file_name,
        points_file_name=points_file_name
    )

    print("===== Data Processing Complete =====")


def find_best_point_correspondence(
        frames_data: Dict[int, Dict[str, Dict[str, np.ndarray]]],
        camera_extrinsics: Dict[str, CameraExtrinsic],
        camera_intrinsics: Dict[str, CameraIntrinsic],
        max_sample_frames: int = 20,
        reproj_error_threshold: float = 5.0
) -> bool:
    """
    Determine the best point correspondence using a few sample frames - Improved version
    Test different point correspondences between cameras, not simply swapping points within the same camera
    """
    # Sort frames by number of shared cameras in descending order
    frames_with_camera_counts = []
    for frame, cam_data in frames_data.items():
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]
        if len(valid_cameras) >= 2:
            frames_with_camera_counts.append((frame, len(valid_cameras)))

    # Sort by camera count in descending order
    frames_with_camera_counts.sort(key=lambda x: x[1], reverse=True)

    # Select the top N frames as samples
    sample_frames = [frame for frame, _ in frames_with_camera_counts[:max_sample_frames]]

    if not sample_frames:
        print("Warning: Not enough sample frames found")
        return False  # Default to not swap

    print(f"Using {len(sample_frames)} sample frames to determine point correspondence")

    # Try two possible correspondence combinations
    errors_combination1 = {'total': 0,
                           'count': 0}  # Combination 1: All cameras' point1 match each other, point2 match each other
    errors_combination2 = {'total': 0,
                           'count': 0}  # Combination 2: First camera remains unchanged, other cameras swap point1 and point2

    for frame in sample_frames:
        cam_data = frames_data[frame]
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]

        if len(valid_cameras) < 2:
            continue

        # Combination 1: All cameras' point1 match each other, all cameras' point2 match each other
        points_match1_1 = {cam_id: cam_data[cam_id]['point1'] for cam_id in valid_cameras}
        points_match1_2 = {cam_id: cam_data[cam_id]['point2'] for cam_id in valid_cameras}

        point3d_match1_1, error_match1_1 = triangulate_point(points_match1_1, camera_extrinsics, camera_intrinsics)
        point3d_match1_2, error_match1_2 = triangulate_point(points_match1_2, camera_extrinsics, camera_intrinsics)

        # Combination 2: Keep first camera unchanged, swap point order for other cameras
        first_cam = valid_cameras[0]
        other_cams = valid_cameras[1:]

        points_match2_1 = {first_cam: cam_data[first_cam]['point1']}
        points_match2_2 = {first_cam: cam_data[first_cam]['point2']}

        for cam_id in other_cams:
            # Swap point order for other cameras
            points_match2_1[cam_id] = cam_data[cam_id]['point2']  # Note the swap here
            points_match2_2[cam_id] = cam_data[cam_id]['point1']  # Note the swap here

        point3d_match2_1, error_match2_1 = triangulate_point(points_match2_1, camera_extrinsics, camera_intrinsics)
        point3d_match2_2, error_match2_2 = triangulate_point(points_match2_2, camera_extrinsics, camera_intrinsics)

        # Calculate and compare total error for both combinations
        valid_combo1 = (not np.isnan(point3d_match1_1).any() and not np.isnan(point3d_match1_2).any() and
                        error_match1_1 < reproj_error_threshold and error_match1_2 < reproj_error_threshold)

        valid_combo2 = (not np.isnan(point3d_match2_1).any() and not np.isnan(point3d_match2_2).any() and
                        error_match2_1 < reproj_error_threshold and error_match2_2 < reproj_error_threshold)

        if valid_combo1:
            errors_combination1['total'] += (error_match1_1 + error_match1_2)
            errors_combination1['count'] += 1

        if valid_combo2:
            errors_combination2['total'] += (error_match2_1 + error_match2_2)
            errors_combination2['count'] += 1

    # Determine best correspondence
    if errors_combination1['count'] == 0 and errors_combination2['count'] == 0:
        print("Warning: Both combinations failed to produce valid triangulation results")
        return False  # Default to not swap

    if errors_combination1['count'] == 0:
        print("Combination 1 produced no valid results, using combination 2")
        return True  # Use combination 2

    if errors_combination2['count'] == 0:
        print("Combination 2 produced no valid results, using combination 1")
        return False  # Use combination 1

    # Calculate average error
    avg_error_combo1 = errors_combination1['total'] / errors_combination1['count']
    avg_error_combo2 = errors_combination2['total'] / errors_combination2['count']

    print(f"Combination 1 average error: {avg_error_combo1:.4f}, Valid frames: {errors_combination1['count']}")
    print(f"Combination 2 average error: {avg_error_combo2:.4f}, Valid frames: {errors_combination2['count']}")

    # Choose the combination with lower average error
    swap_points = avg_error_combo2 < avg_error_combo1

    if swap_points:
        print("Selected combination 2 as best correspondence")
    else:
        print("Selected combination 1 as best correspondence")

    return swap_points


def save_optimization_results(output_dir: str,
                              optimized_camera_extrinsics: Dict,
                              optimized_camera_intrinsics: Dict,
                              optimized_points_3d: Dict,
                              save_format: str = 'json',
                              extrinsic_file_name: str = None,
                              intrinsic_file_name: str = None,
                              points_file_name: str = None
                              ):
    """
    Save optimized camera extrinsic parameters, intrinsic parameters and 3D point cloud data to specified directory

    Args:
        output_dir: Output directory path
        optimized_camera_extrinsics: Dictionary of optimized camera extrinsic parameters
        optimized_camera_intrinsics: Dictionary of optimized camera intrinsic parameters
        optimized_points_3d: Dictionary of optimized 3D point cloud data
        save_format: Save format, options: 'pickle' or 'json'
        extrinsic_file_name: Extrinsic filename (without extension)
        intrinsic_file_name: Intrinsic filename (without extension)
        points_file_name: 3D points filename (without extension)
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    scaled_extrinsics_filename = os.path.join(output_dir, f"scaled_{extrinsic_file_name}.{save_format}")
    scaled_intrinsics_filename = os.path.join(output_dir, f"scaled_{intrinsic_file_name}.{save_format}")
    scaled_points3d_filename = os.path.join(output_dir, f"scaled_{points_file_name}.{save_format}")

    # Choose different save methods based on format
    if save_format.lower() == 'pickle':
        # Save as Pickle format
        # Save camera extrinsic parameters
        with open(scaled_extrinsics_filename, 'wb') as f:
            pickle.dump(optimized_camera_extrinsics, f)

        # Save camera intrinsic parameters
        with open(scaled_intrinsics_filename, 'wb') as f:
            pickle.dump(optimized_camera_intrinsics, f)

        # Save 3D point cloud
        with open(scaled_points3d_filename, 'wb') as f:
            pickle.dump(optimized_points_3d, f)

    elif save_format.lower() == 'json':
        # Save as JSON format
        # Process camera extrinsic parameters
        json_extrinsics = {}
        for cam_id, extrinsic in optimized_camera_extrinsics.items():
            json_extrinsics[cam_id] = {
                'rotation': extrinsic.rotation.tolist(),
                'translation': extrinsic.translation.tolist()
            }

        with open(scaled_extrinsics_filename, 'w') as f:
            json.dump(json_extrinsics, f, indent=2)

        # Process camera intrinsic parameters
        json_intrinsics = {}
        for cam_id, intrinsic in optimized_camera_intrinsics.items():
            json_intrinsics[cam_id] = {
                'camera_matrix': intrinsic.camera_matrix.tolist(),
                'dist_coeffs': intrinsic.dist_coeffs.tolist(),
                'image_size': intrinsic.image_size
            }

        with open(scaled_intrinsics_filename, 'w') as f:
            json.dump(json_intrinsics, f, indent=2)

        # Process 3D point cloud
        json_points3d = {}
        for frame, point in optimized_points_3d.items():
            json_points3d[str(frame)] = point.tolist()

        with open(scaled_points3d_filename, 'w') as f:
            json.dump(json_points3d, f, indent=2)
    else:
        raise ValueError(f"Unsupported save format: {save_format}, supported formats: 'pickle' and 'json'")

    print(f"scaled save dir: {output_dir}")
    print(f"  - camera extrinsic: {os.path.basename(scaled_extrinsics_filename)}")
    print(f"  - camera intrinsic: {os.path.basename(scaled_intrinsics_filename)}")
    print(f"  - 3D points: {os.path.basename(scaled_points3d_filename)}")


def visualize_2d_clusters(all_points_by_camera, all_cluster_labels, camera_intrinsics=None):
    """
    Create and directly display 2D cluster visualization for each camera

    Args:
        all_points_by_camera: All 2D points grouped by camera ID
        all_cluster_labels: Cluster labels grouped by camera ID
        camera_intrinsics: Camera intrinsic parameter dictionary (optional), used to get image size
    """
    import matplotlib.pyplot as plt

    # Create cluster visualization for each camera
    for cam_id, points in all_points_by_camera.items():
        if cam_id not in all_cluster_labels or len(all_cluster_labels[cam_id]) == 0:
            print(f"Camera {cam_id} has no cluster labels, skipping visualization")
            continue

        if len(points) < 2:
            print(f"Camera {cam_id} has only {len(points)} points, not enough for visualization")
            continue

        # Get cluster labels for this camera
        cluster_labels = all_cluster_labels[cam_id]

        # Create new figure
        plt.figure(figsize=(10, 8))

        # Convert points to numpy array
        points_array = np.array(points)

        # Plot points of different clusters (with different colors)
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # Color list

        for cluster_id in sorted(set(cluster_labels)):
            # Get all points of current cluster
            cluster_points = points_array[np.array(cluster_labels) == cluster_id]

            # Plot points
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=colors[cluster_id % len(colors)],
                label=f'Group {cluster_id}',
                alpha=0.7,
                s=30
            )

        # Set figure properties
        plt.title(f'2D Point Clustering for Camera {cam_id}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # If camera intrinsic parameters provided, set appropriate boundaries
        if camera_intrinsics and cam_id in camera_intrinsics:
            if hasattr(camera_intrinsics[cam_id], 'image_size'):
                img_size = camera_intrinsics[cam_id].image_size
                plt.xlim(0, img_size[0])
                plt.ylim(img_size[1], 0)  # Note: Y-axis reversed, origin at top-left

        plt.show()  # Directly display instead of saving
        print(f"Displayed cluster visualization for camera {cam_id}")


def visualize_3d_clusters_with_centers(point1_filtered, point2_filtered, center1, center2,
                                       real_distance, current_distance):
    """
    Visualize two groups of 3D point clouds and their center points, showing the distance between centers

    Args:
        point1_filtered: First group of filtered 3D points
        point2_filtered: Second group of filtered 3D points
        center1: Center of first group of points
        center2: Center of second group of points
        real_distance: Actual distance
        current_distance: Calculated current distance
    """
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot first group of points (red)
    ax.scatter(point1_filtered[:, 0], point1_filtered[:, 1], point1_filtered[:, 2],
               c='red', s=20, alpha=0.6, label='Point Group 1')

    # Plot second group of points (blue)
    ax.scatter(point2_filtered[:, 0], point2_filtered[:, 1], point2_filtered[:, 2],
               c='blue', s=20, alpha=0.6, label='Point Group 2')

    # Plot center points (using larger points and different colors)
    ax.scatter(center1[0], center1[1], center1[2],
               c='darkred', s=100, label='Center Point 1', edgecolor='black')
    ax.scatter(center2[0], center2[1], center2[2],
               c='darkblue', s=100, label='Center Point 2', edgecolor='black')

    # Plot line between center points
    ax.plot([center1[0], center2[0]],
            [center1[1], center2[1]],
            [center1[2], center2[2]],
            'g-', linewidth=2, label=f'Distance: {current_distance:.4f}')

    # Set legend, title and labels
    ax.legend(loc='upper right')
    ax.set_title(
        f'3D Point Cloud Cluster Visualization\nActual Distance: {real_distance:.4f}, Calculated Distance: {current_distance:.4f}, Ratio: {real_distance / current_distance:.4f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set appropriate axis range
    all_points = np.vstack([point1_filtered, point2_filtered])
    max_range = np.max([
        np.max(all_points[:, 0]) - np.min(all_points[:, 0]),
        np.max(all_points[:, 1]) - np.min(all_points[:, 1]),
        np.max(all_points[:, 2]) - np.min(all_points[:, 2])
    ])
    mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) * 0.5
    mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) * 0.5
    mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) * 0.5

    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)

    # Add text annotation for center point coordinates
    ax.text(center1[0], center1[1], center1[2],
            f'({center1[0]:.3f}, {center1[1]:.3f}, {center1[2]:.3f})',
            color='darkred', fontsize=9)
    ax.text(center2[0], center2[1], center2[2],
            f'({center2[0]:.3f}, {center2[1]:.3f}, {center2[2]:.3f})',
            color='darkblue', fontsize=9)

    # Display figure
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    optimization_dir = r'./data_file_4_with_ref/results'
    ref_cam_info_path = r'./data_file_4_with_ref/reference_points/ref_data/cam_info.csv'  # csv
    ref_data2d_path = r'./data_file_4_with_ref/reference_points/ref_data/data2d_distorted.csv'  # csv
    real_distance = 23.61 / 1000  # meters between 2 reference points, mm->m
    output_dir = optimization_dir
    extrinsic_file_name = "optimized_camera_extrinsics"
    intrinsic_file_name = "optimized_camera_intrinsics"
    points_file_name = "optimized_points_3d"
    vis_2D = True
    vis_3D = True

    process_data(
        optimization_dir,
        ref_cam_info_path,
        ref_data2d_path,
        real_distance,
        output_dir,
        extrinsic_file_name=extrinsic_file_name,
        intrinsic_file_name=intrinsic_file_name,
        points_file_name=points_file_name,
        vis_2D=vis_2D,
        vis_3D=vis_3D
    )