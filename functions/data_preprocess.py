import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import yaml
import os
import cv2
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CameraIntrinsic:
    """Camera Intrinsic"""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]


@dataclass
class CameraData:
    """Store camera frame info"""
    frames: List[int]
    x_coords: List[float]
    y_coords: List[float]
    velocities: Optional[List[float]] = None  # for velocity
    is_valid: Optional[List[bool]] = None  # for valid points masks


def load_camera_intrinsics(cam_id: str, yaml_dir: str) -> CameraIntrinsic:
    yaml_path = os.path.join(yaml_dir, f"{cam_id}.yaml")
    with open(yaml_path, 'r') as f:
        cam_info = yaml.safe_load(f)

    required_keys = ['camera_matrix', 'distortion_coefficients', 'image_width', 'image_height']
    for key in required_keys:
        if key not in cam_info:
            raise KeyError(f"No key in YAML file: {key}")

    camera_matrix = np.array(cam_info['camera_matrix']['data']).reshape(3, 3)
    dist_coeffs = np.array(cam_info['distortion_coefficients']['data'])
    image_size = (cam_info['image_width'], cam_info['image_height'])

    return CameraIntrinsic(camera_matrix, dist_coeffs, image_size)


def undistort_points(points_xy: np.ndarray, cam_intrinsic: CameraIntrinsic) -> Tuple[np.ndarray, int]:
    """
    Undistort points and filter out those outside image boundaries
    Returns undistorted points and count of out-of-bounds points
    """
    if len(points_xy) == 0:
        return np.array([]), 0

    points_xy = points_xy.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        points_xy.astype(np.float32),
        cam_intrinsic.camera_matrix,
        cam_intrinsic.dist_coeffs,
        P=cam_intrinsic.camera_matrix
    )
    undistorted = undistorted.reshape(-1, 2)

    # Filter out points outside image boundaries
    in_bounds_mask = np.array([
        is_point_in_bounds(point, cam_intrinsic.image_size)
        for point in undistorted
    ])

    out_of_bounds_count = len(undistorted) - np.sum(in_bounds_mask)
    filtered_points = undistorted[in_bounds_mask]

    return filtered_points, out_of_bounds_count


def count_valid_points(camera_data: CameraData) -> int:
    valid_coords_count = 0
    for x, y in zip(camera_data.x_coords, camera_data.y_coords):
        if not (np.isnan(x) or np.isnan(y)):
            valid_coords_count += 1

    return valid_coords_count


def get_unified_frame_range(all_camera_data: Dict[str, CameraData]) -> Tuple[int, int, Set[int]]:
    """
    确定所有相机的统一帧范围

    起始帧: 首次有至少两台相机有有效数据的帧
    结束帧: 最后一个有至少两台相机有有效数据的帧

    返回: (起始帧, 结束帧, 所有介于两者之间的帧集合)
    """
    # 收集所有帧
    all_frames = set()
    for cam_data in all_camera_data.values():
        all_frames.update(cam_data.frames)

    # 帧号排序
    all_frames_sorted = sorted(all_frames)

    # 找到起始帧（首次有至少两台相机有有效数据的帧）
    start_frame = None
    for frame in all_frames_sorted:
        cameras_with_data = 0
        for cam_data in all_camera_data.values():
            if frame in cam_data.frames:
                idx = cam_data.frames.index(frame)
                if not (np.isnan(cam_data.x_coords[idx]) or np.isnan(cam_data.y_coords[idx])):
                    cameras_with_data += 1

        if cameras_with_data >= 2:
            start_frame = frame
            break

    if start_frame is None:
        start_frame = all_frames_sorted[0]  # 如果没有找到，则使用第一帧

    # 找到结束帧（最后一个有至少两台相机有有效数据的帧）
    end_frame = None
    for frame in reversed(all_frames_sorted):
        cameras_with_data = 0
        for cam_data in all_camera_data.values():
            if frame in cam_data.frames:
                idx = cam_data.frames.index(frame)
                if not (np.isnan(cam_data.x_coords[idx]) or np.isnan(cam_data.y_coords[idx])):
                    cameras_with_data += 1

        if cameras_with_data >= 2:
            end_frame = frame
            break

    if end_frame is None:
        end_frame = all_frames_sorted[-1]  # 如果没有找到，则使用最后一帧

    # 创建起始帧到结束帧之间的所有帧集合
    valid_frame_range = set(range(start_frame, end_frame + 1))

    return start_frame, end_frame, valid_frame_range


def standardize_camera_data(camera_data: Dict[str, CameraData], frame_range: Set[int]) -> Dict[str, CameraData]:
    """
    标准化所有相机数据，确保它们都有相同的帧范围
    对于缺失的帧，用NaN填充
    """
    # 从frame_range创建有序帧列表
    frames_list = sorted(list(frame_range))

    standardized_data = {}

    for cam_id, cam_data in camera_data.items():
        # 创建帧号到数据的映射
        frame_to_data = {}
        for i, frame in enumerate(cam_data.frames):
            frame_to_data[frame] = (cam_data.x_coords[i], cam_data.y_coords[i])

        # 为每一帧创建标准化的数据
        new_x_coords = []
        new_y_coords = []

        for frame in frames_list:
            if frame in frame_to_data:
                new_x_coords.append(frame_to_data[frame][0])
                new_y_coords.append(frame_to_data[frame][1])
            else:
                new_x_coords.append(np.nan)
                new_y_coords.append(np.nan)

        # 创建新的CameraData对象
        standardized_data[cam_id] = CameraData(
            frames=frames_list,
            x_coords=new_x_coords,
            y_coords=new_y_coords
        )

    return standardized_data


def filter_frames_with_multiple_cameras_and_fill_nan(all_camera_data: Dict[str, CameraData]) -> Dict[str, CameraData]:
    """
    找到至少有两台相机有有效数据的帧，对其余帧的坐标设为NaN
    返回处理后的相机数据字典
    """
    # 收集所有帧
    all_frames = set()
    for cam_data in all_camera_data.values():
        all_frames.update(cam_data.frames)

    # 找到有效帧（至少两台相机有数据的帧）
    valid_frames = set()
    for frame in all_frames:
        cameras_with_data = 0
        for cam_data in all_camera_data.values():
            if frame in cam_data.frames:
                idx = cam_data.frames.index(frame)
                if not (np.isnan(cam_data.x_coords[idx]) or np.isnan(cam_data.y_coords[idx])):
                    cameras_with_data += 1

        if cameras_with_data >= 2:
            valid_frames.add(frame)

    # 创建新的相机数据字典，对不在valid_frames中的帧坐标设为NaN
    filtered_camera_data = {}
    for cam_id, cam_data in all_camera_data.items():
        # 创建相同长度的新列表
        new_x_coords = []
        new_y_coords = []

        # 保留原有的velocities和is_valid（如果存在）
        new_velocities = cam_data.velocities.copy() if cam_data.velocities else None
        new_is_valid = cam_data.is_valid.copy() if cam_data.is_valid else None

        # 对每一帧进行处理
        for i, frame in enumerate(cam_data.frames):
            if frame in valid_frames:
                # 有效帧保留原始坐标
                new_x_coords.append(cam_data.x_coords[i])
                new_y_coords.append(cam_data.y_coords[i])
            else:
                # 无效帧设为NaN
                new_x_coords.append(np.nan)
                new_y_coords.append(np.nan)
                # 如果is_valid存在，将无效帧标记为False
                if new_is_valid and i < len(new_is_valid):
                    new_is_valid[i] = False

        # 创建新的CameraData对象
        filtered_camera_data[cam_id] = CameraData(
            frames=cam_data.frames.copy(),  # 保留所有原始帧
            x_coords=new_x_coords,
            y_coords=new_y_coords,
            velocities=new_velocities,
            is_valid=new_is_valid
        )

    print(f"Identified {len(valid_frames)} frames with at least 2 cameras with valid data")
    return filtered_camera_data


def calculate_velocity(frames: List[int], x_coords: List[float], y_coords: List[float]) -> List[float]:
    """speed between frames"""
    velocities = []
    for i in range(len(frames) - 1):
        if (np.isnan(x_coords[i]) or np.isnan(y_coords[i]) or
                np.isnan(x_coords[i + 1]) or np.isnan(y_coords[i + 1])):
            velocities.append(np.nan)
            continue

        dt = frames[i + 1] - frames[i]
        if dt == 0:
            velocities.append(np.nan)
            continue

        dx = x_coords[i + 1] - x_coords[i]
        dy = y_coords[i + 1] - y_coords[i]
        velocity = np.sqrt(dx * dx + dy * dy) / dt
        velocities.append(velocity)

    velocities.append(np.nan)
    return velocities


def detect_anomalies(camera_data: CameraData, velocity_threshold: float = 10.0) -> List[bool]:
    """detect outliers based on speed and spatial"""
    '''
    outlier: speed/spatial error greater than velocity_threshold*std
    '''
    velocities = calculate_velocity(camera_data.frames,
                                    camera_data.x_coords,
                                    camera_data.y_coords)
    camera_data.velocities = velocities

    valid_velocities = [v for v in velocities if not np.isnan(v)]
    if not valid_velocities:
        return [False] * len(camera_data.frames)

    velocity_mean = np.mean(valid_velocities)
    velocity_std = np.std(valid_velocities)

    is_valid = []
    for i in range(len(camera_data.frames)):
        point_valid = True

        if (np.isnan(camera_data.x_coords[i]) or
                np.isnan(camera_data.y_coords[i])):
            point_valid = False

        elif not np.isnan(velocities[i]):
            if abs(velocities[i] - velocity_mean) > velocity_threshold * velocity_std:
                point_valid = False

        if point_valid and i > 0 and i < len(camera_data.frames) - 1:
            prev_valid_idx = next_valid_idx = None

            for j in range(i - 1, -1, -1):
                if not np.isnan(camera_data.x_coords[j]) and not np.isnan(camera_data.y_coords[j]):
                    prev_valid_idx = j
                    break

            for j in range(i + 1, len(camera_data.frames)):
                if not np.isnan(camera_data.x_coords[j]) and not np.isnan(camera_data.y_coords[j]):
                    next_valid_idx = j
                    break

            if prev_valid_idx is not None and next_valid_idx is not None:
                expected_x = (camera_data.x_coords[prev_valid_idx] +
                              camera_data.x_coords[next_valid_idx]) / 2
                expected_y = (camera_data.y_coords[prev_valid_idx] +
                              camera_data.y_coords[next_valid_idx]) / 2

                actual_x = camera_data.x_coords[i]
                actual_y = camera_data.y_coords[i]

                spatial_error = np.sqrt((expected_x - actual_x) ** 2 +
                                        (expected_y - actual_y) ** 2)

                if spatial_error > velocity_threshold * velocity_std:
                    point_valid = False

        is_valid.append(point_valid)

    return is_valid


def apply_ransac_2d(camera_data: CameraData) -> Tuple[List[float], List[float]]:
    """RANSAC 2d trajectory fit"""
    frames_array = np.array(camera_data.frames)
    x_array = np.array(camera_data.x_coords)
    y_array = np.array(camera_data.y_coords)

    valid_points = ~np.isnan(x_array) & ~np.isnan(y_array) & camera_data.is_valid
    valid_frames = frames_array[valid_points]
    valid_x = x_array[valid_points]
    valid_y = y_array[valid_points]

    if len(valid_frames) > 2:
        try:
            ransac = RANSACRegressor(random_state=42, max_trials=10000)
            ransac.fit(valid_x.reshape(-1, 1), valid_y)
            inlier_mask = ransac.inlier_mask_

            new_x = np.full_like(x_array, np.nan)
            new_y = np.full_like(y_array, np.nan)

            valid_indices = np.where(valid_points)[0]
            inlier_indices = valid_indices[inlier_mask]

            new_x[inlier_indices] = x_array[inlier_indices]
            new_y[inlier_indices] = y_array[inlier_indices]

            return new_x.tolist(), new_y.tolist()
        except:
            return camera_data.x_coords, camera_data.y_coords

    return camera_data.x_coords, camera_data.y_coords


def is_point_in_bounds(point, image_size):
    """Check if a point is within image boundaries"""
    x, y = point
    width, height = image_size
    return 0 <= x < width and 0 <= y < height


def load_and_process_data(data2d_path: str, cam_info_path: str, yaml_dir: str, do_undistort: bool = True):
    # read data files
    data2d = pd.read_csv(data2d_path)
    cam_info = pd.read_csv(cam_info_path)

    # cam number to id
    camn_to_id = dict(zip(cam_info['camn'], cam_info['cam_id']))

    # save camera data
    camera_intrinsics = {}  # use cam_id as key
    camera_data = {}  # use cam_id as key
    stats = {
        'original': {},
        'undistorted': {},
        'in_bounds': {},  # New stat for points within image boundaries
        'standardized': {},
        'velocity': {},
        'ransac': {},
        'multicam': {},  # 最终多相机过滤后的统计
        'initial_multicam': {},
        're_standardized': {}
    }

    # Only perform undistortion if do_undistort is True
    if do_undistort:
        # undistort for all cameras
        for cam_id in cam_info['cam_id'].unique():
            try:
                # load intrinsic
                cam_intrinsic = load_camera_intrinsics(cam_id, yaml_dir)
                camera_intrinsics[cam_id] = cam_intrinsic

                # get all camn for 1 cam_id
                cam_camns = cam_info[cam_info['cam_id'] == cam_id]['camn']

                # obtain data for the camera
                cam_mask = data2d['camn'].isin(cam_camns)
                cam_data = data2d[cam_mask]

                if len(cam_data) > 0:
                    # get valid data only
                    valid_mask = ~(cam_data['x'].isna() | cam_data['y'].isna())
                    valid_points = cam_data[valid_mask][['x', 'y']].values

                    if len(valid_points) > 0:
                        # undistort and check bounds
                        undistorted_points, out_of_bounds_count = undistort_points(valid_points, cam_intrinsic)

                        # Store original valid points count for this camera
                        if cam_id not in stats['undistorted']:
                            stats['undistorted'][cam_id] = 0
                        stats['undistorted'][cam_id] += len(valid_points)

                        # Store in-bounds count
                        if cam_id not in stats['in_bounds']:
                            stats['in_bounds'][cam_id] = 0
                        stats['in_bounds'][cam_id] += len(undistorted_points)

                        # Create a new mask for the valid indices that are also in bounds
                        valid_indices = cam_data[valid_mask].index

                        # We need to map the undistorted points back to the original data
                        # First, create a dataframe with the undistorted points
                        if len(undistorted_points) > 0:
                            undistorted_df = pd.DataFrame(undistorted_points, columns=['x', 'y'])

                            # Create a mapping from valid indices to new indices
                            valid_indices_list = valid_indices.tolist()
                            index_mapping = {i: idx for i, idx in
                                             enumerate(valid_indices_list[:len(undistorted_points)])}

                            # Update data2d with undistorted and in-bounds points
                            for i, (x, y) in enumerate(undistorted_points):
                                original_idx = index_mapping[i]
                                data2d.loc[original_idx, 'x'] = x
                                data2d.loc[original_idx, 'y'] = y

                            # Set out-of-bounds points to NaN
                            out_of_bounds_indices = valid_indices_list[len(undistorted_points):]
                            for idx in out_of_bounds_indices:
                                data2d.loc[idx, 'x'] = np.nan
                                data2d.loc[idx, 'y'] = np.nan

            except Exception as e:
                print(f"Warning: Failed to process camera {cam_id}: {e}")
                continue
    else:
        # If not undistorting, load the camera intrinsics anyway (might be needed later)
        for cam_id in cam_info['cam_id'].unique():
            try:
                camera_intrinsics[cam_id] = load_camera_intrinsics(cam_id, yaml_dir)
            except Exception as e:
                print(f"Warning: Failed to load intrinsics for camera {cam_id}: {e}")

    # group by cam_id
    # get stats of orig data and undistorted data
    original_data = pd.read_csv(data2d_path)
    for camn, cam_id in camn_to_id.items():
        # get stats of orig data
        cam_original = original_data[original_data['camn'] == camn]
        if cam_id not in stats['original']:
            stats['original'][cam_id] = 0
        stats['original'][cam_id] += sum(1 for x, y in zip(cam_original['x'], cam_original['y'])
                                         if not (np.isnan(x) or np.isnan(y)))

        # get undistorted data (or original data if not undistorting)
        cam_data = data2d[data2d['camn'] == camn]
        if cam_id not in camera_data:
            camera_data[cam_id] = CameraData(
                frames=[],
                x_coords=[],
                y_coords=[]
            )

        # put all data for one camera together
        camera_data[cam_id].frames.extend(cam_data['frame'].tolist())
        camera_data[cam_id].x_coords.extend(cam_data['x'].tolist())
        camera_data[cam_id].y_coords.extend(cam_data['y'].tolist())

    # sort data
    for cam_id in camera_data:
        # sort by frames
        sorted_indices = np.argsort(camera_data[cam_id].frames)
        camera_data[cam_id].frames = [camera_data[cam_id].frames[i] for i in sorted_indices]
        camera_data[cam_id].x_coords = [camera_data[cam_id].x_coords[i] for i in sorted_indices]
        camera_data[cam_id].y_coords = [camera_data[cam_id].y_coords[i] for i in sorted_indices]

    # 获取统一的帧范围
    start_frame, end_frame, frame_range = get_unified_frame_range(camera_data)
    print(f"Using unified frame range: {start_frame} to {end_frame} ({len(frame_range)} frames)")

    # 标准化所有相机数据以包含相同的帧范围
    camera_data = standardize_camera_data(camera_data, frame_range)

    # 更新标准化后的统计数据
    for cam_id in camera_data:
        stats['standardized'][cam_id] = count_valid_points(camera_data[cam_id])

    # 添加：在处理前先筛选多相机帧
    camera_data = filter_frames_with_multiple_cameras_and_fill_nan(camera_data)

    # 添加：更新多相机初步筛选后的统计
    for cam_id in camera_data:
        stats['initial_multicam'][cam_id] = count_valid_points(camera_data[cam_id])

    # data process for all the cameras - 流程改动：先处理每个相机的数据，最后才筛选多相机帧
    for cam_id in camera_data:
        # speed filter
        camera_data[cam_id].is_valid = detect_anomalies(camera_data[cam_id])
        # 计算速度过滤后仍然有效的点数
        valid_count = sum(1 for i, valid in enumerate(camera_data[cam_id].is_valid)
                          if valid and not (np.isnan(camera_data[cam_id].x_coords[i])
                                            or np.isnan(camera_data[cam_id].y_coords[i])))
        stats['velocity'][cam_id] = valid_count

        # RANSAC
        ransac_x, ransac_y = apply_ransac_2d(camera_data[cam_id])
        camera_data[cam_id].x_coords = ransac_x
        camera_data[cam_id].y_coords = ransac_y
        stats['ransac'][cam_id] = sum(1 for x, y in zip(ransac_x, ransac_y)
                                      if not (np.isnan(x) or np.isnan(y)))

    # 最后，应用多相机数据过滤
    camera_data = filter_frames_with_multiple_cameras_and_fill_nan(camera_data)
    # 更新多相机筛选后的统计数据
    for cam_id in camera_data:
        stats['multicam'][cam_id] = count_valid_points(camera_data[cam_id])

    # 获取统一的帧范围
    start_frame, end_frame, frame_range = get_unified_frame_range(camera_data)
    print(f"Using unified frame range: {start_frame} to {end_frame} ({len(frame_range)} frames)")
    # 标准化所有相机数据以包含相同的帧范围
    camera_data = standardize_camera_data(camera_data, frame_range)
    for cam_id in camera_data:
        stats['re_standardized'][cam_id] = count_valid_points(camera_data[cam_id])

    # If not undistorting, set stats to match original data
    if not do_undistort:
        for cam_id in stats['original'].keys():
            stats['undistorted'][cam_id] = stats['original'][cam_id]
            stats['in_bounds'][cam_id] = stats['original'][cam_id]

    return camera_data, camn_to_id, stats, camera_intrinsics


def data_preprocess(data2d_path: str, cam_info_path: str, yaml_dir: str, do_undistort: bool = True):
    """
    Main data preprocessing function with optional undistortion

    Parameters:
    data2d_path (str): Path to the 2D data CSV file
    cam_info_path (str): Path to the camera info CSV file
    yaml_dir (str): Directory containing the camera intrinsic calibration YAML files
    do_undistort (bool): Whether to perform undistortion (default: True)

    Returns:
    Tuple: Processed camera data, camn to cam_id mapping, and camera intrinsics
    """
    # 1. 加载和处理数据
    camera_data, camn_to_id, stats, camera_intrinsics = load_and_process_data(
        data2d_path, cam_info_path, yaml_dir, do_undistort)

    # 更新打印格式，包含in_bounds列和undistortion状态
    undistort_status = "enabled" if do_undistort else "disabled"
    print(f"\nData process summary (Undistortion: {undistort_status}):")
    print("=" * 155)
    print(f"{'Camera ID':<20} {'OrigPoints':<15} {'UndistortedPts':<15} {'InBoundsPts':<15} {'Standardized':<15} "
          f"{'InitMulticamFilter':<20}{'SpeedFilter':<15} {'RANSAC':<15} {'MulticamFilter':<15} {'ReStandardized':<15}")
    print("-" * 155)

    for cam_id in stats['original'].keys():
        print(f"{cam_id:<20} {stats['original'][cam_id]:<15} "
              f"{stats['undistorted'][cam_id]:<15} {stats['in_bounds'][cam_id]:<15} {stats['standardized'][cam_id]:<15} "
              f"{stats['initial_multicam'].get(cam_id, 0):<20} {stats['velocity'][cam_id]:<15} {stats['ransac'][cam_id]:<15} "
              f"{stats['multicam'].get(cam_id, 0):<15} {stats['re_standardized'].get(cam_id, 0):<15}")

    # 计算有多少帧至少有两台相机有有效数据
    valid_frames = set()
    all_frames = set()
    for cam_data in camera_data.values():
        all_frames.update(cam_data.frames)

    for frame in all_frames:
        cameras_with_data = 0
        for cam_data in camera_data.values():
            if frame in cam_data.frames:
                idx = cam_data.frames.index(frame)
                if not (np.isnan(cam_data.x_coords[idx]) or np.isnan(cam_data.y_coords[idx])):
                    cameras_with_data += 1

        if cameras_with_data >= 2:
            valid_frames.add(frame)

    print(f"Finally, there are {len(valid_frames)} frames in at least 2 cameras")

    # Only print out-of-bounds stats if undistortion was performed
    if do_undistort:
        print("\nOut-of-bounds points after undistortion:")
        print(f"{'Camera ID':<20} {'Out-of-bounds Pts':<20} {'Percentage'}")
        print("-" * 55)
        for cam_id in stats['original'].keys():
            if cam_id in stats['undistorted'] and cam_id in stats['in_bounds']:
                out_of_bounds = stats['undistorted'][cam_id] - stats['in_bounds'][cam_id]
                if stats['undistorted'][cam_id] > 0:
                    percentage = (out_of_bounds / stats['undistorted'][cam_id]) * 100
                else:
                    percentage = 0
                print(f"{cam_id:<20} {out_of_bounds:<20} {percentage:.2f}%")

    return camera_data, camn_to_id, camera_intrinsics


'''example'''
if __name__ == "__main__":
    base_dir = "../data_file"
    data2d_path = os.path.join(base_dir, "20241017_164418/data2d_distorted.csv")
    cam_info_path = os.path.join(base_dir, "20241017_164418/cam_info.csv")
    yaml_dir = os.path.join(base_dir, "intrinsic_calibrations")

    # Example with undistortion enabled (default)
    processed_data_with_undistort, camn_to_id, camera_intrinsics = data_preprocess(
        data2d_path, cam_info_path, yaml_dir, do_undistort=True)

    print("\n" + "=" * 80)
    print("Running without undistortion:")

    # Example with undistortion disabled
    processed_data_no_undistort, camn_to_id, camera_intrinsics = data_preprocess(
        data2d_path, cam_info_path, yaml_dir, do_undistort=False)