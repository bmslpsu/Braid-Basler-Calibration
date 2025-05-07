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
    加载优化后的相机外参、内参和3D点云数据

    Args:
        input_dir: 输入目录路径
        extrinsic_file_name: 外参文件名（不含扩展名）
        intrinsic_file_name: 内参文件名（不含扩展名）
        points_file_name: 3D点文件名（不含扩展名）
        format_type: 文件格式，可选 'pickle' 或 'json'

    Returns:
        tuple: (相机外参字典, 相机内参字典, 3D点云字典)
    """
    # 检查文件名并构建完整路径
    extrinsics_filename = os.path.join(input_dir, f"{extrinsic_file_name}.{format_type}")
    intrinsics_filename = os.path.join(input_dir, f"{intrinsic_file_name}.{format_type}")
    points3d_filename = os.path.join(input_dir, f"{points_file_name}.{format_type}")

    # 检查文件是否存在
    if not os.path.exists(extrinsics_filename):
        raise FileNotFoundError(f"no extrinsic: {extrinsics_filename}")
    if not os.path.exists(intrinsics_filename):
        raise FileNotFoundError(f"no intrinsic: {intrinsics_filename}")
    if not os.path.exists(points3d_filename):
        raise FileNotFoundError(f"no 3D points: {points3d_filename}")

    if format_type.lower() == 'pickle':
        # 加载Pickle格式文件
        with open(extrinsics_filename, 'rb') as f:
            camera_extrinsics = pickle.load(f)

        with open(intrinsics_filename, 'rb') as f:
            camera_intrinsics = pickle.load(f)

        with open(points3d_filename, 'rb') as f:
            points_3d = pickle.load(f)

    elif format_type.lower() == 'json':
        # 加载JSON格式文件
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
    加载相机信息和2D数据点 - 增强版，支持点聚类分组

    Args:
        cam_info_path: 相机信息CSV文件路径
        data2d_path: 2D点数据CSV文件路径
        use_clustering: 是否使用聚类将点分为两组
        min_points_for_clustering: 执行聚类所需的最小点数量

    Returns:
        tuple: (cam_id映射字典, 按相机ID分组的2D点数据，包含聚类标签)
    """
    # 加载相机信息
    cam_info = pd.read_csv(cam_info_path)
    cam_id_mapping = dict(zip(cam_info['camn'], cam_info['cam_id']))

    # 加载2D点数据
    data2d = pd.read_csv(data2d_path)

    # 将camn替换为cam_id
    data2d['cam_id'] = data2d['camn'].map(cam_id_mapping)

    # 检查每个相机的每一帧是否有两个点
    point_counts = data2d.groupby(['cam_id', 'frame']).size().reset_index(name='count')
    valid_frames = point_counts[point_counts['count'] == 2][['cam_id', 'frame']]

    # 创建一个联合键用于过滤
    data2d['key'] = data2d['cam_id'].astype(str) + '_' + data2d['frame'].astype(str)
    valid_frames['key'] = valid_frames['cam_id'].astype(str) + '_' + valid_frames['frame'].astype(str)
    valid_keys = set(valid_frames['key'])

    # 过滤数据，只保留有两个点的帧
    filtered_data = data2d[data2d['key'].isin(valid_keys)]

    print(f"原始数据中有 {len(data2d)} 行2D数据")
    print(f"过滤后保留 {len(filtered_data)} 行2D数据（只保留每帧有两个点的数据）")

    # 检查是否有数据ID或对象ID可用于确保点的顺序一致
    has_obj_id = 'obj_id' in filtered_data.columns
    has_data_id = 'data_id' in filtered_data.columns

    if has_obj_id:
        print("检测到obj_id列，将使用它来确保点的顺序一致")
    elif has_data_id:
        print("检测到data_id列，将使用它来确保点的顺序一致")
    else:
        print("警告：未检测到obj_id或data_id列，将使用原始数据顺序")

    # 按相机ID分组
    grouped_data = {}

    # 存储每个相机的所有点，用于后续聚类
    all_points_by_camera = defaultdict(list)
    all_frames_by_camera = defaultdict(list)
    all_point_indices_by_camera = defaultdict(list)  # 存储每个点在原始顺序中的索引

    # 为每个相机的每一帧构建两个点的坐标 - 严格保持原始顺序
    for cam_id, cam_group in filtered_data.groupby('cam_id'):
        grouped_data[cam_id] = {}
        frame_points = {}
        point_index = 0

        for frame, frame_group in cam_group.groupby('frame'):
            # 确保这一帧真的有两个点
            if len(frame_group) == 2:
                # 获取原始顺序的点坐标 - 绝对不排序!

                # 如果有obj_id，按obj_id排序以确保所有相机中的点顺序一致
                if has_obj_id:
                    frame_group = frame_group.sort_values('obj_id')
                # 如果有data_id，按data_id排序
                elif has_data_id:
                    frame_group = frame_group.sort_values('data_id')
                # 否则，保持原始顺序

                # 注意：保持原始数据文件中的顺序非常重要
                points = frame_group[['x', 'y']].values

                frame_points[frame] = {
                    'point1': points[0],  # 第一个点
                    'point2': points[1],  # 第二个点
                    'group1': None,  # 聚类组1
                    'group2': None  # 聚类组2
                }

                # 收集所有点用于后续聚类
                all_points_by_camera[cam_id].append(points[0])
                all_frames_by_camera[cam_id].append(frame)
                all_point_indices_by_camera[cam_id].append((point_index, 0))  # (点索引, 点在帧内的索引)

                all_points_by_camera[cam_id].append(points[1])
                all_frames_by_camera[cam_id].append(frame)
                all_point_indices_by_camera[cam_id].append((point_index, 1))  # (点索引, 点在帧内的索引)

                point_index += 1

        # 只有当至少有一帧有效数据时才添加这个相机
        if frame_points:
            grouped_data[cam_id] = frame_points

    print(f"加载了 {len(grouped_data)} 个相机的有效数据")
    valid_frames_count = sum(len(frames) for frames in grouped_data.values())
    print(f"总共有 {valid_frames_count} 个有效帧（每帧包含两个点）")

    all_cluster_labels = {}

    # 如果启用聚类，为每个相机执行点聚类
    if use_clustering:
        for cam_id, points in all_points_by_camera.items():
            if len(points) >= min_points_for_clustering:
                print(f"对相机 {cam_id} 的 {len(points)} 个点进行聚类...")

                # 将点转换为numpy数组
                points_array = np.array(points)

                # 使用KMeans聚类
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(points_array)

                # 保存该相机的所有聚类标签
                all_cluster_labels[cam_id] = clusters.tolist()

                # 将聚类标签应用回原始数据
                for i, (frame_idx, point_idx) in enumerate(all_point_indices_by_camera[cam_id]):
                    frame = all_frames_by_camera[cam_id][i]
                    cluster_label = clusters[i]

                    # 更新对应帧和点的聚类标签
                    if point_idx == 0:  # point1
                        grouped_data[cam_id][frame]['group1'] = cluster_label
                    else:  # point2
                        grouped_data[cam_id][frame]['group2'] = cluster_label

                # 打印每个聚类的数量统计
                cluster_counts = np.bincount(clusters)
                print(f"相机 {cam_id} 聚类结果: 组0: {cluster_counts[0]} 个点, 组1: {cluster_counts[1]} 个点")
            else:
                print(f"相机 {cam_id} 只有 {len(points)} 个点，不足以进行可靠聚类")

    return cam_id_mapping, grouped_data, all_points_by_camera, all_cluster_labels


def project_3d_to_2d(point_3d: np.ndarray, camera_extrinsic: CameraExtrinsic,
                     camera_intrinsic: CameraIntrinsic) -> np.ndarray:
    """
    将3D点投影到相机平面上得到2D点

    Args:
        point_3d: 3D点坐标 (3,)
        camera_extrinsic: 相机外参
        camera_intrinsic: 相机内参

    Returns:
        np.ndarray: 投影后的2D点坐标 (2,)
    """
    # 将3D点从世界坐标系转换到相机坐标系
    R = camera_extrinsic.rotation
    t = camera_extrinsic.translation

    # 点在相机坐标系中的坐标
    point_cam = R @ point_3d + t

    # 投影到图像平面
    fx = camera_intrinsic.camera_matrix[0, 0]
    fy = camera_intrinsic.camera_matrix[1, 1]
    cx = camera_intrinsic.camera_matrix[0, 2]
    cy = camera_intrinsic.camera_matrix[1, 2]

    # 简化处理，不考虑畸变
    x = fx * point_cam[0] / point_cam[2] + cx
    y = fy * point_cam[1] / point_cam[2] + cy

    return np.array([x, y])


def calculate_reprojection_error(point_3d: np.ndarray, point_2d: np.ndarray,
                                 camera_extrinsic: CameraExtrinsic,
                                 camera_intrinsic: CameraIntrinsic) -> float:
    """
    计算3D点重投影误差

    Args:
        point_3d: 3D点坐标
        point_2d: 实际观测到的2D点坐标
        camera_extrinsic: 相机外参
        camera_intrinsic: 相机内参

    Returns:
        float: 重投影误差（欧氏距离）
    """
    projected_2d = project_3d_to_2d(point_3d, camera_extrinsic, camera_intrinsic)
    error = np.linalg.norm(projected_2d - point_2d)
    return error


def triangulate_point(points_2d: Dict[str, np.ndarray],
                      camera_extrinsics: Dict[str, CameraExtrinsic],
                      camera_intrinsics: Dict[str, CameraIntrinsic]) -> Tuple[np.ndarray, float]:
    """
    使用多个相机的2D点进行三角测量得到3D点 - 优化版本

    Args:
        points_2d: 各相机观测到的2D点字典 {cam_id: [x, y]}
        camera_extrinsics: 各相机外参字典
        camera_intrinsics: 各相机内参字典

    Returns:
        tuple: (三角测量得到的3D点, 平均重投影误差)
    """
    # 构建系数矩阵A进行三角测量
    A = []
    valid_cameras = []

    for cam_id, point_2d in points_2d.items():
        # 检查相机和点是否有效
        if cam_id not in camera_extrinsics or cam_id not in camera_intrinsics:
            continue

        if np.isnan(point_2d).any():
            continue

        # 获取相机参数
        R = camera_extrinsics[cam_id].rotation
        t = camera_extrinsics[cam_id].translation
        K = camera_intrinsics[cam_id].camera_matrix

        # 内参矩阵逆
        K_inv = np.linalg.inv(K)

        # 将2D点转换为归一化坐标
        uv1 = np.array([point_2d[0], point_2d[1], 1.0])
        xy1 = K_inv @ uv1

        # 构建系数矩阵的两行
        P = np.hstack((R, t.reshape(3, 1)))

        A.append(xy1[0] * P[2, :] - P[0, :])
        A.append(xy1[1] * P[2, :] - P[1, :])
        valid_cameras.append(cam_id)

    # 检查是否至少有2个相机的数据 (至少4个约束方程)
    if len(A) < 4:
        return np.array([np.nan, np.nan, np.nan]), np.inf

    # 求解超定方程组
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    point_3d_homo = Vh[-1, :]

    # 齐次坐标转换为3D坐标
    point_3d = point_3d_homo[:3] / point_3d_homo[3]

    # 计算重投影误差
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
    根据聚类标签确定最佳的点对应关系

    Args:
        frames_data: 按帧整理的数据，包含聚类标签
        camera_extrinsics: 相机外参字典
        camera_intrinsics: 相机内参字典
        max_sample_frames: 用于确定对应关系的最大样本帧数
        reproj_error_threshold: 重投影误差阈值

    Returns:
        Dict: 每个相机的每个聚类组应对应到哪个3D点组的映射
    """
    print("使用聚类标签确定点对应关系...")

    # 按照共享相机数量降序排列帧
    frames_with_camera_counts = []
    for frame, cam_data in frames_data.items():
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]
        if len(valid_cameras) >= 2:
            frames_with_camera_counts.append((frame, len(valid_cameras)))

    # 按相机数量降序排序
    frames_with_camera_counts.sort(key=lambda x: x[1], reverse=True)

    # 选择前N个帧作为样本
    sample_frames = [frame for frame, _ in frames_with_camera_counts[:max_sample_frames]]

    if not sample_frames:
        print("警告: 没有找到足够的样本帧")
        return {}  # 默认映射为空

    print(f"使用 {len(sample_frames)} 个样本帧确定聚类对应关系")

    # 创建相机之间的聚类对应关系分析结构
    cluster_correspondence = {}

    # 选择第一个有效的相机作为参考相机
    reference_cameras = set()
    for frame in sample_frames:
        for cam_id, cam_data in frames_data[frame].items():
            if ('group1' in cam_data and cam_data['group1'] is not None and
                    'group2' in cam_data and cam_data['group2'] is not None):
                reference_cameras.add(cam_id)

    if not reference_cameras:
        print("警告: 没有找到有效的聚类数据")
        return {}

    ref_cam_id = sorted(list(reference_cameras))[0]
    print(f"选择相机 {ref_cam_id} 作为参考相机")

    # 初始化相机间聚类对应关系
    for cam_id in reference_cameras:
        if cam_id != ref_cam_id:
            # 初始化四种可能的匹配组合
            cluster_correspondence[cam_id] = {
                # 组合1: 参考相机0匹配此相机0，参考相机1匹配此相机1
                'combination1': {'error': 0, 'count': 0, 'mapping': {0: 0, 1: 1}},
                # 组合2: 参考相机0匹配此相机1，参考相机1匹配此相机0
                'combination2': {'error': 0, 'count': 0, 'mapping': {0: 1, 1: 0}},
            }

    # 使用样本帧测试每个相机的聚类对应关系
    for frame in sample_frames:
        # 检查参考相机在当前帧是否有有效数据
        if ref_cam_id not in frames_data[frame]:
            continue

        ref_cam_data = frames_data[frame][ref_cam_id]

        # 检查参考相机是否有有效的聚类分组
        if ref_cam_data['group1'] is None or ref_cam_data['group2'] is None:
            continue

        # 参考相机的聚类组
        ref_group1 = ref_cam_data['group1']
        ref_group2 = ref_cam_data['group2']

        # 参考相机的点坐标
        ref_point1 = ref_cam_data['point1']
        ref_point2 = ref_cam_data['point2']

        # 针对每个其他相机测试聚类对应关系
        for cam_id, correspondence in cluster_correspondence.items():
            if cam_id not in frames_data[frame]:
                continue

            cam_data = frames_data[frame][cam_id]

            # 检查当前相机是否有有效的聚类分组
            if cam_data['group1'] is None or cam_data['group2'] is None:
                continue

            # 当前相机的聚类组
            cam_group1 = cam_data['group1']
            cam_group2 = cam_data['group2']

            # 当前相机的点坐标
            cam_point1 = cam_data['point1']
            cam_point2 = cam_data['point2']

            # 测试组合1: 参考相机组0匹配当前相机组0，参考相机组1匹配当前相机组1
            # 构建对应关系
            points_combo1_1 = {
                ref_cam_id: ref_point1 if ref_group1 == 0 else ref_point2,
                cam_id: cam_point1 if cam_group1 == 0 else cam_point2
            }
            points_combo1_2 = {
                ref_cam_id: ref_point2 if ref_group1 == 0 else ref_point1,
                cam_id: cam_point2 if cam_group1 == 0 else cam_point1
            }

            # 三角测量
            _, error_combo1_1 = triangulate_point(points_combo1_1, camera_extrinsics, camera_intrinsics)
            _, error_combo1_2 = triangulate_point(points_combo1_2, camera_extrinsics, camera_intrinsics)

            # 测试组合2: 参考相机组0匹配当前相机组1，参考相机组1匹配当前相机组0
            # 构建对应关系
            points_combo2_1 = {
                ref_cam_id: ref_point1 if ref_group1 == 0 else ref_point2,
                cam_id: cam_point2 if cam_group1 == 0 else cam_point1
            }
            points_combo2_2 = {
                ref_cam_id: ref_point2 if ref_group1 == 0 else ref_point1,
                cam_id: cam_point1 if cam_group1 == 0 else cam_point2
            }

            # 三角测量
            _, error_combo2_1 = triangulate_point(points_combo2_1, camera_extrinsics, camera_intrinsics)
            _, error_combo2_2 = triangulate_point(points_combo2_2, camera_extrinsics, camera_intrinsics)

            # 计算总误差
            total_error_combo1 = error_combo1_1 + error_combo1_2
            total_error_combo2 = error_combo2_1 + error_combo2_2

            # 更新对应关系统计
            if total_error_combo1 < reproj_error_threshold * 2:
                correspondence['combination1']['error'] += total_error_combo1
                correspondence['combination1']['count'] += 1

            if total_error_combo2 < reproj_error_threshold * 2:
                correspondence['combination2']['error'] += total_error_combo2
                correspondence['combination2']['count'] += 1

    # 确定每个相机的最佳聚类对应关系
    final_cluster_mapping = {}

    # 参考相机的映射是固定的
    final_cluster_mapping[ref_cam_id] = {0: 0, 1: 1}  # 自己对自己的映射

    for cam_id, correspondence in cluster_correspondence.items():
        combo1 = correspondence['combination1']
        combo2 = correspondence['combination2']

        # 打印统计信息
        print(f"\n相机 {cam_id} 的聚类对应统计:")
        print(f"  组合1 (0->0, 1->1): 总误差 {combo1['error']:.2f}, 有效帧数 {combo1['count']}")
        print(f"  组合2 (0->1, 1->0): 总误差 {combo2['error']:.2f}, 有效帧数 {combo2['count']}")

        # 根据有效帧数和平均误差选择最佳组合
        if combo1['count'] == 0 and combo2['count'] == 0:
            print(f"  警告: 相机 {cam_id} 没有产生有效的对应关系")
            final_cluster_mapping[cam_id] = {0: 0, 1: 1}  # 默认映射
        elif combo1['count'] == 0:
            print(f"  选择组合2作为最佳对应关系 (组合1无效)")
            final_cluster_mapping[cam_id] = combo2['mapping']
        elif combo2['count'] == 0:
            print(f"  选择组合1作为最佳对应关系 (组合2无效)")
            final_cluster_mapping[cam_id] = combo1['mapping']
        else:
            # 计算平均误差
            avg_error1 = combo1['error'] / combo1['count']
            avg_error2 = combo2['error'] / combo2['count']

            if avg_error1 <= avg_error2:
                print(f"  选择组合1作为最佳对应关系 (平均误差: {avg_error1:.2f} vs {avg_error2:.2f})")
                final_cluster_mapping[cam_id] = combo1['mapping']
            else:
                print(f"  选择组合2作为最佳对应关系 (平均误差: {avg_error2:.2f} vs {avg_error1:.2f})")
                final_cluster_mapping[cam_id] = combo2['mapping']

    return final_cluster_mapping


def remove_outliers(points, threshold=2.0):
    """
    移除离群点 - 加强版

    Args:
        points: 点集数组
        threshold: Z-score阈值

    Returns:
        过滤后的点集数组
    """
    if len(points) < 4:  # 如果点太少，则不进行过滤
        return points

    # 计算每个点与中心点的距离
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)

    # 计算z-score并去除离群点
    z_scores = zscore(distances)
    mask = np.abs(z_scores) < threshold

    # 确保至少保留一半的点
    if np.sum(mask) < len(points) / 2:
        # 如果过滤掉了太多点，那么只过滤掉最远的几个点
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
    主要处理函数 - 使用聚类分组的点对应关系策略
    """
    print("===== 开始处理数据 =====")

    # 1. 加载优化结果
    camera_extrinsics, camera_intrinsics, original_points_3d  = load_optimization_results(
        optimization_dir,
        format_type=format_type,
        extrinsic_file_name=extrinsic_file_name,
        intrinsic_file_name=intrinsic_file_name,
        points_file_name=points_file_name
    )

    # 2. 加载2D数据，并进行聚类分组
    cam_id_mapping, grouped_2d_data, all_points_by_camera, all_cluster_labels = load_2d_data(
        cam_info_path,
        data2d_path,
        use_clustering=use_clustering
    )

    # 2.1 新增：生成聚类可视化（如果启用聚类）
    if use_clustering and all_cluster_labels and vis_2D:
        # 获取相机图像尺寸用于设置可视化边界
        visualize_2d_clusters(all_points_by_camera, all_cluster_labels, camera_intrinsics)
        print("已生成所有相机的2D聚类可视化图像")

    # 3. 按帧整理数据，准备三角测量
    frames_data = defaultdict(dict)

    for cam_id, cam_data in grouped_2d_data.items():
        for frame, points in cam_data.items():
            if frame not in frames_data:
                frames_data[frame] = {}
            frames_data[frame][cam_id] = points

    print(f"准备对 {len(frames_data)} 帧数据进行处理")

    # 4. 确定最佳点对应关系
    if use_clustering:
        # 使用聚类信息确定点对应关系
        cluster_mapping = find_best_cluster_correspondence(
            frames_data,
            camera_extrinsics,
            camera_intrinsics,
            max_sample_frames=20,
            reproj_error_threshold=reproj_error_threshold
        )

        if not cluster_mapping:
            print("警告: 未能确定有效的聚类映射关系，将使用默认的点顺序匹配")
            swap_points = find_best_point_correspondence(
                frames_data, camera_extrinsics, camera_intrinsics,
                max_sample_frames=20, reproj_error_threshold=reproj_error_threshold
            )
            use_clustering = False  # 回退到普通匹配
    else:
        # 使用传统方法确定点对应关系
        swap_points = find_best_point_correspondence(
            frames_data, camera_extrinsics, camera_intrinsics,
            max_sample_frames=20, reproj_error_threshold=reproj_error_threshold
        )

    # 5. 使用确定的点对应关系策略对所有帧进行三角测量
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
        # 获取有效相机
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]

        if len(valid_cameras) < 2:
            continue

        total_processed += 1

        # 准备点数据，根据确定的策略
        if use_clustering and cluster_mapping:
            # 基于聚类映射关系准备点
            points_for_3d1 = {}
            points_for_3d2 = {}

            for cam_id in valid_cameras:
                # 获取当前相机的聚类映射
                if cam_id not in cluster_mapping:
                    print(f"警告: 相机 {cam_id} 没有聚类映射，将跳过")
                    continue

                # 获取当前帧该相机的点和群组信息
                point1 = cam_data[cam_id]['point1']
                point2 = cam_data[cam_id]['point2']
                group1 = cam_data[cam_id]['group1']
                group2 = cam_data[cam_id]['group2']

                # 如果没有聚类信息，则跳过这个相机
                if group1 is None or group2 is None:
                    print(f"警告: 帧 {frame} 的相机 {cam_id} 没有聚类标签")
                    continue

                # 根据聚类映射分配点
                cam_mapping = cluster_mapping[cam_id]

                # 将点映射到正确的3D点组
                if cam_mapping[group1] == 0:
                    points_for_3d1[cam_id] = point1
                    points_for_3d2[cam_id] = point2
                else:
                    points_for_3d1[cam_id] = point2
                    points_for_3d2[cam_id] = point1
        else:
            # 使用原始顺序或交换顺序
            if swap_points:
                # 使用交换顺序
                points_for_3d1 = {cam_id: cam_data[cam_id]['point2'] for cam_id in valid_cameras}
                points_for_3d2 = {cam_id: cam_data[cam_id]['point1'] for cam_id in valid_cameras}
            else:
                # 使用原始顺序
                points_for_3d1 = {cam_id: cam_data[cam_id]['point1'] for cam_id in valid_cameras}
                points_for_3d2 = {cam_id: cam_data[cam_id]['point2'] for cam_id in valid_cameras}

        # 确保至少有两个相机的数据用于三角测量
        if len(points_for_3d1) < 2 or len(points_for_3d2) < 2:
            continue

        # 三角测量
        point_3d1, error1 = triangulate_point(points_for_3d1, camera_extrinsics, camera_intrinsics)
        point_3d2, error2 = triangulate_point(points_for_3d2, camera_extrinsics, camera_intrinsics)

        # 存储有效的三角测量结果
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

    print(f"处理了 {total_processed} 帧，成功三角测量了 {successful_triangulations} 帧")
    print(f"point1 有 {len(points_3d['point1'])} 个3D点")
    print(f"point2 有 {len(points_3d['point2'])} 个3D点")

    if not points_3d['point1'] or not points_3d['point2']:
        print("错误: 三角测量失败，没有生成有效的3D点")
        return

    if residuals['point1'] and residuals['point2']:
        print(f"point1 平均重投影误差: {np.mean(list(residuals['point1'].values())):.4f}")
        print(f"point2 平均重投影误差: {np.mean(list(residuals['point2'].values())):.4f}")

    # 6. 处理点云数据
    point1_array = np.array(list(points_3d['point1'].values()))
    point2_array = np.array(list(points_3d['point2'].values()))

    # 去除离群点
    point1_filtered = remove_outliers(point1_array, outlier_threshold)
    point2_filtered = remove_outliers(point2_array, outlier_threshold)

    print(f"离群点去除后: point1有 {len(point1_filtered)} 个点，point2有 {len(point2_filtered)} 个点")

    # 7. 计算两个点的中心位置
    center1 = np.mean(point1_filtered, axis=0)
    center2 = np.mean(point2_filtered, axis=0)

    # 8. 计算两个中心点之间的距离
    current_distance = np.linalg.norm(center1 - center2)
    print(f"当前估计的两点距离: {current_distance:.4f}")
    print(f"实际两点距离: {real_distance:.4f}")

    # 计算缩放比例
    scale_factor = real_distance / current_distance
    print(f"缩放比例: {scale_factor:.4f}")

    # 9.1 展示3D图像
    if vis_3D:
        # 可视化3D点云和参考点
        print("正在生成3D点云可视化...")
        visualize_3d_clusters_with_centers(
            point1_filtered, point2_filtered,
            center1, center2,
            real_distance, current_distance
        )

    # 10. 缩放相机外参
    scaled_extrinsics = {}
    for cam_id, extrinsic in camera_extrinsics.items():
        scaled_translation = extrinsic.translation * scale_factor
        scaled_extrinsics[cam_id] = CameraExtrinsic(
            rotation=extrinsic.rotation,  # 保持旋转不变
            translation=scaled_translation
        )

    # 11. 缩放最初加载的3D点，而不是使用三角测量得到的点
    scaled_points_3d = {}
    for frame, point in original_points_3d.items():
        scaled_points_3d[frame] = point * scale_factor

    # 12. 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    save_optimization_results(
        output_dir,
        scaled_extrinsics,
        camera_intrinsics,  # 内参保持不变
        scaled_points_3d,
        format_type,
        extrinsic_file_name=extrinsic_file_name,
        intrinsic_file_name=intrinsic_file_name,
        points_file_name=points_file_name
    )

    print("===== 数据处理完成 =====")


def find_best_point_correspondence(
        frames_data: Dict[int, Dict[str, Dict[str, np.ndarray]]],
        camera_extrinsics: Dict[str, CameraExtrinsic],
        camera_intrinsics: Dict[str, CameraIntrinsic],
        max_sample_frames: int = 20,
        reproj_error_threshold: float = 5.0
) -> bool:
    """
    使用少量样本帧确定最佳的点对应关系 - 改进版
    测试不同相机间的点对应关系，而不是简单交换同一相机内的点
    """
    # 按照共享相机数量降序排列帧
    frames_with_camera_counts = []
    for frame, cam_data in frames_data.items():
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]
        if len(valid_cameras) >= 2:
            frames_with_camera_counts.append((frame, len(valid_cameras)))

    # 按相机数量降序排序
    frames_with_camera_counts.sort(key=lambda x: x[1], reverse=True)

    # 选择前N个帧作为样本
    sample_frames = [frame for frame, _ in frames_with_camera_counts[:max_sample_frames]]

    if not sample_frames:
        print("警告: 没有找到足够的样本帧")
        return False  # 默认不交换

    print(f"使用 {len(sample_frames)} 个样本帧确定点对应关系")

    # 尝试两种可能的对应关系组合
    errors_combination1 = {'total': 0, 'count': 0}  # 组合1：所有相机的point1互相匹配，point2互相匹配
    errors_combination2 = {'total': 0, 'count': 0}  # 组合2：第一个相机保持不变，其他相机交换point1和point2

    for frame in sample_frames:
        cam_data = frames_data[frame]
        valid_cameras = [
            cam_id for cam_id, points in cam_data.items()
            if not np.isnan(points['point1']).any() and not np.isnan(points['point2']).any()
               and cam_id in camera_extrinsics and cam_id in camera_intrinsics
        ]

        if len(valid_cameras) < 2:
            continue

        # 组合1：所有相机的point1互相匹配，所有相机的point2互相匹配
        points_match1_1 = {cam_id: cam_data[cam_id]['point1'] for cam_id in valid_cameras}
        points_match1_2 = {cam_id: cam_data[cam_id]['point2'] for cam_id in valid_cameras}

        point3d_match1_1, error_match1_1 = triangulate_point(points_match1_1, camera_extrinsics, camera_intrinsics)
        point3d_match1_2, error_match1_2 = triangulate_point(points_match1_2, camera_extrinsics, camera_intrinsics)

        # 组合2：对第一个相机保持原样，其余相机交换点顺序
        first_cam = valid_cameras[0]
        other_cams = valid_cameras[1:]

        points_match2_1 = {first_cam: cam_data[first_cam]['point1']}
        points_match2_2 = {first_cam: cam_data[first_cam]['point2']}

        for cam_id in other_cams:
            # 交换其他相机的点顺序
            points_match2_1[cam_id] = cam_data[cam_id]['point2']  # 注意这里交换了
            points_match2_2[cam_id] = cam_data[cam_id]['point1']  # 注意这里交换了

        point3d_match2_1, error_match2_1 = triangulate_point(points_match2_1, camera_extrinsics, camera_intrinsics)
        point3d_match2_2, error_match2_2 = triangulate_point(points_match2_2, camera_extrinsics, camera_intrinsics)

        # 计算并比较两种组合的总误差
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

    # 确定最佳对应关系
    if errors_combination1['count'] == 0 and errors_combination2['count'] == 0:
        print("警告: 两种组合都没有产生有效的三角测量结果")
        return False  # 默认不交换

    if errors_combination1['count'] == 0:
        print("组合1没有产生有效结果，使用组合2")
        return True  # 使用组合2

    if errors_combination2['count'] == 0:
        print("组合2没有产生有效结果，使用组合1")
        return False  # 使用组合1

    # 计算平均误差
    avg_error_combo1 = errors_combination1['total'] / errors_combination1['count']
    avg_error_combo2 = errors_combination2['total'] / errors_combination2['count']

    print(f"组合1平均误差: {avg_error_combo1:.4f}，有效帧数: {errors_combination1['count']}")
    print(f"组合2平均误差: {avg_error_combo2:.4f}，有效帧数: {errors_combination2['count']}")

    # 选择平均误差较小的组合
    swap_points = avg_error_combo2 < avg_error_combo1

    if swap_points:
        print("选择组合2作为最佳对应关系")
    else:
        print("选择组合1作为最佳对应关系")

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
    保存优化后的相机外参、内参和3D点云数据到指定目录

    Args:
        output_dir: 输出目录路径
        optimized_camera_extrinsics: 优化后的相机外参数据字典
        optimized_camera_intrinsics: 优化后的相机内参数据字典
        optimized_points_3d: 优化后的3D点云数据字典
        save_format: 保存格式，可选 'pickle' 或 'json'
        extrinsic_file_name: 外参文件名（不含扩展名）
        intrinsic_file_name: 内参文件名（不含扩展名）
        points_file_name: 3D点文件名（不含扩展名）
    """
    # 如果目录不存在，创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    scaled_extrinsics_filename = os.path.join(output_dir, f"scaled_{extrinsic_file_name}.{save_format}")
    scaled_intrinsics_filename = os.path.join(output_dir, f"scaled_{intrinsic_file_name}.{save_format}")
    scaled_points3d_filename = os.path.join(output_dir, f"scaled_{points_file_name}.{save_format}")

    # 根据保存格式选择不同的保存方法
    if save_format.lower() == 'pickle':
        # 保存为Pickle格式
        # 保存相机外参
        with open(scaled_extrinsics_filename, 'wb') as f:
            pickle.dump(optimized_camera_extrinsics, f)

        # 保存相机内参
        with open(scaled_intrinsics_filename, 'wb') as f:
            pickle.dump(optimized_camera_intrinsics, f)

        # 保存3D点云
        with open(scaled_points3d_filename, 'wb') as f:
            pickle.dump(optimized_points_3d, f)

    elif save_format.lower() == 'json':
        # 保存为JSON格式
        # 处理相机外参
        json_extrinsics = {}
        for cam_id, extrinsic in optimized_camera_extrinsics.items():
            json_extrinsics[cam_id] = {
                'rotation': extrinsic.rotation.tolist(),
                'translation': extrinsic.translation.tolist()
            }

        with open(scaled_extrinsics_filename, 'w') as f:
            json.dump(json_extrinsics, f, indent=2)

        # 处理相机内参
        json_intrinsics = {}
        for cam_id, intrinsic in optimized_camera_intrinsics.items():
            json_intrinsics[cam_id] = {
                'camera_matrix': intrinsic.camera_matrix.tolist(),
                'dist_coeffs': intrinsic.dist_coeffs.tolist(),
                'image_size': intrinsic.image_size
            }

        with open(scaled_intrinsics_filename, 'w') as f:
            json.dump(json_intrinsics, f, indent=2)

        # 处理3D点云
        json_points3d = {}
        for frame, point in optimized_points_3d.items():
            json_points3d[str(frame)] = point.tolist()

        with open(scaled_points3d_filename, 'w') as f:
            json.dump(json_points3d, f, indent=2)
    else:
        raise ValueError(f"不支持的保存格式: {save_format}, 支持的格式: 'pickle' 和 'json'")

    print(f"scaled save dir: {output_dir}")
    print(f"  - camera extrinsic: {os.path.basename(scaled_extrinsics_filename)}")
    print(f"  - camera intrinsic: {os.path.basename(scaled_intrinsics_filename)}")
    print(f"  - 3D points: {os.path.basename(scaled_points3d_filename)}")


def visualize_2d_clusters(all_points_by_camera, all_cluster_labels, camera_intrinsics=None):
    """
    为每个相机创建并直接显示2D聚类可视化图像

    Args:
        all_points_by_camera: 按相机ID分组的所有2D点
        all_cluster_labels: 按相机ID分组的聚类标签
        camera_intrinsics: 相机内参字典（可选），用于获取图像尺寸
    """
    import matplotlib.pyplot as plt

    # 为每个相机创建聚类可视化
    for cam_id, points in all_points_by_camera.items():
        if cam_id not in all_cluster_labels or len(all_cluster_labels[cam_id]) == 0:
            print(f"相机 {cam_id} 没有聚类标签，跳过可视化")
            continue

        if len(points) < 2:
            print(f"相机 {cam_id} 只有 {len(points)} 个点，不足以进行可视化")
            continue

        # 获取该相机的聚类标签
        cluster_labels = all_cluster_labels[cam_id]

        # 创建新的图形
        plt.figure(figsize=(10, 8))

        # 将点转换为numpy数组
        points_array = np.array(points)

        # 绘制不同聚类的点（用不同颜色）
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # 颜色列表

        for cluster_id in sorted(set(cluster_labels)):
            # 获取当前聚类的所有点
            cluster_points = points_array[np.array(cluster_labels) == cluster_id]

            # 绘制点
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=colors[cluster_id % len(colors)],
                label=f'组 {cluster_id}',
                alpha=0.7,
                s=30
            )

        # 设置图像属性
        plt.title(f'相机 {cam_id} 的2D点聚类')
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 如果提供了相机内参，则设置适当的边界
        if camera_intrinsics and cam_id in camera_intrinsics:
            if hasattr(camera_intrinsics[cam_id], 'image_size'):
                img_size = camera_intrinsics[cam_id].image_size
                plt.xlim(0, img_size[0])
                plt.ylim(img_size[1], 0)  # 注意：Y轴反转，使原点在左上角

        plt.show()  # 直接显示而不是保存
        print(f"已显示相机 {cam_id} 的聚类可视化图像")


def visualize_3d_clusters_with_centers(point1_filtered, point2_filtered, center1, center2,
                                       real_distance, current_distance):
    """
    可视化两组3D点云及其中心点，并展示中心点之间的距离

    Args:
        point1_filtered: 第一组过滤后的3D点
        point2_filtered: 第二组过滤后的3D点
        center1: 第一组点的中心
        center2: 第二组点的中心
        real_distance: 实际距离
        current_distance: 计算得到的当前距离
    """
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一组点（红色）
    ax.scatter(point1_filtered[:, 0], point1_filtered[:, 1], point1_filtered[:, 2],
               c='red', s=20, alpha=0.6, label='点组1')

    # 绘制第二组点（蓝色）
    ax.scatter(point2_filtered[:, 0], point2_filtered[:, 1], point2_filtered[:, 2],
               c='blue', s=20, alpha=0.6, label='点组2')

    # 绘制中心点（使用更大的点和不同颜色）
    ax.scatter(center1[0], center1[1], center1[2],
               c='darkred', s=100, label='中心点1', edgecolor='black')
    ax.scatter(center2[0], center2[1], center2[2],
               c='darkblue', s=100, label='中心点2', edgecolor='black')

    # 绘制中心点之间的连线
    ax.plot([center1[0], center2[0]],
            [center1[1], center2[1]],
            [center1[2], center2[2]],
            'g-', linewidth=2, label=f'距离: {current_distance:.4f}')

    # 设置图例、标题和标签
    ax.legend(loc='upper right')
    ax.set_title(
        f'3D点云聚类可视化\n实际距离: {real_distance:.4f}, 计算距离: {current_distance:.4f}, 比例: {real_distance / current_distance:.4f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置合适的坐标轴范围
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

    # 添加文本标注中心点坐标
    ax.text(center1[0], center1[1], center1[2],
            f'({center1[0]:.3f}, {center1[1]:.3f}, {center1[2]:.3f})',
            color='darkred', fontsize=9)
    ax.text(center2[0], center2[1], center2[2],
            f'({center2[0]:.3f}, {center2[1]:.3f}, {center2[2]:.3f})',
            color='darkblue', fontsize=9)

    # 显示图形
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