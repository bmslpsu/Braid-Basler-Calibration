import numpy as np
from typing import Dict, Tuple, NamedTuple, Any


class CameraMatrices(NamedTuple):
    """存储相机的所有矩阵参数"""
    P: np.ndarray  # 投影矩阵 (3x4)
    K: np.ndarray  # 内参矩阵 (3x3)
    D: np.ndarray  # distortion matrix (5,)
    R: np.ndarray  # 旋转矩阵 (3x3)
    t: np.ndarray  # 平移向量 (3,)
    skew_value: float  # skew value


class CameraExtrinsic(NamedTuple):
    """存储相机的外参"""
    rotation: np.ndarray  # 旋转矩阵 (3x3)
    translation: np.ndarray  # 平移向量 (3,)


def process_get_skew(P):
    """计算skew值"""
    # 这个函数实现在.get_skew模块中
    # 为了完整性，我们在这里提供一个简单的实现
    return 0.0  # 返回默认值，实际应用中请替换为真实实现


def verify_rotation_matrix(R: np.ndarray) -> bool:
    """验证旋转矩阵是否正交"""
    if R.shape != (3, 3):
        print(f"Error: Rotation matrix should be 3x3, got {R.shape}")
        return False
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        print("Warning: Rotation matrix is not orthogonal")
        return False
    return True


def verify_intrinsic_matrix(intrinsic: np.ndarray) -> bool:
    """验证内参矩阵的格式和值是否正确"""
    if intrinsic.shape != (3, 3):
        print(f"Error: Intrinsic matrix should be 3x3, got {intrinsic.shape}")
        return False
    if not np.allclose(intrinsic[2, :], [0, 0, 1]):
        print("Error: Last row of intrinsic matrix should be [0,0,1]")
        return False
    if not np.allclose(intrinsic[0, 1], 0):
        print("Warning: Skew parameter is not zero")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        print("Error: Focal lengths should be positive")
        return False
    return True


def calculate_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """计算投影矩阵 P = K[R|t]"""
    Rt = np.hstack((R, t.reshape(3, 1)))
    P = K @ Rt
    return P


def calculate_camera_matrices(camera_intrinsics: Dict,
                              camera_extrinsics: Dict,
                              verify: bool = True) -> Dict[str, CameraMatrices]:
    """
    计算所有相机的矩阵参数，返回包含P、K、D、R、t的字典
    接受CameraExtrinsic对象格式的外参

    参数:
    camera_intrinsics: 包含相机内参的字典，键为相机ID，值为包含camera_matrix和dist_coeffs的对象
    camera_extrinsics: 包含相机外参的字典，键为相机ID，值为CameraExtrinsic对象
    verify: 是否验证矩阵的正确性

    返回:
    Dict[str, CameraMatrices]: 包含各相机矩阵参数的字典
    """
    camera_matrices = {}

    for cam_id in camera_extrinsics:
        # 获取相机参数
        K = camera_intrinsics[cam_id].camera_matrix
        D = camera_intrinsics[cam_id].dist_coeffs

        # 直接从对象中获取R和t
        R = camera_extrinsics[cam_id].rotation
        t = camera_extrinsics[cam_id].translation

        # 验证矩阵
        if verify:
            if not verify_intrinsic_matrix(K):
                print(f"Warning: Invalid intrinsic matrix for camera {cam_id}")
            if not verify_rotation_matrix(R):
                print(f"Warning: Invalid rotation matrix for camera {cam_id}")

        # 计算投影矩阵
        P = calculate_projection_matrix(K, R, t)

        # calc skew value
        skew_value = process_get_skew(P)

        # 存储结果
        camera_matrices[cam_id] = CameraMatrices(P=P, K=K, D=D, R=R, t=t, skew_value=skew_value)

        # 打印结果
        print(f"\nCamera {cam_id}:")
        print("Projection matrix P:")
        print(P)
        print("\nDecomposition:")
        print(f"Intrinsic matrix K:\n{K}")
        print(f"Distortion matrix D:\n{D}")
        print(f"Rotation matrix R:\n{R}")
        print(f"Translation vector t:\n{t}")
        print(f"Skew value:\n{skew_value}")

    return camera_matrices


def verify_camera_matrices(matrices: CameraMatrices,
                           tolerance: float = 1e-10) -> bool:
    """验证相机矩阵参数是否正确"""
    # 重新计算投影矩阵
    P_calc = calculate_projection_matrix(matrices.K, matrices.R, matrices.t)

    # 比较差异
    diff = np.abs(matrices.P - P_calc)
    max_diff = np.max(diff)

    if max_diff > tolerance:
        print(f"Warning: Large difference in projection matrix: {max_diff}")
        print("Calculated P:")
        print(P_calc)
        print("Given P:")
        print(matrices.P)
        return False

    return True


def test_projection(matrices: CameraMatrices,
                    test_point: np.ndarray = np.array([1, 1, 1, 1])) -> np.ndarray:
    """使用测试点验证投影矩阵并返回投影点坐标"""
    # 投影点
    point_2d_homog = matrices.P @ test_point
    point_2d = point_2d_homog[:2] / point_2d_homog[2]

    print(f"3D point: {test_point[:3]}")
    print(f"Projected 2D point: {point_2d}")

    return point_2d


# 如果需要兼容旧代码，可以添加一个转换函数
def convert_extrinsic_to_matrix(extrinsic: CameraExtrinsic) -> np.ndarray:
    """将CameraExtrinsic对象转换为4x4外参矩阵"""
    matrix = np.eye(4)
    matrix[:3, :3] = extrinsic.rotation
    matrix[:3, 3] = extrinsic.translation
    return matrix


# 支持4x4矩阵格式的外参的原始函数
def calculate_camera_matrices_4x4(camera_intrinsics: Dict,
                                  camera_extrinsics_4x4: Dict,
                                  verify: bool = True) -> Dict[str, CameraMatrices]:
    """
    使用4x4矩阵格式的外参计算相机矩阵参数
    这是为了兼容原始代码
    """
    camera_matrices = {}

    for cam_id in camera_extrinsics_4x4:
        # 获取相机参数
        K = camera_intrinsics[cam_id].camera_matrix
        D = camera_intrinsics[cam_id].dist_coeffs
        extrinsic = camera_extrinsics_4x4[cam_id]

        # 验证矩阵
        if verify:
            if not verify_intrinsic_matrix(K):
                print(f"Warning: Invalid intrinsic matrix for camera {cam_id}")
            if not verify_extrinsic_matrix(extrinsic):
                print(f"Warning: Invalid extrinsic matrix for camera {cam_id}")

        # 从外参矩阵中提取R和t
        R, t = get_RT_from_extrinsic(extrinsic)

        # 计算投影矩阵
        P = calculate_projection_matrix(K, R, t)

        # calc skew value
        skew_value = process_get_skew(P)

        # 存储结果
        camera_matrices[cam_id] = CameraMatrices(P=P, K=K, D=D, R=R, t=t, skew_value=skew_value)

        # 打印结果
        print(f"\nCamera {cam_id}:")
        print("Projection matrix P:")
        print(P)
        print("\nDecomposition:")
        print(f"Intrinsic matrix K:\n{K}")
        print(f"Distortion matrix D:\n{D}")
        print(f"Rotation matrix R:\n{R}")
        print(f"Translation vector t:\n{t}")
        print(f"Skew value:\n{skew_value}")

    return camera_matrices


def verify_extrinsic_matrix(extrinsic: np.ndarray) -> bool:
    """验证外参矩阵的格式和值是否正确"""
    if extrinsic.shape != (4, 4):
        print(f"Error: Extrinsic matrix should be 4x4, got {extrinsic.shape}")
        return False
    if not np.allclose(extrinsic[3], [0, 0, 0, 1]):
        print("Error: Last row of extrinsic matrix should be [0,0,0,1]")
        return False
    R = extrinsic[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        print("Warning: Rotation matrix is not orthogonal")
        return False
    return True


def get_RT_from_extrinsic(extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从外参矩阵中提取旋转矩阵R和平移向量t"""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return R, t