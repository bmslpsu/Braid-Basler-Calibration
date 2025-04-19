import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


class MultiCameraReconstructor:
    def __init__(self, xml_file):
        """
        初始化多相机重构器

        参数:
            xml_file: 包含相机标定参数的XML文件路径
        """
        self.xml_file = xml_file
        self.cameras = {}
        self.minimum_eccentricity = None
        self.load_calibration_data()

    def load_calibration_data(self):
        """从XML文件加载相机标定数据"""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        # 读取最小离心率
        min_ecc_elem = root.find('minimum_eccentricity')
        if min_ecc_elem is not None:
            self.minimum_eccentricity = float(min_ecc_elem.text)

        # 读取每个相机的标定数据
        for camera in root.findall('single_camera_calibration'):
            cam_id = camera.find('cam_id').text

            # 解析标定矩阵
            calib_matrix_text = camera.find('calibration_matrix').text
            matrix_values = [float(val) for val in calib_matrix_text.replace(';', ' ').split()]
            calibration_matrix = np.array(matrix_values).reshape(3, 4)

            # 解析相机分辨率
            resolution_text = camera.find('resolution').text
            resolution = tuple(map(int, resolution_text.split()))

            # 解析非线性参数
            non_linear_params = {}
            non_linear_elem = camera.find('non_linear_parameters')
            for param in non_linear_elem:
                non_linear_params[param.tag] = float(param.text)

            # 创建相机信息字典
            camera_info = {
                'calibration_matrix': calibration_matrix,
                'resolution': resolution,
                'non_linear_params': non_linear_params
            }

            self.cameras[cam_id] = camera_info

    def project_3d_to_2d(self, cam_id, points_3d):
        """
        将3D点投影到特定相机的2D图像平面

        参数:
            cam_id: 相机ID
            points_3d: 形状为(N, 3)的3D点数组

        返回:
            形状为(N, 2)的2D点数组
        """
        if cam_id not in self.cameras:
            raise ValueError(f"相机ID {cam_id} 不存在")

        camera = self.cameras[cam_id]
        calibration_matrix = camera['calibration_matrix']
        non_linear_params = camera['non_linear_params']

        # 确保points_3d是numpy数组且形状正确
        points_3d = np.array(points_3d)
        if len(points_3d.shape) == 1:
            points_3d = points_3d.reshape(1, -1)

        # 将3D点转换为齐次坐标(x, y, z, 1)
        num_points = points_3d.shape[0]
        homogeneous_points = np.hstack((points_3d, np.ones((num_points, 1))))

        # 使用标定矩阵投影到相机坐标系
        projected_points = np.dot(calibration_matrix, homogeneous_points.T).T

        # 将齐次坐标转换为2D图像坐标
        pixel_points = projected_points[:, :2] / projected_points[:, 2:3]

        # 应用镜头畸变校正
        # 这里可以使用非线性参数进行更精确的校正
        # 但为简化起见，先假设线性投影已足够

        return pixel_points

    def triangulate_3d_point(self, image_points_dict):
        """
        通过多个相机的2D点三角测量3D点

        参数:
            image_points_dict: 字典，键为相机ID，值为相应的2D点坐标

        返回:
            3D点坐标
        """
        # 构建系数矩阵A用于最小二乘法求解
        A = []

        for cam_id, point_2d in image_points_dict.items():
            if cam_id not in self.cameras:
                continue

            P = self.cameras[cam_id]['calibration_matrix']
            x, y = point_2d

            # 对于每个相机添加两行约束
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        A = np.array(A)

        # 使用SVD求解超定线性方程组
        _, _, Vt = np.linalg.svd(A)

        # 最小特征值对应的特征向量就是解
        point_3d_homogeneous = Vt[-1, :]

        # 转换为非齐次坐标
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        return point_3d

    def calculate_reprojection_error(self, point_3d, image_points_dict):
        """
        计算3D点重投影误差

        参数:
            point_3d: 3D点坐标
            image_points_dict: 字典，键为相机ID，值为实际观测到的2D点坐标

        返回:
            每个相机的重投影误差字典和平均重投影误差
        """
        errors = {}
        total_error = 0
        num_cameras = 0

        for cam_id, observed_point in image_points_dict.items():
            if cam_id not in self.cameras:
                continue

            # 将3D点投影回2D
            projected_point = self.project_3d_to_2d(cam_id, [point_3d])[0]

            # 计算欧几里得距离作为误差
            error = np.sqrt(np.sum((projected_point - observed_point) ** 2))
            errors[cam_id] = error

            total_error += error
            num_cameras += 1

        avg_error = total_error / num_cameras if num_cameras > 0 else 0

        return errors, avg_error

    def undistort_points(self, cam_id, points_2d):
        """
        使用相机的非线性参数对2D点进行畸变校正

        参数:
            cam_id: 相机ID
            points_2d: 形状为(N, 2)的2D点数组

        返回:
            校正后的2D点数组
        """
        if cam_id not in self.cameras:
            raise ValueError(f"相机ID {cam_id} 不存在")

        camera = self.cameras[cam_id]
        non_linear_params = camera['non_linear_params']

        # 从非线性参数中提取内参和畸变系数
        fx = non_linear_params['fc1']
        fy = non_linear_params['fc2']
        cx = non_linear_params['cc1']
        cy = non_linear_params['cc2']
        k1 = non_linear_params['k1']
        k2 = non_linear_params['k2']
        p1 = non_linear_params['p1']
        p2 = non_linear_params['p2']

        # 构建相机矩阵
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # 构建畸变系数
        dist_coeffs = np.array([k1, k2, p1, p2, 0])

        # 确保points_2d是正确的格式
        points_2d = np.array(points_2d, dtype=np.float32)
        if len(points_2d.shape) == 1:
            points_2d = points_2d.reshape(1, -1)

        # 使用OpenCV的undistortPoints函数
        points_2d_reshaped = points_2d.reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points_2d_reshaped, camera_matrix, dist_coeffs, None, camera_matrix)

        return undistorted_points.reshape(-1, 2)


def test_reprojection_with_synthetic_data(reconstructor):
    """
    使用合成数据测试重投影精度

    参数:
        reconstructor: MultiCameraReconstructor实例
    """
    # 创建一些简单的3D点
    simple_points_3d = [
        [0, 0, 0],
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 100],
        [100, 100, 100]
    ]

    # 创建一个螺旋轨迹作为更复杂的测试数据
    t = np.linspace(0, 4 * np.pi, 50)  # 50个点的螺旋
    radius = 100
    height = 200
    spiral_points_3d = []

    for ti in t:
        x = radius * np.cos(ti)
        y = radius * np.sin(ti)
        z = height * ti / (4 * np.pi)
        spiral_points_3d.append([x, y, z])

    # 合并所有测试点
    points_3d = simple_points_3d + spiral_points_3d

    # 将3D点投影到每个相机上（添加一些噪声模拟真实情况）
    image_points_dicts = []

    for point_3d in points_3d:
        image_points_dict = {}
        for cam_id in reconstructor.cameras.keys():
            # 将点投影到2D
            point_2d = reconstructor.project_3d_to_2d(cam_id, [point_3d])[0]

            # 添加随机噪声（均值0，标准差0.5像素的高斯噪声）
            noise = np.random.normal(0, 0.5, 2)
            point_2d = point_2d + noise

            image_points_dict[cam_id] = point_2d

        image_points_dicts.append(image_points_dict)

    # 计算重投影误差
    print("使用合成数据的重投影误差:")

    total_errors = {}
    for cam_id in reconstructor.cameras.keys():
        total_errors[cam_id] = []

    # 仅打印前10个点的详细信息，以避免输出过多
    for i, (point_3d, image_points_dict) in enumerate(zip(points_3d[:10], image_points_dicts[:10])):
        errors, avg_error = reconstructor.calculate_reprojection_error(point_3d, image_points_dict)

        print(f"3D点 {i + 1}: {point_3d}")
        for cam_id, error in errors.items():
            print(f"  相机 {cam_id} 重投影误差: {error:.4f} 像素")
            total_errors[cam_id].append(error)
        print(f"  平均重投影误差: {avg_error:.4f} 像素")
        print()

    # 计算并打印所有点的统计信息
    print("Reprojection error:")
    for cam_id in reconstructor.cameras.keys():
        all_errors = []
        for image_points_dict, point_3d in zip(image_points_dicts, points_3d):
            errors, _ = reconstructor.calculate_reprojection_error(point_3d, image_points_dict)
            all_errors.append(errors[cam_id])

        mean_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        std_error = np.std(all_errors)

        print(f"camera {cam_id}:")
        print(f"  mean error: {mean_error:.4f} pixels")
        print(f"  max error: {max_error:.4f} pixels")
        print(f"  min error: {min_error:.4f} pixels")
        print(f"  std: {std_error:.4f} pixels")
        print()


def test_triangulation_accuracy(reconstructor):
    """
    测试三角测量准确性

    参数:
        reconstructor: MultiCameraReconstructor实例
    """
    # 创建一些简单的3D点
    simple_points_3d = [
        [0, 0, 0],
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 100],
        [100, 100, 100]
    ]

    # 创建一条轨迹
    # 创建一个环形轨迹
    circle_points = []
    num_circle_points = 20
    radius = 150
    for i in range(num_circle_points):
        angle = 2 * np.pi * i / num_circle_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 50  # 固定高度
        circle_points.append([x, y, z])

    # 创建一个8字形轨迹
    figure8_points = []
    num_figure8_points = 30
    radius_a = 100
    radius_b = 50
    for i in range(num_figure8_points):
        t = 2 * np.pi * i / num_figure8_points
        x = radius_a * np.sin(t)
        y = radius_b * np.sin(2 * t)
        z = 20 + 10 * np.cos(t)  # 变化的高度
        figure8_points.append([x, y, z])

    # 合并所有测试轨迹点
    all_trajectory_points = simple_points_3d + circle_points + figure8_points

    print("三角测量准确性测试:")

    # 打印前5个点的详细信息
    for i, original_point_3d in enumerate(simple_points_3d):
        # 将点投影到每个相机上，添加一些噪声
        image_points_dict = {}
        for cam_id in reconstructor.cameras.keys():
            point_2d = reconstructor.project_3d_to_2d(cam_id, [original_point_3d])[0]
            # 添加随机噪声
            noise = np.random.normal(0, 0.5, 2)
            point_2d = point_2d + noise
            image_points_dict[cam_id] = point_2d

        # 从2D点重新三角测量3D点
        reconstructed_point_3d = reconstructor.triangulate_3d_point(image_points_dict)

        # 计算原始点和重构点之间的欧几里得距离
        error = np.sqrt(np.sum((np.array(original_point_3d) - reconstructed_point_3d) ** 2))

        print(f"点 {i + 1}:")
        print(f"  原始3D点: {original_point_3d}")
        print(f"  重构3D点: {reconstructed_point_3d}")
        print(f"  重构误差: {error:.4f} 单位")
        print()

    # 测试所有轨迹点并收集统计数据
    all_errors = []
    original_points = []
    reconstructed_points = []

    for original_point_3d in all_trajectory_points:
        # 将点投影到每个相机上，添加一些噪声
        image_points_dict = {}
        for cam_id in reconstructor.cameras.keys():
            point_2d = reconstructor.project_3d_to_2d(cam_id, [original_point_3d])[0]
            # 添加随机噪声
            noise = np.random.normal(0, 0.5, 2)
            point_2d = point_2d + noise
            image_points_dict[cam_id] = point_2d

        # 从2D点重新三角测量3D点
        reconstructed_point_3d = reconstructor.triangulate_3d_point(image_points_dict)

        # 计算误差
        error = np.sqrt(np.sum((np.array(original_point_3d) - reconstructed_point_3d) ** 2))
        all_errors.append(error)

        # 保存原始点和重构点用于可视化
        original_points.append(original_point_3d)
        reconstructed_points.append(reconstructed_point_3d)

    # 计算误差统计
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    max_error = np.max(all_errors)
    min_error = np.min(all_errors)
    std_error = np.std(all_errors)

    # 计算误差小于1像素的点数
    errors_within_1px = np.sum(np.array(all_errors) < 1.0)
    percentage_within_1px = (errors_within_1px / len(all_errors)) * 100

    print("reprojection error for all 3D points:")
    print(f"  num of points: {len(all_errors)}")
    print(f"  mean error: {mean_error:.4f} units")
    print(f"  median error: {median_error:.4f} units")
    print(f"  max error: {max_error:.4f} units")
    print(f"  min error: {min_error:.4f} units")
    print(f"  std: {std_error:.4f} units")
    print(f"  points with error < 1px: {errors_within_1px} ({percentage_within_1px:.2f}%)")
    print()


def main():
    """主函数"""
    # 替换为您的XML文件路径
    xml_file = "F:/BMS Backup/2025-4-13/BMS_Lab/Free_Flight_experiment/Mike_Braid_demoset_calibration_files_step_by_step/MCSC_steps_BRAIDZ_file/MCSC6_best/MCSC6_unaligned.xml"
    xml_file = "./data_file_4_angled/results/braid_calibration.xml"

    try:
        # 初始化重构器
        reconstructor = MultiCameraReconstructor(xml_file)

        print(f"已加载 {len(reconstructor.cameras)} 个相机的标定数据")
        for cam_id, camera in reconstructor.cameras.items():
            print(f"相机 {cam_id}:")
            print(f"  分辨率: {camera['resolution']}")
            print(f"  标定矩阵:\n{camera['calibration_matrix']}")
            print()

        # 测试重投影精度
        test_reprojection_with_synthetic_data(reconstructor)

        # 测试三角测量准确性
        test_triangulation_accuracy(reconstructor)

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()