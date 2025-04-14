import numpy as np


def get_cam_coor(R):
    """
    获取相机坐标系的三个轴向量，尽量保持原始方向同时确保正交

    Args:
        R: 3x3 旋转矩阵

    Returns:
        tuple: 正交化后的 (x_axis, y_axis, z_axis)
    """
    # 从旋转矩阵中获取原始的三个轴向量
    x_orig = R.T[:, 0]
    y_orig = R.T[:, 1]
    z_orig = R.T[:, 2]

    # 使用正交化流程，但尽量保持原始方向
    # 首先保持 z 轴不变
    z_axis = z_orig / np.linalg.norm(z_orig)

    # 计算 y 轴 - 从原始 y 轴出发，但确保与 z 轴正交
    # 从 y_orig 中移除与 z_axis 平行的分量
    y_temp = y_orig - np.dot(y_orig, z_axis) * z_axis
    y_axis = y_temp / np.linalg.norm(y_temp)

    # 计算 x 轴 - 通过叉积确保与 y 和 z 轴都正交
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 检查三个轴是否构成右手坐标系
    # 如果不是，翻转 y 轴
    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
        y_axis = -y_axis

    return x_axis, y_axis, z_axis

def analyze_optimization_results(camera_extrinsics, points_3d=None, initial_extrinsics=None):
    """
    Analyze the optimization results and print useful statistics,
    and visualize the 3D points and camera positions

    Args:
        camera_extrinsics: Dictionary of optimized camera extrinsics (CameraExtrinsic objects)
        points_3d: Optional dictionary of optimized 3D points
        initial_extrinsics: Optional dictionary of initial camera extrinsics (CameraExtrinsic objects)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.spatial.transform import Rotation

    print("\nOptimization Results Analysis:")
    print("-" * 30)

    # 正确计算相机在世界坐标系中的位置
    positions = []
    for ext in camera_extrinsics.values():
        R = ext.rotation
        t = ext.translation
        # 计算相机在世界坐标系中的位置: C = -R^T * t
        camera_pos = -R.T @ t
        positions.append(camera_pos)

    positions = np.array(positions)
    center = np.mean(positions, axis=0)

    print(f"Camera configuration center: {center}")

    # Compute distances between cameras
    n_cameras = len(camera_extrinsics)
    distances = []
    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            pos_i = positions[i]
            pos_j = positions[j]
            dist = np.linalg.norm(pos_i - pos_j)
            distances.append(dist)

    avg_dist = np.mean(distances)
    std_dist = np.std(distances)

    print(f"Average distance between cameras: {avg_dist:.2f} ± {std_dist:.2f}")

    # Analyze camera orientations
    for i, (cam_id, ext) in enumerate(camera_extrinsics.items()):
        # 使用相机在世界坐标系中的旋转矩阵
        R_world = ext.rotation.T  # 世界坐标系下的旋转矩阵
        euler_angles = Rotation.from_matrix(R_world).as_euler('xyz', degrees=True)
        print(f"\nCamera {cam_id} orientation (euler angles in degrees (camera to world)):")
        print(f"  Roll: {euler_angles[0]:.2f}")
        print(f"  Pitch: {euler_angles[1]:.2f}")
        print(f"  Yaw: {euler_angles[2]:.2f}")

    # Analyze 3D points if available
    if points_3d is not None and len(points_3d) > 0:
        points_array = np.array(list(points_3d.values()))
        point_center = np.mean(points_array, axis=0)
        point_std = np.std(points_array, axis=0)

        print("\n3D Points Analysis:")
        print(f"Number of 3D points: {len(points_3d)}")
        print(f"Point cloud center: {point_center}")
        print(f"Point cloud standard deviation: {point_std}")

        # Calculate point cloud dimensions
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)
        dimensions = max_coords - min_coords

        print(f"Point cloud dimensions (x, y, z): {dimensions}")

    if initial_extrinsics is not None:
        print("\nChanges from initial configuration:")
        for cam_id in camera_extrinsics.keys():
            # 正确计算初始和最终相机位置
            init_R = initial_extrinsics[cam_id].rotation
            init_t = initial_extrinsics[cam_id].translation
            init_pos = -init_R.T @ init_t

            final_R = camera_extrinsics[cam_id].rotation
            final_t = camera_extrinsics[cam_id].translation
            final_pos = -final_R.T @ final_t

            position_change = np.linalg.norm(final_pos - init_pos)

            # 计算旋转变化
            R_init_world = init_R.T
            R_final_world = final_R.T
            R_diff = R_final_world @ R_init_world.T  # 从初始旋转到最终旋转的变换
            angle_change = np.arccos((np.trace(R_diff) - 1) / 2)
            angle_change_deg = np.degrees(angle_change)

            print(f"\nCamera {cam_id}:")
            print(f"  Position change: {position_change:.2f}")
            print(f"  Orientation change: {angle_change_deg:.2f} degrees")

    # Visualization of 3D points and camera positions
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the overall range considering both points and cameras
    cam_positions = positions  # 已经计算的正确相机位置
    cam_min = np.min(cam_positions, axis=0)
    cam_max = np.max(cam_positions, axis=0)

    # Initialize overall min and max with camera positions
    overall_min = cam_min.copy()
    overall_max = cam_max.copy()

    # Update with points range if available
    points_array = None
    if points_3d is not None and len(points_3d) > 0:
        points_array = np.array(list(points_3d.values()))
        points_min = np.min(points_array, axis=0)
        points_max = np.max(points_array, axis=0)

        # Update overall min and max to include both cameras and points
        overall_min = np.minimum(overall_min, points_min)
        overall_max = np.maximum(overall_max, points_max)

        # Plot 3D points
        ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2],
                   c='blue', s=1, alpha=0.5, label='3D Points')

    # Plot camera positions (使用正确计算的相机位置)
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
               c='red', s=100, marker='o', label='Cameras')

    # Add camera IDs as text labels
    for i, (cam_id, ext) in enumerate(camera_extrinsics.items()):
        R = ext.rotation
        t = ext.translation
        pos = -R.T @ t  # 正确的相机世界坐标
        ax.text(pos[0], pos[1], pos[2], f" {cam_id}", color='black')

    # Plot camera orientations (principal axes)
    arrow_length = avg_dist * 0.2  # Scale arrows based on average camera distance

    # Create dummy arrows for legend
    dummy_x = ax.quiver(0, 0, 0, 0, 0, 0, color='red', label='Camera X-axis')
    dummy_y = ax.quiver(0, 0, 0, 0, 0, 0, color='green', label='Camera Y-axis')
    dummy_z = ax.quiver(0, 0, 0, 0, 0, 0, color='blue', label='Camera Z-axis (view direction)')
    dummy_direction = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', linewidth=2, label='Camera facing direction')

    # 添加原点坐标轴
    origin = np.array([0, 0, 0])
    axis_length = avg_dist * 0.5  # 使原点坐标轴长度为相机平均距离的50%

    # X轴 - 红色
    ax.quiver(origin[0], origin[1], origin[2],
              axis_length, 0, 0,
              color='darkred', arrow_length_ratio=0.15, linewidth=3, label='World X-axis')

    # Y轴 - 绿色
    ax.quiver(origin[0], origin[1], origin[2],
              0, axis_length, 0,
              color='darkgreen', arrow_length_ratio=0.15, linewidth=3, label='World Y-axis')

    # Z轴 - 蓝色
    ax.quiver(origin[0], origin[1], origin[2],
              0, 0, axis_length,
              color='darkblue', arrow_length_ratio=0.15, linewidth=3, label='World Z-axis')

    # 添加原点文本标签
    ax.text(0, 0, 0, " Origin (0,0,0)", color='black', fontweight='bold')

    for cam_id, ext in camera_extrinsics.items():
        R = ext.rotation
        t = ext.translation
        pos = -R.T @ t  # 正确的相机世界坐标

        # 确保相机朝向正交
        # 通过QR分解或SVD可以获得正交的旋转矩阵
        # 这里使用简单的方法：先确定z轴，然后构建正交的x和y轴

        # # 获取z轴方向 (相机光轴)
        # z_axis = R.T[:, 2]
        # z_axis = z_axis / np.linalg.norm(z_axis)
        #
        # # 选择一个临时向量来与z轴叉乘，生成正交的x轴
        # # 如果z轴与y轴接近平行，则选用x轴作为临时向量，否则选用y轴
        # temp = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.9 else np.array([1, 0, 0])
        #
        # # 生成正交的x轴
        # # x_axis = np.cross(temp, z_axis)
        # x_axis = R.T[:, 0]
        # x_axis = x_axis / np.linalg.norm(x_axis)
        #
        # # 生成正交的y轴
        # # y_axis = np.cross(z_axis, x_axis)
        # y_axis = R.T[:, 1]
        # y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis, y_axis, z_axis = get_cam_coor(R)

        # 构建正交的旋转矩阵
        R_world = np.column_stack((x_axis, y_axis, z_axis))

        # 确保所有轴使用相同长度
        # X axis - Red
        ax.quiver(pos[0], pos[1], pos[2],
                  R_world[0, 0] * arrow_length, R_world[1, 0] * arrow_length, R_world[2, 0] * arrow_length,
                  color='red', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

        # Y axis - Green
        ax.quiver(pos[0], pos[1], pos[2],
                  R_world[0, 1] * arrow_length, R_world[1, 1] * arrow_length, R_world[2, 1] * arrow_length,
                  color='green', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

        # Z axis (viewing direction) - Blue
        ax.quiver(pos[0], pos[1], pos[2],
                  R_world[0, 2] * arrow_length, R_world[1, 2] * arrow_length, R_world[2, 2] * arrow_length,
                  color='blue', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

        # Add a prominent arrow showing the camera facing direction (Positive Z-axis)
        view_dir = R_world[:, 2]  # 相机Z轴在世界坐标系中的方向
        ax.quiver(pos[0], pos[1], pos[2],
                  view_dir[0], view_dir[1], view_dir[2],
                  color='magenta', linewidth=2, arrow_length_ratio=0.15,
                  length=arrow_length * 1.5, normalize=True)

    # Plot initial camera positions if available
    if initial_extrinsics is not None:
        init_positions = []
        for ext in initial_extrinsics.values():
            R = ext.rotation
            t = ext.translation
            init_pos = -R.T @ t
            init_positions.append(init_pos)

        init_positions = np.array(init_positions)
        ax.scatter(init_positions[:, 0], init_positions[:, 1], init_positions[:, 2],
                   c='orange', s=50, marker='x', label='Initial Cameras')

        # Connect initial and final positions with lines
        for i, cam_id in enumerate(camera_extrinsics.keys()):
            init_R = initial_extrinsics[cam_id].rotation
            init_t = initial_extrinsics[cam_id].translation
            init_pos = -init_R.T @ init_t

            final_R = camera_extrinsics[cam_id].rotation
            final_t = camera_extrinsics[cam_id].translation
            final_pos = -final_R.T @ final_t

            ax.plot([init_pos[0], final_pos[0]],
                    [init_pos[1], final_pos[1]],
                    [init_pos[2], final_pos[2]],
                    'k--', alpha=0.3)

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction: Points and Cameras')

    # Calculate the overall range and center based on both points and cameras
    overall_range = np.max(overall_max - overall_min) / 2.0
    mid_x = (overall_max[0] + overall_min[0]) * 0.5
    mid_y = (overall_max[1] + overall_min[1]) * 0.5
    mid_z = (overall_max[2] + overall_min[2]) * 0.5

    # Set equal aspect ratio for all axes using the overall range
    ax.set_xlim(mid_x - overall_range, mid_x + overall_range)
    ax.set_ylim(mid_y - overall_range, mid_y + overall_range)
    ax.set_zlim(mid_z - overall_range, mid_z + overall_range)

    # Add a comprehensive legend with custom handles
    ax.legend(loc='upper right', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.savefig('camera_points_3d.png', dpi=300)
    plt.show()

    print("\nVisualization saved to camera_points_3d.png")

    # Create a second plot with only cameras for better visibility
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    # For second plot, use only camera positions for range calculation
    cam_range = np.array([
        np.max(cam_positions[:, 0]) - np.min(cam_positions[:, 0]),
        np.max(cam_positions[:, 1]) - np.min(cam_positions[:, 1]),
        np.max(cam_positions[:, 2]) - np.min(cam_positions[:, 2])
    ]).max() / 2.0

    cam_mid_x = (np.max(cam_positions[:, 0]) + np.min(cam_positions[:, 0])) * 0.5
    cam_mid_y = (np.max(cam_positions[:, 1]) + np.min(cam_positions[:, 1])) * 0.5
    cam_mid_z = (np.max(cam_positions[:, 2]) + np.min(cam_positions[:, 2])) * 0.5

    # Plot camera positions
    ax2.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
                c='red', s=100, marker='o', label='Cameras')

    # Add camera IDs as text labels
    for i, (cam_id, ext) in enumerate(camera_extrinsics.items()):
        R = ext.rotation
        t = ext.translation
        pos = -R.T @ t  # 正确的相机世界坐标
        ax2.text(pos[0], pos[1], pos[2], f" {cam_id}", color='black')

    # Plot camera orientations (principal axes)
    # Create dummy arrows for legend
    dummy_x = ax2.quiver(0, 0, 0, 0, 0, 0, color='red', label='Camera X-axis')
    dummy_y = ax2.quiver(0, 0, 0, 0, 0, 0, color='green', label='Camera Y-axis')
    dummy_z = ax2.quiver(0, 0, 0, 0, 0, 0, color='blue', label='Camera Z-axis')
    dummy_direction = ax2.quiver(0, 0, 0, 0, 0, 0, color='magenta', linewidth=2,
                                 label='Camera facing direction (positive z in opencv coor)')

    # 添加原点坐标轴（第二个图）
    axis_length = cam_range * 0.5  # 使原点坐标轴长度为相机范围的50%

    # X轴 - 红色
    ax2.quiver(origin[0], origin[1], origin[2],
               axis_length, 0, 0,
               color='darkred', arrow_length_ratio=0.15, linewidth=3, label='World X-axis')

    # Y轴 - 绿色
    ax2.quiver(origin[0], origin[1], origin[2],
               0, axis_length, 0,
               color='darkgreen', arrow_length_ratio=0.15, linewidth=3, label='World Y-axis')

    # Z轴 - 蓝色
    ax2.quiver(origin[0], origin[1], origin[2],
               0, 0, axis_length,
               color='darkblue', arrow_length_ratio=0.15, linewidth=3, label='World Z-axis')

    # 添加原点文本标签
    ax2.text(0, 0, 0, " Origin (0,0,0)", color='black', fontweight='bold')

    for cam_id, ext in camera_extrinsics.items():
        R = ext.rotation
        t = ext.translation
        pos = -R.T @ t  # 正确的相机世界坐标

        # 确保相机朝向正交
        # 通过QR分解或SVD可以获得正交的旋转矩阵
        # 这里使用简单的方法：先确定z轴，然后构建正交的x和y轴

        # # 获取z轴方向 (相机光轴)
        # z_axis = R.T[:, 2]
        # z_axis = z_axis / np.linalg.norm(z_axis)
        #
        # # 选择一个临时向量来与z轴叉乘，生成正交的x轴
        # # 如果z轴与y轴接近平行，则选用x轴作为临时向量，否则选用y轴
        # temp = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.9 else np.array([1, 0, 0])
        #
        # # 生成正交的x轴
        # x_axis = np.cross(temp, z_axis)
        # x_axis = x_axis / np.linalg.norm(x_axis)
        #
        # # 生成正交的y轴
        # y_axis = np.cross(z_axis, x_axis)
        # y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis, y_axis, z_axis = get_cam_coor(R)
        # 构建正交的旋转矩阵
        R_world = np.column_stack((x_axis, y_axis, z_axis))

        # 确保所有轴使用相同长度（第二个图）
        # X axis - Red
        ax2.quiver(pos[0], pos[1], pos[2],
                   R_world[0, 0] * arrow_length, R_world[1, 0] * arrow_length, R_world[2, 0] * arrow_length,
                   color='red', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

        # Y axis - Green
        ax2.quiver(pos[0], pos[1], pos[2],
                   R_world[0, 1] * arrow_length, R_world[1, 1] * arrow_length, R_world[2, 1] * arrow_length,
                   color='green', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

        # Z axis - Blue
        ax2.quiver(pos[0], pos[1], pos[2],
                   R_world[0, 2] * arrow_length, R_world[1, 2] * arrow_length, R_world[2, 2] * arrow_length,
                   color='blue', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

        # Add a prominent arrow showing the camera facing direction (positive Z-axis)
        view_dir = R_world[:, 2]  # 相机Z轴在世界坐标系中的方向
        ax2.quiver(pos[0], pos[1], pos[2],
                   view_dir[0], view_dir[1], view_dir[2],
                   color='magenta', linewidth=2, arrow_length_ratio=0.15,
                   length=arrow_length * 1.5, normalize=True)

    # Plot initial camera positions if available
    if initial_extrinsics is not None:
        init_positions = []
        for ext in initial_extrinsics.values():
            R = ext.rotation
            t = ext.translation
            init_pos = -R.T @ t
            init_positions.append(init_pos)

        init_positions = np.array(init_positions)
        ax2.scatter(init_positions[:, 0], init_positions[:, 1], init_positions[:, 2],
                    c='orange', s=50, marker='x', label='Initial Cameras')

        # Connect initial and final positions with lines
        for i, cam_id in enumerate(camera_extrinsics.keys()):
            init_R = initial_extrinsics[cam_id].rotation
            init_t = initial_extrinsics[cam_id].translation
            init_pos = -init_R.T @ init_t

            final_R = camera_extrinsics[cam_id].rotation
            final_t = camera_extrinsics[cam_id].translation
            final_pos = -final_R.T @ final_t

            ax2.plot([init_pos[0], final_pos[0]],
                     [init_pos[1], final_pos[1]],
                     [init_pos[2], final_pos[2]],
                     'k--', alpha=0.3)

    # Set axis labels and title
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Camera Positions and Orientations')

    # Set equal aspect ratio for second plot using only camera range
    ax2.set_xlim(cam_mid_x - cam_range, cam_mid_x + cam_range)
    ax2.set_ylim(cam_mid_y - cam_range, cam_mid_y + cam_range)
    ax2.set_zlim(cam_mid_z - cam_range, cam_mid_z + cam_range)

    # Add a comprehensive legend with custom handles
    ax2.legend(loc='upper right', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.savefig('cameras_3d.png', dpi=300)
    plt.show()

    print("Camera-only visualization saved to cameras_3d.png")


def save_point_cloud(points_3d, filename="point_cloud.ply"):
    """
    Save 3D points as a PLY file

    Args:
        points_3d: Dictionary of 3D points
        filename: Output filename
    """
    if not points_3d:
        print("No points to save.")
        return

    points_array = np.array(list(points_3d.values()))

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_array)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for point in points_array:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Point cloud saved to {filename}")


import numpy as np
from scipy.spatial.transform import Rotation


class CameraExtrinsic:
    def __init__(self, rotation, translation):
        """
        初始化相机外参

        Args:
            rotation: 3x3 旋转矩阵
            translation: 3x1 平移向量
        """
        self.rotation = rotation
        self.translation = translation


def manual_input_camera_matrix():
    """手动输入相机标定矩阵"""
    num_cameras = int(input("请输入相机数量: "))
    camera_extrinsics = {}

    for i in range(num_cameras):
        print(f"\n===== 相机 {i + 1} =====")
        cam_id = input(f"请输入相机ID (默认为 'cam{i + 1}'): ") or f"cam{i + 1}"

        print("请输入3x3旋转矩阵 (按行输入，以空格分隔):")
        rotation = []
        for j in range(3):
            row = list(map(float, input(f"行 {j + 1}: ").split()))
            rotation.append(row)
        rotation = np.array(rotation)

        print("请输入平移向量 (3个值，以空格分隔):")
        translation = list(map(float, input().split()))
        translation = np.array(translation)

        camera_extrinsics[cam_id] = CameraExtrinsic(rotation, translation)

    return camera_extrinsics


def parse_calibration_matrix():
    """解析标准格式的相机标定矩阵"""
    num_cameras = int(input("请输入相机数量: "))
    camera_extrinsics = {}

    for i in range(num_cameras):
        print(f"\n===== 相机 {i + 1} =====")
        cam_id = input(f"请输入相机ID (默认为 'cam{i + 1}'): ") or f"cam{i + 1}"

        print("请输入4x4标定矩阵 (按行输入，以空格分隔):")
        matrix = []
        for j in range(4):
            row = list(map(float, input(f"行 {j + 1}: ").split()))
            matrix.append(row)
        matrix = np.array(matrix)

        # 从4x4矩阵中提取旋转和平移
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        camera_extrinsics[cam_id] = CameraExtrinsic(rotation, translation)

    return camera_extrinsics


def parse_3x4_calibration_matrix():
    """解析3x4格式的相机标定矩阵（自动添加第四行[0,0,0,1]）"""
    num_cameras = int(input("请输入相机数量: "))
    camera_extrinsics = {}

    for i in range(num_cameras):
        print(f"\n===== 相机 {i + 1} =====")
        cam_id = input(f"请输入相机ID (默认为 'cam{i + 1}'): ") or f"cam{i + 1}"

        print("请输入3x4标定矩阵 (按行输入，以空格分隔或分号分隔):")
        matrix_str = input("请将整个矩阵复制粘贴在这里: ")

        # 处理可能的分号分隔格式
        if ';' in matrix_str:
            rows = matrix_str.strip().split(';')
            matrix = []
            for row_str in rows:
                row = list(map(float, row_str.strip().split()))
                matrix.append(row)
        else:
            # 假设按行输入
            matrix = []
            matrix.append(list(map(float, matrix_str.strip().split())))
            for j in range(1, 3):
                row = list(map(float, input(f"行 {j + 1}: ").strip().split()))
                matrix.append(row)

        # 添加第四行 [0,0,0,1]
        matrix.append([0, 0, 0, 1])
        matrix = np.array(matrix)

        # 从矩阵中提取旋转和平移
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        camera_extrinsics[cam_id] = CameraExtrinsic(rotation, translation)

    return camera_extrinsics


def provide_example_data():
    """提供示例数据进行演示"""
    camera_extrinsics = {}

    # 相机1 - 面向正面
    R1 = np.eye(3)
    t1 = np.array([0, 0, 0])
    camera_extrinsics['cam1'] = CameraExtrinsic(R1, t1)

    # 相机2 - 右侧向内旋转 45 度
    R2 = Rotation.from_euler('y', -45, degrees=True).as_matrix()  # 对于位于右侧的相机，负角度表示向内旋转
    t2 = np.array([2, 0, 2])
    camera_extrinsics['cam2'] = CameraExtrinsic(R2, t2)

    # 相机3 - 左侧 -30 度
    R3 = Rotation.from_euler('y', -30, degrees=True).as_matrix()
    t3 = np.array([-2, 0, 2])
    camera_extrinsics['cam3'] = CameraExtrinsic(R3, t3)

    # 相机4 - 俯视 30 度
    R4 = Rotation.from_euler('x', -30, degrees=True).as_matrix()
    t4 = np.array([0, -2, 2])
    camera_extrinsics['cam4'] = CameraExtrinsic(R4, t4)

    # 相机5 - 仰视 20 度
    R5 = Rotation.from_euler('x', 20, degrees=True).as_matrix()
    t5 = np.array([0, 2, 2])
    camera_extrinsics['cam5'] = CameraExtrinsic(R5, t5)

    return camera_extrinsics


def apply_example_calibration():
    """使用您提供的标定矩阵示例"""
    camera_extrinsics = {}

    # 使用您提供的标定矩阵
    calibration_str = "-1.012936e+02 8.533322e+02 -6.316735e+02 1.866323e+03; 8.306266e+02 9.075726e+01 -5.362521e+02 1.362220e+03; -2.577099e-02 1.358163e-02 -9.995756e-01 3.351540e+00"

    # 解析字符串为矩阵
    rows = calibration_str.strip().split(';')
    matrix = []
    for row_str in rows:
        row = list(map(float, row_str.strip().split()))
        matrix.append(row)

    # 添加第四行 [0,0,0,1]
    matrix.append([0, 0, 0, 1])
    matrix = np.array(matrix)

    # 从矩阵中提取旋转和平移
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    camera_extrinsics['cam_example'] = CameraExtrinsic(rotation, translation)

    print("\n已加载您提供的标定矩阵")

    return camera_extrinsics


if __name__ == "__main__":

    print("相机标定矩阵可视化工具")
    print("=" * 40)

    while True:
        print("\n请选择输入方式:")
        print("1. 手动输入旋转和平移")
        print("2. 输入4x4标定矩阵")
        print("3. 输入3x4标定矩阵（您提供的格式）")
        print("4. 使用示例数据")
        print("5. 使用您提供的标定矩阵示例")
        print("6. 退出")

        choice = input("\n请选择 (1-6): ")

        if choice == '1':
            camera_extrinsics = manual_input_camera_matrix()
        elif choice == '2':
            camera_extrinsics = parse_calibration_matrix()
        elif choice == '3':
            camera_extrinsics = parse_3x4_calibration_matrix()
        elif choice == '4':
            camera_extrinsics = provide_example_data()
            print("\n已加载示例数据，包含5个不同位置和朝向的相机")
        elif choice == '5':
            camera_extrinsics = apply_example_calibration()
        elif choice == '6':
            print("退出程序")
            break
        else:
            print("无效选择，请重试")
            continue

        # 是否创建3D点云
        create_points = input("\n是否创建3D点云? (y/n, 默认为n): ").lower() == 'y'

        points_3d = {}
        if create_points:
            num_points = int(input("请输入点的数量: "))
            print("创建随机3D点...")

            # 根据相机位置计算合适的点云范围
            positions = np.array([ext.translation for ext in camera_extrinsics.values()])
            center = np.mean(positions, axis=0)
            spread = np.std(positions, axis=0) * 2  # 点云范围是相机范围的两倍

            # 创建随机点云
            for i in range(num_points):
                # 生成在相机周围的随机点
                point = center + np.random.uniform(-spread, spread, 3)
                points_3d[f'point_{i}'] = point

        # 运行分析和可视化
        analyze_optimization_results(camera_extrinsics, points_3d)

        # 是否继续
        if input("\n是否继续? (y/n, 默认为y): ").lower() == 'n':
            print("退出程序")
            break