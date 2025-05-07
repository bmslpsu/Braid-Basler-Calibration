import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox
import pickle
from scipy.spatial.transform import Rotation
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, StringVar, Entry, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import sys
from typing import Tuple
from dataclasses import dataclass
matplotlib.use("TkAgg")

@dataclass
class CameraIntrinsic:
    """Camera Intrinsic"""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]


@dataclass
class CameraExtrinsic:
    """Camera Extrinsic Parameters"""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector

def get_cam_coor(R):
    """获取相机坐标系的三个轴在世界坐标系中的表示"""
    # 相机坐标系的三个轴在世界坐标系中的表示
    R_world = R.T  # 将相机到世界的旋转转换为世界到相机
    x_axis = R_world[:, 0]  # 相机的X轴在世界坐标系中的方向
    y_axis = R_world[:, 1]  # 相机的Y轴在世界坐标系中的方向
    z_axis = R_world[:, 2]  # 相机的Z轴（视线方向）在世界坐标系中的方向
    return x_axis, y_axis, z_axis


def load_optimization_results(base_dir, extrinsic_file_name, intrinsic_file_name, points_file_name, format_type='json'):
    """从指定目录加载优化结果"""

    # 检查文件名并构建完整路径
    extrinsics_filename = os.path.join(base_dir, f"{extrinsic_file_name}.{format_type}")
    intrinsics_filename = os.path.join(base_dir, f"{intrinsic_file_name}.{format_type}")
    points3d_filename = os.path.join(base_dir, f"{points_file_name}.{format_type}")

    # 检查文件是否存在
    if not os.path.exists(extrinsics_filename):
        raise FileNotFoundError(f"no extrinsic: {extrinsics_filename}")
    if not os.path.exists(intrinsics_filename):
        raise FileNotFoundError(f"no intrinsic: {intrinsics_filename}")
    if not os.path.exists(points3d_filename):
        raise FileNotFoundError(f"no 3D points: {points3d_filename}")

    # 根据格式加载文件
    if format_type.lower() == 'json':
        # JSON格式加载
        with open(extrinsics_filename, 'r') as f:
            json_extrinsics = json.load(f)

        with open(intrinsics_filename, 'r') as f:
            json_intrinsics = json.load(f)

        with open(points3d_filename, 'r') as f:
            json_points3d = json.load(f)

        # 转换点云数据
        points_3d = {}
        for frame, point in json_points3d.items():
            points_3d[int(frame)] = np.array(point)

        # 转换外参数据到CameraExtrinsic对象
        camera_extrinsics = {}
        for cam_id, ext_data in json_extrinsics.items():
            rotation = np.array(ext_data['rotation'])
            translation = np.array(ext_data['translation'])
            camera_extrinsics[cam_id] = CameraExtrinsicSimple(rotation, translation)

        # 转换内参数据（简化处理，实际应用中可能需要更详细的处理）
        camera_intrinsics = {}
        for cam_id, intr_data in json_intrinsics.items():
            camera_matrix = np.array(intr_data['camera_matrix'])
            dist_coeffs = np.array(intr_data['dist_coeffs'])
            image_size = tuple(intr_data['image_size'])
            camera_intrinsics[cam_id] = CameraIntrinsicSimple(camera_matrix, dist_coeffs, image_size)

    elif format_type.lower() == 'pickle':
        # Pickle格式加载
        with open(extrinsics_filename, 'rb') as f:
            camera_extrinsics = pickle.load(f)

        with open(intrinsics_filename, 'rb') as f:
            camera_intrinsics = pickle.load(f)

        with open(points3d_filename, 'rb') as f:
            points_3d = pickle.load(f)

    else:
        raise ValueError(f"format not support: {format_type}")

    print(f"successfully loaded data:")
    print(f"  - camera extrinsic: {len(camera_extrinsics)} cameras")
    print(f"  - camera intrinsic: {len(camera_intrinsics)} cameras")
    print(f"  - 3D points: {len(points_3d)} points")

    return camera_extrinsics, camera_intrinsics, points_3d


class CameraExtrinsicSimple:
    """简化的相机外参类，用于可视化"""

    def __init__(self, rotation, translation):
        self.rotation = rotation  # 3x3旋转矩阵
        self.translation = translation  # 3x1平移向量

    def to_rt(self):
        """返回(rvec, tvec)元组"""
        import cv2
        rvec, _ = cv2.Rodrigues(self.rotation)
        return rvec, self.translation


class CameraIntrinsicSimple:
    """简化的相机内参类，用于可视化"""

    def __init__(self, camera_matrix, dist_coeffs, image_size):
        self.camera_matrix = camera_matrix  # 3x3内参矩阵
        self.dist_coeffs = dist_coeffs  # 畸变系数
        self.image_size = image_size


class VisualizationApp:
    def __init__(self, root, box_x=1.0, box_y=1.0, box_z=1.0):
        self.root = root
        self.root.title("3D point and camera coordinate adjusting")
        self.root.geometry("1280x800")

        # 初始化数据
        self.camera_extrinsics = None
        self.camera_intrinsics = None
        self.points_3d = None
        self.box_dimensions = (box_x, box_y, box_z)  # 初始长方体尺寸 (x, y, z)

        # 变换参数
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 欧拉角 (度)
        self.translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.scale = 1.0

        # 添加变量以存储输入框的StringVar对象
        self.rx_var = StringVar(value="0.0")
        self.ry_var = StringVar(value="0.0")
        self.rz_var = StringVar(value="0.0")
        self.tx_var = StringVar(value="0.0")
        self.ty_var = StringVar(value="0.0")
        self.tz_var = StringVar(value="0.0")
        self.scale_var = StringVar(value="1.0")

        # 防止频繁更新的变量
        self.update_pending = False
        self.update_delay = 50  # 50毫秒的延迟

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 创建菜单栏
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="file", menu=file_menu)
        file_menu.add_command(label="load optimized results", command=self.load_results)
        file_menu.add_command(label="save camera extrinsics", command=self.save_extrinsics)
        file_menu.add_separator()
        file_menu.add_command(label="quit", command=self.root.quit)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="edit", menu=edit_menu)
        edit_menu.add_command(label="set box dimensions", command=self.set_box_dimensions)
        edit_menu.add_command(label="reset transformation", command=self.reset_transformation)

        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建左侧控制面板
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 旋转控制
        rotation_frame = tk.LabelFrame(control_frame, text="rotation (deg)", padx=10, pady=10)
        rotation_frame.pack(fill=tk.X, pady=5)

        # X旋转控制 - 添加滑块和输入框
        rx_frame = Frame(rotation_frame)
        rx_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(rx_frame, text="X rotation:").pack(side=tk.LEFT)
        self.rx_entry = Entry(rx_frame, textvariable=self.rx_var, width=8)
        self.rx_entry.pack(side=tk.RIGHT)
        self.rx_entry.bind("<Return>", lambda e: self.update_from_entry('rx'))
        self.rx_entry.bind("<FocusOut>", lambda e: self.update_from_entry('rx'))

        self.rx_slider = tk.Scale(rotation_frame, from_=-180, to=180, orient=tk.HORIZONTAL, resolution=0.001,
                                  command=lambda v: self.update_rotation(0, float(v)))
        self.rx_slider.set(0)
        self.rx_slider.grid(row=1, column=0, columnspan=2, sticky=tk.EW)

        # Y旋转控制
        ry_frame = Frame(rotation_frame)
        ry_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(ry_frame, text="Y rotation:").pack(side=tk.LEFT)
        self.ry_entry = Entry(ry_frame, textvariable=self.ry_var, width=8)
        self.ry_entry.pack(side=tk.RIGHT)
        self.ry_entry.bind("<Return>", lambda e: self.update_from_entry('ry'))
        self.ry_entry.bind("<FocusOut>", lambda e: self.update_from_entry('ry'))

        self.ry_slider = tk.Scale(rotation_frame, from_=-180, to=180, orient=tk.HORIZONTAL, resolution=0.001,
                                  command=lambda v: self.update_rotation(1, float(v)))
        self.ry_slider.set(0)
        self.ry_slider.grid(row=3, column=0, columnspan=2, sticky=tk.EW)

        # Z旋转控制
        rz_frame = Frame(rotation_frame)
        rz_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(rz_frame, text="Z rotation:").pack(side=tk.LEFT)
        self.rz_entry = Entry(rz_frame, textvariable=self.rz_var, width=8)
        self.rz_entry.pack(side=tk.RIGHT)
        self.rz_entry.bind("<Return>", lambda e: self.update_from_entry('rz'))
        self.rz_entry.bind("<FocusOut>", lambda e: self.update_from_entry('rz'))

        self.rz_slider = tk.Scale(rotation_frame, from_=-180, to=180, orient=tk.HORIZONTAL, resolution=0.001,
                                  command=lambda v: self.update_rotation(2, float(v)))
        self.rz_slider.set(0)
        self.rz_slider.grid(row=5, column=0, columnspan=2, sticky=tk.EW)

        # 平移控制
        translation_frame = tk.LabelFrame(control_frame, text="translation", padx=10, pady=10)
        translation_frame.pack(fill=tk.X, pady=5)

        # X平移控制
        tx_frame = Frame(translation_frame)
        tx_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(tx_frame, text="X trans:").pack(side=tk.LEFT)
        self.tx_entry = Entry(tx_frame, textvariable=self.tx_var, width=8)
        self.tx_entry.pack(side=tk.RIGHT)
        self.tx_entry.bind("<Return>", lambda e: self.update_from_entry('tx'))
        self.tx_entry.bind("<FocusOut>", lambda e: self.update_from_entry('tx'))

        self.tx_slider = tk.Scale(translation_frame, from_=-10, to=10, orient=tk.HORIZONTAL, resolution=0.001,
                                  command=lambda v: self.update_translation(0, float(v)))
        self.tx_slider.set(0)
        self.tx_slider.grid(row=1, column=0, columnspan=2, sticky=tk.EW)

        # Y平移控制
        ty_frame = Frame(translation_frame)
        ty_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(ty_frame, text="Y trans:").pack(side=tk.LEFT)
        self.ty_entry = Entry(ty_frame, textvariable=self.ty_var, width=8)
        self.ty_entry.pack(side=tk.RIGHT)
        self.ty_entry.bind("<Return>", lambda e: self.update_from_entry('ty'))
        self.ty_entry.bind("<FocusOut>", lambda e: self.update_from_entry('ty'))

        self.ty_slider = tk.Scale(translation_frame, from_=-10, to=10, orient=tk.HORIZONTAL, resolution=0.001,
                                  command=lambda v: self.update_translation(1, float(v)))
        self.ty_slider.set(0)
        self.ty_slider.grid(row=3, column=0, columnspan=2, sticky=tk.EW)

        # Z平移控制
        tz_frame = Frame(translation_frame)
        tz_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(tz_frame, text="Z trans:").pack(side=tk.LEFT)
        self.tz_entry = Entry(tz_frame, textvariable=self.tz_var, width=8)
        self.tz_entry.pack(side=tk.RIGHT)
        self.tz_entry.bind("<Return>", lambda e: self.update_from_entry('tz'))
        self.tz_entry.bind("<FocusOut>", lambda e: self.update_from_entry('tz'))

        self.tz_slider = tk.Scale(translation_frame, from_=-10, to=10, orient=tk.HORIZONTAL, resolution=0.001,
                                  command=lambda v: self.update_translation(2, float(v)))
        self.tz_slider.set(0)
        self.tz_slider.grid(row=5, column=0, columnspan=2, sticky=tk.EW)

        # 缩放控制
        scale_frame = tk.LabelFrame(control_frame, text="scale", padx=10, pady=10)
        scale_frame.pack(fill=tk.X, pady=5)

        # 缩放控制
        scale_input_frame = Frame(scale_frame)
        scale_input_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(scale_input_frame, text="scale:").pack(side=tk.LEFT)
        self.scale_entry = Entry(scale_input_frame, textvariable=self.scale_var, width=8)
        self.scale_entry.pack(side=tk.RIGHT)
        self.scale_entry.bind("<Return>", lambda e: self.update_from_entry('scale'))
        self.scale_entry.bind("<FocusOut>", lambda e: self.update_from_entry('scale'))

        self.scale_slider = tk.Scale(scale_frame, from_=0.1, to=10, orient=tk.HORIZONTAL, resolution=0.001,
                                     command=lambda v: self.update_scale(float(v)))
        self.scale_slider.set(1.0)
        self.scale_slider.grid(row=1, column=0, columnspan=2, sticky=tk.EW)

        # 长方体信息
        box_frame = tk.LabelFrame(control_frame, text="ref box", padx=10, pady=10)
        box_frame.pack(fill=tk.X, pady=5)

        box_info = f"length (X): {self.box_dimensions[0]}m\nwidth (Y): {self.box_dimensions[1]}m\nheight (Z): {self.box_dimensions[2]}m"
        self.box_label = tk.Label(box_frame, text=box_info, anchor=tk.W, justify=tk.LEFT)
        self.box_label.pack(fill=tk.X)

        # 状态信息
        self.status_frame = tk.LabelFrame(control_frame, text="status", padx=10, pady=10)
        self.status_frame.pack(fill=tk.X, pady=5)

        self.status_label = tk.Label(self.status_frame, text="data not loaded", anchor=tk.W, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X)

        # 操作按钮
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.load_button = tk.Button(button_frame, text="load data", command=self.load_results)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame, text="save results", command=self.save_extrinsics)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # 创建图表区域
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加Matplotlib导航工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始化显示
        self.initialize_plot()

    def update_from_entry(self, entry_type):
        """从文本框更新值"""
        try:
            if entry_type == 'rx':
                value = float(self.rx_var.get())
                self.rx_slider.set(value)
                self.update_rotation(0, value)
            elif entry_type == 'ry':
                value = float(self.ry_var.get())
                self.ry_slider.set(value)
                self.update_rotation(1, value)
            elif entry_type == 'rz':
                value = float(self.rz_var.get())
                self.rz_slider.set(value)
                self.update_rotation(2, value)
            elif entry_type == 'tx':
                value = float(self.tx_var.get())
                self.tx_slider.set(value)
                self.update_translation(0, value)
            elif entry_type == 'ty':
                value = float(self.ty_var.get())
                self.ty_slider.set(value)
                self.update_translation(1, value)
            elif entry_type == 'tz':
                value = float(self.tz_var.get())
                self.tz_slider.set(value)
                self.update_translation(2, value)
            elif entry_type == 'scale':
                value = float(self.scale_var.get())
                self.scale_slider.set(value)
                self.update_scale(value)
        except ValueError:
            # 如果输入不是有效的浮点数，重置为当前值
            self.update_entry_values()

    def update_entry_values(self):
        """更新输入框的值"""
        self.rx_var.set(f"{self.rotation[0]:.3f}")
        self.ry_var.set(f"{self.rotation[1]:.3f}")
        self.rz_var.set(f"{self.rotation[2]:.3f}")
        self.tx_var.set(f"{self.translation[0]:.3f}")
        self.ty_var.set(f"{self.translation[1]:.3f}")
        self.tz_var.set(f"{self.translation[2]:.3f}")
        self.scale_var.set(f"{self.scale:.3f}")

    def initialize_plot(self):
        """初始化3D图表"""
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D point and camera coordinate adjusting')

        # 绘制坐标轴和参考长方体
        self.draw_reference_box()

        # 设置初始视图范围
        box_max_dim = max(self.box_dimensions)
        self.ax.set_xlim(-box_max_dim, box_max_dim)
        self.ax.set_ylim(-box_max_dim, box_max_dim)
        self.ax.set_zlim(-box_max_dim, box_max_dim)

        # 初始化视图状态标志
        self.view_initialized = False

        # 设置图例
        self.ax.legend()

        # 刷新画布
        self.canvas.draw()
    def draw_reference_box(self):
        """绘制参考长方体和坐标轴"""
        x_size, y_size, z_size = self.box_dimensions

        # 计算长方体中心
        center_x = x_size / 2
        center_y = y_size / 2
        center_z = z_size / 2

        # 长方体的顶点 (以长方体中心为原点)
        x = [-center_x, center_x, center_x, -center_x, -center_x, center_x, center_x, -center_x]
        y = [-center_y, -center_y, center_y, center_y, -center_y, -center_y, center_y, center_y]
        z = [-center_z, -center_z, -center_z, -center_z, center_z, center_z, center_z, center_z]

        # 绘制长方体的12条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)  # 连接底面和顶面的边
        ]

        for edge in edges:
            self.ax.plot([x[edge[0]], x[edge[1]]],
                         [y[edge[0]], y[edge[1]]],
                         [z[edge[0]], z[edge[1]]],
                         color='gray', linestyle='--', linewidth=1)

        # 添加尺寸标记
        self.ax.text(center_x, 0, 0, f"X: {x_size}m", color='red')
        self.ax.text(0, center_y, 0, f"Y: {y_size}m", color='green')
        self.ax.text(0, 0, center_z, f"Z: {z_size}m", color='blue')

        # 绘制世界坐标系 (原点在长方体中心)
        origin = np.array([0, 0, 0])
        axis_length = max(self.box_dimensions) / 2

        # X轴 - 红色
        self.ax.quiver(origin[0], origin[1], origin[2],
                       axis_length, 0, 0,
                       color='darkred', arrow_length_ratio=0.15, linewidth=3, label='world X')

        # Y轴 - 绿色
        self.ax.quiver(origin[0], origin[1], origin[2],
                       0, axis_length, 0,
                       color='darkgreen', arrow_length_ratio=0.15, linewidth=3, label='world Y')

        # Z轴 - 蓝色
        self.ax.quiver(origin[0], origin[1], origin[2],
                       0, 0, axis_length,
                       color='darkblue', arrow_length_ratio=0.15, linewidth=3, label='world Z')

        # 添加原点文本标签
        self.ax.text(0, 0, 0, " origin (0,0,0)", color='black', fontweight='bold')

    def schedule_update_plot(self):
        """安排延迟更新以避免频繁重绘"""
        # 如果已经有更新在等待，不再安排新的更新
        if self.update_pending:
            return

        self.update_pending = True
        self.root.after(self.update_delay, self.do_update_plot)

    def do_update_plot(self):
        """实际执行更新"""
        self.update_pending = False
        if self.camera_extrinsics is not None and self.points_3d is not None:
            self.update_plot()

    def load_results(self):
        """加载优化结果"""
        # 弹出文件选择对话框
        result_dir = filedialog.askdirectory(title="folder with optimized files")

        if not result_dir:
            return

        try:
            # 尝试加载JSON格式
            self.camera_extrinsics, self.camera_intrinsics, self.points_3d = load_optimization_results(result_dir,
                                                                                                       format_type='json')

            # 保存原始数据副本，用于后续的变换
            self.original_extrinsics = {cam_id: CameraExtrinsicSimple(ext.rotation.copy(), ext.translation.copy())
                                        for cam_id, ext in self.camera_extrinsics.items()}

            self.original_points = {}
            for frame, point in self.points_3d.items():
                self.original_points[frame] = point.copy()

            # 更新状态
            self.update_status()

            # 重置变换
            self.reset_transformation()

            # 更新图表
            self.update_plot()

            messagebox.showinfo("success",
                                f"Loaded files:\n- num cameras: {len(self.camera_extrinsics)} \n- num 3D points: {len(self.points_3d)} ")

        except Exception as e:
            messagebox.showerror("error", f"loading failed: {str(e)}")

    def update_status(self):
        """更新状态信息"""
        if self.camera_extrinsics is None or self.points_3d is None:
            status_text = "data not loaded"
        else:
            status_text = f"data loaded:\n" \
                          f"- num cameras: {len(self.camera_extrinsics)} \n" \
                          f"- num 3D points: {len(self.points_3d)} \n" \
                          f"rotation: ({self.rotation[0]:.3f}, {self.rotation[1]:.3f}, {self.rotation[2]:.3f}) deg\n" \
                          f"translation: ({self.translation[0]:.3f}, {self.translation[1]:.3f}, {self.translation[2]:.3f})\n" \
                          f"scale: {self.scale:.3f}"

        self.status_label.config(text=status_text)

    def update_rotation(self, axis, value):
        """更新旋转参数"""
        self.rotation[axis] = value
        # 更新输入框值
        if axis == 0:
            self.rx_var.set(f"{value:.3f}")
        elif axis == 1:
            self.ry_var.set(f"{value:.3f}")
        elif axis == 2:
            self.rz_var.set(f"{value:.3f}")
        self.update_status()
        # 实时更新图表
        self.schedule_update_plot()

    def update_translation(self, axis, value):
        """更新平移参数"""
        self.translation[axis] = float(value)
        # 更新输入框值
        if axis == 0:
            self.tx_var.set(f"{value:.3f}")
        elif axis == 1:
            self.ty_var.set(f"{value:.3f}")
        elif axis == 2:
            self.tz_var.set(f"{value:.3f}")
        self.update_status()
        # 实时更新图表
        self.schedule_update_plot()

    def update_scale(self, value):
        """更新缩放参数"""
        self.scale = value
        self.scale_var.set(f"{value:.3f}")
        self.update_status()
        # 实时更新图表
        self.schedule_update_plot()

    def reset_transformation(self):
        """重置所有变换参数"""
        self.rotation = np.array([0, 0, 0])
        self.translation = np.array([0, 0, 0])
        self.scale = 1.0

        # 重置滑块
        self.rx_slider.set(0)
        self.ry_slider.set(0)
        self.rz_slider.set(0)
        self.tx_slider.set(0)
        self.ty_slider.set(0)
        self.tz_slider.set(0)
        self.scale_slider.set(1.0)

        # 重置输入框
        self.update_entry_values()

        self.update_status()

        # 重置视图初始化状态
        self.view_initialized = False

        # 如果已加载数据，则更新图表
        if self.camera_extrinsics is not None and self.points_3d is not None:
            self.update_plot()

    def set_box_dimensions(self):
        """设置参考长方体尺寸"""
        try:
            x_size = simpledialog.askfloat("set x", "enter x (m):", initialvalue=self.box_dimensions[0])
            if x_size is None:
                return

            y_size = simpledialog.askfloat("set y", "enter y (m):", initialvalue=self.box_dimensions[1])
            if y_size is None:
                return

            z_size = simpledialog.askfloat("set scale", "enter z (m):", initialvalue=self.box_dimensions[2])
            if z_size is None:
                return

            self.box_dimensions = (x_size, y_size, z_size)

            # 更新长方体信息标签
            box_info = f"length (X): {self.box_dimensions[0]}m\nwidth (Y): {self.box_dimensions[1]}m\nheight (Z): {self.box_dimensions[2]}m"
            self.box_label.config(text=box_info)

            # 更新图表
            self.update_plot()

        except Exception as e:
            messagebox.showerror("error", f"fail to set: {str(e)}")

    def apply_transformation(self):
        """应用当前的变换参数到数据"""
        if self.original_extrinsics is None or self.original_points is None:
            return

        # 创建旋转矩阵 (欧拉角转旋转矩阵)
        rx, ry, rz = np.radians(self.rotation)
        rotation = Rotation.from_euler('xyz', [rx, ry, rz])
        rot_matrix = rotation.as_matrix()

        # 应用变换到相机外参
        for cam_id in self.camera_extrinsics:
            # 获取原始相机位置
            orig_R = self.original_extrinsics[cam_id].rotation
            orig_t = self.original_extrinsics[cam_id].translation
            orig_pos = -orig_R.T @ orig_t

            # 应用变换
            # 1. 缩放
            scaled_pos = orig_pos * self.scale
            # 2. 旋转
            rotated_pos = rot_matrix @ scaled_pos
            # 3. 平移
            transformed_pos = rotated_pos + self.translation

            # 更新相机外参
            new_R = self.original_extrinsics[cam_id].rotation @ rot_matrix.T

            # 从新的位置和旋转计算新的平移向量
            new_t = -new_R @ transformed_pos

            # 更新相机外参
            self.camera_extrinsics[cam_id].rotation = new_R
            self.camera_extrinsics[cam_id].translation = new_t

        # 应用变换到3D点
        for frame in self.points_3d:
            # 获取原始点坐标
            orig_point = self.original_points[frame]

            # 应用变换
            # 1. 缩放
            scaled_point = orig_point * self.scale
            # 2. 旋转
            rotated_point = rot_matrix @ scaled_point
            # 3. 平移
            transformed_point = rotated_point + self.translation

            # 更新点坐标
            self.points_3d[frame] = transformed_point

    def update_plot(self):
        """更新3D图表，同时保留当前视图状态"""
        if self.camera_extrinsics is None or self.points_3d is None:
            return

        # 保存当前的视图状态
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_zlim = self.ax.get_zlim()

        # 记录当前的视角
        elev, azim = self.ax.elev, self.ax.azim

        # 应用变换到数据
        self.apply_transformation()

        # 清除当前图表
        self.ax.clear()

        # 设置图表标题和轴标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D point and camera coordinate adjusting')

        # 绘制参考长方体和坐标轴 (坐标原点在长方体中心)
        self.draw_reference_box()

        # 计算相机位置
        positions = []
        for cam_id, ext in self.camera_extrinsics.items():
            R = ext.rotation
            t = ext.translation
            camera_pos = -R.T @ t
            positions.append(camera_pos)

        positions = np.array(positions)

        # 计算点云位置（如果存在）
        if self.points_3d and len(self.points_3d) > 0:
            points_array = np.array(list(self.points_3d.values()))
            # 绘制3D点
            self.ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2],
                            c='blue', s=1, alpha=0.5, label='3D points')

        # 计算初始视图范围（仅在第一次更新时使用）
        if not hasattr(self, 'view_initialized') or not self.view_initialized:
            if len(positions) > 0:
                # 计算相机位置范围
                cam_min = np.min(positions, axis=0)
                cam_max = np.max(positions, axis=0)

                # 初始化总体范围
                overall_min = cam_min.copy()
                overall_max = cam_max.copy()

                # 如果有点云，更新范围
                if self.points_3d and len(self.points_3d) > 0:
                    points_min = np.min(points_array, axis=0)
                    points_max = np.max(points_array, axis=0)

                    overall_min = np.minimum(overall_min, points_min)
                    overall_max = np.maximum(overall_max, points_max)

                # 计算总体范围
                box_max_dim = max(self.box_dimensions)
                scene_max_dim = max(np.max(overall_max - overall_min) / 2.0, box_max_dim)

                # 设置坐标轴范围 (确保长方体在视图中)
                view_range = scene_max_dim * 1.2  # 增加20%的视图范围
                self.ax.set_xlim(-view_range, view_range)
                self.ax.set_ylim(-view_range, view_range)
                self.ax.set_zlim(-view_range, view_range)

                # 标记视图已初始化
                self.view_initialized = True
            else:
                # 如果没有相机数据，使用参考长方体的尺寸
                box_max_dim = max(self.box_dimensions)
                self.ax.set_xlim(-box_max_dim, box_max_dim)
                self.ax.set_ylim(-box_max_dim, box_max_dim)
                self.ax.set_zlim(-box_max_dim, box_max_dim)

                # 标记视图已初始化
                self.view_initialized = True
        else:
            # 恢复先前的视图范围
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            self.ax.set_zlim(current_zlim)

            # 恢复视角
            self.ax.view_init(elev=elev, azim=azim)

        # 计算相机之间的平均距离，用于箭头长度
        if len(positions) > 0:
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)

            # 如果有多个相机，计算平均距离，否则使用默认值
            if distances:
                avg_dist = np.mean(distances)
                arrow_length = avg_dist * 0.2
            else:
                # 使用场景尺寸或长方体尺寸
                scene_size = max(self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
                                 self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                 self.ax.get_zlim()[1] - self.ax.get_zlim()[0])
                arrow_length = scene_size * 0.1

            # 绘制相机位置
            self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c='red', s=100, marker='o', label='camera')

            # 添加相机ID作为文本标签
            for i, (cam_id, ext) in enumerate(self.camera_extrinsics.items()):
                R = ext.rotation
                t = ext.translation
                pos = -R.T @ t  # 相机的世界坐标
                self.ax.text(pos[0], pos[1], pos[2], f" {cam_id}", color='black')

                # 绘制相机坐标轴
                x_axis, y_axis, z_axis = get_cam_coor(R)
                R_world = np.column_stack((x_axis, y_axis, z_axis))

                # X轴 - 红色
                self.ax.quiver(pos[0], pos[1], pos[2],
                               R_world[0, 0] * arrow_length, R_world[1, 0] * arrow_length,
                               R_world[2, 0] * arrow_length,
                               color='red', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

                # Y轴 - 绿色
                self.ax.quiver(pos[0], pos[1], pos[2],
                               R_world[0, 1] * arrow_length, R_world[1, 1] * arrow_length,
                               R_world[2, 1] * arrow_length,
                               color='green', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

                # Z轴 - 蓝色
                self.ax.quiver(pos[0], pos[1], pos[2],
                               R_world[0, 2] * arrow_length, R_world[1, 2] * arrow_length,
                               R_world[2, 2] * arrow_length,
                               color='blue', arrow_length_ratio=0.1, length=arrow_length, normalize=True)

                # 添加相机朝向箭头
                view_dir = R_world[:, 2]  # 相机Z轴在世界坐标系中的方向
                self.ax.quiver(pos[0], pos[1], pos[2],
                               view_dir[0], view_dir[1], view_dir[2],
                               color='magenta', linewidth=2, arrow_length_ratio=0.15,
                               length=arrow_length * 1.5, normalize=True)

            # 为图例创建虚拟箭头
            self.ax.quiver(0, 0, 0, 0, 0, 0, color='red', label='cam x')
            self.ax.quiver(0, 0, 0, 0, 0, 0, color='green', label='cam y')
            self.ax.quiver(0, 0, 0, 0, 0, 0, color='blue', label='cam z')
            self.ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', linewidth=2, label='camera dir')

        # 添加图例
        self.ax.legend(loc='upper right')

        # 刷新画布
        self.canvas.draw()

    def save_extrinsics(self):
        """保存变换后的相机外参"""
        if self.camera_extrinsics is None:
            messagebox.showwarning("warning", "no extrinsic to save")
            return

        # 弹出文件选择对话框
        output_dir = filedialog.askdirectory(title="select output folder")

        if not output_dir:
            return

        try:
            # 确保目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存相机外参为JSON
            json_extrinsics = {}
            for cam_id, ext in self.camera_extrinsics.items():
                json_extrinsics[cam_id] = {
                    'rotation': ext.rotation.tolist(),
                    'translation': ext.translation.tolist()
                }

            extrinsics_path = os.path.join(output_dir, "transformed_camera_extrinsics.json")
            with open(extrinsics_path, 'w') as f:
                json.dump(json_extrinsics, f, indent=2)

            # 如果有点云数据，也保存
            if self.points_3d is not None and len(self.points_3d) > 0:
                json_points3d = {}
                for frame, point in self.points_3d.items():
                    json_points3d[str(frame)] = point.tolist()

                points3d_path = os.path.join(output_dir, "transformed_points_3d.json")
                with open(points3d_path, 'w') as f:
                    json.dump(json_points3d, f, indent=2)

            # 保存变换参数
            transform_params = {
                'rotation': self.rotation.tolist(),
                'translation': self.translation.tolist(),
                'scale': self.scale,
                'box_dimensions': self.box_dimensions
            }

            params_path = os.path.join(output_dir, "transform_parameters.json")
            with open(params_path, 'w') as f:
                json.dump(transform_params, f, indent=2)

            # 保存当前视图为图像
            image_path = os.path.join(output_dir, "camera_points_visualization.png")
            self.fig.savefig(image_path, dpi=300, bbox_inches='tight')

            messagebox.showinfo("success", f"save successfully:\n{output_dir}")

        except Exception as e:
            messagebox.showerror("error", f"fail to save: {str(e)}")

    def get_transformed_data(self):
        """获取变换后的数据"""
        # 确保数据存在
        if (self.camera_extrinsics is None or
                self.camera_intrinsics is None or
                self.points_3d is None):
            return None, None, None

        # 应用变换以确保数据为最新
        self.apply_transformation()

        # 将CameraExtrinsicSimple转换为新的CameraExtrinsic类
        transformed_extrinsics = {}
        for cam_id, simple_ext in self.camera_extrinsics.items():
            transformed_extrinsics[cam_id] = CameraExtrinsic(
                rotation=simple_ext.rotation.copy(),
                translation=simple_ext.translation.copy()
            )

        # 将CameraIntrinsicSimple转换为新的CameraIntrinsic类
        transformed_intrinsics = {}
        for cam_id, simple_intr in self.camera_intrinsics.items():
            transformed_intrinsics[cam_id] = CameraIntrinsic(
                camera_matrix=simple_intr.camera_matrix.copy(),
                dist_coeffs=simple_intr.dist_coeffs.copy(),
                image_size=simple_intr.image_size
            )

        return transformed_extrinsics, transformed_intrinsics, self.points_3d


def main(data_dir: str = None, length_x: float = 1.0, length_y: float = 1.0, length_z: float = 1.0,
         extrinsic_file_name: str = None, intrinsic_file_name: str = None, points_file_name: str = None):
    """主函数"""

    root = tk.Tk()
    app = VisualizationApp(root, box_x=length_x, box_y=length_y, box_z=length_z)

    # 如果提供了数据目录，自动加载数据
    if data_dir and os.path.exists(data_dir):
        try:
            app.camera_extrinsics, app.camera_intrinsics, app.points_3d = load_optimization_results(data_dir,
                                                                                                    extrinsic_file_name=extrinsic_file_name,
                                                                                                    intrinsic_file_name=intrinsic_file_name,
                                                                                                    points_file_name=points_file_name,
                                                                                                    format_type='json')

            # 保存原始数据副本，用于后续的变换
            app.original_extrinsics = {cam_id: CameraExtrinsicSimple(ext.rotation.copy(), ext.translation.copy())
                                       for cam_id, ext in app.camera_extrinsics.items()}

            app.original_points = {}
            for frame, point in app.points_3d.items():
                app.original_points[frame] = point.copy()

            # 更新状态
            app.update_status()

            # 更新图表
            app.update_plot()

        except Exception as e:
            print(f"error to load automatically: {e}")

    # 设置窗口关闭事件处理
    def on_closing():
        root.quit()  # 结束mainloop循环
        root.destroy()  # 销毁窗口

    # 绑定窗口关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 运行GUI，当窗口关闭时会自动退出此循环
    root.mainloop()

    # 返回变换后的数据
    return app.get_transformed_data()

if __name__ == "__main__":
    # 修改函数调用以返回变换后的数据
    data_dir = r'./data_file_4_with_ref/results'
    extrinsic_file_name = "scaled_optimized_camera_extrinsics"
    intrinsic_file_name = "scaled_optimized_camera_intrinsics"
    points_file_name = "scaled_optimized_points_3d"

    extrinsics, intrinsics, points3d = main(data_dir, 0.9144, 0.9144,
                                            0.3048, extrinsic_file_name=extrinsic_file_name,
                                            intrinsic_file_name=intrinsic_file_name, points_file_name=points_file_name) # m

    if extrinsics is not None:
        print(f"Successfully retrieved transformed data:")
        print(f"  - camera extrinsic: {len(extrinsics)} cameras")
        # print(extrinsics)
        print(f"  - camera intrinsic: {len(intrinsics)} cameras")
        # print(intrinsics)
        print(f"  - 3D points: {len(points3d)} points")
    else:
        print("No data was loaded or GUI was closed without data.")

    import functions.after_hundle_adjustment as after_hundle_adjustment
    import functions.calc_projection_matrix as calc_projection_matrix
    after_hundle_adjustment.analyze_optimization_results(extrinsics, points3d)
    projection_matrices = calc_projection_matrix.calculate_camera_matrices(intrinsics, extrinsics)
    import functions.convert_2_Braid_xml as convert2Braidxml
    xml_path = convert2Braidxml.save_camera_xml(extrinsic_data=extrinsics, intrinsic_data=intrinsics, camera_matrices_data=projection_matrices,
                                                output_folder=data_dir, filename='braid_calibration.xml',
                                                min_eccentricity=1.4)
    print(f"xml saved to: {xml_path}")

