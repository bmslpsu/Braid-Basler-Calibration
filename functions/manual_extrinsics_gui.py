import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Arrow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Callable
import copy
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R


@dataclass
class CameraExtrinsic:
    """Camera Extrinsic Parameters"""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector

    @property
    def projection_matrix(self) -> np.ndarray:
        """Get 3x4 projection matrix [R|t]"""
        return np.hstack((self.rotation, self.translation.reshape(3, 1)))

    @staticmethod
    def from_rt(rvec: np.ndarray, tvec: np.ndarray) -> 'CameraExtrinsic':
        """Create from rotation vector and translation vector"""
        rotation = cv2.Rodrigues(rvec)[0]
        return CameraExtrinsic(rotation=rotation, translation=tvec)

    def to_rt(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to rotation vector and translation vector"""
        rvec = cv2.Rodrigues(self.rotation)[0]
        return rvec, self.translation

    def copy(self) -> 'CameraExtrinsic':
        """Create a deep copy of this object"""
        return CameraExtrinsic(
            rotation=self.rotation.copy(),
            translation=self.translation.copy()
        )


def convert_to_world_to_camera_translation(extrinsics):
    """
    转换相机外参中的translation，从相机在世界坐标系中的位置
    转换为世界坐标系到相机坐标系的变换向量

    Args:
        extrinsics: 包含相机外参的字典

    Returns:
        转换后的外参字典
    """
    converted_extrinsics = copy.deepcopy(extrinsics)

    for cam_id, extrinsic in converted_extrinsics.items():
        # 保存原始的相机到世界的旋转矩阵，用于参考
        extrinsic.original_rotation = extrinsic.rotation.copy()

        # 转换旋转矩阵（世界到相机 = 转置）
        # extrinsic.rotation = extrinsic.rotation.T

        # 计算世界坐标系到相机坐标系的变换向量
        t_world_to_cam = -extrinsic.rotation @ extrinsic.translation

        # 保存原始的相机在世界坐标系中的位置
        extrinsic.original_translation = extrinsic.translation.copy()

        # 更新为世界到相机的变换向量
        extrinsic.translation = t_world_to_cam

    return converted_extrinsics

class ManualExtrinsicsGUI:
    def __init__(self, camera_intrinsics: Dict[str, 'CameraIntrinsic'],
                 existing_extrinsics: Optional[Dict[str, CameraExtrinsic]] = None):
        """
        Initialize GUI for manual camera extrinsics calibration

        Args:
            camera_intrinsics: Dictionary of camera intrinsic parameters
            existing_extrinsics: Dictionary of existing camera extrinsic parameters (optional)
        """
        # Extract camera IDs from camera_intrinsics
        self.camera_ids = sorted(list(camera_intrinsics.keys()))
        self.camera_intrinsics = camera_intrinsics

        # Initialize extrinsics if not provided
        self.camera_extrinsics = {}

        # For all cameras, use provided extrinsics or initialize with default values
        for i, cam_id in enumerate(self.camera_ids):
            if existing_extrinsics and cam_id in existing_extrinsics:
                self.camera_extrinsics[cam_id] = existing_extrinsics[cam_id].copy()
            else:
                # 初始化相机在原点，坐标系与世界坐标系一致
                # 1. 创建单位旋转矩阵（相机坐标系与世界坐标系一致）
                # 注意：在计算机视觉中，通常相机坐标系的z轴指向前方（相机光轴方向）
                # 相机的y轴指向下方，x轴指向右方

                # 创建一个标准的旋转矩阵，使相机坐标系与世界坐标系对齐
                # 相机坐标系：x向右，y向下，z向前
                # 世界坐标系：x向右，y向上，z向后
                # 因此需要将y和z轴反向
                rotation = np.array([
                    [1.0, 0.0, 0.0],  # x轴保持一致
                    [0.0, -1.0, 0.0],  # y轴反向
                    [0.0, 0.0, -1.0]  # z轴反向
                ])

                # 2. 将所有相机放置在原点
                # 由于我们希望所有相机都在原点，所以平移向量为[0,0,0]
                # 如果需要，可以根据索引i给每个相机一个小偏移，避免完全重叠
                # 例如：tx = 0.2 * i, ty = 0.0, tz = 0.0
                tx = 0.0
                ty = 0.0
                tz = 0.0

                self.camera_extrinsics[cam_id] = CameraExtrinsic(
                    rotation=rotation,
                    translation=np.array([tx, ty, tz])
                )

        # Store original extrinsics for reset functionality
        self.original_extrinsics = copy.deepcopy(self.camera_extrinsics)

        # Current Euler angles in camera coordinates for each camera
        self.camera_local_euler = {}
        for cam_id in self.camera_ids:
            self.camera_local_euler[cam_id] = np.zeros(3)

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Manual Camera Extrinsics Calibration")
        self.root.geometry("1200x800")

        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initialize the UI components
        self._setup_ui()

        # Final extrinsics to return
        self.final_extrinsics = None

    def _setup_ui(self):
        """Setup the UI components"""
        # Create left panel for controls
        self.left_panel = ttk.Frame(self.main_frame, width=400)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Create camera selection combobox
        ttk.Label(self.left_panel, text="Select Camera:").pack(pady=(0, 5), anchor=tk.W)
        self.camera_selector = ttk.Combobox(self.left_panel, values=self.camera_ids, state="readonly")
        self.camera_selector.pack(fill=tk.X, pady=(0, 10))
        self.camera_selector.current(0)
        self.camera_selector.bind("<<ComboboxSelected>>", self._on_camera_selected)

        # Camera parameters frame
        param_frame = ttk.LabelFrame(self.left_panel, text="Camera Parameters")
        param_frame.pack(fill=tk.X, pady=(0, 10))

        # Translation controls
        ttk.Label(param_frame, text="Translation").pack(pady=(5, 5), anchor=tk.W)

        tx_frame = ttk.Frame(param_frame)
        tx_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tx_frame, text="X:").pack(side=tk.LEFT, padx=(0, 5))
        self.tx_var = tk.DoubleVar()
        self.tx_scale = ttk.Scale(tx_frame, from_=-10.0, to=10.0, orient=tk.HORIZONTAL,
                                  variable=self.tx_var, command=self._on_translation_changed)
        self.tx_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tx_entry = ttk.Entry(tx_frame, textvariable=self.tx_var, width=6)
        self.tx_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.tx_entry.bind("<Return>", self._on_entry_changed)

        ty_frame = ttk.Frame(param_frame)
        ty_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ty_frame, text="Y:").pack(side=tk.LEFT, padx=(0, 5))
        self.ty_var = tk.DoubleVar()
        self.ty_scale = ttk.Scale(ty_frame, from_=-10.0, to=10.0, orient=tk.HORIZONTAL,
                                  variable=self.ty_var, command=self._on_translation_changed)
        self.ty_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ty_entry = ttk.Entry(ty_frame, textvariable=self.ty_var, width=6)
        self.ty_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.ty_entry.bind("<Return>", self._on_entry_changed)

        tz_frame = ttk.Frame(param_frame)
        tz_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tz_frame, text="Z:").pack(side=tk.LEFT, padx=(0, 5))
        self.tz_var = tk.DoubleVar()
        self.tz_scale = ttk.Scale(tz_frame, from_=-10.0, to=10.0, orient=tk.HORIZONTAL,
                                  variable=self.tz_var, command=self._on_translation_changed)
        self.tz_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tz_entry = ttk.Entry(tz_frame, textvariable=self.tz_var, width=6)
        self.tz_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.tz_entry.bind("<Return>", self._on_entry_changed)

        # Rotation controls (in camera coordinate system)
        ttk.Label(param_frame, text="Rotation (Around Camera Axes, in degrees)").pack(pady=(10, 5), anchor=tk.W)

        rx_frame = ttk.Frame(param_frame)
        rx_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rx_frame, text="X:").pack(side=tk.LEFT, padx=(0, 5))
        self.rx_var = tk.DoubleVar()
        self.rx_scale = ttk.Scale(rx_frame, from_=-180.0, to=180.0, orient=tk.HORIZONTAL,
                                  variable=self.rx_var, command=self._on_camera_rotation_changed)
        self.rx_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rx_entry = ttk.Entry(rx_frame, textvariable=self.rx_var, width=6)
        self.rx_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.rx_entry.bind("<Return>", self._on_entry_changed)

        ry_frame = ttk.Frame(param_frame)
        ry_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ry_frame, text="Y:").pack(side=tk.LEFT, padx=(0, 5))
        self.ry_var = tk.DoubleVar()
        self.ry_scale = ttk.Scale(ry_frame, from_=-180.0, to=180.0, orient=tk.HORIZONTAL,
                                  variable=self.ry_var, command=self._on_camera_rotation_changed)
        self.ry_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ry_entry = ttk.Entry(ry_frame, textvariable=self.ry_var, width=6)
        self.ry_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.ry_entry.bind("<Return>", self._on_entry_changed)

        rz_frame = ttk.Frame(param_frame)
        rz_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rz_frame, text="Z:").pack(side=tk.LEFT, padx=(0, 5))
        self.rz_var = tk.DoubleVar()
        self.rz_scale = ttk.Scale(rz_frame, from_=-180.0, to=180.0, orient=tk.HORIZONTAL,
                                  variable=self.rz_var, command=self._on_camera_rotation_changed)
        self.rz_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rz_entry = ttk.Entry(rz_frame, textvariable=self.rz_var, width=6)
        self.rz_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.rz_entry.bind("<Return>", self._on_entry_changed)

        # World coordinate display
        world_frame = ttk.LabelFrame(self.left_panel, text="World Coordinate Information")
        world_frame.pack(fill=tk.X, pady=(0, 10))

        # Create a text widget to display world coordinate info
        self.world_info = tk.Text(world_frame, height=15, width=46, state=tk.DISABLED)
        self.world_info.pack(fill=tk.X, pady=5, padx=5)

        # Buttons
        btn_frame = ttk.Frame(self.left_panel)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Reset Current Camera", command=self._reset_current_camera).pack(side=tk.LEFT,
                                                                                                    padx=(0, 5))
        ttk.Button(btn_frame, text="Reset All Cameras", command=self._reset_all_cameras).pack(side=tk.LEFT)

        # Accept/Cancel buttons
        action_frame = ttk.Frame(self.left_panel)
        action_frame.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)

        ttk.Button(action_frame, text="Accept", command=self._on_accept).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                              padx=(0, 5))
        ttk.Button(action_frame, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create right panel for 3D visualization
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup the 3D plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 禁用自动缩放
        self.ax.autoscale(enable=False)

        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏用于3D导航
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_panel)
        self.toolbar.update()

        # 设置鼠标事件处理，跟踪视角变化
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)

        # 存储图形元素的引用
        self.camera_objects = {}

        # 更新UI，加载初始相机
        self._load_camera_params(self.camera_ids[0])

        # 初始化3D图而不是更新它
        self._initialize_3d_plot()

    def _initialize_3d_plot(self):
        """初始化3D图，创建所有需要的图形对象"""
        # 清除任何现有内容
        self.ax.clear()

        # 基本设置
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Camera Configuration')

        # 设置一个固定的视图范围
        self.ax.set_xlim(-250, 250)
        self.ax.set_ylim(-250, 250)
        self.ax.set_zlim(-250, 250)

        # 保存初始视图范围，用于计算缩放比例
        self.initial_scale = {
            'x': 4,  # [-2, 2] 范围长度为4
            'y': 4,
            'z': 4
        }
        self.current_scale = self.initial_scale.copy()

        # 设置初始视角
        self.ax.view_init(elev=30, azim=-60)

        # 绘制世界坐标系 - 保存引用，但不设置长度，在_update_axes_scale中设置
        self.world_axes = [
            self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1, normalize=True, label='World X'),
            self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1, normalize=True, label='World Y'),
            self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1, normalize=True, label='World Z')
        ]

        # 为每个相机创建对象，并保存它们的引用
        for cam_id in self.camera_ids:
            # 每个相机创建一组图形对象，但不设置坐标轴长度，在_update_3d_plot中设置
            camera_obj = {
                'position': self.ax.scatter([], [], [], color='black', s=50),
                'highlight': self.ax.scatter([], [], [], color='red', s=100, edgecolor='yellow', linewidth=2,
                                             alpha=0.7),
                'x_axis': self.ax.quiver([], [], [], [], [], [], color='r', length=0.5),
                'y_axis': self.ax.quiver([], [], [], [], [], [], color='g', length=0.5),
                'z_axis': self.ax.quiver([], [], [], [], [], [], color='b', length=0.5),
                'frustum_lines': [],
                'label': self.ax.text(0, 0, 0, cam_id, fontsize=10)
            }

            # 为视锥体创建线条对象
            for _ in range(8):  # 8条线
                line, = self.ax.plot([], [], [], 'k-')
                camera_obj['frustum_lines'].append(line)

            # 存储此相机的对象引用
            self.camera_objects[cam_id] = camera_obj

        # 添加图例
        self.ax.legend(loc='upper right')

        # 强制绘制以创建所有对象
        self.canvas.draw()

        # 保存当前视角
        self._current_elev = self.ax.elev
        self._current_azim = self.ax.azim

        # 更新坐标轴缩放
        self._update_axes_scale()

    def _on_mouse_release(self, event):
        """
        鼠标释放事件处理函数，用于在用户手动旋转视图后更新保存的视角和坐标轴缩放
        """
        # 确保我们跟踪当前视角和轴范围
        if hasattr(self, 'ax'):
            self._current_elev = self.ax.elev
            self._current_azim = self.ax.azim
            self._update_axes_scale()

    def _update_axes_scale(self):
        """
        根据当前视图范围更新坐标轴的缩放比例
        """
        # 获取当前轴范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        # 计算当前范围长度
        self.current_scale = {
            'x': abs(xlim[1] - xlim[0]),
            'y': abs(ylim[1] - ylim[0]),
            'z': abs(zlim[1] - zlim[0])
        }

        # 计算相对于初始比例的比例因子
        # 使用平均缩放比例，以便所有轴保持相同长度
        avg_scale_factor = (
                                   self.current_scale['x'] / self.initial_scale['x'] +
                                   self.current_scale['y'] / self.initial_scale['y'] +
                                   self.current_scale['z'] / self.initial_scale['z']
                           ) / 3

        # 更新世界坐标系轴长度
        axis_scale = 1.0 * avg_scale_factor  # 基础长度 * 缩放因子，可以调整系数
        self.world_axes[0].set_segments([np.array([[0, 0, 0], [axis_scale, 0, 0]])])
        self.world_axes[1].set_segments([np.array([[0, 0, 0], [0, axis_scale, 0]])])
        self.world_axes[2].set_segments([np.array([[0, 0, 0], [0, 0, axis_scale]])])

        # 保存缩放因子供相机坐标轴使用
        self.axis_scale = axis_scale

        # 如果已经有相机，重新绘制它们
        if hasattr(self, 'camera_objects') and self.camera_objects:
            self._update_3d_plot()

    def _load_camera_params(self, cam_id):
        """Load camera parameters into UI"""
        if cam_id not in self.camera_extrinsics:
            return

        # Get current extrinsics
        extrinsic = self.camera_extrinsics[cam_id]

        # Set translation values
        self.tx_var.set(extrinsic.translation[0])
        self.ty_var.set(extrinsic.translation[1])
        self.tz_var.set(extrinsic.translation[2])

        # Set camera local euler angles
        if cam_id in self.camera_local_euler:
            local_euler = self.camera_local_euler[cam_id]
            self.rx_var.set(float(local_euler[0]))
            self.ry_var.set(float(local_euler[1]))
            self.rz_var.set(float(local_euler[2]))
        else:
            # Initialize local euler angles to zeros
            self.camera_local_euler[cam_id] = np.zeros(3)
            self.rx_var.set(0.0)
            self.ry_var.set(0.0)
            self.rz_var.set(0.0)

        # Enable all controls
        self.tx_scale.state(['!disabled'])
        self.ty_scale.state(['!disabled'])
        self.tz_scale.state(['!disabled'])
        self.rx_scale.state(['!disabled'])
        self.ry_scale.state(['!disabled'])
        self.rz_scale.state(['!disabled'])
        self.tx_entry.state(['!disabled'])
        self.ty_entry.state(['!disabled'])
        self.tz_entry.state(['!disabled'])
        self.rx_entry.state(['!disabled'])
        self.ry_entry.state(['!disabled'])
        self.rz_entry.state(['!disabled'])

        # Update world coordinate information
        self._update_world_info(cam_id)

    def _update_world_info(self, cam_id):
        """更新所选相机的世界坐标信息，显示相机坐标系到世界坐标系的变换信息，以及世界坐标系到相机坐标系的变换信息"""
        if cam_id not in self.camera_extrinsics:
            return

        # 获取当前外参
        extrinsic = self.camera_extrinsics[cam_id]

        # 获取旋转矩阵和平移向量
        R_camera_to_world = extrinsic.rotation  # 这是相机坐标到世界坐标的旋转变换
        t_cam_in_world = extrinsic.translation  # 相机在世界坐标系中的位置

        # 计算世界坐标系到相机坐标系的旋转矩阵
        R_world_to_camera = R_camera_to_world.T

        # 计算世界坐标系到相机坐标系的变换向量
        t_world_to_camera = -R_world_to_camera @ t_cam_in_world

        # 从旋转矩阵创建一个Rotation对象
        rot = R.from_matrix(R_camera_to_world)

        # 获取四元数表示（相机坐标到世界坐标）
        quat = rot.as_quat()  # [x, y, z, w]格式

        # 计算欧拉角
        euler_world_to_camera = R.from_matrix(R_world_to_camera).as_euler('xyz', degrees=True)
        euler_camera_to_world = R.from_matrix(R_camera_to_world).as_euler('xyz', degrees=True)

        # 启用文本控件进行编辑
        self.world_info.config(state=tk.NORMAL)
        self.world_info.delete(1.0, tk.END)

        # 添加相机位置信息（世界坐标系中）
        self.world_info.insert(tk.END, f"相机在世界坐标中的位置: \n")
        self.world_info.insert(tk.END,
                               f"[{t_cam_in_world[0]:.3f}, {t_cam_in_world[1]:.3f}, {t_cam_in_world[2]:.3f}]\n\n")

        # 添加世界坐标系到相机坐标系的变换向量
        self.world_info.insert(tk.END, "世界坐标系到相机坐标系的变换向量:\n")
        self.world_info.insert(tk.END,
                               f"[{t_world_to_camera[0]:.3f}, {t_world_to_camera[1]:.3f}, {t_world_to_camera[2]:.3f}]\n\n")

        # 添加欧拉角信息（世界到相机）
        self.world_info.insert(tk.END, "欧拉角 (世界到相机, xyz, 度):\n")
        self.world_info.insert(tk.END, f"Roll (X): {euler_world_to_camera[0]:.2f}°\n")
        self.world_info.insert(tk.END, f"Pitch (Y): {euler_world_to_camera[1]:.2f}°\n")
        self.world_info.insert(tk.END, f"Yaw (Z): {euler_world_to_camera[2]:.2f}°\n\n")

        # 添加欧拉角信息（相机到世界）
        self.world_info.insert(tk.END, "欧拉角 (相机到世界, xyz, 度):\n")
        self.world_info.insert(tk.END, f"Roll (X): {euler_camera_to_world[0]:.2f}°\n")
        self.world_info.insert(tk.END, f"Pitch (Y): {euler_camera_to_world[1]:.2f}°\n")
        self.world_info.insert(tk.END, f"Yaw (Z): {euler_camera_to_world[2]:.2f}°\n\n")

        # 添加相机坐标系到世界坐标系的旋转矩阵
        self.world_info.insert(tk.END, "旋转矩阵 (相机到世界):\n")
        for i in range(3):
            row_text = "["
            for j in range(3):
                row_text += f"{R_camera_to_world[i, j]:.3f}, "
            row_text = row_text[:-2] + "]\n"  # 移除尾随逗号并添加括号
            self.world_info.insert(tk.END, row_text)

        self.world_info.insert(tk.END, '\n')  # 添加空行

        # 添加世界坐标系到相机坐标系的旋转矩阵
        self.world_info.insert(tk.END, "旋转矩阵 (世界到相机):\n")
        for i in range(3):
            row_text = "["
            for j in range(3):
                row_text += f"{R_world_to_camera[i, j]:.3f}, "
            row_text = row_text[:-2] + "]\n"  # 移除尾随逗号并添加括号
            self.world_info.insert(tk.END, row_text)

        self.world_info.insert(tk.END, '\n')  # 添加空行

        # 添加四元数表示（相机到世界）
        self.world_info.insert(tk.END,
                               f"四元数 [x,y,z,w] (相机到世界): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]\n\n")

        # 再次禁用文本控件
        self.world_info.config(state=tk.DISABLED)

    def _on_camera_selected(self, event):
        """Handle camera selection change"""
        cam_id = self.camera_selector.get()
        self._load_camera_params(cam_id)

    def _on_translation_changed(self, *args):
        """Handle translation changes"""
        cam_id = self.camera_selector.get()

        try:
            # Get current values
            tx = float(self.tx_var.get())
            ty = float(self.ty_var.get())
            tz = float(self.tz_var.get())

            # Update extrinsics
            self.camera_extrinsics[cam_id].translation = np.array([tx, ty, tz])

            # Update world coordinate information
            self._update_world_info(cam_id)

            # Update 3D visualization
            self._update_3d_plot()
        except (ValueError, TypeError) as e:
            print(f"Error converting translation values: {e}")
            # Don't update if there was an error

    def _on_camera_rotation_changed(self, *args):
        """确保所有轴都是围绕相机局部坐标系旋转，每次只调整一个轴"""
        cam_id = self.camera_selector.get()

        try:
            # 获取当前角度值（度）
            rx = self.rx_var.get()
            ry = self.ry_var.get()
            rz = self.rz_var.get()

            # 获取之前的角度值
            old_rx, old_ry, old_rz = self.camera_local_euler[cam_id]

            # 计算每个轴的增量旋转（度）
            delta_rx = rx - old_rx
            delta_ry = ry - old_ry
            delta_rz = rz - old_rz

            # 记录新的欧拉角（仅用于显示）
            self.camera_local_euler[cam_id] = np.array([rx, ry, rz])

            # 获取当前相机的旋转矩阵（世界坐标到相机坐标）
            R_world_to_camera = self.camera_extrinsics[cam_id].rotation.copy()

            # 相机坐标到世界坐标的转换
            R_camera_to_world = R_world_to_camera.T

            # 获取相机的局部坐标轴在世界坐标系中的表示
            camera_x = R_camera_to_world[:, 0]  # 相机X轴（在世界坐标系中）
            camera_y = R_camera_to_world[:, 1]  # 相机Y轴（在世界坐标系中）
            camera_z = R_camera_to_world[:, 2]  # 相机Z轴（在世界坐标系中）

            # 创建增量旋转矩阵（按顺序应用 Z->Y->X 以保持稳定性）
            delta_rotation = np.eye(3)  # 初始化为单位矩阵（无旋转）

            # 按特定顺序应用旋转（Z->Y->X）以保持直观性
            # 首先围绕Z轴旋转
            if abs(delta_rz) > 1e-6:
                delta_rz_rad = np.deg2rad(delta_rz)
                rot_z = R.from_rotvec(delta_rz_rad * camera_z).as_matrix()
                delta_rotation = rot_z @ delta_rotation

                # 重要：更新相机坐标系
                temp_camera_to_world = rot_z @ R_camera_to_world
                camera_x = temp_camera_to_world[:, 0]
                camera_y = temp_camera_to_world[:, 1]

            # 然后围绕Y轴旋转
            if abs(delta_ry) > 1e-6:
                delta_ry_rad = np.deg2rad(delta_ry)
                rot_y = R.from_rotvec(delta_ry_rad * camera_y).as_matrix()
                delta_rotation = rot_y @ delta_rotation

                # 更新相机X轴（Z轴不变）
                temp_camera_to_world = rot_y @ (
                    rot_z @ R_camera_to_world if abs(delta_rz) > 1e-6 else R_camera_to_world)
                camera_x = temp_camera_to_world[:, 0]

            # 最后围绕X轴旋转
            if abs(delta_rx) > 1e-6:
                delta_rx_rad = np.deg2rad(delta_rx)
                rot_x = R.from_rotvec(delta_rx_rad * camera_x).as_matrix()
                delta_rotation = rot_x @ delta_rotation

            # 应用增量旋转到当前旋转
            # 在世界坐标系中应用旋转，然后转回相机坐标系
            R_new_camera_to_world = delta_rotation @ R_camera_to_world
            R_new_world_to_camera = R_new_camera_to_world.T

            # 验证旋转矩阵的正交性和行列式为1（非常重要！）
            # 正交检查
            ortho_check = np.allclose(R_new_camera_to_world @ R_new_camera_to_world.T, np.eye(3), atol=1e-6)
            det_check = np.isclose(np.linalg.det(R_new_camera_to_world), 1.0, atol=1e-6)

            if not ortho_check or not det_check:
                print("警告：生成的旋转矩阵不是有效的旋转矩阵!")
                print(f"正交检查: {ortho_check}, 行列式检查: {det_check}")
                # 如果矩阵无效，可以强制正交化
                u, _, vh = np.linalg.svd(R_new_camera_to_world, full_matrices=False)
                R_new_camera_to_world = u @ vh
                R_new_world_to_camera = R_new_camera_to_world.T

            # 更新相机的旋转矩阵
            self.camera_extrinsics[cam_id].rotation = R_new_world_to_camera

            # 更新显示信息和3D可视化
            self._update_world_info(cam_id)
            self._update_3d_plot()

        except (ValueError, TypeError) as e:
            print(f"应用相机旋转时出错: {e}")
            import traceback
            traceback.print_exc()

    def _on_entry_changed(self, event):
        """Handle manual entry of values"""
        # Update both translation and rotation
        self._on_translation_changed()
        self._on_camera_rotation_changed()

    def _update_3d_plot(self):
        """使用对象引用更新3D图，并按当前比例缩放坐标轴"""
        # 更新每个相机的对象
        for cam_id, extrinsic in self.camera_extrinsics.items():
            # 获取此相机的图形对象
            camera_obj = self.camera_objects[cam_id]

            # 获取相机位置（平移）
            pos = extrinsic.translation

            # 获取相机方向（旋转矩阵）
            R = extrinsic.rotation

            # 相机轴（在相机坐标系中）
            x_axis = np.array([1, 0, 0])
            y_axis = np.array([0, 1, 0])
            z_axis = np.array([0, 0, 1])

            # 转换到世界坐标
            x_world = R.T @ x_axis
            y_world = R.T @ y_axis
            z_world = R.T @ z_axis

            # 使用当前缩放因子
            scale = 0.5 * getattr(self, 'axis_scale', 0.5)  # 默认值为0.5，如果axis_scale不存在

            # 更新相机位置点
            camera_obj['position'].set_offsets(np.column_stack([pos[0], pos[1]]))
            camera_obj['position'].set_3d_properties(pos[2], 'z')

            # 更新轴向量，使用当前缩放比例
            camera_obj['x_axis'].set_segments([np.array([[pos[0], pos[1], pos[2]],
                                                         [pos[0] + scale * x_world[0],
                                                          pos[1] + scale * x_world[1],
                                                          pos[2] + scale * x_world[2]]])])

            camera_obj['y_axis'].set_segments([np.array([[pos[0], pos[1], pos[2]],
                                                         [pos[0] + scale * y_world[0],
                                                          pos[1] + scale * y_world[1],
                                                          pos[2] + scale * y_world[2]]])])

            camera_obj['z_axis'].set_segments([np.array([[pos[0], pos[1], pos[2]],
                                                         [pos[0] + scale * z_world[0],
                                                          pos[1] + scale * z_world[1],
                                                          pos[2] + scale * z_world[2]]])])

            # 视锥体定义 - 也按比例缩放
            w, h = 0.3 * scale, 0.2 * scale
            frustum_pts = np.array([
                [-w / 2, -h / 2, scale],  # 左下
                [w / 2, -h / 2, scale],  # 右下
                [w / 2, h / 2, scale],  # 右上
                [-w / 2, h / 2, scale],  # 左上
                [0, 0, 0]  # 相机中心
            ])

            # 转换到世界坐标
            frustum_world = []
            for pt in frustum_pts:
                pt_world = R.T @ pt + pos
                frustum_world.append(pt_world)

            # 视锥体线条定义
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底座矩形
                [4, 0], [4, 1], [4, 2], [4, 3]  # 从中心到角的线
            ]

            # 更新视锥体线条
            for i, line in enumerate(lines):
                pt1, pt2 = frustum_world[line[0]], frustum_world[line[1]]
                camera_obj['frustum_lines'][i].set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])
                camera_obj['frustum_lines'][i].set_3d_properties([pt1[2], pt2[2]], 'z')

            # 更新文本标签位置，同样按比例缩放偏移
            label_offset = 0.2 * scale
            camera_obj['label'].set_position((pos[0], pos[1]))
            camera_obj['label'].set_3d_properties(pos[2] + label_offset, 'z')

            # 更新高亮状态
            if cam_id == self.camera_selector.get():
                camera_obj['highlight'].set_offsets(np.column_stack([pos[0], pos[1]]))
                camera_obj['highlight'].set_3d_properties(pos[2], 'z')
                camera_obj['highlight'].set_alpha(0.7)  # 显示高亮
            else:
                camera_obj['highlight'].set_alpha(0)  # 隐藏高亮

        # 重绘图表
        self.canvas.draw()

    def _reset_current_camera(self):
        """重置当前相机到初始状态"""
        cam_id = self.camera_selector.get()

        if cam_id in self.original_extrinsics:
            # 深拷贝原始外参
            self.camera_extrinsics[cam_id] = self.original_extrinsics[cam_id].copy()
            # 重置局部欧拉角
            self.camera_local_euler[cam_id] = np.zeros(3)
            # 更新UI控件
            self._load_camera_params(cam_id)
            # 更新3D可视化
            self._update_3d_plot()

    def _reset_all_cameras(self):
        """重置所有相机到初始状态"""
        # 深拷贝原始外参
        self.camera_extrinsics = copy.deepcopy(self.original_extrinsics)
        # 重置所有局部欧拉角
        for cam_id in self.camera_ids:
            self.camera_local_euler[cam_id] = np.zeros(3)
        # 更新当前选中相机的UI控件
        self._load_camera_params(self.camera_selector.get())
        # 更新3D可视化
        self._update_3d_plot()

    def _on_accept(self):
        """Handle Accept button click - convert translation to world-to-camera transformation"""
        # 创建相机外参的深拷贝
        temp_extrinsics = copy.deepcopy(self.camera_extrinsics)

        # 将平移向量转换为世界坐标系到相机坐标系的变换向量
        self.final_extrinsics = convert_to_world_to_camera_translation(temp_extrinsics)

        self.root.quit()
        self.root.destroy()

    def _on_cancel(self):
        """Handle Cancel button click"""
        self.final_extrinsics = copy.deepcopy(self.original_extrinsics)
        self.root.quit()
        self.root.destroy()

    def run(self) -> Dict[str, CameraExtrinsic]:
        """
        Run the GUI and return the final camera extrinsics

        Returns:
            Dictionary of camera extrinsics
        """
        self.root.mainloop()

        # Return the final extrinsics
        return self.final_extrinsics
def manual_extrinsics_calibration(camera_intrinsics: Dict[str, 'CameraIntrinsic'],
                                  existing_extrinsics: Optional[Dict[str, CameraExtrinsic]] = None) -> Dict[
    str, CameraExtrinsic]:
    """
    Open GUI for manual camera extrinsics calibration

    Args:
        camera_intrinsics: Dictionary of camera intrinsic parameters
        existing_extrinsics: Dictionary of existing camera extrinsic parameters (optional)

    Returns:
        Dictionary of manually calibrated camera extrinsics
    """
    # Create and run the GUI
    gui = ManualExtrinsicsGUI(camera_intrinsics, existing_extrinsics)
    extrinsics = gui.run()

    return extrinsics


if __name__ == "__main__":
    # Example usage
    # Creating example camera intrinsics
    from dataclasses import dataclass
    from typing import Tuple


    @dataclass
    class CameraIntrinsic:
        """Camera Intrinsic"""
        camera_matrix: np.ndarray
        dist_coeffs: np.ndarray
        image_size: Tuple[int, int]


    # Create some example camera intrinsics
    camera_intrinsics = {}
    for i, cam_id in enumerate(["Basler-62", "Basler-56", "Basler-55", "Basler-61"]):
        camera_intrinsics[cam_id] = CameraIntrinsic(
            camera_matrix=np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]),
            dist_coeffs=np.zeros(5),
            image_size=(640, 480)
        )

    # Create some example extrinsics
    existing_extrinsics = {}
    # Reference camera (identity)
    existing_extrinsics["Basler-62"] = CameraExtrinsic(
        # rotation=np.eye(3),
        rotation=np.array([[1.000, -0.000, 0.000],
                   [0.000, -0.000, -1.000],
                   [0.000, 1.000, 0.000]]),
        translation=np.array([-1.000, -2.000, 0.000])
    )
    # Second camera
    existing_extrinsics["Basler-56"] = CameraExtrinsic(
        # rotation=cv2.Rodrigues(np.array([0, np.pi / 4, 0]))[0],
        rotation=np.array([[1.000, -0.000, 0.000],
                            [0.000, 0.000, -1.000],
                            [0.000, 1.000, 0.000]]),
        translation=np.array([1.000, -2.000, 0.000])
    )
    # Third camera
    existing_extrinsics["Basler-55"] = CameraExtrinsic(
        # rotation=cv2.Rodrigues(np.array([-np.pi / 4, 0, 0]))[0],
        rotation=np.array([[-0.000, -1.000, -0.000],
                            [0.000, 0.000, -1.000],
                            [1.000, -0.000, 0.000]]),
        translation=np.array([-2.000, 1.000, 0.000])
    )
    # Fourth camera
    existing_extrinsics["Basler-61"] = CameraExtrinsic(
        # rotation=cv2.Rodrigues(np.array([0, np.pi, 0]))[0],
        rotation=np.array([[-0.000, -1.000, 0.000],
                            [0.000, -0.000, -1.000],
                            [1.000, -0.000, 0.000]]),
        translation=np.array([-2.000, -1.000, 0.000])
    )

    # Run manual calibration
    calibrated_extrinsics = manual_extrinsics_calibration(camera_intrinsics, existing_extrinsics)

    # Print results
    if calibrated_extrinsics:
        print("Calibrated Extrinsics:")
        for cam_id, extrinsic in calibrated_extrinsics.items():
            print(f"\nCamera: {cam_id}")

            # 计算欧拉角 (世界到相机)
            R_world_to_camera = extrinsic.rotation
            euler_world_to_camera = R.from_matrix(R_world_to_camera).as_euler('xyz', degrees=True)

            # 计算欧拉角 (相机到世界)
            R_camera_to_world = extrinsic.rotation.T
            euler_camera_to_world = R.from_matrix(R_camera_to_world).as_euler('xyz', degrees=True)

            print("Rotation matrix (world to camera):")
            print(extrinsic.rotation)

            print("Euler angles (world to camera, xyz, degrees):")
            print(f"  Roll (X): {euler_world_to_camera[0]:.2f}")
            print(f"  Pitch (Y): {euler_world_to_camera[1]:.2f}")
            print(f"  Yaw (Z): {euler_world_to_camera[2]:.2f}")

            print("Euler angles (camera to world, xyz, degrees):")
            print(f"  Roll (X): {euler_camera_to_world[0]:.2f}")
            print(f"  Pitch (Y): {euler_camera_to_world[1]:.2f}")
            print(f"  Yaw (Z): {euler_camera_to_world[2]:.2f}")

            print("Translation vector (world to camera):")
            print(extrinsic.translation)
    from after_hundle_adjustment import analyze_optimization_results
    analyze_optimization_results(calibrated_extrinsics)

    print(cv2.Rodrigues(np.array([0, np.pi / 4, 0]))[0])