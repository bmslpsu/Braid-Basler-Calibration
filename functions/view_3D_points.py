import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_points(points_3d, scale_factor=1e0, save_path=None):
    """
    只可视化3D点云，适用于非常小的数值

    Args:
        points_3d: 将帧ID映射到3D点坐标的字典
        scale_factor: 缩放因子，用于放大非常小的点坐标
        save_path: 保存图像的可选路径
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 提取点坐标并应用缩放
    points = np.array(list(points_3d.values())) * scale_factor

    if len(points) > 0:
        # 提取坐标进行绘图
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        # 打印一些统计信息
        print(f"点数量: {len(points)}")
        print(f"原始点范围 - X: [{np.min(xs / scale_factor):.2e}, {np.max(xs / scale_factor):.2e}], "
              f"Y: [{np.min(ys / scale_factor):.2e}, {np.max(ys / scale_factor):.2e}], "
              f"Z: [{np.min(zs / scale_factor):.2e}, {np.max(zs / scale_factor):.2e}]")
        print(f"缩放后点范围 - X: [{np.min(xs):.2f}, {np.max(xs):.2f}], "
              f"Y: [{np.min(ys):.2f}, {np.max(ys):.2f}], "
              f"Z: [{np.min(zs):.2f}, {np.max(zs):.2f}]")

        # 绘制点
        ax.scatter(xs, ys, zs, c='blue', s=1, alpha=0.5)
    else:
        print("没有3D点可供可视化")

    # 设置标签和标题
    ax.set_xlabel(f'X (x{scale_factor})')
    ax.set_ylabel(f'Y (x{scale_factor})')
    ax.set_zlabel(f'Z (x{scale_factor})')
    ax.set_title('3D点云可视化 (缩放后)')

    # 添加网格
    ax.grid(True)

    # 设置初始视角
    ax.view_init(elev=20, azim=30)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, ax


# 使用示例
if __name__ == "__main__":
    # 解析您提供的数据样本
    points_3d = {}

    # 这里可以添加您的数据点
    # 例如:
    points_3d[10665] = np.array([-6.92494673e-07, -2.62731633e-06, -1.63156774e-09])
    points_3d[10666] = np.array([-6.92227544e-07, -2.62631001e-06, -1.63158285e-09])
    # ... 添加更多点

    # 可视化点云，应用1e6的缩放因子
    visualize_points(points_3d, scale_factor=1e6, save_path="small_points_visualization.png")