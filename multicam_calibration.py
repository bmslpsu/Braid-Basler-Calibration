import os

from functions.data_preprocess import data_preprocess
from functions.bundle_adjustment import run_sparse_bundle_adjustment_with_distortion
from functions.after_hundle_adjustment import analyze_optimization_results
from functions.calc_projection_matrix import calculate_camera_matrices
from functions.view_3D_points import visualize_points
from functions.init_camera_extrinsic_ransac import data_preprocess_with_choices
import math
if __name__ == '__main__':
    base_dir = "./data_file_4_angled"
    data2d_path = os.path.join(base_dir, "wand_data/data2d_distorted.csv")
    cam_info_path = os.path.join(base_dir, "wand_data/cam_info.csv")

    # base_dir = "./data_file"
    # data2d_path = os.path.join(base_dir, "20241017_164418/data2d_distorted.csv")
    # cam_info_path = os.path.join(base_dir, "20241017_164418/cam_info.csv")

    yaml_dir = os.path.join(base_dir, "intrinsic_calibrations")
    print('Start data processing.....')

    filtered_data, camn_to_id, camera_intrinsics, camera_extrinsics = data_preprocess_with_choices(
        data2d_path,
        cam_info_path,
        yaml_dir,
        min_shared_frames=50,
        verbose=True,
        manual_init_extrinsic=False
    )
    print(camera_intrinsics)
    print(camera_extrinsics)
    print('Done data processing, \nStart bundle adjustment for extrinsics....')

    camera_extrinsics, camera_intrinsics, point_3D = run_sparse_bundle_adjustment_with_distortion(
        filtered_data,
        camera_intrinsics,
        camera_extrinsics,
        min_cameras=2,
        verbose=True,
        robust=True,
        max_iterations=100,
        optimize_distortion=True,  # Enable distortion optimization
        optimize_k1k2=True,  # Optimize radial distortion k1, k2
        optimize_p1p2=False,  # Optimize tangential distortion p1, p2
        optimize_k3=False  # Don't optimize k3 (often unstable)
    )
    print('Done bundle adjustment, \nStart analyzing optimization results....')
    analyze_optimization_results(camera_extrinsics, point_3D)
    print('\nDone analyzeing optimization results, \nStart c alculating projection matrices....')
    print(camera_extrinsics)
    projection_matrices = calculate_camera_matrices(camera_intrinsics, camera_extrinsics)
    # print(f'\nprojection_matrices:{projection_matrices}')
    print('\nvis 3D points')
    visualize_points(point_3D)


