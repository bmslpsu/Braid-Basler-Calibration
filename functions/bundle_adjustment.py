import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
import copy
import os
import pickle
import json

# Import necessary classes from the reconstruction module
try:
    from init_camera_extrinsic_ransac import CameraData, CameraIntrinsic, CameraExtrinsic, find_frame_observations, \
        get_synchronized_frames, optimize_3d_point_hartley_sturm
except ImportError:
    try:
        from .init_camera_extrinsic_ransac import CameraData, CameraIntrinsic, CameraExtrinsic, find_frame_observations, \
            get_synchronized_frames, optimize_3d_point_hartley_sturm
    except ImportError:
        print("Warning: Could not import reconstruction module")


def camera_extrinsics_to_params(camera_extrinsics: Dict[str, CameraExtrinsic]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert camera extrinsics dictionary to parameter vector for optimization

    Args:
        camera_extrinsics: Dictionary of camera extrinsics

    Returns:
        Tuple of (parameter vector, ordered camera IDs)
    """
    camera_params = []
    camera_ids = sorted(list(camera_extrinsics.keys()))

    for cam_id in camera_ids:
        extrinsic = camera_extrinsics[cam_id]
        # Convert rotation matrix to axis-angle representation
        rvec, tvec = extrinsic.to_rt()
        # Flatten and append to parameter vector
        camera_params.extend(rvec.flatten())
        camera_params.extend(tvec.flatten())

    return np.array(camera_params), camera_ids


def camera_intrinsics_to_params(camera_intrinsics: Dict[str, CameraIntrinsic],
                                camera_ids: List[str],
                                optimize_k1k2: bool = True,
                                optimize_p1p2: bool = True,
                                optimize_k3: bool = False) -> Tuple[np.ndarray, List[bool]]:
    """
    Convert camera intrinsics dictionary to parameter vector for optimization

    Args:
        camera_intrinsics: Dictionary of camera intrinsics
        camera_ids: Ordered list of camera IDs
        optimize_k1k2: Whether to optimize radial distortion parameters k1, k2
        optimize_p1p2: Whether to optimize tangential distortion parameters p1, p2
        optimize_k3: Whether to optimize radial distortion parameter k3

    Returns:
        Tuple of (parameter vector, list of parameters to optimize)
    """
    intrinsic_params = []
    param_mask = []  # Indicates which distortion parameters to optimize

    for cam_id in camera_ids:
        intrinsic = camera_intrinsics[cam_id]
        dist_coeffs = intrinsic.dist_coeffs.flatten()

        # Add distortion coefficients based on mask
        # Typically dist_coeffs = [k1, k2, p1, p2, k3, ...]
        if len(dist_coeffs) >= 5:  # Make sure we have enough coefficients
            if optimize_k1k2:
                intrinsic_params.extend(dist_coeffs[0:2])  # k1, k2
                param_mask.extend([True, True])

            if optimize_p1p2:
                intrinsic_params.extend(dist_coeffs[2:4])  # p1, p2
                param_mask.extend([True, True])

            if optimize_k3 and len(dist_coeffs) >= 5:
                intrinsic_params.append(dist_coeffs[4])  # k3
                param_mask.append(True)
        else:
            # If we don't have enough coefficients, use what we have
            intrinsic_params.extend(dist_coeffs)
            param_mask.extend([True] * len(dist_coeffs))

    return np.array(intrinsic_params), param_mask


def params_to_camera_extrinsics(params: np.ndarray, camera_ids: List[str]) -> Dict[str, CameraExtrinsic]:
    """
    Convert parameter vector back to camera extrinsics dictionary

    Args:
        params: Parameter vector
        camera_ids: Ordered list of camera IDs

    Returns:
        Dictionary of camera extrinsics
    """
    camera_extrinsics = {}

    for i, cam_id in enumerate(camera_ids):
        # Each camera has 6 parameters: 3 for rotation and 3 for translation
        param_idx = i * 6
        rvec = params[param_idx:param_idx + 3]
        tvec = params[param_idx + 3:param_idx + 6]

        # Convert back to CameraExtrinsic
        camera_extrinsics[cam_id] = CameraExtrinsic.from_rt(rvec, tvec)

    return camera_extrinsics


def params_to_camera_intrinsics(params: np.ndarray,
                                original_intrinsics: Dict[str, CameraIntrinsic],
                                camera_ids: List[str],
                                dist_param_counts: List[int]) -> Dict[str, CameraIntrinsic]:
    """
    Convert parameter vector back to camera intrinsics dictionary

    Args:
        params: Parameter vector containing distortion parameters
        original_intrinsics: Original camera intrinsics to modify
        camera_ids: Ordered list of camera IDs
        dist_param_counts: Number of distortion parameters per camera

    Returns:
        Dictionary of updated camera intrinsics
    """
    updated_intrinsics = {}
    param_index = 0

    for i, cam_id in enumerate(camera_ids):
        # Get original intrinsic and make a copy
        original = original_intrinsics[cam_id]
        updated = copy.deepcopy(original)

        # Get the number of parameters for this camera
        num_params = dist_param_counts[i]

        # Extract the parameters for this camera
        camera_dist_params = params[param_index:param_index + num_params]
        param_index += num_params

        # Update the distortion coefficients
        # We need to handle the case where we're only updating some of the coefficients
        dist_coeffs = original.dist_coeffs.flatten()

        # Determine which parameters we're updating based on the length
        if num_params == 2:  # k1, k2 only
            dist_coeffs[0:2] = camera_dist_params
        elif num_params == 4:  # k1, k2, p1, p2
            dist_coeffs[0:4] = camera_dist_params
        elif num_params == 5:  # k1, k2, p1, p2, k3
            dist_coeffs[0:5] = camera_dist_params
        else:
            # Custom number of parameters
            dist_coeffs[0:num_params] = camera_dist_params

        # Update the intrinsics with new distortion coefficients
        updated.dist_coeffs = dist_coeffs.reshape(-1, 1)

        updated_intrinsics[cam_id] = updated

    return updated_intrinsics


def points_3d_to_params(points_3d: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[int]]:
    """
    Convert 3D points dictionary to parameter vector for optimization

    Args:
        points_3d: Dictionary mapping frame numbers to 3D point coordinates

    Returns:
        Tuple of (parameter vector, ordered frame numbers)
    """
    point_params = []
    frame_numbers = sorted(list(points_3d.keys()))

    for frame in frame_numbers:
        point_coords = points_3d[frame]
        point_params.extend(point_coords)

    return np.array(point_params), frame_numbers


def params_to_points_3d(params: np.ndarray, frame_numbers: List[int]) -> Dict[int, np.ndarray]:
    """
    Convert parameter vector back to 3D points dictionary

    Args:
        params: Parameter vector
        frame_numbers: Ordered list of frame numbers

    Returns:
        Dictionary mapping frame numbers to 3D point coordinates
    """
    points_3d = {}

    for i, frame in enumerate(frame_numbers):
        # Each point has 3 parameters (x, y, z)
        param_idx = i * 3
        point_coords = params[param_idx:param_idx + 3]

        points_3d[frame] = point_coords

    return points_3d


def project_point_robust(point_3d: np.ndarray, camera_intrinsic: CameraIntrinsic,
                         rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Project a 3D point to a 2D point in camera image with robust error handling

    Args:
        point_3d: 3D point (x, y, z)
        camera_intrinsic: Camera intrinsic parameters
        rvec: Rotation vector
        tvec: Translation vector

    Returns:
        2D projected point (x, y)
    """
    # Project the point
    try:
        # Project point using OpenCV (handles distortion)
        point_3d_reshaped = point_3d.reshape(1, 3)
        projected_point, _ = cv2.projectPoints(
            point_3d_reshaped,
            rvec,
            tvec,
            camera_intrinsic.camera_matrix,
            camera_intrinsic.dist_coeffs
        )
        return projected_point.reshape(2)
    except Exception:
        # Fallback to simple projection (no distortion) if OpenCV fails
        point_3d_homogeneous = np.append(point_3d, 1.0)
        R = cv2.Rodrigues(rvec)[0]
        RT = np.hstack((R, tvec.reshape(3, 1)))
        P = camera_intrinsic.camera_matrix @ RT

        projected = P @ point_3d_homogeneous
        projected = projected / projected[2]  # Normalize
        return projected[:2]


def compute_residuals(params: np.ndarray,
                      camera_ids: List[str],
                      frame_numbers: List[int],
                      camera_intrinsics: Dict[str, CameraIntrinsic],
                      observations: Dict[int, Dict[str, np.ndarray]],
                      optimize_distortion: bool = False,
                      dist_param_counts: Optional[List[int]] = None) -> np.ndarray:
    """
    Compute residuals for bundle adjustment optimization with optional distortion optimization

    Args:
        params: Parameter vector (camera extrinsics + distortion parameters + 3D points)
        camera_ids: Ordered list of camera IDs
        frame_numbers: Ordered list of frame numbers
        camera_intrinsics: Dictionary of camera intrinsics
        observations: Dictionary mapping frames to camera observations
        optimize_distortion: Whether to optimize distortion parameters
        dist_param_counts: List containing number of distortion parameters per camera

    Returns:
        Residual vector
    """
    num_cameras = len(camera_ids)
    num_points = len(frame_numbers)

    # Extract parameters
    camera_params = params[:num_cameras * 6]  # Camera extrinsics

    # If optimizing distortion, extract distortion parameters
    if optimize_distortion and dist_param_counts is not None:
        total_dist_params = sum(dist_param_counts)
        distortion_params = params[num_cameras * 6:num_cameras * 6 + total_dist_params]
        point_params = params[num_cameras * 6 + total_dist_params:]

        # Update camera intrinsics with new distortion parameters
        updated_intrinsics = params_to_camera_intrinsics(
            distortion_params,
            camera_intrinsics,
            camera_ids,
            dist_param_counts
        )
    else:
        # Not optimizing distortion, use original intrinsics
        updated_intrinsics = camera_intrinsics
        point_params = params[num_cameras * 6:]

    # Reconstruct camera extrinsics and 3D points from params
    camera_extrinsics_params = params_to_camera_extrinsics(camera_params, camera_ids)
    points_3d_params = params_to_points_3d(point_params, frame_numbers)

    # Compute residuals
    residuals = []

    for i, frame in enumerate(frame_numbers):
        point_3d = points_3d_params[frame]

        # For each camera that observed this point
        for cam_id in camera_ids:
            # Skip if camera didn't observe this point
            if cam_id not in observations.get(frame, {}):
                continue

            # Get observed 2D point
            observed_point = observations[frame][cam_id]

            # Get camera parameters
            camera_extrinsic = camera_extrinsics_params[cam_id]
            camera_intrinsic = updated_intrinsics[cam_id]  # Use updated intrinsics

            # Get rotation and translation vectors
            rvec, tvec = camera_extrinsic.to_rt()

            # Project 3D point to 2D
            projected_point = project_point_robust(point_3d, camera_intrinsic, rvec, tvec)

            # Compute residual (reprojection error)
            residual = projected_point - observed_point

            # Add to residual vector
            residuals.extend(residual)

    return np.array(residuals)


def compute_bundle_adjustment_jacobian_sparsity(camera_ids: List[str],
                                                frame_numbers: List[int],
                                                observations: Dict[int, Dict[str, np.ndarray]],
                                                optimize_distortion: bool = False,
                                                dist_param_counts: Optional[List[int]] = None) -> csr_matrix:
    """
    Compute the sparsity pattern of the Jacobian matrix for bundle adjustment
    with optional distortion parameter optimization

    Args:
        camera_ids: Ordered list of camera IDs
        frame_numbers: Ordered list of frame numbers
        observations: Dictionary mapping frames to camera observations
        optimize_distortion: Whether to optimize distortion parameters
        dist_param_counts: List containing number of distortion parameters per camera

    Returns:
        Sparse Jacobian matrix structure in CSR format
    """
    num_cameras = len(camera_ids)
    num_points = len(frame_numbers)
    total_dist_params = sum(dist_param_counts) if dist_param_counts is not None else 0

    # Count the number of residuals (2 per observation - x and y components)
    num_residuals = 0
    for frame in frame_numbers:
        for cam_id in camera_ids:
            if cam_id in observations.get(frame, {}):
                num_residuals += 2  # x and y components of reprojection error

    # Total number of parameters
    num_camera_params = num_cameras * 6  # 6 params per camera (3 for rotation, 3 for translation)
    num_dist_params = total_dist_params if optimize_distortion else 0  # Distortion parameters if optimizing
    num_point_params = num_points * 3  # 3 params per 3D point (x, y, z)
    num_params = num_camera_params + num_dist_params + num_point_params

    # Create sparse Jacobian matrix in LIL format (efficient for building)
    jacobian_sparsity = lil_matrix((num_residuals, num_params), dtype=int)

    # Fill the Jacobian sparsity pattern
    row_idx = 0
    for i, frame in enumerate(frame_numbers):
        point_idx = num_camera_params + num_dist_params + i * 3  # Index of the point parameters

        for j, cam_id in enumerate(camera_ids):
            # Skip if camera didn't observe this point
            if cam_id not in observations.get(frame, {}):
                continue

            camera_idx = j * 6  # Index of the camera parameters

            # Each observation contributes to the Jacobian for both the camera and point parameters
            # Mark the entries for x-coordinate residual
            # Camera parameters (rotation and translation)
            jacobian_sparsity[row_idx, camera_idx:camera_idx + 6] = 1

            # Distortion parameters (if optimizing)
            if optimize_distortion and dist_param_counts is not None:
                # Calculate the offset for this camera's distortion parameters
                dist_offset = num_camera_params
                # Add up the number of parameters for all cameras before this one
                for k in range(j):
                    dist_offset += dist_param_counts[k]

                # Number of distortion parameters for this camera
                num_dist = dist_param_counts[j]

                # Mark the entries for distortion parameters
                jacobian_sparsity[row_idx, dist_offset:dist_offset + num_dist] = 1
                jacobian_sparsity[row_idx + 1, dist_offset:dist_offset + num_dist] = 1

            # Point parameters
            jacobian_sparsity[row_idx, point_idx:point_idx + 3] = 1

            # Mark the entries for y-coordinate residual
            # Camera parameters (rotation and translation)
            jacobian_sparsity[row_idx + 1, camera_idx:camera_idx + 6] = 1
            # Point parameters
            jacobian_sparsity[row_idx + 1, point_idx:point_idx + 3] = 1

            # Increment row index by 2 (x and y residuals)
            row_idx += 2

    # Convert to CSR format (efficient for matrix operations)
    return jacobian_sparsity.tocsr()


def robust_loss(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Huber robust loss function

    Args:
        x: Input values
        c: Threshold parameter

    Returns:
        Loss values
    """
    abs_x = np.abs(x)
    mask = abs_x <= c

    result = np.zeros_like(x)
    result[mask] = 0.5 * x[mask] ** 2
    result[~mask] = c * (abs_x[~mask] - 0.5 * c)

    return result


def bundle_adjustment_sparse(camera_data: Dict[str, CameraData],
                             camera_intrinsics: Dict[str, CameraIntrinsic],
                             camera_extrinsics: Dict[str, CameraExtrinsic],
                             min_cameras: int = 2,
                             verbose: bool = True,
                             robust: bool = True,
                             max_iterations: int = 100,
                             optimize_distortion: bool = False,
                             optimize_k1k2: bool = True,
                             optimize_p1p2: bool = True,
                             optimize_k3: bool = False) -> Tuple[Dict[str, CameraExtrinsic],
Dict[str, CameraIntrinsic],
Dict[int, np.ndarray]]:
    """
    Perform bundle adjustment with sparse Jacobian to optimize camera parameters,
    distortion coefficients, and 3D points

    Args:
        camera_data: Dictionary of camera data
        camera_intrinsics: Dictionary of camera intrinsics
        camera_extrinsics: Dictionary of camera extrinsics (initial guess)
        min_cameras: Minimum number of cameras needed to reconstruct a point
        verbose: Whether to output detailed information
        robust: Whether to use robust loss function
        max_iterations: Maximum number of iterations for optimization
        optimize_distortion: Whether to optimize distortion parameters
        optimize_k1k2: Whether to optimize radial distortion parameters k1, k2
        optimize_p1p2: Whether to optimize tangential distortion parameters p1, p2
        optimize_k3: Whether to optimize radial distortion parameter k3

    Returns:
        Tuple of (optimized camera extrinsics, optimized camera intrinsics, optimized 3D points)
    """
    if verbose:
        print("Starting sparse bundle adjustment" +
              (" with distortion optimization..." if optimize_distortion else "..."))

    # Step 1: Reconstruct initial 3D points using the provided camera_extrinsics
    initial_points_3d = {}
    observations_dict = {}

    # Get all synchronized frames
    synced_frames = get_synchronized_frames(camera_data)

    if verbose:
        print(f"Found {len(synced_frames)} synchronized frames")

    # Collect all frame observations
    for frame in synced_frames:
        frame_observations = find_frame_observations(camera_data, frame)

        # Only use frames observed by enough cameras
        if len(frame_observations) >= min_cameras:
            observations_dict[frame] = frame_observations

    if verbose:
        print(f"Using {len(observations_dict)} frames with at least {min_cameras} camera observations")

    # Initialize 3D points using triangulation
    for frame, obs in observations_dict.items():
        try:
            point_3d, filtered_obs = optimize_3d_point_hartley_sturm(
                obs,
                camera_intrinsics,
                camera_extrinsics
            )

            # Check if point is valid
            if isinstance(point_3d, np.ndarray) and point_3d.size > 0 and not np.any(np.isnan(point_3d)):
                initial_points_3d[frame] = point_3d
                observations_dict[frame] = filtered_obs  # Use filtered observations
        except Exception as e:
            if verbose:
                print(f"Error triangulating frame {frame}: {e}")

    if verbose:
        print(f"Initialized {len(initial_points_3d)} 3D points")

    # Convert to parameter vectors
    camera_params, camera_ids = camera_extrinsics_to_params(camera_extrinsics)
    point_params, frame_numbers = points_3d_to_params(initial_points_3d)

    # Handle distortion parameters if optimizing
    if optimize_distortion:
        distortion_params, _ = camera_intrinsics_to_params(
            camera_intrinsics,
            camera_ids,
            optimize_k1k2=optimize_k1k2,
            optimize_p1p2=optimize_p1p2,
            optimize_k3=optimize_k3
        )

        # Determine how many distortion parameters each camera has
        dist_param_counts = []
        for cam_id in camera_ids:
            count = 0
            if optimize_k1k2:
                count += 2
            if optimize_p1p2:
                count += 2
            if optimize_k3:
                count += 1
            dist_param_counts.append(count)

        # Combine parameters into a single vector
        params = np.concatenate([camera_params, distortion_params, point_params])

        if verbose:
            print(f"Parameter vector size: {len(params)}")
            print(f"  Camera parameters: {len(camera_params)} ({len(camera_ids)} cameras)")
            print(f"  Distortion parameters: {len(distortion_params)} ({sum(dist_param_counts)} total)")
            print(f"  Point parameters: {len(point_params)} ({len(frame_numbers)} points)")
    else:
        # Not optimizing distortion, just combine extrinsics and points
        params = np.concatenate([camera_params, point_params])
        dist_param_counts = None

        if verbose:
            print(f"Parameter vector size: {len(params)}")
            print(f"  Camera parameters: {len(camera_params)} ({len(camera_ids)} cameras)")
            print(f"  Point parameters: {len(point_params)} ({len(frame_numbers)} points)")

    # Calculate initial reprojection error
    initial_residuals = compute_residuals(
        params,
        camera_ids,
        frame_numbers,
        camera_intrinsics,
        observations_dict,
        optimize_distortion=optimize_distortion,
        dist_param_counts=dist_param_counts
    )

    initial_rmse = np.sqrt(np.mean(initial_residuals ** 2))

    if verbose:
        print(f"Initial RMSE: {initial_rmse:.4f} pixels")

    # Create Jacobian sparsity pattern
    if verbose:
        print("Computing Jacobian sparsity pattern...")

    sparsity = compute_bundle_adjustment_jacobian_sparsity(
        camera_ids,
        frame_numbers,
        observations_dict,
        optimize_distortion=optimize_distortion,
        dist_param_counts=dist_param_counts
    )

    if verbose:
        nnz = sparsity.nnz
        total_size = sparsity.shape[0] * sparsity.shape[1]
        sparsity_ratio = 100.0 * (1.0 - nnz / total_size)
        print(f"Jacobian sparsity: {sparsity_ratio:.2f}% sparsity")
        print(f"  Shape: {sparsity.shape}")
        print(f"  Non-zero elements: {nnz} out of {total_size}")

    # Define optimization function
    def optimization_function(params):
        residuals = compute_residuals(
            params,
            camera_ids,
            frame_numbers,
            camera_intrinsics,
            observations_dict,
            optimize_distortion=optimize_distortion,
            dist_param_counts=dist_param_counts
        )

        if robust:
            # Apply robust loss function (Huber)
            return robust_loss(residuals, c=1.0)
        else:
            return residuals

    # Run bundle adjustment with sparse Jacobian
    if verbose:
        print("Running sparse bundle adjustment optimization...")

    result = least_squares(
        optimization_function,
        params,
        jac_sparsity=sparsity,  # Use sparse Jacobian
        method='trf',  # Trust Region Reflective algorithm
        loss='linear' if not robust else 'soft_l1',  # Use built-in robust loss if requested
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=max_iterations,
        verbose=2 if verbose else 0
    )

    optimized_params = result.x

    # Extract optimized parameters
    num_cameras = len(camera_ids)

    if optimize_distortion:
        # Extract optimized camera extrinsics, distortion parameters, and 3D points
        optimized_camera_params = optimized_params[:num_cameras * 6]

        total_dist_params = sum(dist_param_counts)
        optimized_dist_params = optimized_params[num_cameras * 6:num_cameras * 6 + total_dist_params]
        optimized_point_params = optimized_params[num_cameras * 6 + total_dist_params:]

        # Convert back to dictionaries
        optimized_camera_extrinsics = params_to_camera_extrinsics(optimized_camera_params, camera_ids)
        optimized_camera_intrinsics = params_to_camera_intrinsics(
            optimized_dist_params,
            camera_intrinsics,
            camera_ids,
            dist_param_counts
        )
    else:
        # Not optimizing distortion, just extract extrinsics and points
        optimized_camera_params = optimized_params[:num_cameras * 6]
        optimized_point_params = optimized_params[num_cameras * 6:]

        # Convert back to dictionaries
        optimized_camera_extrinsics = params_to_camera_extrinsics(optimized_camera_params, camera_ids)
        optimized_camera_intrinsics = camera_intrinsics  # Use original intrinsics

    optimized_points_3d = params_to_points_3d(optimized_point_params, frame_numbers)

    # Calculate final reprojection error using the optimized parameters
    final_residuals = compute_residuals(
        optimized_params,
        camera_ids,
        frame_numbers,
        optimize_distortion=optimize_distortion,
        camera_intrinsics=camera_intrinsics,  # Use original intrinsics for consistency
        observations=observations_dict,
        dist_param_counts=dist_param_counts
    )

    # Calculate the absolute value of residuals (which contains x and y direction errors)
    abs_residuals = np.abs(final_residuals)

    # Calculate Euclidean distance error for each observation point (combine x,y direction errors)
    point_residuals = np.sqrt(np.sum(abs_residuals.reshape(-1, 2) ** 2, axis=1))

    # Calculate statistics
    mean_residual = np.mean(point_residuals)
    median_residual = np.median(point_residuals)
    max_residual = np.max(point_residuals)
    min_residual = np.min(point_residuals)

    # Calculate percentage statistics
    percent_below_1px = np.sum(point_residuals < 1.0) / len(point_residuals) * 100
    percent_below_5px = np.sum(point_residuals < 5.0) / len(point_residuals) * 100

    if verbose:
        print("\n=== Residual Status ===")
        print(f"Mean residual: {mean_residual:.4f} pixels")
        print(f"Median residual: {median_residual:.4f} pixels")
        print(f"Max residual: {max_residual:.4f} pixels")
        print(f"Min residual: {min_residual:.4f} pixels")
        print(f"<1 pixel: {percent_below_1px:.2f}%")
        print(f"<5 pixels: {percent_below_5px:.2f}%")

    final_rmse = np.sqrt(np.mean(final_residuals ** 2))

    if verbose:
        print(f"\nSparse bundle adjustment complete")
        print(f"Final RMSE: {final_rmse:.4f} pixels")
        print(f"Improvement: {initial_rmse - final_rmse:.4f} pixels ({(initial_rmse - final_rmse) / initial_rmse:.2%})")

        # Print camera parameter changes
        print("\nCamera Parameter Changes:")
        for i, cam_id in enumerate(camera_ids):
            initial_extrinsic = camera_extrinsics[cam_id]
            optimized_extrinsic = optimized_camera_extrinsics[cam_id]

            # Calculate rotation difference
            initial_R = initial_extrinsic.rotation
            optimized_R = optimized_extrinsic.rotation

            R_diff = np.matmul(optimized_R, initial_R.T)
            r_diff = Rotation.from_matrix(R_diff)
            degrees = np.degrees(r_diff.magnitude())

            # Calculate translation difference
            initial_t = initial_extrinsic.translation
            optimized_t = optimized_extrinsic.translation
            t_diff = np.linalg.norm(optimized_t - initial_t)

            print(f"  Camera {cam_id}:")
            print(f"    Rotation change: {degrees:.4f} degrees")
            print(f"    Translation change: {t_diff:.4f} units")

            # Print distortion parameter changes if optimizing
            if optimize_distortion:
                initial_dist = camera_intrinsics[cam_id].dist_coeffs.flatten()
                optimized_dist = optimized_camera_intrinsics[cam_id].dist_coeffs.flatten()

                print("    Distortion changes:")
                param_names = ["k1", "k2", "p1", "p2", "k3"]
                for j in range(min(len(initial_dist), len(param_names))):
                    if (optimize_k1k2 and j < 2) or (optimize_p1p2 and 2 <= j < 4) or (optimize_k3 and j == 4):
                        change = optimized_dist[j] - initial_dist[j]
                        percent = (change / initial_dist[j] * 100) if initial_dist[j] != 0 else float('inf')
                        print(
                            f"      {param_names[j]}: {initial_dist[j]:.6f} -> {optimized_dist[j]:.6f} (change: {change:.6f}, {percent:.2f}%)")

    return optimized_camera_extrinsics, optimized_camera_intrinsics, optimized_points_3d


def save_optimization_results(output_dir: str,
                              optimized_camera_extrinsics: Dict,
                              optimized_camera_intrinsics: Dict,
                              optimized_points_3d: Dict,
                              save_format: str = 'json'):
    """
    保存优化后的相机外参、内参和3D点云数据到指定目录

    Args:
        output_dir: 输出目录路径
        optimized_camera_extrinsics: 优化后的相机外参数据字典
        optimized_camera_intrinsics: 优化后的相机内参数据字典
        optimized_points_3d: 优化后的3D点云数据字典
        save_format: 保存格式，可选 'pickle'(默认) 或 'json'
    """
    # 如果目录不存在，创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 根据保存格式选择不同的保存方法
    if save_format.lower() == 'pickle':
        # 保存为Pickle格式
        # 保存相机外参
        extrinsics_path = os.path.join(output_dir, "optimized_camera_extrinsics.pkl")
        with open(extrinsics_path, 'wb') as f:
            pickle.dump(optimized_camera_extrinsics, f)

        # 保存相机内参
        intrinsics_path = os.path.join(output_dir, "optimized_camera_intrinsics.pkl")
        with open(intrinsics_path, 'wb') as f:
            pickle.dump(optimized_camera_intrinsics, f)

        # 保存3D点云
        points3d_path = os.path.join(output_dir, "optimized_points_3d.pkl")
        with open(points3d_path, 'wb') as f:
            pickle.dump(optimized_points_3d, f)

    elif save_format.lower() == 'json':
        # 保存为JSON格式
        # 由于相机外参和内参包含numpy数组和特殊对象，需要进行处理

        # 处理相机外参
        json_extrinsics = {}
        for cam_id, extrinsic in optimized_camera_extrinsics.items():
            json_extrinsics[cam_id] = {
                'rotation': extrinsic.rotation.tolist(),
                'translation': extrinsic.translation.tolist()
            }

        extrinsics_path = os.path.join(output_dir, "optimized_camera_extrinsics.json")
        with open(extrinsics_path, 'w') as f:
            json.dump(json_extrinsics, f, indent=2)

        # 处理相机内参
        json_intrinsics = {}
        for cam_id, intrinsic in optimized_camera_intrinsics.items():
            json_intrinsics[cam_id] = {
                'camera_matrix': intrinsic.camera_matrix.tolist(),
                'dist_coeffs': intrinsic.dist_coeffs.tolist()
            }

        intrinsics_path = os.path.join(output_dir, "optimized_camera_intrinsics.json")
        with open(intrinsics_path, 'w') as f:
            json.dump(json_intrinsics, f, indent=2)

        # 处理3D点云
        json_points3d = {}
        for frame, point in optimized_points_3d.items():
            json_points3d[str(frame)] = point.tolist()

        points3d_path = os.path.join(output_dir, "optimized_points_3d.json")
        with open(points3d_path, 'w') as f:
            json.dump(json_points3d, f, indent=2)
    else:
        raise ValueError(f"saving format not supported: {save_format}, supported formats: 'pickle' 和 'json'")

    print(f"saved dir: {output_dir}")
    print(f"  - camera extrinsic: {os.path.basename(extrinsics_path)}")
    print(f"  - camera intrinsic: {os.path.basename(intrinsics_path)}")
    print(f"  - 3D point cloud: {os.path.basename(points3d_path)}")

def run_sparse_bundle_adjustment_with_distortion(camera_data: Dict[str, CameraData],
                                                 camera_intrinsics: Dict[str, CameraIntrinsic],
                                                 camera_extrinsics: Dict[str, CameraExtrinsic],
                                                 min_cameras: int = 2,
                                                 verbose: bool = True,
                                                 robust: bool = True,
                                                 max_iterations: int = 100,
                                                 optimize_distortion: bool = True,
                                                 optimize_k1k2: bool = True,
                                                 optimize_p1p2: bool = True,
                                                 optimize_k3: bool = False,
                                                 save_dir: str = None) -> Tuple[
    Dict[str, CameraExtrinsic], Dict[str, CameraIntrinsic], Dict[int, np.ndarray]]:
    """
    Run sparse bundle adjustment using the results from your SFM reconstruction code,
    with optional distortion parameter optimization.

    Args:
        camera_data: Dictionary of camera data from sfm
        camera_intrinsics: Dictionary of camera intrinsics from sfm
        camera_extrinsics: Dictionary of camera extrinsics from sfm
        min_cameras: Minimum number of cameras needed to reconstruct a point in BA
        verbose: Whether to output detailed information
        robust: Whether to use robust loss function
        max_iterations: Maximum number of iterations for optimization
        optimize_distortion: Whether to optimize distortion parameters
        optimize_k1k2: Whether to optimize radial distortion parameters k1, k2
        optimize_p1p2: Whether to optimize tangential distortion parameters p1, p2
        optimize_k3: Whether to optimize radial distortion parameter k3

    Returns:
        Tuple of (optimized camera extrinsics, optimized camera intrinsics, optimized 3D points)
    """
    if verbose:
        print("Starting sparse bundle adjustment using provided SFM results" +
              (" with distortion optimization..." if optimize_distortion else "..."))

    # Run bundle adjustment to optimize camera extrinsics, intrinsics, and 3D points
    optimized_camera_extrinsics, optimized_camera_intrinsics, optimized_points_3d = bundle_adjustment_sparse(
        camera_data,
        camera_intrinsics,
        camera_extrinsics,
        min_cameras=min_cameras,
        verbose=verbose,
        robust=robust,
        max_iterations=max_iterations,
        optimize_distortion=optimize_distortion,
        optimize_k1k2=optimize_k1k2,
        optimize_p1p2=optimize_p1p2,
        optimize_k3=optimize_k3
    )

    if save_dir is not None:
        save_optimization_results(save_dir,
                                  optimized_camera_extrinsics,
                                  optimized_camera_intrinsics,
                                  optimized_points_3d)

    return optimized_camera_extrinsics, optimized_camera_intrinsics, optimized_points_3d


if __name__ == "__main__":
    # Example of how to use run_sparse_bundle_adjustment_with_distortion

    # This is just an example - you would replace this with your actual code
    try:
        from init_camera_extrinsic_ransac import data_preprocess_with_incremental_sfm_sync
    except ImportError:
        from .init_camera_extrinsic_ransac import data_preprocess_with_incremental_sfm_sync

    # Example paths
    base_dir = "../data_file"
    data2d_path = os.path.join(base_dir, "20241017_164418/data2d_distorted.csv")
    cam_info_path = os.path.join(base_dir, "20241017_164418/cam_info.csv")
    yaml_dir = os.path.join(base_dir, "intrinsic_calibrations")

    # Get SFM results first
    camera_data, camn_to_id, camera_intrinsics, camera_extrinsics = data_preprocess_with_incremental_sfm_sync(
        data2d_path,
        cam_info_path,
        yaml_dir,
        min_shared_frames=50,
        verbose=True
    )

    # Run sparse bundle adjustment with distortion optimization
    optimized_camera_extrinsics, optimized_camera_intrinsics, optimized_points_3d = run_sparse_bundle_adjustment_with_distortion(
        camera_data,
        camera_intrinsics,
        camera_extrinsics,
        min_cameras=2,
        verbose=True,
        robust=True,
        max_iterations=100,
        optimize_distortion=True,  # Enable distortion optimization
        optimize_k1k2=True,  # Optimize radial distortion k1, k2
        optimize_p1p2=True,  # Optimize tangential distortion p1, p2
        optimize_k3=False  # Don't optimize k3 (often unstable)
    )

    # Output results
    print("\n=== Optimized Camera Extrinsics ===")
    for cam_id, extrinsic in optimized_camera_extrinsics.items():
        print(f"Camera: {cam_id}")
        print("Rotation matrix:")
        print(extrinsic.rotation)
        print("Translation vector:")
        print(extrinsic.translation)
        print("-" * 50)

    print("\n=== Optimized Camera Intrinsics (Distortion) ===")
    for cam_id, intrinsic in optimized_camera_intrinsics.items():
        print(f"Camera: {cam_id}")
        print("Distortion coefficients:")
        dist_coeffs = intrinsic.dist_coeffs.flatten()
        param_names = ["k1", "k2", "p1", "p2", "k3"]
        for i in range(min(len(dist_coeffs), len(param_names))):
            print(f"  {param_names[i]}: {dist_coeffs[i]:.8f}")
        print("-" * 50)

    print(f"\nReconstructed {len(optimized_points_3d)} 3D points")