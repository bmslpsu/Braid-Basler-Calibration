import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import copy
import itertools
import os

# Try to import from data_preprocess module
try:
    from data_preprocess import data_preprocess
except ImportError:
    try:
        from .data_preprocess import data_preprocess
    except ImportError:
        print("Warning: Could not import data_preprocess module")

try:
    from manual_extrinsics_gui import manual_extrinsics_calibration
except ImportError:
    try:
        from .manual_extrinsics_gui import manual_extrinsics_calibration
    except ImportError:
        print("Warning: Could not import manual_extrinsics_calibration module")


@dataclass
class CameraData:
    """Store camera frame info for a single point tracking"""
    frames: List[int]
    x_coords: List[float]  # NaN when not observed
    y_coords: List[float]  # NaN when not observed
    velocities: Optional[List[float]] = None  # for velocity
    is_valid: Optional[List[bool]] = None  # for valid points masks


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


def count_valid_points(camera_data: CameraData) -> int:
    """Count valid coordinate pairs in camera data"""
    valid_coords_count = 0
    for x, y in zip(camera_data.x_coords, camera_data.y_coords):
        if not (np.isnan(x) or np.isnan(y)):
            valid_coords_count += 1

    return valid_coords_count


def find_frame_observations(camera_data: Dict[str, CameraData], frame: int) -> Dict[str, np.ndarray]:
    """
    Find all valid observations for a specific frame across all cameras

    Args:
        camera_data: Dictionary of camera data
        frame: Frame number

    Returns:
        Dictionary mapping camera_id to its 2D point observation
    """
    observations = {}

    for cam_id, cam_data in camera_data.items():
        if frame in cam_data.frames:
            idx = cam_data.frames.index(frame)
            x = cam_data.x_coords[idx]
            y = cam_data.y_coords[idx]

            if not (np.isnan(x) or np.isnan(y)):
                observations[cam_id] = np.array([x, y])

    return observations


def triangulate_hartley_sturm(observations: Dict[str, np.ndarray],
                              camera_intrinsics: Dict[str, CameraIntrinsic],
                              camera_extrinsics: Dict[str, CameraExtrinsic]) -> np.ndarray:
    """
    Triangulate a 3D point using Hartley-Sturm triangulation

    Args:
        observations: Dictionary mapping camera ID to 2D observation
        camera_intrinsics: Dictionary of camera intrinsics
        camera_extrinsics: Dictionary of camera extrinsics

    Returns:
        Triangulated 3D point coordinates (x, y, z)
    """
    if len(observations) < 2:
        raise ValueError("Need at least two camera observations for triangulation")

    # First, get an initial estimate using DLT
    # Build coefficient matrix A for linear system AX = 0
    A = []

    for cam_id, point_2d in observations.items():
        # Get projection matrix P = K[R|t]
        K = camera_intrinsics[cam_id].camera_matrix
        RT = camera_extrinsics[cam_id].projection_matrix
        P = K @ RT

        x, y = point_2d

        # For each point, build two equations
        # x * p^3 - p^1 = 0
        # y * p^3 - p^2 = 0
        # where p^i is the i-th row of projection matrix P
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    # Stack all equations into one big matrix
    A = np.vstack(A)

    # Solve homogeneous equation AX = 0 using SVD
    _, _, Vt = np.linalg.svd(A)

    # Solution is the right singular vector corresponding to the smallest singular value
    X_initial = Vt[-1, :]

    # Convert from homogeneous to non-homogeneous coordinates
    X_initial = X_initial / X_initial[3]
    X_initial = X_initial[:3]

    # Now refine the estimate using Hartley-Sturm optimization
    def reprojection_cost(X):
        X_homogeneous = np.append(X, 1.0)
        total_cost = 0

        for cam_id, point_2d in observations.items():
            K = camera_intrinsics[cam_id].camera_matrix
            RT = camera_extrinsics[cam_id].projection_matrix
            P = K @ RT

            # Project 3D point
            projected = P @ X_homogeneous
            projected = projected / projected[2]  # Normalize by homogeneous coordinate

            # Calculate Euclidean distance
            error = np.linalg.norm(projected[:2] - point_2d)
            total_cost += error ** 2

        return total_cost

    # Optimize using Hartley-Sturm method
    result = least_squares(
        lambda X: np.sqrt(reprojection_cost(X)),  # Take square root to get actual reprojection error
        X_initial,
        method='trf',  # Trust Region Reflective algorithm
        loss='linear',  # Standard least squares
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=100  # Maximum number of function evaluations
    )

    return result.x


def project_point(point_3d: np.ndarray, camera_intrinsic: CameraIntrinsic,
                  camera_extrinsic: CameraExtrinsic) -> np.ndarray:
    """
    Project a 3D point to a 2D point in camera image

    Args:
        point_3d: 3D point (x, y, z)
        camera_intrinsic: Camera intrinsic parameters
        camera_extrinsic: Camera extrinsic parameters

    Returns:
        2D projected point (x, y)
    """
    # Ensure point is 3D
    if len(point_3d) == 4:
        point_3d = point_3d[:3]

    # Convert to homogeneous coordinates
    point_3d_homogeneous = np.append(point_3d, 1)

    # Project: P = K[R|t]
    projection = camera_intrinsic.camera_matrix @ camera_extrinsic.projection_matrix @ point_3d_homogeneous

    # Convert to non-homogeneous coordinates
    projection = projection / projection[2]
    return projection[:2]


def reprojection_error(point_3d: np.ndarray, point_2d: np.ndarray,
                       camera_intrinsic: CameraIntrinsic, camera_extrinsic: CameraExtrinsic) -> float:
    """
    Calculate reprojection error between a 3D point and its observed 2D point

    Args:
        point_3d: 3D point (x, y, z)
        point_2d: Observed 2D point (x, y)
        camera_intrinsic: Camera intrinsic parameters
        camera_extrinsic: Camera extrinsic parameters

    Returns:
        Reprojection error (Euclidean distance)
    """
    projected = project_point(point_3d, camera_intrinsic, camera_extrinsic)
    return np.linalg.norm(projected - point_2d)


def optimize_3d_point_hartley_sturm(observations: Dict[str, np.ndarray],
                                    camera_intrinsics: Dict[str, CameraIntrinsic],
                                    camera_extrinsics: Dict[str, CameraExtrinsic],
                                    max_iterations: int = 5) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Improved 3D point optimization using Hartley-Sturm triangulation with outlier removal

    Args:
        observations: Dictionary mapping camera ID to 2D observation
        camera_intrinsics: Dictionary of camera intrinsics
        camera_extrinsics: Dictionary of camera extrinsics
        max_iterations: Maximum number of RANSAC-like iterations

    Returns:
        (Optimized 3D point, Filtered observations without outliers)
    """
    if len(observations) < 2:
        return np.zeros(3), observations

    # Working copy to avoid modifying input
    current_observations = observations.copy()

    # First get initial 3D point estimate using Hartley-Sturm
    try:
        initial_point = triangulate_hartley_sturm(current_observations, camera_intrinsics, camera_extrinsics)
    except Exception as e:
        print(f"Hartley-Sturm triangulation failed: {e}")
        # If initial triangulation fails, try to initialize with zeros
        initial_point = np.zeros(3)

        # Try a simple least squares to find a possible initial point
        def simple_objective(point):
            errors = []
            for cam_id, obs in current_observations.items():
                proj = project_point(point, camera_intrinsics[cam_id], camera_extrinsics[cam_id])
                error = proj - obs
                errors.extend(error)
            return errors

        try:
            result = least_squares(simple_objective, initial_point, method='lm')
            initial_point = result.x
        except:
            return np.zeros(3), {}

    best_point = initial_point.copy()
    best_error = float('inf')
    best_observations = current_observations.copy()

    # RANSAC-like iterative optimization, removing outliers
    for iteration in range(max_iterations):
        # Calculate reprojection errors for all observations
        errors = {}
        total_error = 0.0

        for cam_id, obs in observations.items():  # Check all original observations
            error = reprojection_error(
                best_point,
                obs,
                camera_intrinsics[cam_id],
                camera_extrinsics[cam_id]
            )
            errors[cam_id] = error
            if cam_id in current_observations:  # Only count error for included observations
                total_error += error

        avg_error = total_error / len(current_observations) if current_observations else float('inf')

        # Detect outliers based on reprojection errors
        if len(errors) >= 3:  # Need at least 3 observations for robust statistics
            error_values = np.array(list(errors.values()))
            median_error = np.median(error_values)
            # Use median absolute deviation (MAD) for outlier detection
            mad = np.median(np.abs(error_values - median_error))
            # Modified Z-score threshold (adaptive threshold based on data)
            dynamic_threshold = median_error + 1.4826 * 3 * mad  # Approx equals 3 standard deviations
        else:
            # Default to a reasonable threshold if not enough observations for robust stats
            dynamic_threshold = 5.0

        # Identify outliers
        outliers = []
        for cam_id, error in errors.items():
            if error > dynamic_threshold:
                outliers.append(cam_id)

        # If no outliers or can't remove more observations (need at least 2)
        if not outliers or len(current_observations) - len(outliers) < 2:
            break

        # Remove the worst outlier
        if outliers:
            worst_outlier = max(outliers, key=lambda x: errors[x])
            current_observations.pop(worst_outlier, None)

        # Re-triangulate with the filtered observations
        try:
            best_point = triangulate_hartley_sturm(current_observations, camera_intrinsics, camera_extrinsics)
            best_observations = current_observations.copy()
        except Exception as e:
            print(f"Hartley-Sturm re-triangulation failed: {e}")
            break

    # Check if 3D point is in front of all cameras
    valid_observations = {}
    for cam_id, obs in best_observations.items():
        # Transform 3D point to camera coordinates
        point_in_cam = camera_extrinsics[cam_id].rotation @ best_point + camera_extrinsics[cam_id].translation

        # Only keep observations where the point is in front of the camera (Z > 0)
        if np.isscalar(point_in_cam[2]) and point_in_cam[2] > 0:
            valid_observations[cam_id] = obs

    # If valid observations less than 2, use original best observations
    if len(valid_observations) < 2:
        return best_point, best_observations

    return best_point, valid_observations


def get_synchronized_frames(camera_data: Dict[str, CameraData]) -> List[int]:
    """
    Find valid frames with synchronous observations across cameras

    Args:
        camera_data: Dictionary of camera data

    Returns:
        List of synchronized frame numbers with valid observations
    """
    if not camera_data:
        return []

    first_cam = next(iter(camera_data.values()))
    all_frames = first_cam.frames

    # Keep frames with at least two cameras having valid observations
    synchronized_frames = []
    for frame in all_frames:
        observations = find_frame_observations(camera_data, frame)
        if len(observations) >= 2:  # At least two cameras with valid observations
            synchronized_frames.append(frame)

    return synchronized_frames


def get_shared_frames_between_cameras(camera_data: Dict[str, CameraData],
                                      cam_id1: str,
                                      cam_id2: str) -> List[int]:
    """
    Find frames observed by both cameras in synchronized data

    Args:
        camera_data: Dictionary of camera data
        cam_id1: First camera ID
        cam_id2: Second camera ID

    Returns:
        List of frame numbers observed by both cameras
    """
    if cam_id1 not in camera_data or cam_id2 not in camera_data:
        return []

    shared_frames = []

    for i, (frame, x1, y1, x2, y2) in enumerate(zip(
            camera_data[cam_id1].frames,
            camera_data[cam_id1].x_coords,
            camera_data[cam_id1].y_coords,
            camera_data[cam_id2].x_coords,
            camera_data[cam_id2].y_coords
    )):
        # Only add frames where both cameras have valid observations
        if (not np.isnan(x1) and not np.isnan(y1) and
                not np.isnan(x2) and not np.isnan(y2)):
            shared_frames.append(frame)

    return shared_frames


def estimate_essential_matrix(observations1: List[np.ndarray],
                              observations2: List[np.ndarray],
                              camera_intrinsic1: CameraIntrinsic,
                              camera_intrinsic2: CameraIntrinsic,
                              method: int = cv2.RANSAC,
                              prob: float = 0.999) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate essential matrix using normalized DLT and multiple thresholds

    Args:
        observations1: List of 2D points from camera 1
        observations2: List of 2D points from camera 2
        camera_intrinsic1: Camera 1 intrinsic parameters
        camera_intrinsic2: Camera 2 intrinsic parameters
        method: Method for estimating essential matrix (RANSAC, etc.)
        prob: Confidence probability

    Returns:
        (Essential matrix, mask of inliers)
    """
    # Normalize point coordinates
    points1_normalized = []
    points2_normalized = []

    K1_inv = np.linalg.inv(camera_intrinsic1.camera_matrix)
    K2_inv = np.linalg.inv(camera_intrinsic2.camera_matrix)

    for p1, p2 in zip(observations1, observations2):
        p1_homogeneous = np.append(p1, 1.0)
        p2_homogeneous = np.append(p2, 1.0)

        p1_normalized = K1_inv @ p1_homogeneous
        p2_normalized = K2_inv @ p2_homogeneous

        points1_normalized.append(p1_normalized[:2])
        points2_normalized.append(p2_normalized[:2])

    # Convert to numpy arrays
    points1_normalized = np.array(points1_normalized, dtype=np.float32)
    points2_normalized = np.float32(points2_normalized)

    # Try multiple thresholds, select best result
    best_E = None
    best_mask = None
    best_inlier_ratio = 0

    # Multiple thresholds to try
    thresholds = [0.5, 1.0, 2.0, 3.0]

    for thresh in thresholds:
        E, mask = cv2.findEssentialMat(
            points1_normalized,
            points2_normalized,
            focal=1.0,
            pp=(0, 0),
            method=method,
            prob=prob,
            threshold=thresh
        )

        # Ensure E and mask are valid
        if E is None or mask is None:
            continue

        # Calculate inlier ratio
        inlier_count = np.sum(mask)
        inlier_ratio = inlier_count / len(mask) if len(mask) > 0 else 0

        # If found better result
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_E = E
            best_mask = mask

    # If all attempts failed, use default threshold
    if best_E is None:
        best_E, best_mask = cv2.findEssentialMat(
            points1_normalized,
            points2_normalized,
            focal=1.0,
            pp=(0, 0),
            method=method,
            prob=prob,
            threshold=1.0  # Default threshold
        )

    # Validate essential matrix against algebraic constraints
    if best_E is not None:
        # SVD of essential matrix
        u, s, vt = np.linalg.svd(best_E)

        # Essential matrix should have two equal singular values and one zero
        # Force to [1,1,0]
        s = np.array([1.0, 1.0, 0.0])

        # Rebuild essential matrix
        best_E = u @ np.diag(s) @ vt

    return best_E, best_mask


def decompose_essential_matrix(E: np.ndarray,
                               points1: List[np.ndarray],
                               points2: List[np.ndarray],
                               K1: np.ndarray,
                               K2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose essential matrix to rotation and translation

    Args:
        E: Essential matrix
        points1: List of 2D points from camera 1
        points2: List of 2D points from camera 2
        K1: Camera 1 intrinsic matrix
        K2: Camera 2 intrinsic matrix

    Returns:
        (Rotation matrix, Translation vector)
    """
    # Convert points to numpy arrays
    points1_np = np.array(points1, dtype=np.float32)
    points2_np = np.array(points2, dtype=np.float32)

    # Reshape arrays to match OpenCV expected input format
    if points1_np.ndim == 2 and points1_np.shape[1] == 2:
        points1_np = points1_np.reshape(-1, 1, 2)
    if points2_np.ndim == 2 and points2_np.shape[1] == 2:
        points2_np = points2_np.reshape(-1, 1, 2)

    # Normalize points
    points1_normalized = cv2.undistortPoints(points1_np, K1, None)
    points2_normalized = cv2.undistortPoints(points2_np, K2, None)

    # Format for recoverPose
    points1_normalized = points1_normalized.reshape(-1, 2)
    points2_normalized = points2_normalized.reshape(-1, 2)

    # Recover pose from essential matrix
    _, R, t, mask = cv2.recoverPose(E, points1_normalized, points2_normalized)

    return R, t


def find_best_camera_pair(camera_data: Dict[str, CameraData],
                          camera_intrinsics: Dict[str, CameraIntrinsic],
                          min_shared_frames: int = 50,
                          verbose: bool = True) -> Tuple[str, str, List[int]]:
    """
    Find the best initial camera pair considering observation quality and distribution

    Args:
        camera_data: Dictionary of camera data
        camera_intrinsics: Dictionary of camera intrinsics
        min_shared_frames: Minimum number of shared frames required
        verbose: Whether to output detailed information

    Returns:
        (Camera 1 ID, Camera 2 ID, List of shared frames)
    """
    camera_ids = sorted(list(camera_data.keys()))
    best_pair = (None, None)
    best_frames = []
    best_score = 0

    # Evaluate each camera pair
    for i, cam_id1 in enumerate(camera_ids):
        for j in range(i + 1, len(camera_ids)):
            cam_id2 = camera_ids[j]

            # Get shared frames and observations
            valid_points_indices = []
            for k, (frame, x1, y1, x2, y2) in enumerate(zip(
                    camera_data[cam_id1].frames,
                    camera_data[cam_id1].x_coords,
                    camera_data[cam_id1].y_coords,
                    camera_data[cam_id2].x_coords,
                    camera_data[cam_id2].y_coords
            )):
                if (not np.isnan(x1) and not np.isnan(y1) and
                        not np.isnan(x2) and not np.isnan(y2)):
                    valid_points_indices.append(k)

            shared_frame_count = len(valid_points_indices)

            # Skip if not enough shared frames
            if shared_frame_count < min_shared_frames:
                continue

            # Extract valid 2D coordinates
            points1 = []
            points2 = []
            shared_frames_list = []

            for idx in valid_points_indices:
                frame = camera_data[cam_id1].frames[idx]
                x1 = camera_data[cam_id1].x_coords[idx]
                y1 = camera_data[cam_id1].y_coords[idx]
                x2 = camera_data[cam_id2].x_coords[idx]
                y2 = camera_data[cam_id2].y_coords[idx]

                points1.append([x1, y1])
                points2.append([x2, y2])
                shared_frames_list.append(frame)

            points1 = np.array(points1)
            points2 = np.array(points2)

            # 1. Evaluate point distribution using standard deviation
            std1 = np.std(points1, axis=0)
            std2 = np.std(points2, axis=0)

            # Calculate area of point distribution
            area1 = std1[0] * std1[1]
            area2 = std2[0] * std2[1]

            # Normalize to image size
            img_size1 = camera_intrinsics[cam_id1].image_size
            img_size2 = camera_intrinsics[cam_id2].image_size

            norm_area1 = area1 / (img_size1[0] * img_size1[1])
            norm_area2 = area2 / (img_size2[0] * img_size2[1])

            # 2. Try estimating preliminary essential matrix to evaluate geometric constraints
            try:
                E, mask = estimate_essential_matrix(
                    points1,
                    points2,
                    camera_intrinsics[cam_id1],
                    camera_intrinsics[cam_id2]
                )

                # Calculate inlier ratio
                inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
            except Exception:
                inlier_ratio = 0

            # 3. Composite score: combine shared frames, point distribution and geometric quality
            # Adjustable weights
            w1 = 0.4  # Shared frames weight
            w2 = 0.3  # Point distribution weight
            w3 = 0.3  # Geometric constraint weight

            # Normalize shared frame count
            norm_frame_count = min(1.0, shared_frame_count / (2 * min_shared_frames))

            # Calculate composite score
            distribution_score = (norm_area1 + norm_area2) / 2
            score = w1 * norm_frame_count + w2 * distribution_score + w3 * inlier_ratio

            if verbose:
                print(f"Camera pair {cam_id1}-{cam_id2}: shared frames={shared_frame_count}, " +
                      f"distribution score={distribution_score:.4f}, inlier ratio={inlier_ratio:.4f}, " +
                      f"total score={score:.4f}")

            # Update best result
            if score > best_score:
                best_score = score
                best_pair = (cam_id1, cam_id2)
                best_frames = shared_frames_list

    if verbose and best_pair[0] is not None:
        print(f"Found best camera pair: {best_pair[0]} and {best_pair[1]}")
        print(f"Score: {best_score:.4f}, shared valid observation frames: {len(best_frames)}")

    return best_pair[0], best_pair[1], best_frames


def incremental_sfm_calibration_with_manual_init(camera_data: Dict[str, CameraData],
                                                 camera_intrinsics: Dict[str, CameraIntrinsic],
                                                 min_shared_frames: int = 50,
                                                 verbose: bool = True) -> Dict[str, CameraExtrinsic]:
    """
    Incremental SFM calibration with manual initialization of camera extrinsics

    This function replaces the automatic initial camera pair and essential matrix calculation
    with manual camera extrinsics setting through a GUI.

    Args:
        camera_data: Dictionary of camera data
        camera_intrinsics: Dictionary of camera intrinsics
        min_shared_frames: Minimum number of shared frames required
        verbose: Whether to output detailed information

    Returns:
        Dictionary of camera extrinsics
    """
    # Import manual extrinsics calibration module
    try:
        from manual_extrinsics_gui import manual_extrinsics_calibration
    except ImportError:
        try:
            from .manual_extrinsics_gui import manual_extrinsics_calibration
        except ImportError:
            print("Warning: Could not import data_preprocess module")

    # Get all camera IDs
    camera_ids = sorted(list(camera_data.keys()))
    if len(camera_ids) < 2:
        print("Error: Need at least two cameras")

    if verbose:
        print(f"Starting incremental SFM calibration with manual initialization")
        print(f"Found {len(camera_ids)} cameras")
        for cam_id in camera_ids:
            valid_points = count_valid_points(camera_data[cam_id])
            total_points = len(camera_data[cam_id].frames)
            print(f"  Camera {cam_id}: {valid_points}/{total_points} valid points ({valid_points / total_points:.2%})")
            print(
                f"  Camera resolution: {camera_intrinsics[cam_id].image_size[0]}x{camera_intrinsics[cam_id].image_size[1]}")

    # Launch GUI for manual camera extrinsics setting
    if verbose:
        print("\nLaunching manual camera extrinsic calibration GUI...")
        print("Please set the initial positions and orientations of all cameras")
        print("The first camera will be used as the reference (fixed at origin)")
        print("Press 'Accept' when done or 'Cancel' to abort")

    camera_extrinsics = manual_extrinsics_calibration(camera_intrinsics)

    if not camera_extrinsics or len(camera_extrinsics) < 2:
        print("Error: Manual calibration failed or was cancelled")
        return {}

    if verbose:
        print("\nManual initialization completed successfully")
        print(f"Initialized {len(camera_extrinsics)} cameras")

        # Log the initial extrinsics
        for cam_id, extrinsic in camera_extrinsics.items():
            print(f"\nCamera: {cam_id}")
            print("Rotation matrix:")
            print(extrinsic.rotation)
            print("Translation vector:")
            print(extrinsic.translation)

    # Now we can skip the initial camera pair and essential matrix calculation
    # and proceed with the 3D reconstruction using the manual extrinsics

    # Get synchronized frames for all cameras
    synchronized_frames = get_synchronized_frames(camera_data)
    if verbose:
        print(f"\nFound {len(synchronized_frames)} synchronized frames across cameras")

    # Initialize for 3D reconstruction
    reconstructed_points_3d = {}
    observations_dict = {}

    # 3D triangulation for all synchronized frames
    for frame in synchronized_frames:
        observations = find_frame_observations(camera_data, frame)

        # Only consider frames with enough observations
        if len(observations) >= 2:
            try:
                # Triangulate using Hartley-Sturm method with the manually set extrinsics
                point_3d, filtered_obs = optimize_3d_point_hartley_sturm(
                    observations,
                    camera_intrinsics,
                    camera_extrinsics
                )

                # Check if optimized 3D point is valid
                if isinstance(point_3d, np.ndarray) and np.isscalar(point_3d.size):
                    point_is_valid = point_3d.size > 0 and not np.any(np.isnan(point_3d))
                else:
                    point_is_valid = point_3d is not None

                # Only keep good triangulation results
                if point_is_valid and len(filtered_obs) >= 2:
                    reconstructed_points_3d[frame] = point_3d
                    observations_dict[frame] = filtered_obs
            except Exception as e:
                if verbose:
                    print(f"Error triangulating frame {frame}: {e}")

    if verbose:
        print(f"Triangulated {len(reconstructed_points_3d)} 3D points with manual extrinsics")

        # Calculate and report reprojection errors
        total_error = 0.0
        total_points = 0

        for frame, point_3d in reconstructed_points_3d.items():
            for cam_id, observed_point in observations_dict[frame].items():
                error = reprojection_error(
                    point_3d,
                    observed_point,
                    camera_intrinsics[cam_id],
                    camera_extrinsics[cam_id]
                )
                total_error += error
                total_points += 1

        if total_points > 0:
            avg_error = total_error / total_points
            print(f"Average reprojection error with manual extrinsics: {avg_error:.4f} pixels")

    # Skip the incremental camera addition step, as we already have all camera extrinsics
    # from manual calibration

    return camera_extrinsics


def reconstruct_3d_trajectory(camera_data: Dict[str, CameraData],
                              camera_intrinsics: Dict[str, CameraIntrinsic],
                              camera_extrinsics: Dict[str, CameraExtrinsic],
                              min_cameras: int = 2,
                              verbose: bool = True) -> Dict[int, np.ndarray]:
    """
    Reconstruct 3D trajectory of a point across all frames

    Args:
        camera_data: Dictionary of camera data
        camera_intrinsics: Dictionary of camera intrinsics
        camera_extrinsics: Dictionary of camera extrinsics
        min_cameras: Minimum number of cameras needed to reconstruct a point
        verbose: Whether to output detailed information

    Returns:
        Dictionary mapping frame numbers to 3D point coordinates
    """
    # Get all unique frames across all cameras
    all_frames = set()
    for cam_data in camera_data.values():
        all_frames.update(cam_data.frames)
    all_frames = sorted(list(all_frames))

    if verbose:
        print(f"Reconstructing 3D trajectory across {len(all_frames)} frames")

    trajectory_3d = {}
    frame_count = 0

    for frame in all_frames:
        # Find all observations for this frame
        frame_observations = find_frame_observations(camera_data, frame)

        # Skip frames with too few observations
        if len(frame_observations) < min_cameras:
            if verbose and frame_count % 100 == 0:
                print(
                    f"  Skipping frame {frame}: only {len(frame_observations)} camera(s) available (need {min_cameras})")
            continue

        try:
            # Triangulate and optimize the 3D point using Hartley-Sturm method
            point_3d, filtered_observations = optimize_3d_point_hartley_sturm(
                frame_observations,
                camera_intrinsics,
                camera_extrinsics
            )

            # Check if optimized 3D point is valid
            if isinstance(point_3d, np.ndarray) and np.isscalar(point_3d.size):
                point_is_valid = point_3d.size > 0 and not np.any(np.isnan(point_3d))
            else:
                point_is_valid = point_3d is not None

            # Only accept points with enough valid observations after filtering
            if point_is_valid and len(filtered_observations) >= min_cameras:
                trajectory_3d[frame] = point_3d
                frame_count += 1

                if verbose and frame_count % 100 == 0:
                    print(f"  Processed {frame_count} frames, current frame: {frame}")
        except Exception as e:
            if verbose:
                print(f"  Error processing frame {frame}: {e}")
            continue

    if verbose:
        print(
            f"Successfully reconstructed {len(trajectory_3d)}/{len(all_frames)} frames ({len(trajectory_3d) / len(all_frames):.2%})")

    return trajectory_3d

def incremental_sfm_calibration_sync(camera_data: Dict[str, CameraData],
                                     camera_intrinsics: Dict[str, CameraIntrinsic],
                                     min_shared_frames: int = 50,
                                     verbose: bool = True) -> Dict[str, CameraExtrinsic]:
    """
    Incremental SFM calibration for synchronized data: starting with the best camera pair,
    then gradually adding more cameras

    Args:
        camera_data: Dictionary of camera data
        camera_intrinsics: Dictionary of camera intrinsics
        min_shared_frames: Minimum number of shared frames required
        verbose: Whether to output detailed information

    Returns:
        Dictionary of optimized camera extrinsics
    """
    # Get all camera IDs
    camera_ids = sorted(list(camera_data.keys()))
    if len(camera_ids) < 2:
        print("Error: Need at least two cameras")
        return {}

    if verbose:
        print(f"Starting incremental SFM calibration for synchronized data with {len(camera_ids)} cameras")
        for cam_id in camera_ids:
            valid_points = count_valid_points(camera_data[cam_id])
            total_points = len(camera_data[cam_id].frames)
            print(f"  Camera {cam_id}: {valid_points}/{total_points} valid points ({valid_points / total_points:.2%})")
            print(
                f"  Camera resolution: {camera_intrinsics[cam_id].image_size[0]}x{camera_intrinsics[cam_id].image_size[1]}")

    # Initialize camera extrinsics dictionary
    camera_extrinsics = {}

    # 1. Find best initial camera pair
    ref_cam_id, second_cam_id, shared_frames = find_best_camera_pair(
        camera_data,
        camera_intrinsics,
        min_shared_frames=min_shared_frames,
        verbose=verbose
    )

    if not ref_cam_id or not second_cam_id:
        print("Error: Could not find suitable initial camera pair")
        return {}

    if verbose:
        print(f"Selected initial camera pair: {ref_cam_id} and {second_cam_id}")
        print(f"They share {len(shared_frames)} valid observation frames")

    # 2. Compute essential matrix for initial camera pair and decompose to R,t
    # Collect corresponding points from shared frames
    points1 = []  # Reference camera points
    points2 = []  # Second camera points

    for frame in shared_frames:
        observations = find_frame_observations(camera_data, frame)
        if ref_cam_id in observations and second_cam_id in observations:
            points1.append(observations[ref_cam_id])
            points2.append(observations[second_cam_id])

    # Estimate essential matrix
    E, mask = estimate_essential_matrix(
        points1,
        points2,
        camera_intrinsics[ref_cam_id],
        camera_intrinsics[second_cam_id]
    )

    # Keep only inliers
    inlier_mask = mask.ravel().astype(bool)
    points1_inliers = [points1[i] for i in range(len(points1)) if inlier_mask[i]]
    points2_inliers = [points2[i] for i in range(len(points2)) if inlier_mask[i]]

    if verbose:
        print(
            f"Essential matrix estimation: {len(points1_inliers)}/{len(points1)} inliers ({len(points1_inliers) / len(points1):.2%})")

    # Decompose essential matrix to R,t
    R, t = decompose_essential_matrix(
        E,
        points1_inliers,
        points2_inliers,
        camera_intrinsics[ref_cam_id].camera_matrix,
        camera_intrinsics[second_cam_id].camera_matrix
    )

    # Set reference camera to identity matrix (world coordinate system)
    camera_extrinsics[ref_cam_id] = CameraExtrinsic(
        rotation=np.eye(3),
        translation=np.zeros(3)
    )

    # Set second camera extrinsics
    camera_extrinsics[second_cam_id] = CameraExtrinsic(
        rotation=R,
        translation=t.flatten()
    )

    if verbose:
        print(f"Initialized camera {ref_cam_id} as reference (identity matrix)")
        print(f"Calibrated camera {second_cam_id} from essential matrix")
        print(f"R = \n{R}\nt = {t.flatten()}")

    # 3. Triangulate 3D points for the calibrated camera pair using Hartley-Sturm
    reconstructed_points_3d = {}
    observations_dict = {}

    for frame in shared_frames:
        observations = find_frame_observations(camera_data, frame)
        if ref_cam_id in observations and second_cam_id in observations:
            current_obs = {
                ref_cam_id: observations[ref_cam_id],
                second_cam_id: observations[second_cam_id]
            }

            try:
                # Triangulate and optimize 3D point using Hartley-Sturm
                point_3d, filtered_obs = optimize_3d_point_hartley_sturm(
                    current_obs,
                    camera_intrinsics,
                    camera_extrinsics
                )

                # Check if optimized 3D point is valid
                if isinstance(point_3d, np.ndarray) and np.isscalar(point_3d.size):
                    point_is_valid = point_3d.size > 0 and not np.any(np.isnan(point_3d))
                else:
                    point_is_valid = point_3d is not None

                # Only keep good triangulation results
                if point_is_valid and len(filtered_obs) >= 2:
                    reconstructed_points_3d[frame] = point_3d
                    observations_dict[frame] = filtered_obs
            except Exception as e:
                if verbose:
                    print(f"Error triangulating frame {frame}: {e}")

    if verbose:
        print(f"Triangulated {len(reconstructed_points_3d)} 3D points from initial camera pair")

    # 4. Incrementally add remaining cameras
    calibrated_cameras = {ref_cam_id, second_cam_id}
    remaining_cameras = set(camera_ids) - calibrated_cameras

    # Track which 3D points are available for each new camera
    while remaining_cameras:
        best_cam_id = None
        best_observations_count = 0
        best_cam_score = 0

        # Find camera with most observations of already triangulated points
        for cam_id in remaining_cameras:
            valid_observations = []
            for frame in reconstructed_points_3d:
                frame_observations = find_frame_observations(camera_data, frame)
                if cam_id in frame_observations:
                    valid_observations.append((frame, frame_observations[cam_id]))

            observation_count = len(valid_observations)

            # Skip if too few observations
            if observation_count < min_shared_frames // 2:
                continue

            # Evaluate observation distribution quality
            if observation_count > 0:
                points_2d = np.array([obs[1] for obs in valid_observations])
                # Calculate point distribution (using standard deviation)
                std_xy = np.std(points_2d, axis=0)
                distribution_score = std_xy[0] * std_xy[1]

                # Normalize to image size
                img_size = camera_intrinsics[cam_id].image_size
                norm_distribution = distribution_score / (img_size[0] * img_size[1])

                # Composite score: combine observation count and distribution
                w1 = 0.7  # Observation count weight
                w2 = 0.3  # Distribution weight

                # Normalize observation count
                norm_count = min(1.0, observation_count / (2 * min_shared_frames))

                # Final score
                cam_score = w1 * norm_count + w2 * norm_distribution
            else:
                cam_score = 0

            if cam_score > best_cam_score:
                best_cam_score = cam_score
                best_observations_count = observation_count
                best_cam_id = cam_id

        if best_cam_id is None or best_observations_count < min_shared_frames // 2:
            if verbose:
                print(f"Warning: Could not find more cameras with enough shared observations")
            break

        if verbose:
            print(
                f"Adding camera {best_cam_id}, which shares {best_observations_count} observation points with calibrated cameras")

        # Collect 2D-3D correspondences for PnP
        object_points = []  # 3D points in world coordinate system
        image_points = []  # Corresponding 2D points in new camera

        for frame, point_3d in reconstructed_points_3d.items():
            frame_observations = find_frame_observations(camera_data, frame)
            if best_cam_id in frame_observations:
                object_points.append(point_3d)
                image_points.append(frame_observations[best_cam_id])

        # Convert to numpy arrays
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Use PnP to solve initial extrinsics for new camera
        # Use adaptive threshold for PnP RANSAC based on image size
        img_width, img_height = camera_intrinsics[best_cam_id].image_size
        pnp_threshold = min(img_width, img_height) * 0.05  # 5% of image dimension

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_intrinsics[best_cam_id].camera_matrix,
            camera_intrinsics[best_cam_id].dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=200,  # Increase iterations
            reprojectionError=pnp_threshold,
            confidence=0.999  # Increase confidence
        )

        if not success or inliers is None or len(inliers) < 10:  # Ensure enough inliers
            if verbose:
                print(f"Could not calibrate camera {best_cam_id} or insufficient inliers")
            remaining_cameras.remove(best_cam_id)
            continue

        # Use only RANSAC inliers to refine pose
        inlier_indices = inliers.ravel()
        inlier_object_points = object_points[inlier_indices]
        inlier_image_points = image_points[inlier_indices]

        if verbose:
            print(
                f"RANSAC inliers: {len(inlier_indices)}/{len(object_points)} ({len(inlier_indices) / len(object_points):.2%})")

            # Calculate initial reprojection error
            initial_errors = []
            for i in range(len(inlier_object_points)):
                obj_pt = inlier_object_points[i]
                img_pt = inlier_image_points[i]

                # Project 3D point
                proj_pt, _ = cv2.projectPoints(
                    obj_pt.reshape(1, 3),
                    rvec,
                    tvec,
                    camera_intrinsics[best_cam_id].camera_matrix,
                    camera_intrinsics[best_cam_id].dist_coeffs
                )

                # Calculate error
                error = np.linalg.norm(proj_pt.reshape(2) - img_pt)
                initial_errors.append(error)

            avg_initial_error = np.mean(initial_errors)
            print(f"RANSAC initial average reprojection error: {avg_initial_error:.4f} pixels")

        # Refine pose using LM optimization on inliers
        refined_success, refined_rvec, refined_tvec = cv2.solvePnP(
            inlier_object_points,
            inlier_image_points,
            camera_intrinsics[best_cam_id].camera_matrix,
            camera_intrinsics[best_cam_id].dist_coeffs,
            rvec,  # Use RANSAC result as initial guess
            tvec,  # Use RANSAC result as initial guess
            useExtrinsicGuess=True,  # Use provided initial guess
            flags=cv2.SOLVEPNP_ITERATIVE  # Use iterative LM optimization
        )

        if not refined_success:
            if verbose:
                print(f"Could not refine camera {best_cam_id} pose with inliers, using RANSAC result")
        else:
            if verbose:
                # Calculate refined reprojection error
                refined_errors = []
                for i in range(len(inlier_object_points)):
                    obj_pt = inlier_object_points[i]
                    img_pt = inlier_image_points[i]

                    # Project 3D point
                    proj_pt, _ = cv2.projectPoints(
                        obj_pt.reshape(1, 3),
                        refined_rvec,
                        refined_tvec,
                        camera_intrinsics[best_cam_id].camera_matrix,
                        camera_intrinsics[best_cam_id].dist_coeffs
                    )

                    # Calculate error
                    error = np.linalg.norm(proj_pt.reshape(2) - img_pt)
                    refined_errors.append(error)

                avg_refined_error = np.mean(refined_errors)
                print(f"Refined average reprojection error: {avg_refined_error:.4f} pixels")
                print(
                    f"Error improvement: {avg_initial_error - avg_refined_error:.4f} pixels ({(avg_initial_error - avg_refined_error) / avg_initial_error:.2%})")

            # Use refined result
            rvec = refined_rvec
            tvec = refined_tvec

        # Add new camera to calibrated set
        camera_extrinsics[best_cam_id] = CameraExtrinsic.from_rt(rvec, tvec)
        calibrated_cameras.add(best_cam_id)
        remaining_cameras.remove(best_cam_id)

        if verbose:
            print(f"Successfully calibrated camera {best_cam_id}")

        # Update 3D reconstruction with new camera
        new_points_3d = {}
        new_observations_dict = {}

        # First update existing points with new camera observations
        for frame, point_3d in reconstructed_points_3d.items():
            frame_observations = find_frame_observations(camera_data, frame)

            # Add observation from new camera if available
            current_obs = observations_dict.get(frame, {}).copy()
            if best_cam_id in frame_observations:
                current_obs[best_cam_id] = frame_observations[best_cam_id]

            if len(current_obs) >= 2:  # At least 2 cameras
                try:
                    # Re-triangulate and optimize using all available cameras with Hartley-Sturm
                    optimized_point, filtered_obs = optimize_3d_point_hartley_sturm(
                        current_obs,
                        camera_intrinsics,
                        camera_extrinsics
                    )

                    # Check if optimized 3D point is valid
                    if isinstance(optimized_point, np.ndarray) and np.isscalar(optimized_point.size):
                        point_is_valid = optimized_point.size > 0 and not np.any(np.isnan(optimized_point))
                    else:
                        point_is_valid = optimized_point is not None

                    if point_is_valid and len(filtered_obs) >= 2:
                        new_points_3d[frame] = optimized_point
                        new_observations_dict[frame] = filtered_obs
                    else:
                        # Keep original reconstruction
                        new_points_3d[frame] = reconstructed_points_3d[frame]
                        new_observations_dict[frame] = observations_dict.get(frame, {})
                except Exception as e:
                    if verbose:
                        print(f"Error updating point for frame {frame}: {e}")
                    # If optimization fails, keep original triangulation
                    new_points_3d[frame] = reconstructed_points_3d[frame]
                    new_observations_dict[frame] = observations_dict.get(frame, {})

        # Then find new points visible in new camera and at least one calibrated camera
        for frame in get_synchronized_frames(camera_data):
            # Skip already reconstructed frames
            if frame in new_points_3d:
                continue

            frame_observations = find_frame_observations(camera_data, frame)
            current_obs = {}

            # Only consider observations from calibrated cameras
            for cam_id in calibrated_cameras:
                if cam_id in frame_observations:
                    current_obs[cam_id] = frame_observations[cam_id]

            if len(current_obs) >= 2:  # At least 2 calibrated cameras observe this point
                try:
                    # Triangulate and optimize 3D point with Hartley-Sturm
                    point_3d, filtered_obs = optimize_3d_point_hartley_sturm(
                        current_obs,
                        camera_intrinsics,
                        camera_extrinsics
                    )

                    # Check if optimized 3D point is valid
                    if isinstance(point_3d, np.ndarray) and np.isscalar(point_3d.size):
                        point_is_valid = point_3d.size > 0 and not np.any(np.isnan(point_3d))
                    else:
                        point_is_valid = point_3d is not None

                    if point_is_valid and len(filtered_obs) >= 2:
                        new_points_3d[frame] = point_3d
                        new_observations_dict[frame] = filtered_obs
                except Exception as e:
                    if verbose:
                        print(f"Error triangulating new point for frame {frame}: {e}")

        # Update reconstructed points with new and updated points
        reconstructed_points_3d = new_points_3d
        observations_dict = new_observations_dict

        if verbose:
            print(f"Updated reconstruction: {len(reconstructed_points_3d)} 3D points")

        # Calculate average reprojection error for all current calibrated cameras
        total_reprojection_error = 0.0
        total_points = 0

        for frame, point_3d in reconstructed_points_3d.items():
            for cam_id in calibrated_cameras:
                if cam_id in observations_dict.get(frame, {}):
                    observed_point = observations_dict[frame][cam_id]
                    error = reprojection_error(
                        point_3d,
                        observed_point,
                        camera_intrinsics[cam_id],
                        camera_extrinsics[cam_id]
                    )
                    total_reprojection_error += error
                    total_points += 1

        if total_points > 0:
            avg_error = total_reprojection_error / total_points
            if verbose:
                print(f"Current average reprojection error across all cameras: {avg_error:.4f} pixels")

    if verbose:
        print("\nFinal camera extrinsics:")
        for cam_id, extrinsic in camera_extrinsics.items():
            print(f"Camera: {cam_id}")
            print(f"Rotation matrix R:\n{extrinsic.rotation}")
            print(f"Translation vector t: {extrinsic.translation}")
            print("-" * 30)

        # Calculate final average reprojection error across all cameras
        final_total_error = 0.0
        final_total_points = 0

        for frame, point_3d in reconstructed_points_3d.items():
            for cam_id in calibrated_cameras:
                if cam_id in observations_dict.get(frame, {}):
                    observed_point = observations_dict[frame][cam_id]
                    error = reprojection_error(
                        point_3d,
                        observed_point,
                        camera_intrinsics[cam_id],
                        camera_extrinsics[cam_id]
                    )
                    final_total_error += error
                    final_total_points += 1

        if final_total_points > 0:
            final_avg_error = final_total_error / final_total_points
            print(f"\nFinal system average reprojection error: {final_avg_error:.4f} pixels")
            print(f"Used {final_total_points} observation points for evaluation")

    return camera_extrinsics

def data_preprocess_with_incremental_sfm_sync(data2d_path: str, cam_info_path: str, yaml_dir: str,
                                              min_cameras: int,
                                              min_shared_frames: int = 50,
                                              verbose: bool = True) -> Tuple[
    Dict[str, CameraData], Dict, Dict[str, CameraIntrinsic], Dict[str, CameraExtrinsic]]:
    """
    Preprocess data and complete camera calibration using incremental SFM for synchronized data

    Args:
        data2d_path: Path to 2D data file
        cam_info_path: Path to camera info file
        yaml_dir: Directory of camera intrinsic YAML files
        min_shared_frames: Minimum number of shared frames required
        verbose: Whether to output detailed information

    Returns:
        (Camera data, camn to ID mapping, camera intrinsics, camera extrinsics)
    """
    # Call original data_preprocess function to get basic processed data
    camera_data, camn_to_id, camera_intrinsics = data_preprocess(data2d_path, cam_info_path, yaml_dir,
                                                                 min_cameras=min_cameras, do_undistort=False)

    if verbose:
        print("\nOriginal data statistics:")
        for cam_id, cam_data in camera_data.items():
            valid_points = count_valid_points(cam_data)
            total_points = len(cam_data.frames)
            print(f"  Camera {cam_id}: {valid_points}/{total_points} valid points ({valid_points / total_points:.2%})")
            print(
                f"  Camera resolution: {camera_intrinsics[cam_id].image_size[0]}x{camera_intrinsics[cam_id].image_size[1]}")

    # Use incremental SFM method for camera calibration
    if verbose:
        print("\nStarting camera calibration using incremental SFM method for synchronized data...")
        print(f"Parameters:")
        print(f"  Minimum shared frames: {min_shared_frames}")

    camera_extrinsics = incremental_sfm_calibration_sync(
        camera_data,
        camera_intrinsics,
        min_shared_frames=min_shared_frames,
        verbose=verbose
    )

    return camera_data, camn_to_id, camera_intrinsics, camera_extrinsics

def data_preprocess_with_manual_extrinsics(data2d_path: str, cam_info_path: str, yaml_dir: str,
                                           min_cameras: int, verbose: bool = True) -> Tuple[
    Dict[str, CameraData], Dict, Dict[str, CameraIntrinsic], Dict[str, CameraExtrinsic]]:
    """
    Preprocess data and use manual GUI for camera extrinsic calibration, then proceed with SFM

    Args:
        data2d_path: Path to 2D data file
        cam_info_path: Path to camera info file
        yaml_dir: Directory of camera intrinsic YAML files
        verbose: Whether to output detailed information

    Returns:
        (Camera data, camn to ID mapping, camera intrinsics, camera extrinsics)
    """
    # Call original data_preprocess function to get basic processed data
    camera_data, camn_to_id, camera_intrinsics = data_preprocess(data2d_path, cam_info_path, yaml_dir,
                                                                 min_cameras=min_cameras, do_undistort=False)

    if verbose:
        print("\nOriginal data statistics:")
        for cam_id, cam_data in camera_data.items():
            valid_points = count_valid_points(cam_data)
            total_points = len(cam_data.frames)
            print(f"  Camera {cam_id}: {valid_points}/{total_points} valid points ({valid_points / total_points:.2%})")
            print(
                f"  Camera resolution: {camera_intrinsics[cam_id].image_size[0]}x{camera_intrinsics[cam_id].image_size[1]}")

    # Use incremental SFM with manual initialization
    camera_extrinsics = incremental_sfm_calibration_with_manual_init(
        camera_data,
        camera_intrinsics,
        verbose=verbose
    )

    return camera_data, camn_to_id, camera_intrinsics, camera_extrinsics

def data_preprocess_with_choices(data2d_path: str, cam_info_path: str, yaml_dir: str,
                                              min_shared_frames: int = 50,
                                              verbose: bool = True,
                                              manual_init_extrinsic: bool = False,
                                              min_cameras: int = 2) -> Tuple[
    Dict[str, CameraData], Dict, Dict[str, CameraIntrinsic], Dict[str, CameraExtrinsic]]:
    if manual_init_extrinsic:
        print("manual init extrinsic")
        camera_data, camn_to_id, camera_intrinsics, camera_extrinsics = data_preprocess_with_manual_extrinsics(
            data2d_path,
            cam_info_path,
            yaml_dir,
            min_cameras=min_cameras,
            verbose=verbose
        )
        return camera_data, camn_to_id, camera_intrinsics, camera_extrinsics
    elif not manual_init_extrinsic:
        print("calc essential extrinsic")
        camera_data, camn_to_id, camera_intrinsics, camera_extrinsics = data_preprocess_with_incremental_sfm_sync(
            data2d_path,
            cam_info_path,
            yaml_dir,
            min_shared_frames=min_shared_frames,
            min_cameras=min_cameras,
            verbose=verbose
        )
        return camera_data, camn_to_id, camera_intrinsics, camera_extrinsics




if __name__ == "__main__":
    # Example usage
    # base_dir = "../data_file"
    # data2d_path = os.path.join(base_dir, "20241017_164418/data2d_distorted.csv")
    # cam_info_path = os.path.join(base_dir, "20241017_164418/cam_info.csv")

    base_dir = "../data_file_me"
    data2d_path = os.path.join(base_dir, "20250211_032959/data2d_distorted.csv")
    cam_info_path = os.path.join(base_dir, "20250211_032959/cam_info.csv")

    yaml_dir = os.path.join(base_dir, "intrinsic_calibrations")

    # Use manual extrinsics calibration method instead of automatic
    camera_data, camn_to_id, camera_intrinsics, camera_extrinsics = data_preprocess_with_choices(
        data2d_path,
        cam_info_path,
        yaml_dir,
        min_shared_frames=50,
        min_cameras=2,
        verbose=True
    )

    # Output camera extrinsics
    print("\n=== Final Camera Extrinsics ===")
    for cam_id, extrinsic in camera_extrinsics.items():
        print(f"Camera: {cam_id}")
        print("Rotation matrix:")
        print(extrinsic.rotation)
        print("Translation vector:")
        print(extrinsic.translation)
        print("-" * 50)