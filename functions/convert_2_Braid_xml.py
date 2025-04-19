import numpy as np
import xml.dom.minidom as md
import xml.etree.ElementTree as ET
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional


# Define data structures to match your data format
@dataclass
class CameraExtrinsic:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class CameraIntrinsic:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]


@dataclass
class CameraMatrices:
    P: np.ndarray
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    skew_value: float


def create_camera_xml(camera_matrices: Dict[str, CameraMatrices],
                      intrinsic_params: Dict[str, CameraIntrinsic],
                      extrinsic_params: Dict[str, CameraExtrinsic],
                      resolution: Optional[Dict[str, Tuple[int, int]]] = None,
                      min_eccentricity: float = 1.4) -> str:
    """
    Create XML file based on camera parameters

    Parameters:
    camera_matrices: Dictionary containing camera P matrices
    intrinsic_params: Dictionary containing camera intrinsic parameters
    extrinsic_params: Dictionary containing camera extrinsic parameters
    resolution: Dictionary of custom resolutions, if None use image_size from intrinsic_params
    min_eccentricity: Minimum eccentricity, default is 1.4

    Returns:
    Generated XML string
    """
    # Create root element
    root = ET.Element("multi_camera_reconstructor")

    # Loop through all cameras
    for cam_id in camera_matrices.keys():
        # Create single camera calibration element
        single_cam = ET.SubElement(root, "single_camera_calibration")

        # Add camera ID
        cam_id_elem = ET.SubElement(single_cam, "cam_id")
        cam_id_elem.text = cam_id

        # Add calibration matrix (P matrix)
        p_matrix = camera_matrices[cam_id].P

        # Convert P matrix first three rows to string format with scientific notation
        p_str = ""
        for i in range(3):  # Only take first 3 rows
            row_str = " ".join([f"{p_matrix[i, j]:.6e}" for j in range(4)])
            if i < 2:
                p_str += row_str + "; "
            else:
                p_str += row_str

        calib_matrix = ET.SubElement(single_cam, "calibration_matrix")
        calib_matrix.text = p_str

        # Add resolution
        if resolution and cam_id in resolution:
            res = resolution[cam_id]
        else:
            res = intrinsic_params[cam_id].image_size

        res_elem = ET.SubElement(single_cam, "resolution")
        res_elem.text = f"{res[0]} {res[1]}"

        # Add non-linear parameters
        nonlinear = ET.SubElement(single_cam, "non_linear_parameters")

        # Get parameters from camera intrinsics
        intrinsic = intrinsic_params[cam_id]
        K = intrinsic.camera_matrix
        D = intrinsic.dist_coeffs

        # Add focal length parameters (fc1, fc2)
        fc1 = ET.SubElement(nonlinear, "fc1")
        fc1.text = f"{K[0, 0]:.6f}"

        fc2 = ET.SubElement(nonlinear, "fc2")
        fc2.text = f"{K[1, 1]:.6f}"

        # Add principal point coordinates (cc1, cc2)
        cc1 = ET.SubElement(nonlinear, "cc1")
        cc1.text = f"{K[0, 2]:.6f}"

        cc2 = ET.SubElement(nonlinear, "cc2")
        cc2.text = f"{K[1, 2]:.6f}"

        # Add radial distortion parameters (k1, k2)
        k1 = ET.SubElement(nonlinear, "k1")
        # Format distortion coefficients with appropriate precision
        k1.text = f"{D[0, 0]:.6f}".rstrip('0').rstrip('.') if '.' in f"{D[0, 0]:.6f}" else f"{D[0, 0]:.6f}"

        k2 = ET.SubElement(nonlinear, "k2")
        k2.text = f"{D[1, 0]:.6f}".rstrip('0').rstrip('.') if '.' in f"{D[1, 0]:.6f}" else f"{D[1, 0]:.6f}"

        # Add tangential distortion parameters (p1, p2)
        p1 = ET.SubElement(nonlinear, "p1")
        # Format p1 to match the example format (scientific notation for small values)
        p1_val = D[2, 0]
        if abs(p1_val) < 0.0001 and p1_val != 0:
            p1.text = f"{p1_val:.1e}"
        else:
            p1.text = f"{p1_val:.6f}".rstrip('0').rstrip('.') if '.' in f"{p1_val:.6f}" else f"{p1_val:.6f}"

        p2 = ET.SubElement(nonlinear, "p2")
        # Format p2 to match the example format
        p2_val = D[3, 0]
        if abs(p2_val) < 0.0001 and p2_val != 0:
            p2.text = f"{p2_val:.1e}"
        else:
            p2.text = f"{p2_val:.6f}".rstrip('0').rstrip('.') if '.' in f"{p2_val:.6f}" else f"{p2_val:.6f}"

        # Add alpha_c (always 0)
        alpha_c = ET.SubElement(nonlinear, "alpha_c")
        alpha_c.text = "0.0"

        # Add additional fc1p, fc2p, cc1p, cc2p (same as fc1, fc2, cc1, cc2)
        fc1p = ET.SubElement(nonlinear, "fc1p")
        fc1p.text = f"{K[0, 0]:.6f}"

        fc2p = ET.SubElement(nonlinear, "fc2p")
        fc2p.text = f"{K[1, 1]:.6f}"

        cc1p = ET.SubElement(nonlinear, "cc1p")
        cc1p.text = f"{K[0, 2]:.6f}"

        cc2p = ET.SubElement(nonlinear, "cc2p")
        cc2p.text = f"{K[1, 2]:.6f}"

    # Add minimum eccentricity
    min_ecc = ET.SubElement(root, "minimum_eccentricity")
    min_ecc.text = str(min_eccentricity)

    # Create XML string and format it
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = md.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_camera_xml(extrinsic_data: Dict[str, CameraExtrinsic],
                    intrinsic_data: Dict[str, CameraIntrinsic],
                    camera_matrices_data: Dict[str, CameraMatrices],
                    output_folder: str,
                    filename: str = "camera_calibration.xml",
                    resolution: Optional[Dict[str, Tuple[int, int]]] = None,
                    min_eccentricity: float = 1.4) -> str:
    """
    Process camera data and save as XML file

    Parameters:
    extrinsic_data: Dictionary of camera extrinsic parameters
    intrinsic_data: Dictionary of camera intrinsic parameters
    camera_matrices_data: Dictionary of camera matrices
    output_folder: Output folder path
    filename: Output filename, default is "camera_calibration.xml"
    resolution: Dictionary of custom resolutions, if None use image_size from intrinsic_data
    min_eccentricity: Minimum eccentricity, default is 1.4

    Returns:
    Full path of the saved file
    """
    # Create output folder (if it doesn't exist)
    os.makedirs(output_folder, exist_ok=True)

    # Generate XML content
    xml_content = create_camera_xml(
        camera_matrices=camera_matrices_data,
        intrinsic_params=intrinsic_data,
        extrinsic_params=extrinsic_data,
        resolution=resolution,
        min_eccentricity=min_eccentricity
    )

    # Save to file
    file_path = os.path.join(output_folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    return file_path


def convert_xml_example():
    """
    Example usage
    """
    # Set output folder
    output_folder = os.path.join(os.getcwd(), "camera_calibration_output")

    # Parse camera parameters from input data
    # Note: In actual use, you need to replace this part with your actual data

    # Example data format (matching the format you provided)
    extrinsic_data = {
        'Basler-24795855': CameraExtrinsic(
            rotation=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            translation=np.array([-0.00093726, -0.0012975, 0.04655965])
        ),
        # ... other camera extrinsics
    }

    intrinsic_data = {
        'Basler-24795855': CameraIntrinsic(
            camera_matrix=np.array([
                [4.67003056e+03, 0.00000000e+00, 8.09607482e+02],
                [0.00000000e+00, 4.65937504e+03, 5.48468705e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ]),
            dist_coeffs=np.array([
                [0.02074622], [-3.2403369], [0.00496982], [0.00782044], [0.]
            ]),
            image_size=(1200, 850)
        ),
        # ... other camera intrinsics
    }

    camera_matrices_data = {
        'Basler-24795855': CameraMatrices(
            P=np.array([
                [4.67003056e+03, 0.00000000e+00, 8.09607482e+02, 3.33180239e+01],
                [0.00000000e+00, 4.65937504e+03, 5.48468705e+02, 1.94909578e+01],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 4.65596499e-02]
            ]),
            K=np.array([
                [4.67003056e+03, 0.00000000e+00, 8.09607482e+02],
                [0.00000000e+00, 4.65937504e+03, 5.48468705e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ]),
            D=np.array([
                [0.02074622], [-3.2403369], [0.00496982], [0.00782044], [0.]
            ]),
            R=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            t=np.array([-0.00093726, -0.0012975, 0.04655965]),
            skew_value=0.0
        ),
        # ... other camera matrices
    }

    # Optional: Set custom resolutions (if not set, will use image_size from intrinsic_data)
    custom_resolution = {
        'Basler-24795855': (800, 800),
        'Basler-24795856': (800, 800),
        'Basler-24795861': (800, 800),
        'Basler-24795862': (800, 800)
    }

    # Call function to save XML
    file_path = save_camera_xml(
        extrinsic_data=extrinsic_data,
        intrinsic_data=intrinsic_data,
        camera_matrices_data=camera_matrices_data,
        output_folder=output_folder,
        filename="camera_calibration.xml",
        resolution=custom_resolution,
        min_eccentricity=1.4
    )

    print(f"Camera calibration data has been saved to: {file_path}")


if __name__ == "__main__":
    convert_xml_example()