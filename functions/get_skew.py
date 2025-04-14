import numpy as np
from scipy.linalg import rq
from scipy.spatial.transform import Rotation as R

def rq_decomposition(matrix):
    # Perform RQ decomposition using scipy
    if matrix.shape[1] == 4:
        matrix = matrix[:,:3]
    intrin, q = rq(matrix)
    # Ensure the diagonal elements of the upper triangular matrix are non-negative
    for i in range(3):
        if intrin[i, i] < 0:
            intrin[:, i] = -intrin[:, i]
            q[i, :] = -q[i, :]

    # Convert Q to a unit quaternion
    rquat = R.from_matrix(q).as_quat()

    return rquat, intrin

def process_get_skew(matrix):
    rquat, intrin = rq_decomposition(matrix)
    k = intrin * (1 / intrin[2, 2])
    expected_alpha_c = k[0, 1] / k[0, 0]
    return expected_alpha_c