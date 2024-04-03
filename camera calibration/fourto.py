import numpy as np

from scipy.spatial.transform import Rotation as R

def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot

def create_4x4_matrix(rotation_matrix, translation_vector):
    matrix_4x4 = np.eye(4)
    matrix_4x4[:3, :3] = rotation_matrix
    matrix_4x4[:3, 3] = translation_vector
    return matrix_4x4

