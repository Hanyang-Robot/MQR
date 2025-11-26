import numpy as np
from dm_control.utils import transformations

def sample_pose(x_range, y_range, z_range, roll_range=[0, 2*np.pi], pitch_range=[0, 2*np.pi], yaw_range=[0, 2*np.pi]):
    x = np.random.uniform(low=x_range[0], high=x_range[1])
    y = np.random.uniform(low=y_range[0], high=y_range[1])
    z = np.random.uniform(low=z_range[0], high=z_range[1])
    roll = np.random.uniform(low=roll_range[0], high=roll_range[1])
    pitch = np.random.uniform(low=pitch_range[0], high=pitch_range[1])
    yaw = np.random.uniform(low=yaw_range[0], high=yaw_range[1])
    quat = list(transformations.euler_to_quat([roll, pitch, yaw]))

    pose = [x, y, z] + quat

    return pose

def pixel_to_xyz(pixel, depth, camera_matrices):
    # Intrinsic matrix (3X3)
    intrinsic_matrix = camera_matrices.image @ camera_matrices.focal
    intrinsic_matrix = intrinsic_matrix[0:3, 0:3]
    # Transform image coordinates to normalized image coordinates
    normalized_image_coordinates = np.linalg.inv(intrinsic_matrix) @ np.append(pixel, 1)
    # Transform normalized image coordinates to camera coordinates {c}
    scale = camera_matrices.translation[2, 3]
    camera_coordinates = scale * normalized_image_coordinates
    camera_coordinates[2] = depth
    camera_coordinates = np.append(camera_coordinates, 1)
    # Transform camera coordinates {c} to world coordinates {w}
    OPTICAL_AXIS_ROTATION = transformations.euler_to_rmat([np.pi, 0, 0], ordering='XYZ', full=True)
    transformation = camera_matrices.translation @ camera_matrices.rotation @ OPTICAL_AXIS_ROTATION
    world_coordinates = transformation @ camera_coordinates
    # xyz = [round(i, 3) for i in world_coordinates[:-1]]
    xyz = world_coordinates[:-1]

    return xyz

def xyz_to_pixel(xyz, camera_matrices):
    image = camera_matrices.image.copy()
    focal = camera_matrices.focal.copy()
    rotation = camera_matrices.rotation.copy()
    translation = camera_matrices.translation.copy()
    focal[0, 0] *= -1
    rotation[0:3, 0:3] = rotation[0:3, 0:3].T
    translation[0:3, 3] *= -1
    camera_matrix = image @ focal @ rotation @ translation
    xs, ys, s = camera_matrix.dot(np.append(xyz, 1.0))
    pixel = [round(i) for i in [xs/s, ys/s]]

    return pixel
