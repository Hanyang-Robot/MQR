import cv2
import numpy as np
from dm_control.utils import transformations


class CameraMatrices:
    def __init__(self, image_matrix, focal_matrix, rotation_matrix, translation_matrix):
        self._image_matrix = image_matrix
        self._focal_matrix = focal_matrix
        self._rotation_matrix = rotation_matrix
        self._translation_matrix = translation_matrix

    @property
    def image(self):
        return self._image_matrix
    
    @property
    def focal(self):
        return self._focal_matrix
    
    @property
    def rotation(self):
        return self._rotation_matrix
    
    @property
    def translation(self):
        return self._translation_matrix


class Camera():
    OPTICAL_AXIS_ROTATION = transformations.euler_to_rmat([np.pi, 0, 0], ordering='XYZ', full=True)

    def __init__(self, physics, config, camera_id):
        self.image_width = config['camera']['resolution']['width']
        self.image_height = config['camera']['resolution']['height']
        self.camera_depth_noise_mean = config['camera']['depth_noise']['mean']
        self.camera_depth_noise_std = config['camera']['depth_noise']['std']
        self.camera_id = camera_id

        # Image matrix (3x3)
        self.image_matrix = np.eye(3)
        self.image_matrix[0, 2] = (self.image_width - 1)/2.0
        self.image_matrix[1, 2] = (self.image_height - 1)/2.0
        # Focal transformation matrix (3x4)
        fov = physics.named.model.cam_fovy[self.camera_id]
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2))*self.image_height/2.0
        self.focal_matrix = np.diag([focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Rotation matrix (4x4)
        rot = physics.named.data.cam_xmat[self.camera_id].reshape(3, 3)
        self.rotation_matrix = np.eye(4)
        self.rotation_matrix[0:3, 0:3] = rot
        # Translation matrix (4x4)
        pos = physics.named.data.cam_xpos[self.camera_id]
        self.translation_matrix = np.eye(4)
        self.translation_matrix[0:3, 3] = pos

    def matrices(self):
        return CameraMatrices(self.image_matrix, self.focal_matrix, self.rotation_matrix, self.translation_matrix)

    # def adjusted_matrices(self, crop_x, crop_y, resize_x, resize_y):
    #     # Crop
    #     adjusted_image_matrix = self.image_matrix.copy()
    #     adjusted_image_matrix[0, 2] -= crop_x
    #     adjusted_image_matrix[1, 2] -= crop_y
    #     # Resize
    #     adjusted_focal_matrix = self.focal_matrix.copy()
    #     adjusted_focal_matrix[0, 0] *= resize_x
    #     adjusted_focal_matrix[1, 1] *= resize_y
    #     adjusted_image_matrix[0, 2] *= resize_x
    #     adjusted_image_matrix[1, 2] *= resize_y

    #     return CameraMatrices(adjusted_image_matrix, adjusted_focal_matrix, self.rotation_matrix, self.translation_matrix)

    def get_color_image(self, physics):
        color_image = physics.render(self.image_height, self.image_width, self.camera_id) # BGR
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # RGB
        return color_image

    def get_depth_image(self, physics, noise=True):
        depth_image = physics.render(self.image_height, self.image_width, self.camera_id, depth=True)
        if noise:
            noise = np.random.normal(self.camera_depth_noise_mean,
                                     self.camera_depth_noise_std,
                                     size=(self.image_height, self.image_width))
            depth_image = depth_image + noise
        return depth_image

    def get_point_cloud(self, physics, noise=True):
        color_image = self.get_color_image(physics)
        depth_image = self.get_depth_image(physics, noise)

        intrinsic_matrix = self.image_matrix @ self.focal_matrix

        mask = np.where(depth_image > 0)
        
        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - intrinsic_matrix[0,2])/intrinsic_matrix[0,0]
        normalized_y = (y.astype(np.float32) - intrinsic_matrix[1,2])/intrinsic_matrix[1,1]

        world_x = normalized_x * depth_image[y, x]
        world_y = normalized_y * depth_image[y, x]
        world_z = depth_image[y, x]
        
        point_cloud = np.vstack((world_x, world_y, world_z)).T

        colors = color_image[y, x, :]

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # pcd.colors = o3d.utility.Vector3dVector(colors/255.0)

        # coordinate_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        # lookat = np.array([0.0, 1.0, 0.0])
        # up = np.array([0.0, -1.0, -1.0])
        # front = np.array([0.0, 1.0, -1.0])
        # zoom = 0.1
        # o3d.visualization.draw_geometries([pcd, coordinate_frame], lookat=lookat, up=up, front=front, zoom=zoom)

        return point_cloud, colors
    
    def get_heightmap(self, physics, workspace, heightmap_resolution, noise=True):
        # Compute heightmap size
        heightmap_size = np.round(((workspace[1][1] - workspace[1][0])/heightmap_resolution, (workspace[0][1] - workspace[0][0])/heightmap_resolution)).astype(int)

        # Get 3D point cloud from RGB-D images
        point_cloud, colors = self.get_point_cloud(physics, noise)

        # Transform 3D point cloud from camera coordinates {c} to robot coordinates {w}
        camera_pose = self.translation_matrix @ self.rotation_matrix @ self.OPTICAL_AXIS_ROTATION
        point_cloud = np.transpose(np.dot(camera_pose[0:3, 0:3],np.transpose(point_cloud)) + np.tile(camera_pose[0:3, 3:], (1, point_cloud.shape[0])))
        
        # Sort surface points by z value
        sorted_z_index = np.argsort(point_cloud[:, 2])
        point_cloud = point_cloud[sorted_z_index]
        colors = colors[sorted_z_index]

        # Filter out surface points outside heightmap
        heightmap_valid_index = np.logical_and(np.logical_and(np.logical_and(point_cloud[:, 0] >= workspace[0][0], point_cloud[:, 0] < workspace[0][1]), point_cloud[:, 1] >= workspace[1][0]), point_cloud[: ,1] < workspace[1][1])
        point_cloud = point_cloud[heightmap_valid_index]
        colors = colors[heightmap_valid_index]

        # Create orthographic top-down-view RGB-D heightmaps
        heightmap_pixel_x = np.floor((point_cloud[:, 0] - workspace[0][0])/heightmap_resolution).astype(int)
        heightmap_pixel_y = np.floor((point_cloud[:, 1] - workspace[1][0])/heightmap_resolution).astype(int)
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_r[heightmap_pixel_x, heightmap_pixel_y] = colors[:, [0]]
        color_heightmap_g[heightmap_pixel_x, heightmap_pixel_y] = colors[:, [1]]
        color_heightmap_b[heightmap_pixel_x, heightmap_pixel_y] = colors[:, [2]]
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)

        depth_heightmap = np.zeros(heightmap_size)
        depth_heightmap[heightmap_pixel_x,heightmap_pixel_y] = point_cloud[:,2]
        depth_heightmap[depth_heightmap < 0] = 0

        return color_heightmap, depth_heightmap