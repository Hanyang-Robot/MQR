import numpy as np
import pykinect_azure as pykinect
from pykinect_azure import k4a_float2_t, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH
from pykinect_azure.k4a import _k4a


class AzureKinect():
    def __init__(self, config):
        self.color_resolution = config['camera']['color_resolution']
        self.depth_mode = config['camera']['depth_mode']
        
        self.initialize()
        
        self.intrinsic_matrix = np.array(self.device.calibration.get_matrix(_k4a.K4A_CALIBRATION_TYPE_COLOR))
        self.extrinsic_matrix = np.array(config['camera']['pose'])
        self.rotation_matrix = np.vstack((np.hstack((self.extrinsic_matrix[:3, :3], np.zeros((3, 1)))), np.array([0, 0, 0, 1])))

    def initialize(self):
        pykinect.initialize_libraries()
        config = pykinect.default_configuration

        if self.color_resolution == "OFF":
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        elif self.color_resolution == 720:
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        elif self.color_resolution == 1080:
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        elif self.color_resolution == 1440:
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_1440P
        elif self.color_resolution == 1536:
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_1536P
        elif self.color_resolution == 2160:
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
        elif self.color_resolution == 3072:
            color_resolution = pykinect.K4A_COLOR_RESOLUTION_3072P
        else:
            raise Exception("Invalidcolor resolution")
        config.color_resolution = color_resolution

        if self.depth_mode == "OFF":
            depth_mode = pykinect.K4A_DEPTH_MODE_OFF
        elif self.depth_mode == "NFOV_2X2BINNED":
            depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        elif self.depth_mode == "NFOV_UNBINNED":
            depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
        elif self.depth_mode == "WFOV_2X2BINNED":
            depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        elif self.depth_mode == "WFOV_UNBINNED":
            depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
        elif self.depth_mode == "PASSIVE_IR":
            depth_mode = pykinect.K4A_DEPTH_MODE_PASSIVE_IR
        else:
            raise Exception("Invalid depth mode")
        config.depth_mode = depth_mode

        self.device = pykinect.start_device(config=config)

    def get_image(self):
        while True:
            capture = self.device.update()
            ret_color, color_image = capture.get_color_image()
            # ret_depth, depth_image = capture.get_depth_image()
            ret_depth, depth_image = capture.get_transformed_depth_image()
            if not ret_color or not ret_depth:
                continue
            break
        return color_image, depth_image/1000

    def get_point_cloud(self):
        color_image, depth_image = self.get_image()
        
        mask = np.where(depth_image > 0)
        x = mask[1]
        y = mask[0]
        
        normalized_x = (x.astype(np.float32) - self.intrinsic_matrix[0,2])/self.intrinsic_matrix[0,0]
        normalized_y = (y.astype(np.float32) - self.intrinsic_matrix[1,2])/self.intrinsic_matrix[1,1]

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
    
    def get_heightmap(self, workspace_limits, heightmap_resolution):
        # Compute heightmap size
        heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)
        
        # Get 3D point cloud from RGB-D images
        surface_pts, color_pts = self.get_point_cloud()

        # Transform 3D point cloud from camera coordinates {c} to robot coordinates {w}
        cam_pose = self.extrinsic_matrix
        surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))
        
        # Sort surface points by z value
        sort_z_ind = np.argsort(surface_pts[:,2])
        surface_pts = surface_pts[sort_z_ind]
        color_pts = color_pts[sort_z_ind]

        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1])
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]
        
        # Create orthographic top-down-view RGB-D heightmaps
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)
        heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
        color_heightmap_r[heightmap_pix_x,heightmap_pix_y] = color_pts[:,[0]]
        color_heightmap_g[heightmap_pix_x,heightmap_pix_y] = color_pts[:,[1]]
        color_heightmap_b[heightmap_pix_x,heightmap_pix_y] = color_pts[:,[2]]
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_x,heightmap_pix_y] = surface_pts[:,2]
        depth_heightmap[depth_heightmap < 0] = 0

        return color_heightmap, depth_heightmap
    
if __name__ == "__main__":
    config = {
    # [-0.316, 0.130], [-0.772, -0.326]]
        'heightmap_resolution': 0.002,
        'robot': {
            # 'workspace': [[-0.312, 0.136], [-0.679, -0.231]]
            'workspace': [[-0.316, 0.132], [-0.772, -0.324]]
        },
        'camera': {
            'color_resolution': 720,
            'depth_mode': 'NFOV_2X2BINNED',
            'pose': [[0.0043, 0.9998, 0.0193, -0.09277],
                     [0.9995, -0.0041, -0.0088, -0.46301],
                     [-0.0087, 0.0193, -0.9997, 0.87484],
                     [0, 0, 0, 1]]
        }
    }

    camera = AzureKinect(config)
    
    # color_image, depth_image = camera.get_image()
    # import cv2
    # cv2.imshow('color_image', color_image)
    # cv2.imshow('depth_image', depth_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # point_cloud, color_image = camera.get_point_cloud()

    color_heightmap, depth_heightmap = camera.get_heightmap(config['robot']['workspace'], config['heightmap_resolution'])
    depth_heightmap_norm = (depth_heightmap-depth_heightmap.min())/(depth_heightmap.max()-depth_heightmap.min())
    print(f"height max: {depth_heightmap.max()}, min: {depth_heightmap.min()}")
    
    import cv2
    cv2.imshow('color_heightmap', color_heightmap)
    cv2.imshow('depth_heightmap', depth_heightmap_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()