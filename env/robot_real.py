import time
import copy

import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from scipy.spatial.transform import Rotation


class UR5e():
    def __init__(self, config):
        self.rtde_c = RTDEControl(config['robot']['ip'])
        self.rtde_r = RTDEReceive(config['robot']['ip'])
        
        self.home_pose = config['robot']['home_pose']
        self.placing_pose = config['robot']['placing_pose']
        
    @property
    def tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def movej(self, q, a=0.5, v=1.1):
        """Move to position (linear in joint-space).
        Args:
            q: joint positions [q1, q2, q3, q4, q5, q6].
            a: joint acceleration of leading axis [rad/s^2].
            v: joint speed of leading axis [rad/s].
        """
        self.rtde_c.moveJ(q, acceleration=a, speed=v)

    def movel(self, pose, a=0.5, v=0.25, check_collision=False):
        """Move to position (linear in tool-space).
        Args:
            pose = target pose [x, y, z, rx, ry, rz].
            a: joint acceleration of leading axis [rad/s^2].
            v: joint speed of leading axis [rad/s].
            check_collision: check collision
        """
        rotvec_pose = copy.deepcopy(pose)
        rpy = rotvec_pose[3:]
        rotvec = self.rpy2rotvec(rpy)
        rotvec_pose[3:] = rotvec

        if check_collision:
            direction = np.array(pose) - np.array(self.tcp_pose)
            distance = np.linalg.norm(pose[:3] - np.array(self.tcp_pose)[:3])
            self.rtde_c.moveL(rotvec_pose, acceleration=a, speed=v, asynchronous=True)
            while distance >= 0.001:
                distance = np.linalg.norm(pose[:3] - np.array(self.tcp_pose)[:3])
                if self.rtde_c.toolContact(direction) != 0:
                    return True
                time.sleep(0.01)
            return False
        else:
            self.rtde_c.moveL(rotvec_pose, acceleration=a, speed=v)
    
    def rpy2rotvec(self, rpy):
        r = Rotation.from_euler("xyz", rpy, degrees=False)
        rotvec = r.as_rotvec()
        return rotvec

# if __name__ == "__main__":
#     config = {
#         'robot': {
#             'ip': '192.168.1.28',
#             'home_pose': [0.100, -0.200, 0.600, 3.14, 0.0, 3.14],
#             'placing_pose': [0.450, -0.450, 0.400, 3.14, 0.0, 3.14]
#         }
#     }
    
#     robot = UR5e(config)
    
#     robot.movel(robot.home_pose)
