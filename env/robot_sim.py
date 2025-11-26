import numpy as np
from dm_control.utils import transformations
from dm_control.utils import inverse_kinematics as ik


class UR5e():
    NAME = 'ur5e'
    JOINT_NAMES = ['ur5e/shoulder_pan_joint',
                   'ur5e/shoulder_lift_joint',
                   'ur5e/elbow_joint',
                   'ur5e/wrist_1_joint',
                   'ur5e/wrist_2_joint',
                   'ur5e/wrist_3_joint']
    ACTUATOR_NAMES = ['ur5e/shoulder_pan',
                      'ur5e/shoulder_lift',
                      'ur5e/elbow',
                      'ur5e/wrist_1',
                      'ur5e/wrist_2',
                      'ur5e/wrist_3']
    SITE_NAME = f'{NAME}/attachment_site'

    HOME_JOINT_POSITION = [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    INITIAL_JOINT_POSITION = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

    def __init__(self):
        pass

    def get_joint_position(self, physics):
        return physics.named.data.qpos[self.JOINT_NAMES]

    def get_end_effector_pose(self, physics):
        pose = np.eye(4)
        pose[0:3, 3] = physics.named.data.site_xpos[self.SITE_NAME]
        pose[0:3, 0:3] = physics.named.data.site_xmat[self.SITE_NAME].reshape(3, 3)
        return pose

    def set_control(self, physics, joint_position):
        physics.named.data.ctrl[self.ACTUATOR_NAMES] = joint_position

    def set(self, physics, pose):
        pose = np.array(pose)
        if pose.shape == (4, 4):
            ik_result = self.inverse_kinematics(physics, pose)
            joint_position = ik_result.qpos[0:6]
        elif pose.shape == (6,):
            joint_position = pose
        else:
            raise Exception("Invalid action shape")
        
        physics.named.data.qpos[self.JOINT_NAMES] = joint_position
        self.set_control(physics, joint_position)
        physics.step_with_rendering(iteration=1, nstep=5)

    def movej(self, physics, target_pose):
        target_pose = np.array(target_pose)
        if target_pose.shape == (4, 4):
            ik_result = self.inverse_kinematics(physics, target_pose)
            target_joint_position = ik_result.qpos[0:6]
        elif target_pose.shape == (6,):
            target_joint_position = target_pose
        else:
            raise Exception("Invalid action shape")

        self.set_control(physics, target_joint_position)

        for _ in range(1000):
            current_joint_position = self.get_joint_position(physics)
            if np.linalg.norm(current_joint_position - target_joint_position) < 1e-3:
                break
            physics.step_with_rendering(iteration=1, nstep=5)
            next_joint_position = self.get_joint_position(physics)
            if np.linalg.norm(current_joint_position - next_joint_position) < 1e-4:
                break

    def movel(self, physics, target_pose, check_collision=False, geoms_1=None, geoms_2=None):
        current_pose = self.get_end_effector_pose(physics)
        trajectory = self.cartesian_trajectory(Xstart=current_pose,
                                               Xend=target_pose,
                                               Tf=2,
                                               N=2,
                                               method=5)
        
        for pose in trajectory:
            ik_result = self.inverse_kinematics(physics, pose)
            target_joint_position = ik_result.qpos[0:6]
            self.set_control(physics, target_joint_position)
            for _ in range(1000):
                current_pose = self.get_end_effector_pose(physics)
                if np.linalg.norm(current_pose[0:3, 3] - pose[0:3, 3]) < 1e-3:
                    break
                physics.step_with_rendering(iteration=1, nstep=5)
                if check_collision:
                    collision = physics.check_contact(geoms_1, geoms_2)
                    if collision:
                        return collision
                next_pose = self.get_end_effector_pose(physics)
                if np.linalg.norm(current_pose[0:3, 3] - next_pose[0:3, 3]) < 1e-4:
                    break

    def inverse_kinematics(self, physics, target_pose):
        """
        Args:
            physics: A `mujoco.Physics` instance.
            site_name: A string specifying the name of the target site.
            target_pos: A (3,) numpy array specifying the desired Cartesian position of
              the site, or None if the position should be unconstrained (default).
              One or both of `target_pos` or `target_quat` must be specified.
            target_quat: A (4,) numpy array specifying the desired orientation of the
              site as a quaternion, or None if the orientation should be unconstrained
              (default). One or both of `target_pos` or `target_quat` must be specified.
            joint_names: (optional) A list, tuple or numpy array specifying the names of
              one or more joints that can be manipulated in order to achieve the target
              site pose. If None (default), all joints may be manipulated.
            tol: (optional) Precision goal for `qpos` (the maximum value of `err_norm`
              in the stopping criterion).
            rot_weight: (optional) Determines the weight given to rotational error
              relative to translational error.
            regularization_threshold: (optional) L2 regularization will be used when
              inverting the Jacobian whilst `err_norm` is greater than this value.
            regularization_strength: (optional) Coefficient of the quadratic penalty
              on joint movements.
            max_update_norm: (optional) The maximum L2 norm of the update applied to
              the joint positions on each iteration. The update vector will be scaled
              such that its magnitude never exceeds this value.
            progress_thresh: (optional) If `err_norm` divided by the magnitude of the
              joint position update is greater than this value then the optimization
              will terminate prematurely. This is a useful heuristic to avoid getting
              stuck in local minima.
            max_steps: (optional) The maximum number of iterations to perform.
            inplace: (optional) If True, `physics.data` will be modified in place.
              Default value is False, i.e. a copy of `physics.data` will be made.

          Returns:
            An `IKResult` namedtuple with the following fields:
              qpos: An (nq,) numpy array of joint positions.
              err_norm: A float, the weighted sum of L2 norms for the residual
                translational and rotational errors.
              steps: An int, the number of iterations that were performed.
              success: Boolean, True if we converged on a solution within `max_steps`,
                False otherwise.
        """
        target_position = target_pose[0:3, 3]
        target_orientation_quat = transformations.mat_to_quat(target_pose)

        ik_result = ik.qpos_from_site_pose(physics=physics,
                                           site_name=self.SITE_NAME,
                                           target_pos=target_position,
                                           target_quat=target_orientation_quat,
                                           joint_names=self.JOINT_NAMES,
                                           regularization_strength=3e-3)
        return ik_result

    def cartesian_trajectory(self, Xstart, Xend, Tf, N, method):
        """Computes a trajectory as a list of N SE(3) matrices corresponding to
        the origin of the end-effector frame following a straight line

        :param Xstart: The initial end-effector configuration
        :param Xend: The final end-effector configuration
        :param Tf: Total time of the motion in seconds from rest to rest
        :param N: The number of points N > 1 (Start and stop) in the discrete
                representation of the trajectory
        :param method: The time-scaling method, where 3 indicates cubic (third-
                    order polynomial) time scaling and 5 indicates quintic
                    (fifth-order polynomial) time scaling
        :return: The discretized trajectory as a list of N matrices in SE(3)
                separated in time by Tf/(N-1). The first in the list is Xstart
                and the Nth is Xend
        """
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = [[None]] * N
        Rstart, pstart = self.TransToRp(Xstart)
        Rend, pend = self.TransToRp(Xend)
        for i in range(N):
            if method == 3:
                s = self.CubicTimeScaling(Tf, timegap * i)
            else:
                s = self.QuinticTimeScaling(Tf, timegap * i)
            traj[i] \
            = np.r_[np.c_[np.dot(Rstart, \
            self.MatrixExp3(self.MatrixLog3(np.dot(np.array(Rstart).T,Rend)) * s)), \
                s * np.array(pend) + (1 - s) * np.array(pstart)], \
                [[0, 0, 0, 1]]]
        return traj

    def TransToRp(self, T):
        T = np.array(T)
        return T[0: 3, 0: 3], T[0: 3, 3]

    def CubicTimeScaling(self, Tf, t):
        return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

    def QuinticTimeScaling(self, Tf, t):
        return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
            + 6 * (1.0 * t / Tf) ** 5

    def MatrixExp3(self, so3mat):
        omgtheta = self.so3ToVec(so3mat)
        if self.NearZero(np.linalg.norm(omgtheta)):
            return np.eye(3)
        else:
            theta = self.AxisAng3(omgtheta)[1]
            omgmat = so3mat / theta
            return np.eye(3) + np.sin(theta) * omgmat \
                + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

    def MatrixLog3(self, R):
        acosinput = (np.trace(R) - 1) / 2.0
        if acosinput >= 1:
            return np.zeros((3, 3))
        elif acosinput <= -1:
            if not self.NearZero(1 + R[2][2]):
                omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                    * np.array([R[0][2], R[1][2], 1 + R[2][2]])
            elif not self.NearZero(1 + R[1][1]):
                omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                    * np.array([R[0][1], 1 + R[1][1], R[2][1]])
            else:
                omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                    * np.array([1 + R[0][0], R[1][0], R[2][0]])
            return self.VecToso3(np.pi * omg)
        else:
            theta = np.arccos(acosinput)
            return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)
    
    def NearZero(self, z):
        return abs(z) < 1e-6
    
    def Normalize(self, V):
        return V / np.linalg.norm(V)

    def VecToso3(self, omg):
        return np.array([[0,      -omg[2],  omg[1]],
                        [omg[2],       0, -omg[0]],
                        [-omg[1], omg[0],       0]])

    def so3ToVec(self, so3mat):
        return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

    def AxisAng3(self, expc3):
        return (self.Normalize(expc3), np.linalg.norm(expc3))