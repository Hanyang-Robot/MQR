class Robotiq2F85():
    NAME = 'robotiq_2f85'
    JOINT_NAMES = ['robotiq_2f85/right_driver_joint',
                   'robotiq_2f85/left_driver_joint']
    ACTUATOR_NAME = 'robotiq_2f85/fingers_actuator'
    PAD_NAMES = ['robotiq_2f85/right_pad1',
                 'robotiq_2f85/right_pad2',
                 'robotiq_2f85/left_pad1',
                 'robotiq_2f85/left_pad2']

    STROKE = 0.085
    OPEN_LENGTH = 0.149
    CLOSE_LENGTH = 0.163
    FINGER_WIDTH = 0.0272 

    OPEN_POSITION = 0
    CLOSE_POSITION = 255

    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.joint_names = [f'{self.robot_name}/{joint_name}' for joint_name in self.JOINT_NAMES]
        self.pad_names = [f'{self.robot_name}/{pad_name}' for pad_name in self.PAD_NAMES]

    def set_control(self, physics, control):
        physics.named.data.ctrl[f'{self.robot_name}/{self.ACTUATOR_NAME}'] = control

    def set(self, physics, position):
        physics.named.data.qpos[self.joint_names] = [position]*2
        self.set_control(physics, position)
        physics.step_with_rendering(iteration=1, nstep=5)

    def open(self, physics):
        self.set_control(physics, self.OPEN_POSITION)
        physics.step_with_rendering(iteration=50, nstep=5)

    def close(self, physics):
        self.set_control(physics, self.CLOSE_POSITION)
        physics.step_with_rendering(iteration=50, nstep=5)