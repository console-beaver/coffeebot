letter_waypoints = {
        'a': ((0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)),
        'b': ((0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)),
        'c': ((1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0))
}

# robot joint movement
# 1. base position: robot.base.translate_by(x_m=...)
# 2. base-rotation: robot.base.rotate_by(...)
# 3. arm lift: robot.lift.move_to(...) or robot.lift.move_by(x_m=...)
# 4. arm extend: robot.arm.move_to(...) or robot.arm.move_by(x_m=...)
# 5. wrist yaw: robot.end_of_arm.move_to('wrist_yaw', ...) or robot.end_of_arm.move_by('wrist_yaw', ...)
# 6. wrist pitch: robot.end_of_arm.move_to('wrist_pitch' ...) or robot.end_of_arm.move_by('wrist_pitch', ...)
# 7. wrist roll: robot.end_of_arm.move_to('wrist_roll', ...) or robot.end_of_arm.move_by('wrist_roll', ...)

# EE control: robot.end_of_arm.move_to('stretch_gripper', ...)

# head camera movement:
# 1. head pan: robot.head.move_to('head_pan', ...) or robot.head.move_by('head_pan', ...)
# 2. head tilt: robot.head.move_to('head_tilt', ...) or robot.head.move_by('head_tilt', ...)

import stretch_body.robot

class LetterControl():
    def __init__(self):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        # q_arm is the 6 joint positions of the arm (excludes 2-dof from base)
        self.q_arm = [self.robot.lift.status['pos'],
                      self.robot.arm.status['pos'],
                      self.robot.end_of_arm.get_joint('wrist_yaw').status,
                      self.robot.end_of_arm.get_joint('wrist_pitch').status,
                      self.robot.end_of_arm.get_joint('wrist_roll').status,
                      self.robot.end_of_arm.get_joint('stretch_gripper').status]

    def __del__(self):
        self.robot.stop()

    # takes 6-vector (5 dof for arm, 1 dof for gripper movement) excluding base
    def EE_q_control(robot, q, wait=True):  # takes 6-vector of desired joint positions
        assert len(q) == 6

        self.robot.lift.move_to(q[0])
        self.robot.arm.move_to(q[1])
        self.robot.end_of_arm.move_to('wrist_yaw', q[2])
        self.robot.end_of_arm.move_to('wrist_pitch', q[3])
        self.robot.end_of_arm.move_to('wrist_roll', q[4])
        self.robot.end_of_arm.move_to('stretch_gripper', q[5])

        self.robot.push_command()
        if wait: self.robot.wait_command()

if __name__ == '__main__':
    robot = LetterControl()


