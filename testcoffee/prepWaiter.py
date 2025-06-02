import stretch_body.robot as rb

if __name__ == '__main__':
    robot = rb.Robot()
    robot.startup()

    # robot.end_of_arm.get_joint('wrist_pitch').move_to(-2)
    robot.end_of_arm.get_joint('wrist_yaw').move_to(2)
    robot.arm.move_to(0)
    robot.push_command()
    robot.wait_command()
    robot.lift.move_to(1.2)
    robot.push_command()
    robot.wait_command()
    robot.lift.move_to(1.00)
    robot.push_command()
    robot.wait_command()
    robot.end_of_arm.get_joint('wrist_yaw').move_to(0)
    robot.end_of_arm.get_joint('wrist_pitch').move_to(0.02)
    robot.end_of_arm.get_joint('wrist_roll').move_to(0.08)
    robot.push_command()
    robot.wait_command()

    # jn = ['wrist_extension', 'joint_lift', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_head_pan', 'joint_head_tilt', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_finger_left', 'joint_gripper_finger_right']
    # jp = [0.08143853896850516, 1.0000066485575705, 0.02035963474212629, 0.02035963474212629, 0.02035963474212629, 0.02035963474212629, -0.05555673457665447, -0.08170828510003944, 0.13997574689456474, 0.009203884727313847, 0.07976700097005335, 0.0034732062973397947, 0.0034732062973397947]

    robot.lift.move_to(1.00)
    robot.arm.move_to(0.02035963474212629 + 0.02035963474212629 + 0.02035963474212629 + 0.02035963474212629)
    robot.push_command()
    robot.wait_command()
