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


    


