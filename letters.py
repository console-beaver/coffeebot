letter_waypoints = {
        'a': [(0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)],
        'b': [(0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)],
        'c': [(1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0)]
}

def move_joints(q_vec):  # takes 7-vector of desired joint positions
    assert len(q_vec) == 7

    robot.base.translate_by(x_m=q[0])  # move base forward _ meters
    robot.base.rotate_by(q[1])  # rotate base by 


if __name__ == '__main__':
    import stretch_body.robot

    robot = stretch_body.robot.Robot()
    robot.startup()
    
    # Open gripper (range is typically 0.0 to 1.0, but you can fine-tune)
    robot.end_of_arm.move_to("stretch_gripper", 0.0)
    robot.push_command()
    robot.wait_command()
    
    # Close gripper
    robot.end_of_arm.move_to("stretch_gripper", 2.0)
    robot.push_command()
    robot.wait_command()
    
    robot.stop()
