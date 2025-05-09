# USAGE: run
# `ros2 launch stretch_core stretch_driver.launch.py mode:=position`
# before this script

# letters are defined relative to bottom-left corner, (x, y, z) (where z is out of the page)
letter_waypoints = {
        'a': ((0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)),
        'b': ((0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)),
        'c': ((1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0))
}

# arm q-vector: lift, telescoping arm, wrist pitch, roll, yaw, gripper
def EE_joint_control(q, node):
    assert len(q) == 6

    node.move_to_pose({
        'joint_lift': q[0],
        'joint_arm': q[1],
        'joint_wrist_pitch': q[2],
        'join_wrist_roll': q[3],
        'joint_wrist_yaw': q[4],
        'joint_gripper_finger_left': q[5]  # TODO: is gripper control correct?
    }, blocking=True)

# takes X = (x, y, z), uses inverse kinematics to call joint control
def EE_position_control(x, node):
    # TODO: use IK to find the q for this x, call EE_joint_control
    raise NotImplementedError

import sys
if __name__ == '__main__':
    assert len(sys.argv) > 1, 'forgot letter argument'
    letter = sys.argv[1][0].lower()
    if letter in letter_waypoints.keys():
        import hello_helpers.hello_misc as hm
        node = hm.HelloNode.quick_create('letters_demo')
        for point in letter_waypoints[letter]:
            EE_position_control(point, node)
    else:
        print('invalid letter argument')
