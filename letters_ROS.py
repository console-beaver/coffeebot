# USAGE: run
# `ros2 launch stretch_core stretch_driver.launch.py mode:=position`
# then, run this script while specifying a letter in {a, b, c}

# letters are defined relative to bottom-left corner, (x, y, z) (where z is out of the page)
LETTER_WAYPOINTS = {
        'a': ((0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)),
        'b': ((0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)),
        'c': ((1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0))
}

BANNED_LINK_NAMES = ['link_right_wheel', 'link_left_wheel', 'caster_link', 'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_head', 'link_head_pan', 'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 'laser', 'respeaker_base']
BANNED_JOINT_NAMES = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint', 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 'joint_laser', 'joint_respeaker']

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
def EE_position_control(x, node, chain):
    q_soln = chain.inverse_kinematics(x)  # TODO: initial_q parameter? (see tutorial for this)
    # TODO: may need to remove/rearrange elements from q_soln to match expected input of EE_joint_control
    EE_joint_control(q_soln, node)

import sys
import os 
if __name__ == '__main__':
    assert len(sys.argv) > 1, 'forgot letter argument'
    letter = sys.argv[1][0].lower()
    if letter in LETTER_WAYPOINTS.keys():
        import hello_helpers.hello_misc as hm

        # TODO: verify that this is the correct urdf to use (Michelle?)
        urdf_path = '/home/cs225a1/.local/lib/python3.10/site-packages/stretch_urdf/RE2V0/stretch_description_RE2V0_tool_stretch_gripper.urdf'
        chain = prep_chain(urdf_path)
        node = hm.HelloNode.quick_create('letters_demo')

        for point in LETTER_WAYPOINTS[letter]:
            # TODO: may need to add transformation to get waypoints into stretch base frame
            EE_position_control(point, node)
    else:
        print('invalid letter argument')
