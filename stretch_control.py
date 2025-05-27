# IK written using https://github.com/hello-robot/stretch_tutorials/blob/master/stretch_body/jupyter/inverse_kinematics.ipynb as reference

import numpy as np
import time
import math

BANNED_LINK_NAMES = ['link_right_wheel', 'link_left_wheel', 'caster_link', 'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_head', 'link_head_pan', 'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 'laser', 'respeaker_base', 'base_imu', 'link_puller']

BANNED_JOINT_NAMES = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint', 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 'joint_laser', 'joint_respeaker', 'joint_base_imu', 'joint_puller']

INCHES_PER_METER = 39.37
EE_LENGTH = 5.45 * 2 / INCHES_PER_METER
clamp = lambda x, l, h : max(min(x, h), l)
sign = lambda x : (x > 0) - (x < 0)

# implemented from scratch, using only 3 joints (arm lift, telescoping arm extension, and wrist yaw
def EE_position_control_2(X, node, sleep_time=0, blocking=True, closed=True):
    # my base-frame coordinates: arm is connected to base at (0,0,0), x,y,z
    # +x points in the direction of the telescoping arm
    # +z points upward, that means +y points in direction of base's front (INKY label)

    # need to restrict position to reachable workspace
    clamp_X = (clamp(X[0], 0, 0.13 * 4 + 0.13),
               clamp(X[1], -0.2, EE_LENGTH),  # change negative bound because camera collision
               clamp(X[2], 0, 1.1))  # rough estimate of reachable workspace

    for i, v in enumerate('xyz'):
        if clamp_X[i] != X[i]: print('coordinate '+v+' was clamped')
        else: print('coordinate '+v+' was not clamped')

    theta = math.asin(abs(X[1]) / EE_LENGTH)
    # print(f'calculated theta for this movement: {theta}, {theta * sign(X[1])}')
    x_offset = EE_LENGTH * math.cos(theta)

    node.move_to_pose({'joint_wrist_pitch': 0.0, 'joint_wrist_roll': 0.0})
    if closed: node.move_to_pose({'joint_gripper_finger_left': 0.0})
    node.move_to_pose({
        'joint_lift': X[2],
        'joint_arm': X[0] - x_offset,
        'joint_wrist_yaw': theta * sign(float(X[1])),
    }, blocking=blocking)

    print(f'finished movement')

    time.sleep(sleep_time)

# problem: node.joint_state reports different joints than the chain
# so, reorder elements from the node.joint_state to be understood by IK chain
# example: here is node.joint_state.names:
# wrist_extension	0.0999952846055835
# joint_lift	0.5995599066271622
# joint_arm_l3	0.024998821151395876
# joint_arm_l2	0.024998821151395876
# joint_arm_l1	0.024998821151395876
# joint_arm_l0	0.024998821151395876
# joint_head_pan	-0.0030191405333959767
# joint_head_tilt	-0.06322773496357809
# joint_wrist_yaw	0.0025566346464760688
# joint_wrist_pitch	-0.6381360077604268
# joint_wrist_roll	-0.0030679615757712823
# joint_gripper_finger_left	0.5434139284070234
# joint_gripper_finger_right	0.5434139284070234
def get_current_joint_pos(node):
    desired_order = [None,
                     None,
                     None,
                     'joint_lift',
                     None,
                     'joint_arm_l3',
                     'joint_arm_l2',
                     'joint_arm_l1',
                     'joint_arm_l0',
                     'joint_wrist_yaw',
                     None,
                     None
                    ]

    q_out = [0.0] * 12
    for i, name in enumerate(node.joint_state.name):
        if name in desired_order:
            q_out[desired_order.index(name)] = node.joint_state.position[i]
    return q_out

# arm q-vector: lift, telescoping arm, wrist pitch, roll, yaw, gripper
def EE_joint_control(q, node):
    assert len(q) == 12
    q_base = q[1]
    q_lift = q[3]
    q_arm = q[5] + q[6] + q[7] + q[8]
    q_yaw = q[9]
    # TODO: get the q_init
    node.move_to_pose({
        'joint_lift': q_lift,
        'joint_arm': q_arm,
        'joint_wrist_yaw': q_yaw
    })
    time.sleep(WAIT_TIME)
    print('REACHED POSITION...')
    time.sleep(WAIT_TIME)

# takes X = (x, y, z), uses inverse kinematics to call joint control
def EE_position_control(x, node, chain):
    if node is not None: node.get_logger().info(f'starting ee position control, attempting to solve for q from x={x}')
    q_init = get_current_joint_pos(node)
    q_soln = chain.inverse_kinematics(x, initial_position=q_init)
    q_soln = chain.inverse_kinematics(x)
    if node is not None: node.get_logger().info(f'solved for q, attempting to move to found q')
    # TODO: may need to remove/rearrange elements from q_soln to match expected input of EE_joint_control
    EE_joint_control(q_soln, node)
    if node is not None: node.get_logger().info(f'finished ee position control')

# create a chain object used by IK (should only need to be called once
def prep_chain(urdf_path):
    import ikpy.chain
    import urdfpy

    modified_urdf = urdfpy.URDF.load(urdf_path)
    banned_links = [link for link in modified_urdf._links if link.name in BANNED_LINK_NAMES]
    for link in banned_links: modified_urdf._links.remove(link)
    banned_joints = [joint for joint in modified_urdf._joints if joint.name in BANNED_JOINT_NAMES]
    for joint in banned_joints: modified_urdf._joints.remove(joint)

    joint_base_translation = urdfpy.Joint(name='joint_base_translation',
                                  parent='base_link',
                                  child='link_base_translation',
                                  joint_type='prismatic',
                                  axis=np.array([1.0, 0.0, 0.0]),
                                  origin=np.eye(4, dtype=np.float64),
                                  limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
    modified_urdf._joints.append(joint_base_translation)
    link_base_translation = urdfpy.Link(name='link_base_translation',
                                inertial=None,
                                visuals=None,
                                collisions=None)
    modified_urdf._links.append(link_base_translation)

    for joint in modified_urdf._joints:
        if joint.name == 'joint_mast':
            joint.parent = 'link_base_translation'

    modified_urdf_path = '/tmp/modified_urdf.urdf'
    modified_urdf.save(modified_urdf_path)

    # TEMP DEBUG, TODO: remove this
    '''import ikpy.urdf.utils
    tree = ikpy.urdf.utils.get_urdf_tree(modified_urdf_path, "base_link")[0]
    from IPython import display
    display.display_png(tree)
    tree.format = 'png'
    tree.render('mytree', view=True)'''
    # END OF DEUBG

    chain = ikpy.chain.Chain.from_urdf_file(modified_urdf_path)
    return chain
