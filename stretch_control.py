# IK written using https://github.com/hello-robot/stretch_tutorials/blob/master/stretch_body/jupyter/inverse_kinematics.ipynb as reference

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
    chain = ikpy.chain.Chain.from_urdf_file(modified_urdf_path)
    return chain
