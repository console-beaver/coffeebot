import argparse
import time
import zmq
import yaml

import numpy as np
import cv2
import torch
import pyrealsense2 as rs

import d405_helpers as dh
import normalized_velocity_control as nvc
import loop_timer as lt
import stretch_body.robot as rb
from stretch_body import robot_params
from stretch_body import hello_utils as hu

from scipy.spatial.transform import Rotation
from yaml.loader import SafeLoader

# SAM / Grounding DINO imports
from groundingdino.models import build_model
from groundingdino.util import load_model, transform as gdino_transform
from segment_anything import sam_model_registry, SamPredictor

# --------------------------------------------------
# Abstract Detector Interface
# --------------------------------------------------
class Detector:
    def __init__(self, camera_info):
        self.camera_info = camera_info
    def detect(self, color_image: np.ndarray, depth_frame) -> dict:
        raise NotImplementedError

# --------------------------------------------------
# YOLO Fingertip Detector
# --------------------------------------------------
class YoloFingerDetector(Detector):
    def __init__(self, camera_info, zmq_address, target_rate_hz, history_s, gain):
        super().__init__(camera_info)
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(zmq_address)
        self.reg = RegulatePollTimeout(target_rate_hz, history_s, gain)

    def detect(self, color_image, depth_frame):
        timeout = self.reg.get_poll_timeout()
        features = {}
        if self.socket.poll(timeout):
            msg = self.socket.recv_pyobj()
            self.reg.run_after_polling()
            for side, f in msg.get('fingertips', {}).items():
                features[f"{side}_fingertip"] = f['pos']
        return features

# --------------------------------------------------
# Text-Prompted Object Detector (Grounding DINO + SAM)
# --------------------------------------------------
class TextPromptDetector(Detector):
    def __init__(self, camera_info, text_prompt, sam_ckpt, gdino_cfg, gdino_ckpt, device='cuda'):
        super().__init__(camera_info)
        self.text_prompt = text_prompt
        self.sam = sam_model_registry['vit_h'](checkpoint=sam_ckpt).to(device)
        self.predictor = SamPredictor(self.sam)
        self.gd_model = build_model(gdino_cfg)
        load_model(self.gd_model, gdino_ckpt)
        self.gd_model.to(device).eval()
        self.device = device

    def detect(self, color_image: np.ndarray, depth_frame):
        img_t = gdino_transform(color_image).to(self.device)[None]
        with torch.no_grad():
            outputs = self.gd_model(img_t, captions=[self.text_prompt])
        logits = outputs['pred_logits'][0]
        boxes  = outputs['pred_boxes'][0][logits.max(-1).values > 0.3]
        if len(boxes)==0:
            return {}
        box = boxes[0].cpu().numpy()
        self.predictor.set_image(color_image)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box = box[None, :],
            multimask_output=False
        )
        mask = masks[0]
        ys, xs = np.nonzero(mask)
        cx, cy = xs.mean(), ys.mean()
        d = depth_frame.get_distance(int(cx), int(cy))
        xyz = dh.point_cloud_from_pixel(cx, cy, d, self.camera_info)
        return {self.text_prompt: xyz}

# --------------------------------------------------
# Poll Timeout Regulator
# --------------------------------------------------
class RegulatePollTimeout:
    def __init__(self, target_hz, history_s, gain, debug_on=False):
        self.target_period_ms = 1000.0 / target_hz
        self.timeout_ms = self.target_period_ms
        self.gain = gain
        self.history_len = int(history_s * target_hz)
        self.debug_on = debug_on
        self.poll_times = []
        self.nonpoll_times = []
        self.t_before = None
        self.t_after = None
    def get_poll_timeout(self):
        now = time.time()
        if self.t_after is not None and self.t_before is not None:
            self.nonpoll_times.append(now - self.t_after)
            if len(self.nonpoll_times)>self.history_len:
                self.nonpoll_times.pop(0)
        self.t_before = now
        return max(int(round(self.timeout_ms)), 1)
    def run_after_polling(self):
        now = time.time()
        if self.t_before is not None:
            self.poll_times.append(now - self.t_before)
            if len(self.poll_times)>self.history_len:
                self.poll_times.pop(0)
        self.t_after = now
        if self.poll_times and self.nonpoll_times:
            mean_poll = np.mean(self.poll_times)*1000
            mean_non  = np.mean(self.nonpoll_times)*1000
            error = self.target_period_ms - (mean_poll+mean_non)
            self.timeout_ms += self.gain * error

# --------------------------------------------------
# Dynamixel limits & recenter (from original)
# --------------------------------------------------
def get_dxl_joint_limits(joint):
    params = robot_params.RobotParams().get_params()[1][joint]
    rng = []
    gr = params['gr']; zero = params['zero_t']
    polarity = -1.0 if params['flip_encoder_polarity'] else 1.0
    for t in params['range_t']:
        x = t-zero
        rng.append(polarity*hu.deg_to_rad(360.0*x/4096.0)/gr)
    return rng

joint_state_center = {
    'lift_pos':0.7,'arm_pos':0.01,'wrist_yaw_pos':0.0,
    'wrist_pitch_pos':0.0,'wrist_roll_pos':0.0,'gripper_pos':10.46
}

min_joint_state = {
    'base_odom_theta':-0.8,'lift_pos':0.1,'arm_pos':0.01,
    'wrist_yaw_pos':-0.20,'wrist_pitch_pos':-1.2,'wrist_roll_pos':-0.1,
    'gripper_pos':3.0
}
max_joint_state = {
    'base_odom_theta':0.8,'lift_pos':1.05,'arm_pos':0.45,
    'wrist_yaw_pos':1.0,'wrist_pitch_pos':0.2,'wrist_roll_pos':0.1,
    'gripper_pos':get_dxl_joint_limits('stretch_gripper')[1]
}
zero_vel = {k:0.0 for k in [
    'base_counterclockwise','lift_up','arm_out',
    'wrist_yaw_counterclockwise','wrist_pitch_up',
    'wrist_roll_counterclockwise','gripper_open'
]}
pos_to_vel = {
    'base_odom_theta':'base_counterclockwise','lift_pos':'lift_up',
    'arm_pos':'arm_out','wrist_yaw_pos':'wrist_yaw_counterclockwise',
    'wrist_pitch_pos':'wrist_pitch_up','wrist_roll_pos':'wrist_roll_counterclockwise',
    'gripper_pos':'gripper_open'
}
vel_to_pos = {v:k for k,v in pos_to_vel.items()}

def recenter_robot(robot):
    pan, tilt = np.pi/2, -np.pi/2
    robot.head.move_to('head_pan', pan); robot.head.move_to('head_tilt', tilt)
    robot.push_command(); robot.wait_command()
    robot.end_of_arm.get_joint('wrist_yaw').move_to(joint_state_center['wrist_yaw_pos'])
    robot.end_of_arm.get_joint('wrist_pitch').move_to(joint_state_center['wrist_pitch_pos'])
    robot.push_command(); robot.wait_command()
    robot.arm.move_to(joint_state_center['arm_pos'])
    robot.push_command(); robot.wait_command()
    robot.lift.move_to(joint_state_center['lift_pos'])
    robot.push_command(); robot.wait_command()
    robot.end_of_arm.get_joint('stretch_gripper').move_to(joint_state_center['gripper_pos'])
    robot.push_command(); robot.wait_command()

# --------------------------------------------------
# Main application
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog='Stretch Visual Servoing Demo',
        description='Combine YOLO fingertips + SAM object detection'
    )
    parser.add_argument('--yolo', action='store_true')
    parser.add_argument('--sam', action='store_true')
    parser.add_argument('--object', type=str, default='bottle')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('-e','--exposure', type=str, default='low')
    args = parser.parse_args()
    if not dh.exposure_argument_is_valid(args.exposure):
        raise argparse.ArgumentTypeError(f'Invalid exposure {args.exposure}')

    # Robot & controller
    robot = rb.Robot(); robot.startup()
    recenter_robot(robot)
    controller = nvc.NormalizedVelocityControl(robot)
    controller.reset_base_odometry()

    # Camera
    pipeline, profile = dh.start_d405(args.exposure)
    depth_info = dh.get_camera_info(pipeline.wait_for_frames().get_depth_frame())

    # Detector setup
    detectors = []
    if args.yolo:
        addr = f"tcp://{yn.remote_computer_ip if args.remote else '127.0.0.1'}:{yn.yolo_port}"
        detectors.append(YoloFingerDetector(depth_info, addr,
                            target_control_loop_rate_hz, seconds_of_timing_history,
                            timeout_proportional_gain))
    if args.sam:
        detectors.append(TextPromptDetector(depth_info, args.object,
                            sam_ckpt="sam_vit_h_4b8939.pth",
                            gdino_cfg="GroundingDINO_SwinB.cfg",
                            gdino_ckpt="GroundingDINO_SwinB.pth"))

    # Grasp parameters from original
    toy_depth_m = 0.055; toy_width_m = 0.0542
    grasp_if_error_below_this = 0.02
    gripper_open_speed = 1.0; gripper_close_speed = 1.0
    lost_ball_target_error_too_large = 0.10
    lost_ball_fingertips_too_close = 0.038
    successful_grasp_effort = -14.0
    successful_grasp_max_fingertip_distance = 0.085
    successful_grasp_min_fingertip_distance = 0.05
    default_between_fingertips = np.array([0.01,0.035,0.17])
    distance_between_fully_open_fingertips = 0.16
    max_toy_z_for_default_fingertips = 0.12
    max_distance_for_attempted_reach = 0.5
    arm_retraction_speedup = 5.0
    max_gripper_length = 0.26
    overall_visual_servoing_velocity_scale = 1.0
    joint_visual_servoing_velocity_scale = {
        'base_counterclockwise':4.0,'lift_up':6.0,'arm_out':6.0,
        'wrist_yaw_counterclockwise':4.0,'wrist_pitch_up':6.0,
        'wrist_roll_counterclockwise':1.0,'gripper_open':1.0
    }
    print_timing = True
    stop_if_toy_not_detected_this_many_frames = 10
    stop_if_fingers_not_detected_this_many_frames = 10
    max_retract_state_count = 60
    min_base_speed = 0.05

    # State variables
    first_frame = True
    frames_since_toy_detected = 0
    frames_since_fingers_detected = 0
    distance_between_fingertips = distance_between_fully_open_fingertips
    prev_distance_between_fingertips = distance_between_fully_open_fingertips
    behavior = 'reach'; prev_behavior = 'reach'
    grasping_the_target = False; pre_reach = True

    loop_timer = lt.LoopTimer()
    try:
        while True:
            loop_timer.start_of_iteration()
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            image = np.copy(color_image)

            # Detection via all detectors
            feats = {}
            for det in detectors:
                feats.update(det.detect(color_image, depth_frame))
            toy_target = feats.get(args.object)
            fingertip_left_pos  = feats.get('left_fingertip')
            fingertip_right_pos = feats.get('right_fingertip')

            # Frame counters
            frames_since_toy_detected = 0 if toy_target is not None else frames_since_toy_detected+1
            if fingertip_left_pos is not None and fingertip_right_pos is not None:
                frames_since_fingers_detected = 0
                between_fingertips = (fingertip_left_pos + fingertip_right_pos)/2.0
                prev_distance_between_fingertips = distance_between_fingertips
                distance_between_fingertips = np.linalg.norm(fingertip_left_pos-fingertip_right_pos)
            elif toy_target is not None and toy_target[2]<max_toy_z_for_default_fingertips:
                between_fingertips = default_between_fingertips
                distance_between_fingertips = prev_distance_between_fingertips
                frames_since_fingers_detected = frames_since_fingers_detected+1
            else:
                frames_since_fingers_detected = frames_since_fingers_detected+1

            joint_state = controller.get_joint_state()
            joint_state['base_odom_theta'] = hu.angle_diff_rad(joint_state['base_odom_theta'], 0.0)

            # Logging
            print(f"Grasp effort = {joint_state['gripper_eff']:.2f}")
            if distance_between_fingertips is not None:
                print(f"Distance between fingertips = {100*distance_between_fingertips:.2f} cm")
            if toy_target is None:
                print(f"{args.object} Detection: FAILED")
            else:
                print(f"{args.object} Detection: SUCCEEDED")

            # Check for lost object
            if distance_between_fingertips is not None and distance_between_fingertips<lost_ball_fingertips_too_close and grasping_the_target:
                print("I LOST THE OBJECT!!!")
                grasping_the_target=False
            if toy_target is not None and between_fingertips is not None:
                position_error = toy_target - between_fingertips
                if np.linalg.norm(position_error)>lost_ball_target_error_too_large and grasping_the_target:
                    print("I LOST THE OBJECT!!!")
                    grasping_the_target=False

            print(f"Behavior={behavior}, Pre-reach={pre_reach}")

            # State machine
            if behavior=='retract':
                if prev_behavior!='retract': retract_state_count=0
                prev_behavior='retract'
                cmd={'lift_up':0.3,'arm_out':-1.0}
                if not grasping_the_target or retract_state_count>max_retract_state_count or joint_state['arm_pos']<min_joint_state['arm_pos']+0.01:
                    cmd=zero_vel.copy()
                    behavior='celebrate' if grasping_the_target else 'disappointed'
                if True:
                    cmd={k:(0.0 if (v<0 and joint_state[vel_to_pos[k]]<min_joint_state[vel_to_pos[k]]) else v) for k,v in cmd.items()}
                    cmd={k:(0.0 if (v>0 and joint_state[vel_to_pos[k]]>max_joint_state[vel_to_pos[k]]) else v) for k,v in cmd.items()}
                    controller.set_command(cmd)
                retract_state_count+=1

            elif behavior=='celebrate':
                if prev_behavior!='celebrate': celebrate_state_count=0; pitch_ready=yaw_ready=False; ready_to_waggle=False; waggle_count=0
                prev_behavior='celebrate'
                with controller.lock:
                    pitch=joint_state['wrist_pitch_pos']; yaw=joint_state['wrist_yaw_pos']
                    pitch_ready=abs(pitch-0.1)<=0.1; yaw_ready=abs(yaw)<=0.1
                    ready_to_waggle=pitch_ready and yaw_ready
                    if not ready_to_waggle:
                        if abs(pitch-0.1)>0.05: robot.end_of_arm.get_joint('wrist_pitch').move_to(0.1)
                        if abs(yaw)>0.05:    robot.end_of_arm.get_joint('wrist_yaw').move_to(0.0)
                    if ready_to_waggle:
                        dir=int(waggle_count/4)%2
                        disp=0.05 if dir==0 else -0.05
                        robot.end_of_arm.get_joint('wrist_yaw').move_by(disp, v_des=3.0, a_des=10.0)
                        waggle_count+=1
                    robot.push_command()
                if waggle_count>16 or celebrate_state_count>100:
                    controller.set_command(zero_vel); behavior='reach'; pre_reach=True
                if not grasping_the_target:
                    controller.set_command(zero_vel); behavior='disappointed'
                celebrate_state_count+=1

            elif behavior=='disappointed':
                if prev_behavior!='disappointed': disappointed_state_count=0
                prev_behavior='disappointed'
                with controller.lock:
                    if joint_state['wrist_pitch_pos']>-1.0:
                        robot.end_of_arm.get_joint('wrist_pitch').move_to(-0.8, v_des=0.5, a_des=1.0)
                    robot.push_command()
                if disappointed_state_count>40:
                    controller.set_command(zero_vel); behavior='reach'; pre_reach=True
                disappointed_state_count+=1

            elif behavior=='reach':
                prev_behavior='reach'
                if pre_reach:
                    cmd={}
                    if joint_state['gripper_pos']>=0.9*max_joint_state['gripper_pos']:
                        cmd['gripper_open']=0.0; pre_reach=False
                    else:
                        cmd['gripper_open']=gripper_open_speed
                    if cmd:
                        cmd={k:overall_visual_servoing_velocity_scale*v for k,v in cmd.items()}
                        cmd={k:joint_visual_servoing_velocity_scale[k]*v for k,v in cmd.items()}
                        controller.set_command(cmd)
                elif between_fingertips is not None and toy_target is not None and np.linalg.norm(toy_target-between_fingertips)<=max_distance_for_attempted_reach:
                    position_error=toy_target-between_fingertips
                    x_err,y_err,z_err=position_error
                    yaw_vel=-x_err; pitch_vel=-y_err; roll_vel=-joint_state['wrist_roll_pos']
                    yaw=joint_state['wrist_yaw_pos']; pitch=-joint_state['wrist_pitch_pos']; roll=-joint_state['wrist_roll_pos']
                    r=Rotation.from_euler('yxz',[yaw,pitch,roll]).as_matrix()
                    lift_vel=np.dot(r[:,1]*-1, position_error)
                    arm_vel=np.dot(r[:,2], position_error)
                    base_rot=np.dot(r[:,0]*-1, position_error)
                    if abs(base_rot)<min_base_speed: base_rot=0.0
                    if arm_vel<0: arm_vel*=arm_retraction_speedup
                    cmd={
                        'lift_up':lift_vel,'arm_out':arm_vel,
                        'wrist_yaw_counterclockwise':yaw_vel,'wrist_pitch_up':pitch_vel,
                        'wrist_roll_counterclockwise':roll_vel,'base_counterclockwise':base_rot
                    }
                    if np.linalg.norm(position_error)<grasp_if_error_below_this:
                        cmd['gripper_open']=-gripper_close_speed
                        if not grasping_the_target and joint_state['gripper_eff']<successful_grasp_effort and distance_between_fingertips<successful_grasp_max_fingertip_distance and distance_between_fingertips>successful_grasp_min_fingertip_distance:
                            print("I GOT THE OBJECT!!!"); grasping_the_target=True; behavior='retract'
                    else:
                        cmd['gripper_open']=gripper_open_speed
                    cmd={k:overall_visual_servoing_velocity_scale*v for k,v in cmd.items()}
                    cmd={k:joint_visual_servoing_velocity_scale[k]*v for k,v in cmd.items()}
                    controller.set_command(cmd)
                else:
                    stop_cmd=zero_vel.copy()
                    if frames_since_toy_detected>=stop_if_toy_not_detected_this_many_frames:
                        stop_cmd['gripper_open']=gripper_open_speed
                    controller.set_command(stop_cmd)

            # Visualization
            if toy_target is not None:
                dh.draw_origin(image, self.camera_info, toy_target, (255,0,0))
                x,y,z=toy_target*100
                lines=[f"{toy_width_m*100:.1f}cm wide", f"{x:.1f},{y:.1f},{z:.1f}cm"]
                center=np.round(dh.pixel_from_3d(toy_target,self.camera_info)).astype(int)
                dh.draw_text(image,center,lines)
            if 'between_fingertips' in locals():
                dh.draw_origin(image, self.camera_info, between_fingertips,(255,255,255))
            cv2.imshow('Servoing',image); cv2.waitKey(1)

            loop_timer.end_of_iteration()
            if print_timing: loop_timer.pretty_print(minimum=True)
    finally:
        controller.stop(); robot.stop(); pipeline.stop()

if __name__=='__main__':
    main()
