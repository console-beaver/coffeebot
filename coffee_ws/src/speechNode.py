#!/usr/bin/env python3

import rclpy
import time
import subprocess
from hello_helpers.hello_misc import HelloNode
from speech_recognition_msgs.msg import SpeechRecognitionCandidates


class MotionLoopNode(HelloNode):
    """
    MotionLoopNode: serves as a base node to help control the motion commands over time
    """
    def __init__(self):
        super().__init__()
        self.rate = 0.5  # seconds
        self.total_time = 60  # seconds
        self.state = 0
        self.poses = [
            {'joint_lift': 0.4, 'joint_wrist_yaw': -1.0, 'joint_wrist_pitch': -1.0},
            {'joint_lift': 1.0, 'joint_wrist_yaw': 0.0, 'joint_wrist_pitch': 0.0},
        ]
        self.start_time = None
        self.motion_timer = None

        # Start ROS node with Hello Robot's main system
        HelloNode.main(self, 'motion_loop_node', 'motion_loop_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('✅ Node is ready. Switching to position mode...')
        self.switch_to_position_mode()
        time.sleep(0.5)

        # 🎤 Subscribe to speech input (not used yet, just set up)
        self.create_subscription(SpeechRecognitionCandidates, '/speech_to_text', self.speech_callback, 1)

        # 🔊 Speak once
        self.get_logger().info('🔊 Saying: "May I take your order?"')
        self.say("May I take your order?")

        # ⏱️ Start motion loop
        self.start_time = time.time()
        self.get_logger().info(f'⏱️ Starting {self.rate}-second motion timer.')
        self.motion_timer = self.create_timer(self.rate, self.motion_loop)

    def motion_loop(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.total_time:
            self.get_logger().info(f"✅ {self.total_time} sec complete. Shutting down.")
            self.motion_timer.cancel()
            rclpy.shutdown()
            return

        pose = self.poses[self.state]
        self.get_logger().info(f'🤖 Moving to pose: {pose}')
        try:
            self.move_to_pose(pose, blocking=True)
        except Exception as e:
            self.get_logger().error(f'❌ Motion failed: {e}')
            rclpy.shutdown()

        self.state = 1 - self.state

    def speech_callback(self, msg):
        transcript = ' '.join(msg.transcript)
        self.get_logger().info(f'🎤 Heard (but not used yet): {transcript}')

    def say(self, text):
        try:
            subprocess.run(['espeak', text], check=True)
        except Exception as e:
            self.get_logger().error(f'❌ Failed to speak: {e}')


def main(args=None):
    try:
        node = MotionLoopNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupt received. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
