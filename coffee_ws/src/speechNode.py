#!/usr/bin/env python3

import rclpy
import time
from hello_helpers.hello_misc import HelloNode
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

class MotionLoopNode(HelloNode):
    """
    MotionLoopNode: Beeps 3x, then moves lift up/down until 60s or "test" is heard.
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
        self.interrupted = False

        HelloNode.main(self, 'motion_loop_node', 'motion_loop_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('✅ Node is ready. Switching to position mode...')
        self.switch_to_position_mode()
        time.sleep(0.5)

        # Subscribe to speech input
        self.create_subscription(SpeechRecognitionCandidates, '/speech_to_text', self.speech_callback, 1)

        # Beep 3x
        self.get_logger().info('🔊 Beeping: "Say test to stop"')
        self.beep()

        # Start motion timer
        self.start_time = time.time()
        self.get_logger().info(f'⏱️ Starting {self.rate}-second motion loop.')
        self.motion_timer = self.create_timer(self.rate, self.motion_loop)

    def motion_loop(self):
        if self.interrupted:
            self.get_logger().warn('🛑 Motion interrupted by keyword "test".')
            self.motion_timer.cancel()
            rclpy.shutdown()
            return

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
        transcript = ' '.join(msg.transcript).lower()
        self.get_logger().info(f'🎤 Heard: {transcript}')
        if 'test' in transcript:
            self.get_logger().warn('🗣️ Detected keyword: "test"')
            self.interrupted = True

    def beep(self):
        try:
            for i in range(3):
                self.robot.pimu.trigger_beep()
                self.get_logger().info(f'✅ Beep {i+1} triggered.')
                time.sleep(0.3)
        except Exception as e:
            self.get_logger().error(f'❌ Failed to beep: {e}')

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
