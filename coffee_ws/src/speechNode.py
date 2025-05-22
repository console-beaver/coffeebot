#!/usr/bin/env python3

import rclpy
import subprocess
from rclpy.node import Node
from hello_helpers.hello_misc import HelloNode
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

INTRO_SOUND = 'sounds/ask_order.wav'
CONFIRM_SOUND_1 = 'sounds/order1.wav'
CONFIRM_SOUND_2 = 'sounds/order2.wav'
CONFIRM_SOUND_3 = 'sounds/order3.wav'

class SpeakerTestNode(HelloNode):
    def __init__(self):
        super().__init__()
        self.voice_command = None

        # Start the ROS node system
        HelloNode.main(self, 'speaker_test_node', 'speaker_test_node', wait_for_first_pointcloud=False)

        # Subscribe to speech input
        self.create_subscription(
            SpeechRecognitionCandidates,
            '/speech_to_text',
            self.speech_callback,
            1
        )

    def main(self):
        self.get_logger().info('🔊 Playing initial audio...')
        self.play_audio(INTRO_SOUND)

        self.get_logger().info('🗣️ Waiting for voice input: 1, 2, or 3...')
        self.create_timer(0.5, self.check_voice_command)

    def speech_callback(self, msg):
        transcript = ' '.join(msg.transcript).strip().lower()
        self.get_logger().info(f'🎤 Heard: "{transcript}"')
        self.voice_command = transcript

    def check_voice_command(self):
        if self.voice_command == 'one':
            self.get_logger().info('🔈 Playing sound for "1"')
            self.play_audio(CONFIRM_SOUND_1)
            rclpy.shutdown()
        elif self.voice_command == 'two':
            self.get_logger().info('🔈 Playing sound for "2"')
            self.play_audio(CONFIRM_SOUND_2)
            rclpy.shutdown()
        elif self.voice_command == 'three':
            self.get_logger().info('🔈 Playing sound for "3"')
            self.play_audio(CONFIRM_SOUND_3)
            rclpy.shutdown()

    def play_audio(self, audio_path):
        try:
            subprocess.run(['aplay', audio_path], check=True)
        except Exception as e:
            self.get_logger().error(f'❌ Failed to play audio: {e}') 

def main(args=None):
    try:
        node = SpeakerTestNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupted. Shutting down...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
