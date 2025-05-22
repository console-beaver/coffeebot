#!/usr/bin/env python3

import rclpy
import subprocess
import hello_helpers.hello_misc as hm

class SpeakerTestNode(hm.HelloNode):
    """
    SpeakerTestNode: Plays a test audio file on Stretch 3's speaker.
    """
    def __init__(self):
        super().__init__()
        hm.HelloNode.main(self, 'speaker_test_node', 'speaker_test_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('🔊 Playing audio...')
        self.play_audio()
        self.get_logger().info('✅ Done playing audio.')
        rclpy.shutdown()

    def play_audio(self):
        try:
            # Replace with your own WAV file path
            audio_path = '/tmp/speaker_test.wav'
            subprocess.run(['aplay', audio_path])
        except Exception as e:
            self.get_logger().error(f'❌ Failed to play audio: {e}')

def main(args=None):
    try:
        node = SpeakerTestNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupted.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
