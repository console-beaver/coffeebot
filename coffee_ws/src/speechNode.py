#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import subprocess

class MotionLoopNode(hm.HelloNode):
    """
    MotionLoopNode: Records audio and plays it back via Stretch speaker.
    """
    def __init__(self):
        super().__init__()
        self.filename = '/tmp/recorded_audio.wav'
        self.sample_rate = 16000  # Stretch audio default
        self.duration = 5.0  # seconds to record

        hm.HelloNode.main(self, 'motion_loop_node', 'motion_loop_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('✅ Node is ready.')

        # === RECORD ===
        self.get_logger().info(f'🎙️ Recording {self.duration} seconds of audio...')
        self.record_audio()
        self.get_logger().info('✅ Done recording.')

        # === PLAY ===
        self.get_logger().info('🔊 Playing back...')
        self.play_audio()
        self.get_logger().info('✅ Playback complete.')

        # === Optional: shutdown after ===
        rclpy.shutdown()

    def record_audio(self):
        try:
            audio = sd.rec(int(self.sample_rate * self.duration), samplerate=self.sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished
            wav.write(self.filename, self.sample_rate, audio)
        except Exception as e:
            self.get_logger().error(f'❌ Failed to record audio: {e}')

    def play_audio(self):
        try:
            # Stretch OS has `aplay` preinstalled and wired to its speaker system
            subprocess.run(['aplay', self.filename])
        except Exception as e:
            self.get_logger().error(f'❌ Failed to play audio: {e}')

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

