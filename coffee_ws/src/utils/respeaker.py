import subprocess
from speech_recognition_msgs.msg import SpeechRecognitionCandidates
import os

INTRO_SOUND = '../sounds/ask_order.wav'
CONFIRM_SOUND_1 = '../sounds/order1.wav'
CONFIRM_SOUND_2 = '../sounds/order2.wav'
CONFIRM_SOUND_3 = '../sounds/order3.wav'

class ReSpeaker:
    def __init__(self, node):
        self.node = node
        self.voice_command = None 

        base_dir = os.path.dirname(os.path.abspath(__file__)) 

        self.intro_path = os.path.join(base_dir, INTRO_SOUND)
        self.confirm_paths = [os.path.join(base_dir, CONFIRM_SOUND_1), 
                              os.path.join(base_dir, CONFIRM_SOUND_2), 
                              os.path.join(base_dir,CONFIRM_SOUND_3)]

        self.node.create_subscription(
            SpeechRecognitionCandidates,
            '/speech_to_text',
            self.speech_callback,
            1
        )
    
    def speech_callback(self, msg):
        self.voice_command = ' '.join(msg.transcript).strip().lower()
        self.node.get_logger().info(f'🎤 Heard: "{self.voice_command}"')

    def play_audio(self, path):
        try:
            subprocess.run(['aplay', path], check=True)
        except Exception as e:
            self.node.get_logger().error(f'❌ Failed to play {path}: {e}')

    def ask_for_order(self):
        self.play_audio(self.intro_path) 

    def check_for_order(self):
        if self.voice_command:
            if 'one' in self.voice_command:
                self.play_audio(self.confirm_paths[0])
                return 1
            elif 'two' in self.voice_command:
                self.play_audio(self.confirm_paths[1])
                return 2
            elif 'three' in self.voice_command:
                self.play_audio(self.confirm_paths[2])
                return 3
        return None
