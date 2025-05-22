import subprocess
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

INTRO_SOUND = '../sounds/ask_order.wav'
CONFIRM_SOUND_1 = '../sounds/order1.wav'
CONFIRM_SOUND_2 = '../sounds/order2.wav'
CONFIRM_SOUND_3 = '../sounds/order3.wav'

class ReSpeaker:
    def __init__(self, node):
        self.node = node
        self.voice_command = None
        self.intro_path = INTRO_SOUND
        self.confirm_paths = [CONFIRM_SOUND_1, CONFIRM_SOUND_2, CONFIRM_SOUND_3]

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
                self.play_audio(self.confirm_paths['1'])
                return 1
            elif 'two' in self.voice_command:
                self.play_audio(self.confirm_paths['2'])
                return 2
            elif 'three' in self.voice_command:
                self.play_audio(self.confirm_paths['3'])
                return 3
        return None
