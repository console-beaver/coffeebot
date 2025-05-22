import sounddevice as sd
import scipy.io.wavfile as wav
import subprocess

filename = 'sounds/recorded.wav'
duration = 5  # seconds
sample_rate = 16000  # Hz

print("🎙️ Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
wav.write(filename, sample_rate, audio)
print(f'✅ Saved to {filename}')

# === Play the recorded audio using aplay ===
print("🔊 Playing back...")
try:
    subprocess.run(['aplay', filename], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to play audio: {e}")
