import sounddevice as sd
import scipy.io.wavfile as wav

filename = 'recorded.wav'
duration = 5  # seconds
sample_rate = 16000  # Hz

print("🎙️ Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
wav.write(filename, sample_rate, audio)
print(f'✅ Saved to {filename}')
