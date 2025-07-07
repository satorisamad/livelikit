import wave
import struct
import math
import os

def generate_sine_wave(frequency, duration, sample_rate, amplitude=32000):
    num_samples = int(sample_rate * duration)
    wave_data = []
    for i in range(num_samples):
        value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        wave_data.append(value)
    return wave_data

def write_wav_file(filename, sample_rate, wave_data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for sample in wave_data:
            wf.writeframes(struct.pack('<h', sample))

output_dir = os.path.dirname(os.path.abspath(__file__))

sample_rate = 16000
duration = 1.0 # seconds

# Generate and save test_audio1.wav (440 Hz sine wave)
wave_data1 = generate_sine_wave(440, duration, sample_rate)
write_wav_file(os.path.join(output_dir, 'test_audio1.wav'), sample_rate, wave_data1)
print(f"Generated test_audio1.wav in {output_dir}")

# Generate and save test_audio2.wav (660 Hz sine wave)
wave_data2 = generate_sine_wave(660, duration, sample_rate)
write_wav_file(os.path.join(output_dir, 'test_audio2.wav'), sample_rate, wave_data2)
print(f"Generated test_audio2.wav in {output_dir}")

# Generate and save test_audio3.wav (880 Hz sine wave)
wave_data3 = generate_sine_wave(880, duration, sample_rate)
write_wav_file(os.path.join(output_dir, 'test_audio3.wav'), sample_rate, wave_data3)
print(f"Generated test_audio3.wav in {output_dir}")
