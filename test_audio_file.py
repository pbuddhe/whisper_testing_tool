import os
import time
import whisper
import torchaudio

# ---------- CONFIG ----------
MODEL_SIZE = "medium"  # tiny, base, small, medium, large
# ----------------------------

print(f"Loading Whisper model ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)

import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <audio_file>")
    sys.exit(1)

AUDIO_DIR = sys.argv[1]
waveform, sample_rate = torchaudio.load(AUDIO_DIR)
duration = waveform.shape[1] / sample_rate

start_time = time.time()
result = model.transcribe(AUDIO_DIR)
end_time = time.time()

print(result["text"])

latency = end_time - start_time
rtf = latency / duration

print(f"Latency: {latency:.2f} seconds")
print(f"RTF: {rtf:.2f}")
