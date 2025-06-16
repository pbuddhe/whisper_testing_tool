import os
import time
import whisper
import torchaudio
from jiwer import wer, cer
from glob import glob

# ---------- CONFIG ----------
AUDIO_DIR = "Dataset_1"  # directory of .wav files
TRANSCRIPT_FILE = "Dataset_1/transcripts.txt"  # tab-separated: filename<TAB>ground_truth
MODEL_SIZE = "medium"  # tiny, base, small, medium, large
# ----------------------------

# Load model
print(f"Loading Whisper model ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)

# Load reference transcriptions
references = {}
with open(TRANSCRIPT_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            # Split on first space
            parts = line.split(' ', 1)
            if len(parts) == 2:
                filename, transcription = parts
                references[filename] = transcription
            else:
                print(f"Warning: Skipping malformed line: {line}")

# Initialize metrics
total_wer, total_cer, total_rtf, total_latency = 0, 0, 0, 0
file_count = 0

# Process each audio file
for audio_file in glob(os.path.join(AUDIO_DIR, "*.wav")):
    filename = os.path.basename(audio_file)
    if filename not in references:
        continue

    # Load audio to get duration
    waveform, sample_rate = torchaudio.load(audio_file)
    duration = waveform.shape[1] / sample_rate

    # Transcribe with Whisper
    start_time = time.time()
    result = model.transcribe(audio_file)
    end_time = time.time()

    predicted = result["text"].strip().lower()
    reference = references[filename].strip().lower()

    # Metrics
    latency = end_time - start_time
    rtf = latency / duration
    file_wer = wer(reference, predicted)
    file_cer = cer(reference, predicted)

    print(f"[{filename}] WER: {file_wer:.2f}, CER: {file_cer:.2f}, Latency: {latency:.2f}s, RTF: {rtf:.2f}")

    total_wer += file_wer
    total_cer += file_cer
    total_latency += latency
    total_rtf += rtf
    file_count += 1

# Averages
if file_count > 0:
    print("\n=== AVERAGE PERFORMANCE ===")
    print(f"Average WER: {total_wer / file_count:.2f}")
    print(f"Average CER: {total_cer / file_count:.2f}")
    print(f"Average Latency: {total_latency / file_count:.2f} seconds")
    print(f"Average RTF: {total_rtf / file_count:.2f}")
else:
    print("No matching audio files with transcripts found.")
