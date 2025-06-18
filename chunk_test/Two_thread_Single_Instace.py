#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torchaudio
import psutil
import numpy as np
from typing import List, Tuple
import warnings
import whisper
import threading  # Threading for parallel chunk processing
import queue      # Thread-safe queue

# Force CPU usage and suppress CUDA-related warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
os.environ["WHISPER_FORCE_CPU"] = "true"  # Force CPU for Whisper
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def parse_args():
    parser = argparse.ArgumentParser(description='Transcribe audio file using Whisper with chunking (CPU only)')
    parser.add_argument('audio_file', type=str, help='Path to the audio file (WAV format)')
    parser.add_argument('chunk_size', type=int, help='Chunk size in seconds')
    parser.add_argument('--num_threads', type=int, default=2, help='Number of transcription threads')
    return parser.parse_args()

def get_peak_memory() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_peak_cpu() -> float:
    return psutil.cpu_percent()

def process_chunk(model, chunk: Tuple[torch.Tensor, int], lock: threading.Lock) -> dict:
    waveform, sample_rate = chunk
    audio_np = waveform.squeeze().numpy().astype(np.float32)

    start_time = time.time()
    with lock:
        result = model.transcribe(audio_np, fp16=False)
    process_time = time.time() - start_time

    confidence = -2.1
    if "segments" in result and len(result["segments"]) > 0:
        confidence = result["segments"][0].get("avg_logprob", -2.1)

    return {
        'text': result.get("text", "").strip(),
        'duration': waveform.size(1) / sample_rate,
        'process_time': process_time,
        'confidence': confidence,
        'rtf': process_time / (waveform.size(1) / sample_rate) if waveform.size(1) > 0 else 0
    }

def split_audio(audio_path: str, chunk_size: int) -> List[Tuple[torch.Tensor, int]]:
    waveform, sample_rate = torchaudio.load(audio_path)
    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    samples_per_chunk = chunk_size * sample_rate
    total_samples = waveform.size(1)
    chunks = []

    for start in range(0, total_samples, samples_per_chunk):
        end = min(start + samples_per_chunk, total_samples)
        chunk = waveform[:, start:end]
        chunks.append((chunk, sample_rate))

    return chunks

def worker(thread_id: int, model, input_queue: queue.Queue, output_queue: queue.Queue, lock: threading.Lock):
    print(f"Thread {thread_id}: Started.")
    while True:
        chunk_data = input_queue.get()
        if chunk_data is None:
            break
        chunk_index, chunk = chunk_data
        print(f"Thread {thread_id}: Processing chunk {chunk_index}...")
        result = process_chunk(model, chunk, lock)
        output_queue.put((chunk_index, result))
        input_queue.task_done()
    print(f"Thread {thread_id}: Exiting.")

def main():
    args = parse_args()
    model = whisper.load_model("medium", device="cpu")
    model_lock = threading.Lock()

    metrics = {
        'total_duration': 0.0,
        'total_process_time': 0.0,
        'total_latency': 0.0,
        'total_rtf': 0.0,
        'total_confidence': 0.0,
        'file_count': 0,
        'peak_cpu': 0.0,
        'peak_memory': 0.0
    }

    if not os.path.isfile(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)

    start_memory = get_peak_memory()
    start_cpu = get_peak_cpu()

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    print(f"Splitting audio file: {args.audio_file} with chunk size: {args.chunk_size}s")
    chunks = split_audio(args.audio_file, args.chunk_size)
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        input_queue.put((i, chunk))

    threads = []
    for i in range(args.num_threads):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, model, input_queue, output_queue, model_lock)
        )
        thread.start()
        threads.append(thread)

    input_queue.join()

    for _ in range(args.num_threads):
        input_queue.put(None)

    for thread in threads:
        thread.join()

    processed_results = [None] * total_chunks
    while not output_queue.empty():
        chunk_index, result = output_queue.get()
        processed_results[chunk_index] = result

    with open("transcribe.txt", "w") as f:
        for i, result in enumerate(processed_results):
            if result:
                print(f"\nResult for chunk {i+1}:")
                print(result)
                f.write(result['text'] + "\n")
                metrics['total_duration'] += result['duration']
                metrics['total_process_time'] += result['process_time']
                metrics['total_latency'] += result['process_time']
                metrics['total_rtf'] += result['rtf']
                metrics['total_confidence'] += result['confidence']
                metrics['file_count'] += 1
                metrics['peak_memory'] = max(metrics['peak_memory'], get_peak_memory() - start_memory)
                metrics['peak_cpu'] = max(metrics['peak_cpu'], get_peak_cpu() - start_cpu)

    if metrics['file_count'] > 0:
        end_time = time.time()
        overall_wall_clock_time = end_time - start_time_main
        print("\n=== PROCESSING COMPLETE (CPU ONLY) ===")
        print(f" - Total Audio Duration: {metrics['total_duration']:.2f} seconds")
        print(f" - Total Processing Time (sum of all chunk processing times across threads): {metrics['total_process_time']:.2f} seconds")
        print(f" - Overall Wall-Clock Time: {overall_wall_clock_time:.2f} seconds")

        if metrics['total_duration'] > 0:
            overall_rtf_wall_clock = overall_wall_clock_time / metrics['total_duration']
            print(f" - Overall Wall-Clock RTF: {overall_rtf_wall_clock:.2f} (lower is better)")
            print(f" - Overall Processing Speed (Wall-Clock): {metrics['total_duration'] / overall_wall_clock_time:.2f}x real-time")

        print(f" - Avg Latency Per Chunk: {metrics['total_latency'] / metrics['file_count']:.2f} seconds")
        print(f" - Avg RTF Per Chunk: {metrics['total_rtf'] / metrics['file_count']:.2f}")
        print(f" - Avg Confidence: {metrics['total_confidence'] / metrics['file_count']:.2f}")
        print(f" - Peak CPU Usage: {metrics['peak_cpu']:.1f}%")
        print(f" - Peak Memory Usage: {metrics['peak_memory']:.2f} MB")
    else:
        print("No chunks were processed.")

if __name__ == "__main__":
    start_time_main = time.time()
    main()
