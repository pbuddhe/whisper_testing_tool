#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torchaudio
import psutil
import numpy as np
from typing import List, Dict, Tuple
# Add this import at the top of your file with other imports
import warnings

import whisper


# Force CPU usage and suppress CUDA-related warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
os.environ["WHISPER_FORCE_CPU"] = "true"  # Force CPU for Whisper
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def parse_args():
    parser = argparse.ArgumentParser(description='Transcribe audio file using Whisper with chunking (CPU only)')
    parser.add_argument('audio_file', type=str, help='Path to the audio file (WAV format)')
    parser.add_argument('chunk_size', type=int, help='Chunk size in seconds')
    return parser.parse_args()

def get_peak_memory() -> float:
    """Get peak memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def get_peak_cpu() -> float:
    """Get peak CPU usage percentage."""
    return psutil.cpu_percent()

def process_chunk(model, chunk: Tuple[torch.Tensor, int]) -> dict:
    """Process a single audio chunk with Whisper using in-memory processing (CPU only)."""
    waveform, sample_rate = chunk
    
    # Convert to numpy array in memory
    audio_np = waveform.squeeze().numpy().astype(np.float32)
    
    # Process with Whisper using in-memory numpy array (CPU only)
    start_time = time.time()
    result = model.transcribe(audio_np, fp16=False)  # Force CPU mode
    process_time = time.time() - start_time
    
    # Use logprob values directly for confidence
    confidence = -2.1  # Default to 'Likely incorrect' if no segments
    if "segments" in result and len(result["segments"]) > 0:
        confidence = result["segments"][0].get("avg_logprob", -2.1)  # Return the raw logprob value
    
    return {
        'text': result.get("text", "").strip(),
        'duration': waveform.size(1) / sample_rate,
        'process_time': process_time,
        'confidence': confidence,
        'rtf': process_time / (waveform.size(1) / sample_rate) if waveform.size(1) > 0 else 0
    }

def split_audio(audio_path: str, chunk_size: int) -> List[Tuple[torch.Tensor, int]]:
    """Split audio into chunks of specified duration."""
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
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

def main():
    args = parse_args()
    
    # Initialize metrics
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
    
    # Check if file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    # Load Whisper model (CPU only)
    print("Loading Whisper model (CPU only)...")
    model = whisper.load_model("medium", device="cpu")  # Force CPU usage
    
    # Monitor resources
    start_memory = get_peak_memory()
    start_cpu = get_peak_cpu()
    
    # Process audio chunks
    print(f"Processing audio file: {args.audio_file} with chunk size: {args.chunk_size}s")
    chunks = split_audio(args.audio_file, args.chunk_size)
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{total_chunks}...")
        
        # Process chunk
        result = process_chunk(model, chunk)

        print(result)
        
        # Update metrics
        metrics['total_duration'] += result['duration']
        metrics['total_process_time'] += result['process_time']
        metrics['total_latency'] += result['process_time']
        metrics['total_rtf'] += result['rtf']
        metrics['total_confidence'] += result['confidence']
        metrics['file_count'] += 1
        
        # Update peak resources
        metrics['peak_memory'] = max(metrics['peak_memory'], get_peak_memory() - start_memory)
        metrics['peak_cpu'] = max(metrics['peak_cpu'], get_peak_cpu() - start_cpu)
        
        with open("transcribe.txt", "a") as f:
            f.write(result['text'] + "\n")
    
    # Calculate final metrics
    if metrics['file_count'] > 0:
        print("\n=== PROCESSING COMPLETE (CPU ONLY) ===")
        print(f" - Total Audio Duration: {metrics['total_duration']:.2f} seconds")
        print(f" - Total Processing Time: {metrics['total_process_time']:.2f} seconds")
        
        if metrics['total_duration'] > 0:
            overall_rtf = metrics['total_process_time'] / metrics['total_duration']
            print(f" - Overall RTF: {overall_rtf:.2f} (lower is better)")
            print(f" - Processing Speed: {metrics['total_duration'] / metrics['total_process_time']:.2f}x real-time")
        
        print(f" - Avg Latency: {metrics['total_latency'] / metrics['file_count']:.2f} seconds")
        print(f" - Avg RTF: {metrics['total_rtf'] / metrics['file_count']:.2f}")
        print(f" - Avg Confidence: {metrics['total_confidence'] / metrics['file_count']:.2f}")
        print(f" - Peak CPU Usage: {metrics['peak_cpu']:.1f}%")
        print(f" - Peak Memory Usage: {metrics['peak_memory']:.2f} MB")
    else:
        print("No chunks were processed.")

if __name__ == "__main__":
    main()