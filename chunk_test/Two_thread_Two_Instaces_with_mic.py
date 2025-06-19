#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torchaudio
import psutil
import numpy as np
from typing import List, Tuple, Optional
import warnings
import whisper
import threading
import queue
import logging
import sounddevice as sd
from dataclasses import dataclass
from collections import deque

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [Thread %(threadName)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Force CPU usage and suppress CUDA-related warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WHISPER_FORCE_CPU"] = "true"
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp: float

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stream = None
        self.recording = False
        self.audio_buffer = []
        self.start_time = 0

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        if self.recording:  # Only buffer if we're actually recording
            self.audio_buffer.append(indata.copy())

    def start_recording(self):
        if self.recording:
            return
            
        self.recording = True
        self.audio_buffer = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=self.audio_callback
        )
        self.start_time = time.time()
        self.stream.start()
        logging.info("Microphone recording started...")

    def stop_recording(self):
        if not self.recording:
            return None
            
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            chunk = AudioChunk(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=self.start_time
            )
            self.audio_buffer = []  # Clear the buffer after creating the chunk
            self.audio_queue.put(chunk)  # Add chunk to the queue for processing
            return chunk
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time audio transcription using Whisper with chunking (CPU only)')
    parser.add_argument('--chunk_size', type=int, default=5, help='Chunk size in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--channels', type=int, default=1, help='Number of audio channels')
    parser.add_argument('--num_threads', type=int, default=2, help='Number of transcription threads')
    parser.add_argument('--device', type=int, default=None, help='Input device (numeric ID or substring)')
    return parser.parse_args()

def get_peak_memory() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_peak_cpu() -> float:
    return psutil.cpu_percent()

def process_chunk(model, chunk: Tuple[torch.Tensor, int]) -> dict:
    waveform, sample_rate = chunk
    audio_np = waveform.squeeze().numpy().astype(np.float32)

    start_time = time.time()
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

def record_audio_chunks(recorder: AudioRecorder, chunk_size: int, stop_event: threading.Event) -> List[AudioChunk]:
    chunks = []
    try:
        logging.info("Press Ctrl+C to stop recording...")
        
        while not stop_event.is_set():
            # Start a new recording
            recorder.start_recording()
            start_time = time.time()
            
            # Sleep for chunk_size seconds, but check stop_event frequently
            while time.time() - start_time < chunk_size and not stop_event.is_set():
                time.sleep(0.1)  # Check stop_event every 100ms
            
            if stop_event.is_set():
                break
                
            # Stop recording to get the chunk
            chunk = recorder.stop_recording()
            if chunk and len(chunk.data) > 0:
                chunk_duration = len(chunk.data) / chunk.sample_rate
                chunks.append(chunk)
                logging.info(f"Recorded chunk {len(chunks)}: {chunk_duration:.2f}s")
                
    except KeyboardInterrupt:
        logging.info("Stopping recording...")
    finally:
        # Get any remaining audio
        final_chunk = recorder.stop_recording()
        if final_chunk and len(final_chunk.data) > 0:
            chunks.append(final_chunk)
            logging.info(f"Recorded final chunk {len(chunks)}: {len(final_chunk.data)/final_chunk.sample_rate:.2f}s")
        
    return chunks

def worker(thread_id: int, input_queue: queue.Queue, output_queue: queue.Queue, model):
    logging.info(f"Thread {thread_id}: Starting worker with pre-loaded model")

    while True:
        chunk_data = input_queue.get()
        if chunk_data is None:
            break

        chunk_index, chunk = chunk_data
        logging.info(f"Thread {thread_id}: Processing chunk {chunk_index}...")

        try:
            result = process_chunk(model, chunk)
        except Exception as e:
            logging.error(f"Thread {thread_id}: Error processing chunk {chunk_index}: {str(e)}")
            output_queue.put((chunk_index, {
                'text': f"[Error: {str(e)}]",
                'duration': 0,
                'process_time': 0,
                'confidence': -1,
                'rtf': 0
            }))
        else:
            # Log the result summary (not writing to file)
            logging.info(
                f"[Chunk {chunk_index + 1}] Text: \"{result['text']}\" | "
                f"Duration: {result['duration']:.2f}s | Time: {result['process_time']:.2f}s | "
                f"RTF: {result['rtf']:.2f} | Confidence: {result['confidence']:.2f}"
            )

            output_queue.put((chunk_index, result))
        finally:
            input_queue.task_done()
            logging.info(f"Thread {thread_id}: Done chunk {chunk_index}...")
    logging.info(f"Thread {thread_id}: Exiting.")


def main():
    args = parse_args()
    
    # List available audio devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']})")
    
    if args.device is not None:
        print(f"\nUsing device: {devices[args.device]['name']}")
        sd.default.device = args.device

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

    start_memory = get_peak_memory()
    start_cpu = get_peak_cpu()

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Initialize Whisper models for each thread
    logging.info(f"Loading {args.num_threads} Whisper model(s)...")
    models = []
    for i in range(args.num_threads):
        logging.info(f"Loading model {i+1}/{args.num_threads}...")
        model = whisper.load_model("medium", device="cpu")
        models.append(model)
    logging.info("All models loaded successfully")
    
    # Create and start worker threads
    threads = []
    for i in range(args.num_threads):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, input_queue, output_queue, models[i]),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Create and start audio recorder
    recorder = AudioRecorder(
        sample_rate=args.sample_rate,
        channels=args.channels
    )
    
    # Start recording in a separate thread
    logging.info("Starting recording thread...")
    recording_thread = threading.Thread(
        target=record_audio_chunks,
        args=(recorder, args.chunk_size, stop_event),
        daemon=True
    )
    recording_thread.start()
    
    # Give recording thread a moment to start
    time.sleep(1)
    
    # Process recorded chunks from the queue
    try:
        while not stop_event.is_set():
            # Wait for chunks to be available
            time.sleep(0.1)  # Small sleep to prevent busy waiting
            
            # Process any available chunks
            while not recorder.audio_queue.empty() and not stop_event.is_set():
                chunk = recorder.audio_queue.get()
                if chunk and len(chunk.data) > 0:
                    waveform = torch.from_numpy(chunk.data.T).float()
                    input_queue.put((metrics['file_count'], (waveform, chunk.sample_rate)))
                    metrics['file_count'] += 1
    except KeyboardInterrupt:
        logging.info("Stopping recording...")
        stop_event.set()
    except Exception as e:
        logging.error(f"Error in main processing loop: {str(e)}")
        stop_event.set()
    
    total_chunks = metrics['file_count']
    logging.info(f"Recorded {total_chunks} chunks for processing")

    threads = []
    for i in range(args.num_threads):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, input_queue, output_queue)
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

    # Process results as they come in
    processed_results = [None] * total_chunks
    results_received = 0
    
    # Start worker threads
    threads = []
    for i in range(args.num_threads):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, input_queue, output_queue),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Process results in real-time
    with open("transcribe.txt", "w") as f:
        while results_received < total_chunks and not stop_event.is_set():
            try:
                chunk_index, result = output_queue.get(timeout=1.0)
                if result:
                    print(f"\nResult for chunk {chunk_index + 1}:")
                    print(result['text'])
                    f.write(result['text'] + "\n")
                    f.flush()  # Ensure text is written to file immediately
                    
                    metrics['total_duration'] += result['duration']
                    metrics['total_process_time'] += result['process_time']
                    metrics['total_latency'] += result['process_time']
                    metrics['total_rtf'] += result['rtf']
                    metrics['total_confidence'] += result['confidence']
                    results_received += 1
                    metrics['peak_memory'] = max(metrics['peak_memory'], get_peak_memory() - start_memory)
                    metrics['peak_cpu'] = max(metrics['peak_cpu'], get_peak_cpu() - start_cpu)
                    
            except queue.Empty:
                continue
    
    # Clean up
    stop_event.set()
    for _ in range(args.num_threads):
        input_queue.put(None)
    
    for thread in threads:
        thread.join(timeout=1.0)
    
    recording_thread.join(timeout=1.0)

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
