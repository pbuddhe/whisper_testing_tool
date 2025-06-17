import os
import time
import re
import warnings
import numpy as np
import whisper
import torch
import torchaudio
import psutil
from jiwer import wer, cer, process_words
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import single_meteor_score
import nltk
from glob import glob
from typing import Dict, Tuple

# Force CPU usage and suppress CUDA-related warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
os.environ["WHISPER_FORCE_CPU"] = "true"  # Force CPU for Whisper
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Download required NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def calculate_punctuation_accuracy(reference, hypothesis):
    # Extract punctuation from reference and hypothesis
    ref_punct = re.findall(r'[^\w\s]', reference)
    hyp_punct = re.findall(r'[^\w\s]', hypothesis)
    
    # Calculate precision and recall for punctuation
    if not ref_punct and not hyp_punct:
        return 1.0, 1.0, 1.0  # Perfect score if no punctuation in either
    
    # Convert to sets for comparison
    ref_set = set((i, p) for i, p in enumerate(ref_punct))
    hyp_set = set((i, p) for i, p in enumerate(hyp_punct[:len(ref_punct)]))  # Align by position
    
    # Calculate matches
    matches = ref_set & hyp_set
    
    precision = len(matches) / len(hyp_punct) if hyp_punct else 0
    recall = len(matches) / len(ref_punct) if ref_punct else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_semantic_similarity(reference, hypothesis):
    # Tokenize sentences
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Calculate BLEU score
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    # Calculate METEOR score with proper tokenization
    meteor = single_meteor_score(ref_tokens, hyp_tokens)
    
    return {
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'meteor': meteor
    }

def get_process_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / (1024 ** 2),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 ** 2),  # Virtual Memory Size
        'cpu_percent': process.cpu_percent(interval=0.1)
    }

def monitor_resources() -> Dict[str, float]:
    """Monitor system and process resources."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    virtual_memory = psutil.virtual_memory()
    
    process_info = get_process_memory_usage()
    
    return {
        'system_cpu_percent': cpu_percent,
        'system_memory_percent': virtual_memory.percent,
        'system_memory_used_gb': virtual_memory.used / (1024 ** 3),
        **process_info,
    }

def print_resource_usage(metrics: Dict[str, float], prefix: str = ""):
    """Print resource usage metrics in a readable format."""
    print(f"\n{prefix}Resource Usage:")
    print("  System:")
    print(f"    CPU: {metrics['system_cpu_percent']:.1f}%")
    print(f"    Memory: {metrics['system_memory_used_gb']:.2f} GB used ({metrics['system_memory_percent']:.1f}%)")
    
    print("\n  Process:")
    print(f"    CPU: {metrics['cpu_percent']:.1f}%")
    print(f"    Memory: {metrics['rss_mb']:.2f} MB (RSS), {metrics['vms_mb']:.2f} MB (VMS)")

# ---------- CONFIG ----------
AUDIO_DIR = "Dataset_1"  # directory of .wav files
TRANSCRIPT_FILE = "Dataset_1/transcripts.txt"  # tab-separated: filename<TAB>ground_truth
MODEL_SIZE = "medium"  # tiny, base, small, medium, large
# ----------------------------

# Load model with resource monitoring
print(f"Loading Whisper model ({MODEL_SIZE})...")
print("\n=== RESOURCE USAGE BEFORE MODEL LOAD ===")
before_load = monitor_resources()
print_resource_usage(before_load)

# Ensure we're using CPU
if torch.cuda.is_available():
    torch.cuda.is_available = lambda: False

start_time = time.time()
model = whisper.load_model(MODEL_SIZE, device="cpu")
model = model.to("cpu")  # Ensure model is on CPU
end_time = time.time()
latency = end_time - start_time

print("\n=== RESOURCE USAGE AFTER MODEL LOAD ===")
after_load = monitor_resources()
print_resource_usage(after_load)

print(f"\nModel load time: {latency:.2f} seconds")

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
total_wec, total_cec = 0, 0
total_precision, total_recall, total_f1 = 0, 0, 0
total_bleu, total_rouge1, total_rouge2, total_rougeL, total_meteor = 0, 0, 0, 0, 0
total_confidence = 0
file_count = 0

# Resource monitoring metrics
peak_cpu = 0
peak_memory = 0
total_inference_time = 0
total_audio_duration = 0

# Process each audio file
for audio_file in glob(os.path.join(AUDIO_DIR, "*.wav")):
    filename = os.path.basename(audio_file)
    if filename not in references:
        continue

    # Load audio to get duration
    waveform, sample_rate = torchaudio.load(audio_file)
    duration = waveform.shape[1] / sample_rate

    # Transcribe with Whisper and monitor resources
    print(f"\n=== PROCESSING: {filename} ===")
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Get resources before transcription
    before_transcribe = monitor_resources()
    
    # Perform transcription
    start_time = time.time()
    result = model.transcribe(audio_file)
    end_time = time.time()
    
    # Get resources after transcription
    after_transcribe = monitor_resources()
    
    # Calculate resource usage during transcription
    transcribe_time = end_time - start_time
    cpu_usage = after_transcribe['cpu_percent']
    memory_usage = after_transcribe['rss_mb'] - before_transcribe['rss_mb']
    
    # Update peak metrics
    peak_cpu = max(peak_cpu, cpu_usage)
    peak_memory = max(peak_memory, after_transcribe['rss_mb'])
    
    # Update total inference time and audio duration for RTF calculation
    total_inference_time += transcribe_time
    total_audio_duration += duration
    
    print(f"\nTranscription completed in {transcribe_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.1f}%")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    predicted = result["text"].strip().lower()
    reference = references[filename].strip().lower()

    # Basic metrics
    latency = end_time - start_time
    rtf = latency / duration
    
    # Calculate WER, CER, WEC, CEC
    file_wer = wer(reference, predicted)
    file_cer = cer(reference, predicted)
    
    # Calculate Word Error Count (WEC) and Character Error Count (CEC)
    from jiwer import process_words, process_characters
    
    # Calculate WER and get error counts
    wer_result = process_words(reference, predicted)
    file_wec = wer_result.substitutions + wer_result.deletions + wer_result.insertions
    
    # Calculate CER and get error counts
    cer_result = process_characters(reference, predicted)
    file_cec = cer_result.substitutions + cer_result.deletions + cer_result.insertions
    
    # Punctuation accuracy
    punct_precision, punct_recall, punct_f1 = calculate_punctuation_accuracy(reference, predicted)
    
    # Semantic similarity metrics
    semantic_scores = calculate_semantic_similarity(reference, predicted)
    
    # Initialize optional metrics
    file_language = result.get("language", "")
    file_confidence = 0.0
    
    # Get confidence from segments if available
    if "segments" in result and len(result["segments"]) > 0:
        file_confidence = result["segments"][0].get("avg_logprob", 0.0)
    
    # Print detailed metrics for this file
    print(f"\n[{filename}]")
    print(f"  - WER: {file_wer:.2f}, WEC: {file_wec}")
    print(f"  - CER: {file_cer:.2f}, CEC: {file_cec}")
    print(f"  - Latency: {latency:.2f}s, RTF: {rtf:.2f}")
    print(f"  - Punctuation (P/R/F1): {punct_precision:.2f}/{punct_recall:.2f}/{punct_f1:.2f}")
    print(f"  - BLEU: {semantic_scores['bleu']:.2f}, METEOR: {semantic_scores['meteor']:.2f}")
    print(f"  - ROUGE-1/2/L: {semantic_scores['rouge1']:.2f}/{semantic_scores['rouge2']:.2f}/{semantic_scores['rougeL']:.2f}")
    print(f"  - Language: {file_language}, Confidence: {file_confidence:.2f}")
          
    # Update totals
    total_wer += file_wer
    total_cer += file_cer
    total_wec += file_wec
    total_cec += file_cec
    total_latency += latency
    total_rtf += rtf
    total_precision += punct_precision
    total_recall += punct_recall
    total_f1 += punct_f1
    total_bleu += semantic_scores['bleu']
    total_rouge1 += semantic_scores['rouge1']
    total_rouge2 += semantic_scores['rouge2']
    total_rougeL += semantic_scores['rougeL']
    total_meteor += semantic_scores['meteor']
    total_confidence += file_confidence
    file_count += 1

# Averages and resource summary
if file_count > 0:
    print("\n=== AVERAGE PERFORMANCE ===")
    print(f"Word-level Metrics:")
    print(f"  - WER: {total_wer / file_count:.2f}, Avg WEC: {total_wec / file_count:.1f}")
    print(f"  - CER: {total_cer / file_count:.2f}, Avg CEC: {total_cec / file_count:.1f}")
    
    print("\nPunctuation Accuracy:")
    print(f"  - Precision: {total_precision / file_count:.2f}")
    print(f"  - Recall: {total_recall / file_count:.2f}")
    print(f"  - F1: {total_f1 / file_count:.2f}")
    
    print("\nSemantic Similarity:")
    print(f"  - BLEU: {total_bleu / file_count:.4f}")
    print(f"  - METEOR: {total_meteor / file_count:.4f}")
    print(f"  - ROUGE-1/2/L: {total_rouge1 / file_count:.4f}/{total_rouge2 / file_count:.4f}/{total_rougeL / file_count:.4f}")
    
    print("\nPerformance Metrics:")
    print(f"  - Avg Latency: {total_latency / file_count:.2f} seconds")
    print(f"  - Avg RTF: {total_rtf / file_count:.2f}")
    print(f"  - Avg Confidence: {total_confidence / file_count:.2f}")
    
    print("\nResource Usage Summary:")
    print(f"  - Peak CPU Usage: {peak_cpu:.1f}%")
    print(f"  - Peak Memory Usage: {peak_memory:.2f} MB")
    
    # Calculate and print real-time factor (RTF) metrics
    if total_audio_duration > 0:
        overall_rtf = total_inference_time / total_audio_duration
        print(f"\nEfficiency Metrics:")
        print(f"  - Total Audio Duration: {total_audio_duration:.2f} seconds")
        print(f"  - Total Processing Time: {total_inference_time:.2f} seconds")
        print(f"  - Overall RTF: {overall_rtf:.2f} (lower is better)")
        print(f"  - Processing Speed: {total_audio_duration / total_inference_time:.2f}x real-time")
else:
    print("No matching audio files with transcripts found.")
