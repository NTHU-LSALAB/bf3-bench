#!/usr/bin/env python3
"""
GPU WordPiece Tokenization Benchmark using RAPIDS nvtext

Usage:
    python wordpiece_benchmark_gpu.py [--payload-kb 8] [--iterations 10]
"""

import time
import argparse
import random
import string
import os

# Import RAPIDS
try:
    import cudf
    from pylibcudf.nvtext.wordpiece_tokenize import wordpiece_tokenize, WordPieceVocabulary
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    print("ERROR: cudf/pylibcudf not available")
    exit(1)


def generate_test_text(size_kb):
    """Generate random test text of specified size"""
    target_size = size_kb * 1024
    words = []
    current_size = 0

    while current_size < target_size:
        word_len = random.randint(4, 8)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        current_size += word_len + 1

    text = ' '.join(words)
    return text[:target_size]


def download_vocab(path):
    """Download BERT vocabulary"""
    import urllib.request
    url = "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
    print(f"Downloading BERT vocab to {path}...")
    urllib.request.urlretrieve(url, path)
    print("Done.")


def benchmark_wordpiece_gpu(text, vocab_path, iterations=10, warmup=3, max_seq_len=512):
    """Benchmark GPU WordPiece tokenization"""
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    vocab_series = cudf.Series(vocab_list)
    vocab = WordPieceVocabulary(vocab_series._column.to_pylibcudf(mode='read'))
    print(f"Vocab size: {len(vocab_series)}")

    # Prepare input
    series = cudf.Series([text])
    input_col = series._column.to_pylibcudf(mode='read')

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        result = wordpiece_tokenize(input_col, vocab, max_seq_len)

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []

    for i in range(iterations):
        # Synchronize GPU
        import cupy
        cupy.cuda.Device().synchronize()

        start = time.perf_counter()
        result = wordpiece_tokenize(input_col, vocab, max_seq_len)
        cupy.cuda.Device().synchronize()
        elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
        times.append(elapsed)

    # Get token count from last result
    try:
        if hasattr(result, 'values'):
            tokens = cudf.Series(result.values()).to_numpy()
        else:
            import numpy as np
            tokens = np.array(result.to_arrow().to_pylist(), dtype=np.int32).flatten()
        token_count = len(tokens)
    except:
        token_count = max_seq_len  # fallback

    return {
        'mean_us': sum(times) / len(times),
        'min_us': min(times),
        'max_us': max(times),
        'std_us': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'tokens': token_count,
        'times': times,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU WordPiece Tokenization Benchmark")
    parser.add_argument('--payload-kb', type=int, default=8,
                        help='Payload size in KB (default: 8)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations (default: 3)')
    parser.add_argument('--vocab', type=str, default='/tmp/bert_vocab.txt',
                        help='Path to BERT vocab.txt file')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU WordPiece Tokenization Benchmark (RAPIDS nvtext)")
    print("=" * 60)
    print(f"Payload: {args.payload_kb} KB")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print("=" * 60)

    # Download vocab if needed
    if not os.path.exists(args.vocab):
        download_vocab(args.vocab)

    # Generate test text
    print("\nGenerating test text...")
    random.seed(42)  # Fixed seed for reproducibility
    text = generate_test_text(args.payload_kb)
    print(f"Text size: {len(text)} bytes ({len(text)/1024:.1f} KB)")

    # Run benchmark
    print()
    result = benchmark_wordpiece_gpu(text, args.vocab, args.iterations, args.warmup)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Platform:   GPU (CUDA)")
    print(f"Algorithm:  WordPiece (BERT)")
    print(f"Library:    RAPIDS nvtext")
    print(f"Tokens:     {result['tokens']}")
    print("-" * 60)
    print(f"Mean:       {result['mean_us']:.0f} µs")
    print(f"Min:        {result['min_us']:.0f} µs")
    print(f"Max:        {result['max_us']:.0f} µs")
    print(f"Std:        {result['std_us']:.0f} µs")
    print("=" * 60)

    # Machine-readable output
    print(f"\nWORDPIECE_GPU_TIME: {result['mean_us']:.2f}")
    print(f"WORDPIECE_GPU_TOKENS: {result['tokens']}")


if __name__ == "__main__":
    main()
