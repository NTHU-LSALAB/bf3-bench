#!/usr/bin/env python3
"""
WordPiece Tokenization Benchmark

Benchmarks HuggingFace WordPiece tokenizer on any platform (DPU ARM or Host CPU).
Uses the same code for fair comparison.

Usage:
    python wordpiece_benchmark.py [--payload-kb 8] [--iterations 10]
"""

import time
import argparse
import random
import string
import platform

# Import HuggingFace tokenizers
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("ERROR: HuggingFace tokenizers not installed.")
    print("Install with: pip install tokenizers")
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


def benchmark_wordpiece(text, iterations=10, warmup=3, tokenizer_path=None):
    """Benchmark WordPiece tokenization"""
    # Load BERT tokenizer
    if tokenizer_path:
        print(f"Loading BERT tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Loading BERT tokenizer from HuggingFace hub...")
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        tokenizer.encode(text)

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    token_count = 0

    for i in range(iterations):
        start = time.perf_counter()
        result = tokenizer.encode(text)
        elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
        times.append(elapsed)
        token_count = len(result.ids)

    return {
        'mean_us': sum(times) / len(times),
        'min_us': min(times),
        'max_us': max(times),
        'std_us': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'tokens': token_count,
        'times': times,
    }


def main():
    parser = argparse.ArgumentParser(description="WordPiece Tokenization Benchmark")
    parser.add_argument('--payload-kb', type=int, default=8,
                        help='Payload size in KB (default: 8)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations (default: 3)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer.json file (default: download from HuggingFace)')
    args = parser.parse_args()

    # System info
    arch = platform.machine()
    processor = platform.processor() or arch

    print("=" * 60)
    print("WordPiece Tokenization Benchmark")
    print("=" * 60)
    print(f"Platform: {arch} ({processor})")
    print(f"Payload: {args.payload_kb} KB")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print("=" * 60)

    # Generate test text
    print("\nGenerating test text...")
    random.seed(42)  # Fixed seed for reproducibility
    text = generate_test_text(args.payload_kb)
    print(f"Text size: {len(text)} bytes ({len(text)/1024:.1f} KB)")

    # Run benchmark
    print()
    result = benchmark_wordpiece(text, args.iterations, args.warmup, args.tokenizer)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Platform:   {arch}")
    print(f"Algorithm:  WordPiece (BERT)")
    print(f"Library:    HuggingFace Tokenizers (Rust)")
    print(f"Tokens:     {result['tokens']}")
    print("-" * 60)
    print(f"Mean:       {result['mean_us']:.0f} µs")
    print(f"Min:        {result['min_us']:.0f} µs")
    print(f"Max:        {result['max_us']:.0f} µs")
    print(f"Std:        {result['std_us']:.0f} µs")
    print("=" * 60)

    # Machine-readable output
    print(f"\nWORDPIECE_TIME: {result['mean_us']:.2f}")
    print(f"WORDPIECE_TOKENS: {result['tokens']}")


if __name__ == "__main__":
    main()
