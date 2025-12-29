#!/usr/bin/env python3
"""
Host CPU Tokenization Benchmark

Benchmarks tokenization on Host CPU using HuggingFace tokenizers (Rust backend).
This provides a third comparison point alongside DPU and GPU.

Tokenizers tested:
  - BPE (GPT-2): 50,257 vocab, sequential algorithm
  - WordPiece (BERT): 30,522 vocab, parallelizable algorithm

Usage:
    python cpu_tokenizer_benchmark.py [--payload-kb 8] [--iterations 10]
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

# Try importing tokenizers
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordPiece
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("ERROR: HuggingFace tokenizers not installed.")
    print("Install with: pip install tokenizers")
    sys.exit(1)


class CPUTokenizerBenchmark:
    """Benchmark Host CPU tokenization"""

    def __init__(self, vocab_dir=None):
        """
        Initialize tokenizers

        Args:
            vocab_dir: Directory containing vocab files (default: ../vocab)
        """
        if vocab_dir is None:
            vocab_dir = Path(__file__).parent.parent / "vocab"
        self.vocab_dir = Path(vocab_dir)

        self.bpe_tokenizer = None
        self.wordpiece_tokenizer = None

        self._init_bpe()
        self._init_wordpiece()

    def _init_bpe(self):
        """Initialize GPT-2 BPE tokenizer"""
        try:
            # Try loading from HuggingFace hub
            self.bpe_tokenizer = Tokenizer.from_pretrained("gpt2")
            print(f"BPE (GPT-2) tokenizer loaded from HuggingFace hub")
            print(f"  Vocab size: {self.bpe_tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"WARNING: Could not load GPT-2 tokenizer: {e}")
            self.bpe_tokenizer = None

    def _init_wordpiece(self):
        """Initialize BERT WordPiece tokenizer"""
        try:
            # Try loading from HuggingFace hub
            self.wordpiece_tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
            print(f"WordPiece (BERT) tokenizer loaded from HuggingFace hub")
            print(f"  Vocab size: {self.wordpiece_tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"WARNING: Could not load BERT tokenizer: {e}")
            self.wordpiece_tokenizer = None

    def generate_test_text(self, size_kb):
        """Generate random test text of specified size"""
        import random
        import string

        target_size = size_kb * 1024
        words = []
        current_size = 0

        while current_size < target_size:
            # Generate random word (4-8 chars)
            word_len = random.randint(4, 8)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
            words.append(word)
            current_size += word_len + 1  # +1 for space

        text = ' '.join(words)
        return text[:target_size]

    def benchmark_tokenizer(self, tokenizer, text, name, iterations=10, warmup=3):
        """
        Benchmark a single tokenizer

        Returns:
            dict with timing results
        """
        if tokenizer is None:
            return None

        # Warmup
        for _ in range(warmup):
            tokenizer.encode(text)

        # Benchmark
        times = []
        token_counts = []

        for _ in range(iterations):
            start = time.perf_counter()
            result = tokenizer.encode(text)
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            times.append(elapsed)
            token_counts.append(len(result.ids))

        return {
            'name': name,
            'platform': 'Host CPU',
            'implementation': 'HuggingFace Tokenizers (Rust)',
            'mean_us': sum(times) / len(times),
            'min_us': min(times),
            'max_us': max(times),
            'std_us': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            'tokens': token_counts[0],
            'iterations': iterations,
            'text_bytes': len(text),
        }

    def run_benchmark(self, payload_kb=8, iterations=10):
        """
        Run full benchmark suite

        Args:
            payload_kb: Text payload size in KB
            iterations: Number of iterations per test

        Returns:
            dict with all results
        """
        print(f"\n{'='*60}")
        print(f"Host CPU Tokenization Benchmark")
        print(f"{'='*60}")
        print(f"Payload: {payload_kb} KB")
        print(f"Iterations: {iterations}")
        print(f"{'='*60}\n")

        # Generate test text
        print("Generating test text...")
        text = self.generate_test_text(payload_kb)
        print(f"Text size: {len(text)} bytes ({len(text)/1024:.1f} KB)\n")

        results = {
            'payload_kb': payload_kb,
            'text_bytes': len(text),
            'iterations': iterations,
            'tokenizers': {}
        }

        # Benchmark BPE (GPT-2)
        print("Benchmarking BPE (GPT-2)...")
        bpe_result = self.benchmark_tokenizer(
            self.bpe_tokenizer, text, "BPE", iterations
        )
        if bpe_result:
            results['tokenizers']['bpe'] = bpe_result
            print(f"  Time: {bpe_result['mean_us']:.0f} +/- {bpe_result['std_us']:.0f} µs")
            print(f"  Tokens: {bpe_result['tokens']}")
        else:
            print("  SKIPPED (tokenizer not available)")

        # Benchmark WordPiece (BERT)
        print("\nBenchmarking WordPiece (BERT)...")
        wp_result = self.benchmark_tokenizer(
            self.wordpiece_tokenizer, text, "WordPiece", iterations
        )
        if wp_result:
            results['tokenizers']['wordpiece'] = wp_result
            print(f"  Time: {wp_result['mean_us']:.0f} +/- {wp_result['std_us']:.0f} µs")
            print(f"  Tokens: {wp_result['tokens']}")
        else:
            print("  SKIPPED (tokenizer not available)")

        return results

    def print_summary(self, results):
        """Print formatted summary"""
        print(f"\n{'='*60}")
        print("SUMMARY: Host CPU Tokenization ({} KB)".format(results['payload_kb']))
        print(f"{'='*60}")
        print(f"{'Tokenizer':<15} {'Time (µs)':<15} {'Tokens':<10} {'Implementation'}")
        print("-" * 60)

        for name, data in results['tokenizers'].items():
            print(f"{data['name']:<15} {data['mean_us']:>10.0f}     {data['tokens']:<10} {data['implementation']}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Host CPU Tokenization Benchmark")
    parser.add_argument('--payload-kb', type=int, default=8,
                        help='Payload size in KB (default: 8)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    # Run benchmark
    benchmark = CPUTokenizerBenchmark()
    results = benchmark.run_benchmark(args.payload_kb, args.iterations)
    benchmark.print_summary(results)

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    # Print machine-readable output for scripts
    if 'bpe' in results['tokenizers']:
        print(f"CPU_BPE_TIME: {results['tokenizers']['bpe']['mean_us']:.2f}")
    if 'wordpiece' in results['tokenizers']:
        print(f"CPU_WORDPIECE_TIME: {results['tokenizers']['wordpiece']['mean_us']:.2f}")


if __name__ == "__main__":
    main()
