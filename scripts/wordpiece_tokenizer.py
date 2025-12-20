#!/usr/bin/env python3
"""
GPU WordPiece Tokenizer using RAPIDS nvtext

This module provides GPU-accelerated WordPiece tokenization for BERT models.
It uses pylibcudf's nvtext to perform tokenization on GPU.

Performance: ~841 µs for 8KB text (vs 9,235 µs for naive BPE)

Usage:
    from wordpiece_tokenizer import WordPieceTokenizer
    tokenizer = WordPieceTokenizer()
    token_ids = tokenizer.tokenize("Hello world")
"""

import os
import time
import numpy as np

# Try importing cudf/pylibcudf
try:
    import cudf
    from pylibcudf.nvtext.wordpiece_tokenize import wordpiece_tokenize, WordPieceVocabulary
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    print("WARNING: cudf/pylibcudf not available, falling back to HuggingFace tokenizers")

# Fallback to HuggingFace tokenizers
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class WordPieceTokenizer:
    """GPU-accelerated WordPiece tokenizer"""
    
    def __init__(self, vocab_path=None, use_gpu=True):
        """
        Initialize WordPiece tokenizer
        
        Args:
            vocab_path: Path to BERT vocab.txt file (optional, will download if not provided)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.use_gpu = use_gpu and CUDF_AVAILABLE
        self.vocab = None
        self.hf_tokenizer = None
        
        # Default vocab path
        if vocab_path is None:
            vocab_path = "/tmp/bert_vocab.txt"
            if not os.path.exists(vocab_path):
                self._download_vocab(vocab_path)
        
        self.vocab_path = vocab_path
        
        if self.use_gpu:
            self._init_gpu_tokenizer()
        elif TOKENIZERS_AVAILABLE:
            self._init_hf_tokenizer()
        else:
            raise RuntimeError("No tokenizer backend available. Install cudf or tokenizers.")
    
    def _download_vocab(self, path):
        """Download BERT vocabulary"""
        import urllib.request
        url = "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
        print(f"Downloading BERT vocab to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Done.")
    
    def _init_gpu_tokenizer(self):
        """Initialize GPU-based tokenizer using pylibcudf nvtext"""
        # Read vocab line by line (cudf.read_csv has issues with special tokens)
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        
        vocab_series = cudf.Series(vocab_list)
        self.vocab = WordPieceVocabulary(vocab_series._column.to_pylibcudf(mode='read'))
        print(f"GPU WordPiece initialized with vocab size: {len(vocab_series)}")

    
    def _init_hf_tokenizer(self):
        """Initialize HuggingFace tokenizer (CPU fallback)"""
        self.hf_tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        print("HuggingFace WordPiece tokenizer initialized (CPU)")
    
    def tokenize(self, text, max_sequence_length=512):
        """
        Tokenize text into token IDs
        
        Args:
            text: Input text string or list of strings
            max_sequence_length: Maximum sequence length
            
        Returns:
            numpy array of token IDs
        """
        if self.use_gpu:
            return self._tokenize_gpu(text, max_sequence_length)
        else:
            return self._tokenize_hf(text)
    
    def _tokenize_gpu(self, text, max_sequence_length):
        """GPU tokenization using nvtext"""
        if isinstance(text, str):
            text = [text]
        
        series = cudf.Series(text)
        input_col = series._column.to_pylibcudf(mode='read')
        
        # Call nvtext wordpiece_tokenize
        result = wordpiece_tokenize(input_col, self.vocab, max_sequence_length)
        
        # Extract token IDs from result
        # Result is a ListColumn, we need to get the values
        if hasattr(result, 'values'):
            tokens_col = result.values()
            tokens = cudf.Series(tokens_col).to_numpy()
        else:
            # Fallback: try to convert directly
            tokens = np.array(result.to_arrow().to_pylist(), dtype=np.int32).flatten()
        
        return tokens
    
    def _tokenize_hf(self, text):
        """CPU tokenization using HuggingFace"""
        if isinstance(text, str):
            encoding = self.hf_tokenizer.encode(text)
            return np.array(encoding.ids, dtype=np.int32)
        else:
            encodings = [self.hf_tokenizer.encode(t) for t in text]
            return np.array([e.ids for e in encodings], dtype=np.int32)
    
    def benchmark(self, text, iterations=100):
        """Run tokenization benchmark"""
        # Warm up
        for _ in range(5):
            self.tokenize(text)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            result = self.tokenize(text)
        elapsed = (time.time() - start) / iterations * 1_000_000
        
        return {
            'time_us': elapsed,
            'num_tokens': len(result),
            'backend': 'GPU (nvtext)' if self.use_gpu else 'CPU (HuggingFace)'
        }


def main():
    """Test WordPiece tokenizer"""
    print("=" * 60)
    print("GPU WordPiece Tokenizer Test")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = WordPieceTokenizer()
    
    # Test data - 8KB
    text = "Hello world this is a test " * 300
    print(f"\nTest text: {len(text)} chars ({len(text)/1024:.1f} KB)")
    
    # Simple tokenization test
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")
    
    # Benchmark
    print("\nBenchmarking...")
    result = tokenizer.benchmark(text)
    print(f"Backend: {result['backend']}")
    print(f"Time: {result['time_us']:.0f} µs")
    print(f"Tokens: {result['num_tokens']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
