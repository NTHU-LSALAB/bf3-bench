# BF3-Bench: DPU Tokenization Offload for LLM Inference

Benchmark for evaluating tokenization offloading strategies using NVIDIA BlueField-3 DPU with GPUDirect RDMA.

---

## Background

Large Language Model (LLM) inference pipelines require tokenization—converting raw text into token IDs—before model computation. This preprocessing stage traditionally runs on CPU or GPU.

This project explores offloading tokenization to the NVIDIA BlueField-3 DPU. By performing tokenization on the DPU's ARM cores before transferring data via RDMA, GPU resources can be reserved for inference workloads.

---

## System Architecture

| Component | Specification |
|:----------|:--------------|
| DPU | NVIDIA BlueField-3 (ARM Cortex-A78, 16 cores) |
| Host CPU | x86_64 (benchmark comparison) |
| GPU | NVIDIA A100X (ConnectX-integrated) |
| Network | PCIe Gen4 x16, GPUDirect RDMA |

```
┌────────────────────┐                    ┌─────────────────────────────────┐
│   BlueField-3 DPU  │   PCIe Gen4 x16    │         Host System             │
│    (ARM Cores)     │◄──────────────────►│  ┌─────────┐    ┌───────────┐  │
│   192.168.200.2    │  GPUDirect RDMA    │  │ Host CPU│    │ A100X GPU │  │
└────────────────────┘                    │  └─────────┘    └───────────┘  │
                                          │  192.168.200.1                  │
                                          └─────────────────────────────────┘
```

---

## Experimental Design

### Comparison Scenarios

| Scenario | Pipeline |
|:---------|:---------|
| **DPU-Based** | Tokenize on DPU ARM → RDMA (tokens) → GPU Embed |
| **CPU-Based** | Tokenize on Host CPU → Copy to GPU → GPU Embed |
| **GPU-Based** | RDMA (raw text) → Tokenize on GPU → GPU Embed |

### Tokenization Algorithms

Two algorithms were tested to evaluate different computational characteristics:

| Algorithm | Vocabulary | Nature | Compatible Models |
|:----------|:-----------|:-------|:------------------|
| **BPE** | 50,257 (GPT-2) | Sequential | GPT-2/3, LLaMA |
| **WordPiece** | 30,522 (BERT) | Parallelizable | BERT, DistilBERT |

### Test Configuration

- **Payload**: 8 KB text
- **Iterations**: 10 per configuration
- **Measurement**: `clock_gettime(CLOCK_MONOTONIC)` per stage

---

## Results

### Three-Way Tokenization Comparison (8KB Payload)

![Tokenization Comparison](charts/tokenization_comparison_3way.png)

| Algorithm | DPU ARM | Host CPU | GPU | Best Platform |
|:----------|:--------|:---------|:----|:--------------|
| **BPE** | 531 µs | 2,680 µs | 9,235 µs | **DPU (17× vs GPU, 5× vs CPU)** |
| **WordPiece** | 1,275 µs | 4,756 µs | 1,316 µs | **DPU/GPU (~equal)** |

### BPE Speedup Analysis

![BPE Speedup](charts/bpe_speedup_3way.png)

For sequential BPE tokenization (GPT-2/LLaMA models):
- **DPU ARM is 17× faster than GPU** (sequential CUDA kernel)
- **DPU ARM is 5× faster than Host CPU** (same HuggingFace backend)

### Pipeline Timeline (BPE)

![BPE Pipeline](charts/bpe_pipeline_3way.png)

| Stage | DPU-Based | CPU-Based | GPU-Based |
|:------|:----------|:----------|:----------|
| Tokenization | 531 µs | 2,680 µs | 9,235 µs |
| RDMA Transfer | 1,011 µs | 1,011 µs | 1,011 µs |
| GPU Embedding | 46 µs | 46 µs | 46 µs |
| **Total** | **1,588 µs** | **3,737 µs** | **10,292 µs** |

### End-to-End Latency Breakdown

![Latency Breakdown](charts/latency_breakdown_3way.png)

### WordPiece Comparison

![WordPiece Comparison](charts/wordpiece_comparison_3way.png)

For parallelizable WordPiece tokenization (BERT models):
- DPU and GPU achieve similar performance (~1,300 µs)
- Host CPU is slower (4,756 µs) due to lack of GPU acceleration

---

## Implementation Details

### Tokenizers

| Platform | Algorithm | Implementation | Time (8KB) |
|:---------|:----------|:---------------|:-----------|
| **DPU ARM** | BPE | C sequential (`bpe_tokenizer.c`) | 531 µs |
| **DPU ARM** | WordPiece | HuggingFace Tokenizers (Rust) | 1,275 µs |
| **Host CPU** | BPE | HuggingFace Tokenizers (Rust) | 2,680 µs |
| **Host CPU** | WordPiece | HuggingFace Tokenizers (Rust) | 4,756 µs |
| **GPU** | BPE | CUDA sequential (`tokenizer_kernel.cu`) | 9,235 µs |
| **GPU** | WordPiece | RAPIDS nvtext | 1,316 µs |

### DPU BPE Tokenizer

```c
// src/dpu_client/bpe_tokenizer.c
BPEContext ctx;
bpe_init(&ctx, "vocab.json", "merges.txt");  // 50,257 tokens, 49,992 merges
int num_tokens = bpe_encode(&ctx, text, text_len, output_ids, max_len);
```

### Host CPU Benchmark

```python
# scripts/cpu_tokenizer_benchmark.py
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")  # or "bert-base-uncased"
result = tokenizer.encode(text)
```

### GPU Tokenizers

```python
# WordPiece: RAPIDS nvtext (GPU-accelerated)
from pylibcudf.nvtext.wordpiece_tokenize import wordpiece_tokenize
result = wordpiece_tokenize(input_col, vocab, max_sequence_length)
```

```cuda
// BPE: src/host_server/tokenizer_kernel.cu (sequential, single thread)
__global__ void bpe_tokenize_kernel(const char* input, int32_t* output, ...);
```

### Embedding Kernel

```cuda
// src/host_server/embedding_kernel.cu
output[i] = word_embed[token_id] + pos_embed[position];
```

---

## Key Findings

1. **BPE on DPU is optimal**: 17× faster than GPU, 5× faster than Host CPU
   - Sequential algorithm doesn't benefit from GPU parallelism
   - DPU ARM cores with optimized C code outperform all alternatives

2. **WordPiece: DPU ≈ GPU**: Both achieve ~1,300 µs
   - Parallelizable algorithm benefits from GPU (RAPIDS nvtext)
   - Choose based on system architecture preferences

3. **Recommendation**:
   - **GPT-2/LLaMA models**: Use DPU-based tokenization
   - **BERT models**: Either DPU or GPU works well

---

## Build & Run

See [docs/Build.md](docs/Build.md) for repository structure, build instructions, and benchmark execution.

### Quick CPU Benchmark

```bash
python scripts/cpu_tokenizer_benchmark.py --payload-kb 8 --iterations 10
```

---

## About

**National Tsing Hua University (NTHU)**
**Large-Scale System Architecture Laboratory (LSAlab)**

Advisor: Prof. Jerry (Chi-Yuan) Chou (周志遠) - jchou@cs.nthu.edu.tw

This code is provided for academic and research purposes.
