# BF3-Bench Benchmark Results

Complete benchmark results for tokenization comparison across platforms.

## Test Configuration

| Parameter | Value |
|:----------|:------|
| Payload Size | 8 KB |
| Iterations | 10 |
| Warmup | 100 (C), 3 (Python) |
| Timing | `clock_gettime(CLOCK_MONOTONIC)` / `time.perf_counter()` |

---

## BPE Tokenization (Fair Comparison)

**All platforms use the same Greedy Longest-Match algorithm** for fair comparison.

### Algorithm Description

The Greedy Longest-Match algorithm:
1. Split text into words at whitespace
2. For each word, try to find it in vocabulary
3. If not found, try progressively shorter substrings
4. Fall back to character-level tokenization

### Results

| Platform | Implementation | Time (µs) | Tokens | Speedup vs GPU |
|:---------|:---------------|----------:|-------:|---------------:|
| **Host CPU (x86-64)** | C Greedy | 332 | 5,481 | **28×** |
| **DPU ARM (Cortex-A78)** | C Greedy | 531 | 5,481 | **17×** |
| **GPU (A100X)** | CUDA Greedy | 9,235 | 5,481 | 1× |

### Analysis

1. **Host CPU is fastest** for sequential work (better single-thread performance)
   - Higher clock speed (~3.0+ GHz vs 2.0 GHz ARM)
   - Better branch prediction and larger cache
   - x86 is 1.6× faster than ARM for same algorithm

2. **GPU is 28× slower** than CPU for sequential work
   - CUDA kernel runs with single thread `<<<1, 1>>>`
   - GPU architecture is optimized for SIMD, not sequential
   - Memory access patterns inefficient for sequential algorithm

3. **DPU advantage is offloading**, not raw performance
   - Frees CPU resources for other tasks
   - Direct RDMA to GPU without host involvement
   - Lower end-to-end latency in offload scenarios

---

## WordPiece Tokenization

**Different implementations on each platform** (no simple greedy equivalent).

### Results

| Platform | Implementation | Time (µs) | Tokens | Notes |
|:---------|:---------------|----------:|-------:|:------|
| **DPU ARM** | HuggingFace (Rust) | 1,275 | ~2,000 | **Needs verification** |
| **GPU** | RAPIDS nvtext | 1,316 | ~2,000 | GPU-accelerated |
| **Host CPU** | HuggingFace (Rust) | 4,768 | ~2,000 | Actual benchmark |

### Analysis

1. **GPU uses RAPIDS nvtext** (1,316 µs) - GPU-accelerated WordPiece

2. **DPU number (1,275 µs) needs verification**
   - No DPU WordPiece benchmark exists in codebase
   - Number may be estimated or from previous test conditions

3. **Host CPU (4,768 µs) uses HuggingFace**
   - Actual benchmark result from this codebase
   - If DPU uses same library, Host CPU should be faster (x86 > ARM)

---

## HuggingFace BPE Comparison (Reference)

For reference, HuggingFace's full BPE implementation (with merge rules):

| Platform | Implementation | Time (µs) |
|:---------|:---------------|----------:|
| Host CPU | HuggingFace (Rust) | 2,631 |
| DPU ARM | HuggingFace (Rust) | ~2,600 (estimated) |

Note: HuggingFace BPE is ~8× slower than Greedy due to merge operations.

---

## Pipeline End-to-End Latency

Complete pipeline with tokenization + RDMA + GPU embedding:

| Scenario | Tokenization | RDMA | GPU Embed | **Total** |
|:---------|-------------:|-----:|----------:|----------:|
| CPU-Based | 332 µs | 1,011 µs | 46 µs | **1,389 µs** |
| DPU-Based | 531 µs | 1,011 µs | 46 µs | **1,588 µs** |
| GPU-Based | 9,235 µs | 1,011 µs | 46 µs | **10,292 µs** |

---

## Key Findings

### 1. Sequential Algorithm Performance

For sequential algorithms like Greedy BPE:
```
Host CPU (x86) > DPU ARM > GPU
    332 µs       531 µs    9,235 µs
```

### 2. When to Use Each Platform

| Use Case | Recommended | Reason |
|:---------|:------------|:-------|
| Lowest latency | Host CPU | Best single-thread perf |
| CPU offloading | DPU | Frees CPU resources |
| Parallel algorithms | GPU | SIMD acceleration |
| RDMA integration | DPU | Direct network-to-GPU |

### 3. GPU is Bad for Sequential Work

Never use GPU for sequential tokenization:
- 28× slower than CPU
- 17× slower than DPU
- Wastes GPU resources that could run inference

---

## Files

| File | Description |
|:-----|:------------|
| `benchmarks/host_cpu_greedy_tokenizer.c` | Host CPU greedy benchmark |
| `src/dpu_client/bpe_tokenizer.c` | DPU ARM greedy tokenizer |
| `src/host_server/tokenizer_kernel.cu` | GPU CUDA greedy tokenizer |
| `scripts/cpu_tokenizer_benchmark.py` | HuggingFace benchmark |
| `scripts/generate_charts_v4.py` | Chart generator |

---

## Running Benchmarks

### Host CPU Greedy (C)
```bash
cd benchmarks
gcc -O3 -o host_cpu_greedy_tokenizer host_cpu_greedy_tokenizer.c
./host_cpu_greedy_tokenizer ../vocab/vocab.json --payload-kb 8 --iterations 10
```

### Host CPU HuggingFace (Python)
```bash
python scripts/cpu_tokenizer_benchmark.py --payload-kb 8 --iterations 10
```

### DPU (requires BlueField-3)
```bash
ssh dpu
./dpu_client --benchmark --payload-kb 8
```

### GPU (requires CUDA)
```bash
./gpu_tokenizer_benchmark --payload-kb 8
```
