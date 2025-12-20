# Build Instructions

## Repository Structure

```
bf3-bench/
├── README.md
├── vocab/                    # GPT-2 vocabulary files
│   ├── vocab.json            # Token-to-ID mapping (50,257 tokens)
│   └── merges.txt            # BPE merge rules (49,992 rules)
├── src/
│   ├── dpu_client/           # DPU-side code (ARM)
│   │   ├── dpu_client_sample.c
│   │   ├── dpu_client_main.c
│   │   ├── bpe_tokenizer.c   # Real GPT-2 BPE implementation
│   │   ├── bpe_tokenizer.h
│   │   ├── rdma_common.h
│   │   ├── common.c
│   │   └── meson.build
│   └── host_server/          # Host GPU code (x86_64 + CUDA)
│       ├── host/
│       │   └── gpunetio_rdma_client_server_write_sample.c
│       ├── device/
│       │   └── gpunetio_rdma_client_server_write_kernel.cu
│       ├── gpunetio_rdma_client_server_write_main.c
│       ├── tokenizer_kernel.cu
│       ├── tokenizer_common.h
│       ├── embedding_kernel.cu
│       ├── embedding_common.h
│       ├── rdma_common.c
│       ├── rdma_common.h
│       ├── common.c
│       └── meson.build
├── scripts/
│   └── wordpiece_tokenizer.py  # GPU WordPiece using RAPIDS nvtext
├── results/
│   └── *.csv                 # Benchmark results
├── charts/
│   └── *.png
└── docs/
    └── DPU_Tokenization_Offload_Report.md
```

---

## Requirements

- NVIDIA DOCA SDK 2.9 or later
- CUDA Toolkit 12.0 or later
- Meson 0.61+
- Ninja build system

---

## Building the Host Server

```bash
cd src/host_server
meson setup build
cd build
ninja
```

The output binary is `doca_gpunetio_rdma_client_server_write`.

---

## Building the DPU Client

On the DPU (access via SSH to 192.168.200.2):

```bash
cd src/dpu_client
meson setup build
cd build
ninja
```

The output binary is `dpu_rdma_client`.

---

## Running the Benchmark

### Quick Test

```bash
# Terminal 1 (Host): Start server
sudo PAYLOAD_SIZE_KB=8 ./doca_gpunetio_rdma_client_server_write \
    -d mlx5_2 -gpu 2a:00.0 --gid-index 1

# Terminal 2 (DPU via SSH): Run client
sudo PAYLOAD_SIZE_KB=8 ./dpu_rdma_client \
    -d mlx5_2 -s 192.168.200.1 -g 1
```

### Mode Selection

| Mode | Host Flag | DPU Flag | Description |
|:-----|:----------|:---------|:------------|
| DPU-Based | (default) | (default) | DPU tokenizes, GPU embeds only |
| GPU-Based | `--gpu-tokenize` | `--send-text` | DPU sends raw text, GPU tokenizes + embeds |

### Batch Processing

Both host and DPU support batch processing via the `BATCH_SIZE` environment variable:

```bash
# Terminal 1 (Host): Start server with batch_size=4
sudo PAYLOAD_SIZE_KB=8 BATCH_SIZE=4 ./doca_gpunetio_rdma_client_server_write \
    -d mlx5_2 -gpu 2a:00.0 --gid-index 1

# Terminal 2 (DPU via SSH): Run client with batch_size=4
sudo PAYLOAD_SIZE_KB=8 BATCH_SIZE=4 ./dpu_rdma_client \
    -d mlx5_2 -s 192.168.200.1 -g 1
```

| Parameter | Range | Description |
|:----------|:------|:------------|
| BATCH_SIZE | 1-32 | Number of sequences to process in parallel |

