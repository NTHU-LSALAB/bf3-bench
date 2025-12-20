#include <stdio.h>
#include <cuda_runtime.h>
#include "embedding_common.h"
#include <doca_log.h>
#include <math.h>

DOCA_LOG_REGISTER(EMBED::KERNEL);

// BERT Base Params
#define VOCAB_SIZE 30522
#define HIDDEN_DIM 768
#define MAX_POS_EMBED 8192
#define TYPE_VOCAB_SIZE 2

// Tables
static float *d_word_table = NULL;
static float *d_pos_table = NULL;
static float *d_type_table = NULL;
static float *d_output = NULL;
// Layer Norm Params
static float *d_gamma = NULL;
static float *d_beta = NULL;

static int current_batch = 0;
static int current_seq = 0;

__global__ void bert_embedding_kernel(
    int32_t *token_ids,
    float *word_table,
    float *pos_table,
    float *type_table,
    float *gamma,
    float *beta,
    float *output,
    int batch_size,
    int seq_len,
    int hidden_dim)
{
    // Thread Layout: 
    // We parallelize over (Batch, Seq, Dim) or (Batch, Seq) -> Loop Dim?
    // HiddenDim 768 fits in a block? No, usually block is 256/512.
    // Let's use 1D Grid where each thread handles ONE element of one vector?
    // Total Elements = Batch * Seq * Dim.
    // idx = global thread index.
    
    int total_elements = batch_size * seq_len * hidden_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    int dim_idx = idx % hidden_dim;
    int token_idx = (idx / hidden_dim); // Index in Token Stream (Batch * Seq)
    int seq_pos = token_idx % seq_len;  // Position in Sequence (0..127)
    
    int32_t tid = token_ids[token_idx];
    if (tid >= VOCAB_SIZE) tid = 0; // Boundary check
    
    // 1. Lookup Sum
    float val = word_table[tid * hidden_dim + dim_idx] +
                pos_table[seq_pos * hidden_dim + dim_idx] + 
                type_table[0 * hidden_dim + dim_idx]; // Always Type 0 for simplicity
                
    // 2. Layer Norm (Naive Implementation - usually needs reduction within warp)
    // For strictly correct LayerNorm, we need Mean/Var of the *whole vector*.
    // Doing it per-thread is inefficient without shared mem.
    // BUT for "Simulating Calculation Load", we can approximate or use a simplified Norm.
    // Or we do a 2-pass kernel?
    // Let's just do a math-heavy operation per element to simulate cost.
    // E.g., sigmoid or exp?
    // Let's stick to true structure but assuming pre-calculated mean/var? No that's cheating.
    
    // Let's implement a lighter "Element-wise transform" that mimics LN cost.
    // val = (val - Beta) / Gamma * ... 
    // Let's just do: Output = val * Gamma + Beta (Scale/Shift only).
    // And add some math: output = tanh(val).
    
    float res = val * gamma[dim_idx] + beta[dim_idx];
    output[idx] = tanhf(res); // Activation
}

doca_error_t init_embedding_resources(void) {
    cudaError_t err;
    
    // 1. Alloc Tables
    size_t word_sz = VOCAB_SIZE * HIDDEN_DIM * sizeof(float);
    size_t pos_sz = MAX_POS_EMBED * HIDDEN_DIM * sizeof(float);
    size_t type_sz = TYPE_VOCAB_SIZE * HIDDEN_DIM * sizeof(float);
    size_t param_sz = HIDDEN_DIM * sizeof(float);
    
    err = cudaMalloc((void**)&d_word_table, word_sz);
    if(err) return DOCA_ERROR_NO_MEMORY;
    cudaMemset(d_word_table, 0, word_sz); // Zero init (simulated)

    err = cudaMalloc((void**)&d_pos_table, pos_sz);
    if(err) return DOCA_ERROR_NO_MEMORY;

    err = cudaMalloc((void**)&d_type_table, type_sz);
    if(err) return DOCA_ERROR_NO_MEMORY;
    
    err = cudaMalloc((void**)&d_gamma, param_sz);
    err = cudaMalloc((void**)&d_beta, param_sz);
    
    DOCA_LOG_INFO("Initialized BERT Embedding Tables (Vocab: %d, Dim: %d)", VOCAB_SIZE, HIDDEN_DIM);
    return DOCA_SUCCESS;
}

void cleanup_embedding_resources(void) {
    if (d_word_table) cudaFree(d_word_table);
    if (d_pos_table) cudaFree(d_pos_table);
    if (d_type_table) cudaFree(d_type_table);
    if (d_output) cudaFree(d_output);
    if (d_gamma) cudaFree(d_gamma);
    if (d_beta) cudaFree(d_beta);
}

static doca_error_t prepare_output(int BatchSize, int SeqLen) {
    if (d_output && current_batch == BatchSize && current_seq == SeqLen) return DOCA_SUCCESS;
    if (d_output) cudaFree(d_output);
    
    size_t out_sz = BatchSize * SeqLen * HIDDEN_DIM * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&d_output, out_sz);
    if (err != cudaSuccess) return DOCA_ERROR_NO_MEMORY;
    
    current_batch = BatchSize;
    current_seq = SeqLen;
    return DOCA_SUCCESS;
}

doca_error_t launch_bert_embedding(cudaStream_t stream, int32_t *d_token_ids, int BatchSize, int SeqLen) {
    if (!d_word_table) return DOCA_ERROR_INITIALIZATION;
    
    doca_error_t res = prepare_output(BatchSize, SeqLen);
    if (res != DOCA_SUCCESS) return res;
    
    int total_elements = BatchSize * SeqLen * HIDDEN_DIM;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    bert_embedding_kernel<<<blocks, threads, 0, stream>>>(
        d_token_ids,
        d_word_table,
        d_pos_table,
        d_type_table,
        d_gamma,
        d_beta,
        d_output,
        BatchSize,
        SeqLen,
        HIDDEN_DIM
    );
    
    if (cudaGetLastError() != cudaSuccess) return DOCA_ERROR_DRIVER;
    return DOCA_SUCCESS;
}
