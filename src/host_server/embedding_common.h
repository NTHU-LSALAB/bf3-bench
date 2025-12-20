#ifndef EMBEDDING_COMMON_H
#define EMBEDDING_COMMON_H

#include <doca_error.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Embedding tables (Word, Pos, Type)
doca_error_t init_embedding_resources(void);

// Destroy resources
void cleanup_embedding_resources(void);

// Launch BERT Embedding: Output = Norm(Word[id] + Pos[idx] + Type[0])
// Stream: CUDA Stream
// d_token_ids: Input IDs from DPU (BatchSize * SeqLen)
// BatchSize: num sequences
// SeqLen: tokens per sequence
doca_error_t launch_bert_embedding(cudaStream_t stream, int32_t *d_token_ids, int BatchSize, int SeqLen);

#ifdef __cplusplus
}
#endif

#endif // EMBEDDING_COMMON_H
