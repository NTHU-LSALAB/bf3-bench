#ifndef TOKENIZER_COMMON_H
#define TOKENIZER_COMMON_H

#include <doca_error.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize tokenizer resources (vocabulary, etc.)
 * @return DOCA_SUCCESS on success
 */
doca_error_t init_tokenizer_resources(void);

/**
 * Cleanup tokenizer resources
 */
void cleanup_tokenizer_resources(void);

/**
 * Launch GPU tokenization kernel
 * @param stream CUDA stream for async execution
 * @param d_input_text Input text buffer on GPU
 * @param d_output_ids Output token IDs on GPU
 * @param total_len Length of input text in bytes
 * @return DOCA_SUCCESS on success
 */
doca_error_t launch_gpu_tokenizer(cudaStream_t stream, char *d_input_text,
                                   int32_t *d_output_ids, int total_len);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_COMMON_H */
