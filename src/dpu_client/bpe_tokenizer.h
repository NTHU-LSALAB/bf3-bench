/*
 * GPT-2 BPE Tokenizer for DPU (ARM)
 * 
 * This implementation is compatible with HuggingFace's GPT-2 tokenizer.
 * It uses the standard GPT-2 vocabulary (50,257 tokens) and merge rules.
 */

#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>
#include <stdlib.h>

#define GPT2_VOCAB_SIZE 50257
#define MAX_TOKEN_LEN 256
#define MAX_MERGES 50000

/* Token structure */
typedef struct {
    char* text;
    int32_t id;
} BPEToken;

/* Merge rule structure */
typedef struct {
    char* pair;     // "ab" means merge tokens a and b
    int priority;   // Lower priority = earlier merge
} BPEMerge;

/* BPE Tokenizer context */
typedef struct {
    BPEToken* vocab;        // Token text -> ID mapping
    int vocab_size;
    BPEMerge* merges;       // Merge rules
    int num_merges;
    int32_t* byte_encoder;  // Byte -> Unicode mapping (GPT-2 specific)
} BPEContext;

/* Initialize tokenizer with vocabulary and merge files */
int bpe_init(BPEContext* ctx, const char* vocab_path, const char* merges_path);

/* Free tokenizer resources */
void bpe_cleanup(BPEContext* ctx);

/* Encode text to token IDs
 * Returns: number of tokens, or -1 on error
 * Output tokens are written to output_ids (must be pre-allocated)
 */
int bpe_encode(BPEContext* ctx, const char* text, int text_len, 
               int32_t* output_ids, int max_output_len);

/* Decode token IDs to text
 * Returns: length of decoded text, or -1 on error
 */
int bpe_decode(BPEContext* ctx, const int32_t* token_ids, int num_tokens,
               char* output_text, int max_output_len);

#endif /* BPE_TOKENIZER_H */
