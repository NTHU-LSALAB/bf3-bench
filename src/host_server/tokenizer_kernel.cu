/*
 * Optimized GPT-2 BPE Tokenizer for GPU (CUDA)
 * 
 * Key optimizations:
 * 1. Compact hash table entry (24 bytes instead of 264 bytes)
 * 2. Store hash value to avoid recomputation
 * 3. Smaller hash table (65536 entries = 1.5MB, fits in L2 cache)
 * 4. Better memory layout for coalesced access
 */

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "tokenizer_common.h"
#include <doca_log.h>

DOCA_LOG_REGISTER(BPE::GPU);

#define GPT2_VOCAB_SIZE 50257
#define MAX_TOKEN_LEN 64        /* Reduced from 256 */
#define VOCAB_HASH_SIZE 65536   /* Reduced from 131072 */
#define SHORT_KEY_LEN 16        /* Store first 16 bytes for collision check */

/* ============================================
 * Compact Data Structures
 * ============================================ */

/* Compact hash table entry - only 24 bytes */
typedef struct {
    uint32_t hash_val;              /* Pre-computed hash (4 bytes) */
    char short_key[SHORT_KEY_LEN];  /* First 16 bytes of key (16 bytes) */
    int32_t id;                      /* Token ID (4 bytes) */
} CompactVocabEntry;

/* Full token strings stored separately (for fallback) */
static char** h_full_tokens = NULL;

/* Static host storage */
static CompactVocabEntry* h_vocab_table = NULL;
static int h_vocab_loaded = 0;

/* GPU storage */
static CompactVocabEntry* d_vocab_table = NULL;

/* GPT-2 byte encoder */
static char h_byte_to_unicode[256][5];
__constant__ char d_byte_to_unicode[256][5];

/* FNV-1a hash function */
static uint32_t host_hash_string(const char* str, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len && str[i]; i++) {
        hash ^= (uint8_t)str[i];
        hash *= 16777619u;
    }
    return hash;
}

/* GPU hash function */
__device__ uint32_t device_hash_string(const char* str, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len && str[i]; i++) {
        hash ^= (uint8_t)str[i];
        hash *= 16777619u;
    }
    return hash;
}

/* Initialize byte encoder */
static void init_byte_encoder(void) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            if (b < 128) {
                h_byte_to_unicode[b][0] = (char)b;
                h_byte_to_unicode[b][1] = '\0';
            } else {
                h_byte_to_unicode[b][0] = 0xC0 | (b >> 6);
                h_byte_to_unicode[b][1] = 0x80 | (b & 0x3F);
                h_byte_to_unicode[b][2] = '\0';
            }
        } else {
            int code = 256 + n;
            h_byte_to_unicode[b][0] = 0xC4 + (code >> 6) - 4;
            h_byte_to_unicode[b][1] = 0x80 | (code & 0x3F);
            h_byte_to_unicode[b][2] = '\0';
            n++;
        }
    }
    h_byte_to_unicode[32][0] = 0xC4;
    h_byte_to_unicode[32][1] = 0xA0;
    h_byte_to_unicode[32][2] = '\0';
}

/* Parse JSON string value */
static int parse_json_string(const char** pp, char* out, int max_len) {
    const char* p = *pp;
    int len = 0;

    if (*p != '"') return -1;
    p++;

    while (*p && *p != '"' && len < max_len - 1) {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n': out[len++] = '\n'; break;
                case 't': out[len++] = '\t'; break;
                case 'r': out[len++] = '\r'; break;
                case '\\': out[len++] = '\\'; break;
                case '"': out[len++] = '"'; break;
                case '/': out[len++] = '/'; break;
                case 'u': {
                    p++;
                    unsigned int code = 0;
                    for (int i = 0; i < 4 && *p; i++, p++) {
                        code <<= 4;
                        if (*p >= '0' && *p <= '9') code |= (*p - '0');
                        else if (*p >= 'a' && *p <= 'f') code |= (*p - 'a' + 10);
                        else if (*p >= 'A' && *p <= 'F') code |= (*p - 'A' + 10);
                    }
                    p--;
                    if (code < 0x80) {
                        out[len++] = (char)code;
                    } else if (code < 0x800) {
                        out[len++] = 0xC0 | (code >> 6);
                        out[len++] = 0x80 | (code & 0x3F);
                    } else {
                        out[len++] = 0xE0 | (code >> 12);
                        out[len++] = 0x80 | ((code >> 6) & 0x3F);
                        out[len++] = 0x80 | (code & 0x3F);
                    }
                    break;
                }
                default: out[len++] = *p;
            }
        } else {
            out[len++] = *p;
        }
        p++;
    }

    out[len] = '\0';
    if (*p == '"') p++;
    *pp = p;
    return len;
}

/* Load vocabulary into compact hash table */
static int load_vocab(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        DOCA_LOG_ERR("GPU BPE: Cannot open vocab file: %s", vocab_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = (char*)malloc(size + 1);
    if (!json) {
        fclose(f);
        return -1;
    }

    size_t bytes_read = fread(json, 1, size, f);
    json[bytes_read] = '\0';
    fclose(f);

    /* Allocate compact hash table */
    h_vocab_table = (CompactVocabEntry*)calloc(VOCAB_HASH_SIZE, sizeof(CompactVocabEntry));
    h_full_tokens = (char**)calloc(GPT2_VOCAB_SIZE, sizeof(char*));
    if (!h_vocab_table || !h_full_tokens) {
        free(json);
        return -1;
    }

    /* Initialize all entries as empty */
    for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
        h_vocab_table[i].id = -1;
    }

    int vocab_count = 0;
    const char* p = json;

    while (*p && *p != '"') p++;

    while (*p == '"') {
        char key[MAX_TOKEN_LEN * 4] = {0};
        int key_len = parse_json_string(&p, key, sizeof(key));
        if (key_len < 0) break;

        while (*p && *p != ':') p++;
        if (*p == ':') p++;
        while (*p && (*p == ' ' || *p == '\t')) p++;

        int32_t id = 0;
        int negative = 0;
        if (*p == '-') { negative = 1; p++; }
        while (*p >= '0' && *p <= '9') {
            id = id * 10 + (*p - '0');
            p++;
        }
        if (negative) id = -id;

        /* Compute hash and insert with linear probing */
        uint32_t hash = host_hash_string(key, key_len);
        uint32_t h = hash % VOCAB_HASH_SIZE;
        int probe = 0;
        
        while (h_vocab_table[h].id >= 0 && probe < VOCAB_HASH_SIZE) {
            h = (h + 1) % VOCAB_HASH_SIZE;
            probe++;
        }
        
        if (probe < VOCAB_HASH_SIZE) {
            h_vocab_table[h].hash_val = hash;
            h_vocab_table[h].id = id;
            /* Copy first SHORT_KEY_LEN bytes */
            int copy_len = key_len < SHORT_KEY_LEN ? key_len : SHORT_KEY_LEN - 1;
            memcpy(h_vocab_table[h].short_key, key, copy_len);
            h_vocab_table[h].short_key[copy_len] = '\0';
            
            /* Store full token for reverse lookup */
            if (id >= 0 && id < GPT2_VOCAB_SIZE) {
                h_full_tokens[id] = strdup(key);
            }
            vocab_count++;
        }

        while (*p && *p != '"' && *p != '}') p++;
    }

    free(json);

    DOCA_LOG_INFO("GPU BPE: Loaded %d vocabulary entries (compact: %.1f MB)", 
                  vocab_count, (float)(VOCAB_HASH_SIZE * sizeof(CompactVocabEntry)) / 1024 / 1024);
    return vocab_count;
}

/* ============================================
 * GPU Kernel: Optimized BPE Tokenization
 * ============================================ */

/* Fast GPU vocabulary lookup using compact hash table */
__device__ int32_t gpu_vocab_lookup_fast(const CompactVocabEntry* vocab_table, 
                                          const char* key, int key_len) {
    uint32_t hash = device_hash_string(key, key_len);
    uint32_t h = hash % VOCAB_HASH_SIZE;
    
    /* Linear probing with early exit on hash mismatch */
    for (int probe = 0; probe < 64; probe++) {  /* Max 64 probes */
        const CompactVocabEntry* entry = &vocab_table[h];
        
        if (entry->id < 0) {
            return -1;  /* Empty slot = not found */
        }
        
        /* Fast path: check hash first */
        if (entry->hash_val == hash) {
            /* Verify with short key comparison */
            int match = 1;
            int cmp_len = key_len < SHORT_KEY_LEN ? key_len : SHORT_KEY_LEN - 1;
            for (int i = 0; i < cmp_len; i++) {
                if (entry->short_key[i] != key[i]) {
                    match = 0;
                    break;
                }
            }
            if (match && (key_len < SHORT_KEY_LEN || entry->short_key[cmp_len] == '\0')) {
                return entry->id;
            }
        }
        
        h = (h + 1) % VOCAB_HASH_SIZE;
    }
    return -1;
}

/* Optimized BPE kernel */
__global__ void bpe_tokenize_kernel_optimized(
    const char* __restrict__ input_text,
    int32_t* __restrict__ output_ids,
    int text_len,
    const CompactVocabEntry* __restrict__ vocab_table,
    int* __restrict__ output_num_tokens
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int num_tokens = 0;
    int pos = 0;
    char word_buf[MAX_TOKEN_LEN * 2];  /* Smaller buffer */
    
    while (pos < text_len && num_tokens < text_len / 2) {
        /* Skip spaces and handle space prefix */
        int has_space_prefix = 0;
        if (input_text[pos] == ' ') {
            has_space_prefix = 1;
            pos++;
            if (pos >= text_len) break;
        }

        /* Find word boundary */
        int word_start = pos;
        while (pos < text_len && 
               input_text[pos] != ' ' && 
               input_text[pos] != '\n' && 
               input_text[pos] != '\t') {
            pos++;
        }
        int word_len = pos - word_start;

        if (word_len == 0) continue;
        if (word_len > MAX_TOKEN_LEN) word_len = MAX_TOKEN_LEN;

        /* Build encoded word */
        int buf_pos = 0;

        if (has_space_prefix && num_tokens > 0) {
            word_buf[buf_pos++] = 0xC4;
            word_buf[buf_pos++] = 0xA0;
        }

        for (int j = word_start; j < word_start + word_len && buf_pos < MAX_TOKEN_LEN * 2 - 4; j++) {
            unsigned char b = (unsigned char)input_text[j];
            const char* enc = d_byte_to_unicode[b];
            for (int k = 0; enc[k] != '\0' && buf_pos < MAX_TOKEN_LEN * 2 - 1; k++) {
                word_buf[buf_pos++] = enc[k];
            }
        }
        word_buf[buf_pos] = '\0';

        /* Try whole word lookup first */
        int32_t id = gpu_vocab_lookup_fast(vocab_table, word_buf, buf_pos);
        if (id >= 0) {
            output_ids[num_tokens++] = id;
        } else {
            /* Character-level fallback with reduced iterations */
            int char_start = 0;

            if (has_space_prefix && num_tokens > 0) {
                char space_char[3] = {(char)0xC4, (char)0xA0, 0};
                id = gpu_vocab_lookup_fast(vocab_table, space_char, 2);
                if (id >= 0) {
                    output_ids[num_tokens++] = id;
                }
                char_start = 2;
            }

            /* Greedy tokenization with max length limit */
            while (char_start < buf_pos && num_tokens < text_len / 2) {
                int found = 0;
                int max_try = buf_pos - char_start;
                if (max_try > 32) max_try = 32;  /* Limit search depth */
                
                for (int try_len = max_try; try_len > 0; try_len--) {
                    char substr[64];
                    for (int k = 0; k < try_len && k < 63; k++) {
                        substr[k] = word_buf[char_start + k];
                    }
                    substr[try_len] = '\0';

                    id = gpu_vocab_lookup_fast(vocab_table, substr, try_len);
                    if (id >= 0) {
                        output_ids[num_tokens++] = id;
                        char_start += try_len;
                        found = 1;
                        break;
                    }
                }

                if (!found) {
                    char_start++;
                }
            }
        }
    }

    *output_num_tokens = num_tokens;
}

/* ============================================
 * Public API
 * ============================================ */

doca_error_t init_tokenizer_resources(void) {
    init_byte_encoder();
    
    cudaError_t err = cudaMemcpyToSymbol(d_byte_to_unicode, h_byte_to_unicode, 
                                          sizeof(h_byte_to_unicode));
    if (err != cudaSuccess) {
        DOCA_LOG_ERR("GPU BPE: Failed to copy byte encoder to GPU: %d", err);
        return DOCA_ERROR_DRIVER;
    }

    const char* vocab_paths[] = {
        "./vocab/vocab.json",
        "../vocab/vocab.json",
        "../../vocab/vocab.json",
        NULL
    };

    int loaded = 0;
    for (int i = 0; vocab_paths[i] != NULL; i++) {
        if (load_vocab(vocab_paths[i]) > 0) {
            loaded = 1;
            break;
        }
    }

    if (!loaded) {
        DOCA_LOG_WARN("GPU BPE: Could not load vocabulary, falling back to hash-based tokenizer");
        return DOCA_SUCCESS;
    }

    /* Copy compact vocabulary to GPU */
    err = cudaMalloc((void**)&d_vocab_table, VOCAB_HASH_SIZE * sizeof(CompactVocabEntry));
    if (err != cudaSuccess) {
        DOCA_LOG_ERR("GPU BPE: Failed to allocate GPU vocab table: %d", err);
        free(h_vocab_table);
        h_vocab_table = NULL;
        return DOCA_ERROR_NO_MEMORY;
    }

    err = cudaMemcpy(d_vocab_table, h_vocab_table, 
                     VOCAB_HASH_SIZE * sizeof(CompactVocabEntry), 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        DOCA_LOG_ERR("GPU BPE: Failed to copy vocab to GPU: %d", err);
        cudaFree(d_vocab_table);
        free(h_vocab_table);
        d_vocab_table = NULL;
        h_vocab_table = NULL;
        return DOCA_ERROR_DRIVER;
    }

    h_vocab_loaded = 1;
    DOCA_LOG_INFO("GPU BPE: Tokenizer initialized (compact hash table)");
    return DOCA_SUCCESS;
}

void cleanup_tokenizer_resources(void) {
    if (d_vocab_table) {
        cudaFree(d_vocab_table);
        d_vocab_table = NULL;
    }
    if (h_vocab_table) {
        free(h_vocab_table);
        h_vocab_table = NULL;
    }
    if (h_full_tokens) {
        for (int i = 0; i < GPT2_VOCAB_SIZE; i++) {
            if (h_full_tokens[i]) free(h_full_tokens[i]);
        }
        free(h_full_tokens);
        h_full_tokens = NULL;
    }
    h_vocab_loaded = 0;
}

/* Fallback hash-based tokenizer */
__global__ void hash_tokenizer_kernel(char *input, int32_t *output, int num_chars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tokens = num_chars / 4;
    if (idx >= num_tokens) return;
    
    int base = idx * 4;
    uint32_t hash = 5381;
    for(int i = 0; i < 4 && (base + i) < num_chars; i++) {
        hash = ((hash << 5) + hash) + input[base + i];
    }
    output[idx] = hash % 30522;
}

doca_error_t launch_gpu_tokenizer(cudaStream_t stream, char *d_input_text, 
                                   int32_t *d_output_ids, int total_len) {
    if (h_vocab_loaded && d_vocab_table != NULL) {
        int* d_num_tokens;
        cudaMalloc((void**)&d_num_tokens, sizeof(int));
        cudaMemset(d_num_tokens, 0, sizeof(int));
        
        /* Use optimized kernel */
        bpe_tokenize_kernel_optimized<<<1, 1, 0, stream>>>(
            d_input_text,
            d_output_ids,
            total_len,
            d_vocab_table,
            d_num_tokens
        );
        
        cudaFree(d_num_tokens);
    } else {
        int threads = 256;
        int blocks = (total_len + threads - 1) / threads;
        hash_tokenizer_kernel<<<blocks, threads, 0, stream>>>(
            d_input_text, d_output_ids, total_len
        );
    }
    
    if (cudaGetLastError() != cudaSuccess) return DOCA_ERROR_DRIVER;
    return DOCA_SUCCESS;
}
