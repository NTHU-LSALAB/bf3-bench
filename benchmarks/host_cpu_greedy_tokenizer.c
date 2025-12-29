/*
 * Host CPU Greedy Tokenizer Benchmark
 *
 * Uses the SAME greedy longest-match algorithm as DPU and GPU
 * for fair three-way comparison.
 *
 * Build: gcc -O3 -o host_cpu_greedy_tokenizer host_cpu_greedy_tokenizer.c
 * Run:   ./host_cpu_greedy_tokenizer ../vocab/vocab.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#define GPT2_VOCAB_SIZE 50257
#define MAX_TOKEN_LEN 256
#define VOCAB_HASH_SIZE 131072

/* Hash table for vocabulary lookup */
typedef struct VocabEntry {
    char* key;
    int32_t id;
    struct VocabEntry* next;
} VocabEntry;

static VocabEntry* vocab_hash[VOCAB_HASH_SIZE];

/* GPT-2 byte encoder: maps bytes 0-255 to unicode characters */
static char byte_to_unicode[256][5];

/* FNV-1a hash function */
static uint32_t hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;
    }
    return hash % VOCAB_HASH_SIZE;
}

/* Add entry to hash table */
static void hash_insert(const char* key, int32_t id) {
    uint32_t h = hash_string(key);
    VocabEntry* entry = malloc(sizeof(VocabEntry));
    entry->key = strdup(key);
    entry->id = id;
    entry->next = vocab_hash[h];
    vocab_hash[h] = entry;
}

/* Lookup in hash table, returns -1 if not found */
static int32_t hash_lookup(const char* key) {
    uint32_t h = hash_string(key);
    VocabEntry* entry = vocab_hash[h];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->id;
        }
        entry = entry->next;
    }
    return -1;
}

/* Initialize GPT-2 byte encoder */
static void init_byte_encoder(void) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            if (b < 128) {
                byte_to_unicode[b][0] = (char)b;
                byte_to_unicode[b][1] = '\0';
            } else {
                byte_to_unicode[b][0] = 0xC0 | (b >> 6);
                byte_to_unicode[b][1] = 0x80 | (b & 0x3F);
                byte_to_unicode[b][2] = '\0';
            }
        } else {
            int code = 256 + n;
            byte_to_unicode[b][0] = 0xC4 + (code >> 6) - 4;
            byte_to_unicode[b][1] = 0x80 | (code & 0x3F);
            byte_to_unicode[b][2] = '\0';
            n++;
        }
    }
    /* Special case: space (32) maps to Ġ (U+0120) */
    byte_to_unicode[32][0] = 0xC4;
    byte_to_unicode[32][1] = 0xA0;
    byte_to_unicode[32][2] = '\0';
}

/* Parse a JSON string value */
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

/* Load vocabulary from vocab.json */
static int load_vocab(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open vocab file: %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = malloc(size + 1);
    if (!json) {
        fclose(f);
        return -1;
    }

    size_t read = fread(json, 1, size, f);
    json[read] = '\0';
    fclose(f);

    memset(vocab_hash, 0, sizeof(vocab_hash));
    int vocab_size = 0;

    const char* p = json;
    while (*p && *p != '"') p++;

    while (*p == '"') {
        char key[MAX_TOKEN_LEN * 4] = {0};
        if (parse_json_string(&p, key, sizeof(key)) < 0) break;

        while (*p && *p != ':') p++;
        if (*p == ':') p++;
        while (*p && (*p == ' ' || *p == '\t')) p++;

        int32_t id = 0;
        bool negative = false;
        if (*p == '-') { negative = true; p++; }
        while (*p >= '0' && *p <= '9') {
            id = id * 10 + (*p - '0');
            p++;
        }
        if (negative) id = -id;

        hash_insert(key, id);
        vocab_size++;

        while (*p && *p != '"' && *p != '}') p++;
    }

    free(json);
    fprintf(stderr, "Loaded %d vocabulary entries\n", vocab_size);
    return vocab_size;
}

/* Greedy longest-match tokenizer - SAME algorithm as DPU */
static int greedy_tokenize(const char* text, int text_len,
                           int32_t* output_ids, int max_output_len) {
    int num_tokens = 0;
    int i = 0;

    while (i < text_len && num_tokens < max_output_len) {
        /* Check for space at current position */
        bool has_space_prefix = false;
        if (text[i] == ' ') {
            has_space_prefix = true;
            i++;
            if (i >= text_len) break;
        }

        /* Find word boundary */
        int word_start = i;
        while (i < text_len && text[i] != ' ' && text[i] != '\n' && text[i] != '\t') {
            i++;
        }
        int word_len = i - word_start;

        if (word_len == 0) continue;

        /* Build GPT-2 encoded word with optional space prefix */
        char word_buf[MAX_TOKEN_LEN * 4];
        int buf_pos = 0;

        if (has_space_prefix && num_tokens > 0) {
            /* Add space prefix (Ġ = U+0120) */
            word_buf[buf_pos++] = 0xC4;
            word_buf[buf_pos++] = 0xA0;
        }

        /* Encode word bytes */
        for (int j = word_start; j < word_start + word_len && buf_pos < MAX_TOKEN_LEN * 4 - 4; j++) {
            unsigned char b = (unsigned char)text[j];
            const char* enc = byte_to_unicode[b];
            int enc_len = strlen(enc);
            memcpy(word_buf + buf_pos, enc, enc_len);
            buf_pos += enc_len;
        }
        word_buf[buf_pos] = '\0';

        /* Try to find whole word in vocabulary */
        int32_t id = hash_lookup(word_buf);
        if (id >= 0) {
            output_ids[num_tokens++] = id;
        } else {
            /* Fall back to character-level tokenization */
            int char_start = 0;

            /* Handle space prefix separately if present */
            if (has_space_prefix && num_tokens > 0) {
                char space_char[4] = {0xC4, 0xA0, 0};
                id = hash_lookup(space_char);
                if (id >= 0) {
                    output_ids[num_tokens++] = id;
                }
                char_start = 2;
            }

            /* Tokenize remaining characters - greedy longest match */
            while (char_start < buf_pos && num_tokens < max_output_len) {
                bool found = false;
                for (int try_len = buf_pos - char_start; try_len > 0; try_len--) {
                    char substr[MAX_TOKEN_LEN * 4];
                    memcpy(substr, word_buf + char_start, try_len);
                    substr[try_len] = '\0';

                    id = hash_lookup(substr);
                    if (id >= 0) {
                        output_ids[num_tokens++] = id;
                        char_start += try_len;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    char_start++;
                }
            }
        }
    }

    return num_tokens;
}

/* Generate test text (same as DPU and GPU benchmarks) */
static void generate_test_text(char* text, int len) {
    for (int i = 0; i < len; i++) {
        text[i] = 'a' + (i % 26);
        if (i % 5 == 4) text[i] = ' ';
    }
    text[len] = '\0';
}

int main(int argc, char* argv[]) {
    const char* vocab_path = "../vocab/vocab.json";
    int payload_kb = 8;
    int iterations = 10;
    int warmup = 100;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--payload-kb") == 0 && i + 1 < argc) {
            payload_kb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            vocab_path = argv[i];
        }
    }

    printf("Host CPU Greedy Tokenizer Benchmark\n");
    printf("====================================\n");
    printf("Vocab: %s\n", vocab_path);
    printf("Payload: %d KB\n", payload_kb);
    printf("Iterations: %d\n", iterations);
    printf("Warmup: %d\n\n", warmup);

    /* Initialize */
    init_byte_encoder();

    if (load_vocab(vocab_path) < 0) {
        return 1;
    }

    /* Generate test text */
    int text_len = payload_kb * 1024;
    char* text = malloc(text_len + 1);
    generate_test_text(text, text_len);

    int32_t* tokens = malloc(text_len * sizeof(int32_t));

    /* Warmup */
    printf("Warming up (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        greedy_tokenize(text, text_len, tokens, text_len);
    }

    /* Benchmark */
    printf("Running benchmark (%d iterations)...\n\n", iterations);

    struct timespec start, end;
    double total_us = 0;
    int num_tokens = 0;

    for (int iter = 0; iter < iterations; iter++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        num_tokens = greedy_tokenize(text, text_len, tokens, text_len);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed_us = (end.tv_sec - start.tv_sec) * 1e6 +
                           (end.tv_nsec - start.tv_nsec) / 1e3;
        total_us += elapsed_us;

        printf("  Iteration %d: %.0f µs (%d tokens)\n", iter + 1, elapsed_us, num_tokens);
    }

    double avg_us = total_us / iterations;

    printf("\n====================================\n");
    printf("Results:\n");
    printf("  Average: %.0f µs\n", avg_us);
    printf("  Tokens: %d\n", num_tokens);
    printf("  Throughput: %.2f MB/s\n", (text_len / 1024.0 / 1024.0) / (avg_us / 1e6));
    printf("\n");
    printf("Host CPU Greedy BPE: %.0f µs\n", avg_us);

    free(text);
    free(tokens);

    return 0;
}
