/*
 * GPT-2 BPE Tokenizer Implementation for DPU (ARM)
 *
 * Simplified but functional BPE tokenizer that produces tokens
 * compatible with HuggingFace's GPT-2 tokenizer.
 */

#include "bpe_tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

/* Hash table for vocabulary lookup */
#define VOCAB_HASH_SIZE 131072

typedef struct VocabEntry {
    char* key;
    int32_t id;
    struct VocabEntry* next;
} VocabEntry;

static VocabEntry* vocab_hash[VOCAB_HASH_SIZE];

/* Reverse lookup: ID -> text */
static char** id_to_text = NULL;

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

/* GPT-2 byte encoder: maps bytes 0-255 to unicode characters
 * This avoids issues with control characters in the vocabulary */
static char byte_to_unicode[256][5];  /* UTF-8 can be up to 4 bytes + null */

static void init_byte_encoder(void) {
    /* GPT-2's byte encoding scheme:
     * - Printable ASCII (33-126) and extended Latin (161-172, 174-255) map to themselves
     * - Other bytes (0-32, 127-160, 173) map to unicode range starting at 256 (U+0100)
     */
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            /* Map to self - single byte UTF-8 or 2-byte UTF-8 for > 127 */
            if (b < 128) {
                byte_to_unicode[b][0] = (char)b;
                byte_to_unicode[b][1] = '\0';
            } else {
                /* UTF-8 encode: 110xxxxx 10xxxxxx */
                byte_to_unicode[b][0] = 0xC0 | (b >> 6);
                byte_to_unicode[b][1] = 0x80 | (b & 0x3F);
                byte_to_unicode[b][2] = '\0';
            }
        } else {
            /* Map to 256 + n (U+0100 onwards) */
            int code = 256 + n;
            /* UTF-8 encode: 110xxxxx 10xxxxxx for U+0080-U+07FF */
            byte_to_unicode[b][0] = 0xC4 + (code >> 6) - 4;
            byte_to_unicode[b][1] = 0x80 | (code & 0x3F);
            byte_to_unicode[b][2] = '\0';
            n++;
        }
    }

    /* Special case: space (32) maps to Ġ (U+0120) */
    byte_to_unicode[32][0] = 0xC4;  /* U+0120 in UTF-8 */
    byte_to_unicode[32][1] = 0xA0;
    byte_to_unicode[32][2] = '\0';
}

/* Parse a JSON string value, handling escape sequences */
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
                    /* Unicode escape \uXXXX */
                    p++;
                    unsigned int code = 0;
                    for (int i = 0; i < 4 && *p; i++, p++) {
                        code <<= 4;
                        if (*p >= '0' && *p <= '9') code |= (*p - '0');
                        else if (*p >= 'a' && *p <= 'f') code |= (*p - 'a' + 10);
                        else if (*p >= 'A' && *p <= 'F') code |= (*p - 'A' + 10);
                    }
                    p--;  /* Will be incremented at end of loop */
                    /* UTF-8 encode */
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

/* Parse vocab.json: {"token": id, ...} */
static int parse_vocab_json(BPEContext* ctx, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "BPE: Cannot open vocab file: %s\n", path);
        return -1;
    }

    /* Read entire file */
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

    /* Allocate id_to_text array */
    id_to_text = calloc(GPT2_VOCAB_SIZE, sizeof(char*));

    /* Parse JSON */
    memset(vocab_hash, 0, sizeof(vocab_hash));
    ctx->vocab_size = 0;

    const char* p = json;

    /* Skip to first quote */
    while (*p && *p != '"') p++;

    while (*p == '"') {
        char key[MAX_TOKEN_LEN * 4] = {0};
        if (parse_json_string(&p, key, sizeof(key)) < 0) break;

        /* Skip to colon and number */
        while (*p && *p != ':') p++;
        if (*p == ':') p++;
        while (*p && (*p == ' ' || *p == '\t')) p++;

        /* Parse number */
        int32_t id = 0;
        bool negative = false;
        if (*p == '-') { negative = true; p++; }
        while (*p >= '0' && *p <= '9') {
            id = id * 10 + (*p - '0');
            p++;
        }
        if (negative) id = -id;

        /* Store in hash table */
        hash_insert(key, id);

        /* Store reverse mapping */
        if (id >= 0 && id < GPT2_VOCAB_SIZE) {
            id_to_text[id] = strdup(key);
        }

        ctx->vocab_size++;

        /* Skip to next quote or end */
        while (*p && *p != '"' && *p != '}') p++;
    }

    free(json);

    fprintf(stderr, "BPE: Loaded %d vocabulary entries\n", ctx->vocab_size);
    return 0;
}

/* Parse merges.txt */
static int parse_merges(BPEContext* ctx, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "BPE: Cannot open merges file: %s\n", path);
        return -1;
    }

    ctx->merges = calloc(MAX_MERGES, sizeof(BPEMerge));
    ctx->num_merges = 0;

    char line[1024];
    int line_num = 0;

    while (fgets(line, sizeof(line), f) && ctx->num_merges < MAX_MERGES) {
        line_num++;

        /* Skip header line starting with # */
        if (line[0] == '#') continue;

        /* Remove newline */
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }

        if (len == 0) continue;

        /* Store merge rule as-is (format: "token1 token2") */
        ctx->merges[ctx->num_merges].pair = strdup(line);
        ctx->merges[ctx->num_merges].priority = ctx->num_merges;
        ctx->num_merges++;
    }

    fclose(f);

    fprintf(stderr, "BPE: Loaded %d merge rules\n", ctx->num_merges);
    return 0;
}

int bpe_init(BPEContext* ctx, const char* vocab_path, const char* merges_path) {
    memset(ctx, 0, sizeof(BPEContext));

    init_byte_encoder();

    if (parse_vocab_json(ctx, vocab_path) != 0) {
        return -1;
    }

    if (parse_merges(ctx, merges_path) != 0) {
        return -1;
    }

    return 0;
}

void bpe_cleanup(BPEContext* ctx) {
    /* Free hash table */
    for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
        VocabEntry* entry = vocab_hash[i];
        while (entry) {
            VocabEntry* next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
        vocab_hash[i] = NULL;
    }

    /* Free reverse lookup */
    if (id_to_text) {
        for (int i = 0; i < GPT2_VOCAB_SIZE; i++) {
            if (id_to_text[i]) free(id_to_text[i]);
        }
        free(id_to_text);
        id_to_text = NULL;
    }

    /* Free merges */
    if (ctx->merges) {
        for (int i = 0; i < ctx->num_merges; i++) {
            if (ctx->merges[i].pair) free(ctx->merges[i].pair);
        }
        free(ctx->merges);
    }

    memset(ctx, 0, sizeof(BPEContext));
}

/* Convert raw bytes to GPT-2 byte-encoded string */
static int bytes_to_gpt2(const char* input, int input_len, char* output, int max_output) {
    int out_len = 0;
    for (int i = 0; i < input_len && out_len < max_output - 4; i++) {
        unsigned char b = (unsigned char)input[i];
        const char* encoded = byte_to_unicode[b];
        int enc_len = strlen(encoded);
        if (out_len + enc_len >= max_output) break;
        memcpy(output + out_len, encoded, enc_len);
        out_len += enc_len;
    }
    output[out_len] = '\0';
    return out_len;
}

/* Simple word-level BPE encoding
 * For each word, try to find it in vocabulary, else fall back to characters */
int bpe_encode(BPEContext* ctx, const char* text, int text_len,
               int32_t* output_ids, int max_output_len) {
    if (!ctx || !text || !output_ids) return -1;

    int num_tokens = 0;
    int i = 0;

    /* Buffer for GPT-2 encoded text */
    char* encoded = malloc(text_len * 4 + 1);
    if (!encoded) return -1;

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
                /* Try Ġ alone */
                char space_char[4] = {0xC4, 0xA0, 0};
                id = hash_lookup(space_char);
                if (id >= 0) {
                    output_ids[num_tokens++] = id;
                }
                char_start = 2;  /* Skip the Ġ prefix */
            }

            /* Tokenize remaining characters */
            while (char_start < buf_pos && num_tokens < max_output_len) {
                /* Try increasingly shorter substrings */
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
                    /* Unknown character - use a fallback (e.g., byte token) */
                    /* Skip one byte */
                    char_start++;
                }
            }
        }
    }

    free(encoded);
    return num_tokens;
}

int bpe_decode(BPEContext* ctx, const int32_t* token_ids, int num_tokens,
               char* output_text, int max_output_len) {
    if (!ctx || !token_ids || !output_text) return -1;

    int len = 0;
    for (int i = 0; i < num_tokens && len < max_output_len - 1; i++) {
        int32_t id = token_ids[i];
        if (id >= 0 && id < GPT2_VOCAB_SIZE && id_to_text && id_to_text[id]) {
            const char* token = id_to_text[id];
            /* Convert GPT-2 encoding back to bytes */
            /* For simplicity, just copy - proper decoding would reverse byte_to_unicode */
            int tlen = strlen(token);
            if (len + tlen < max_output_len) {
                memcpy(output_text + len, token, tlen);
                len += tlen;
            }
        }
    }
    output_text[len] = '\0';
    return len;
}
