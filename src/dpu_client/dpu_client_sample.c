/* DPU Client Sample - RoCE v2 Fix */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <doca_log.h>
#include <doca_error.h>
#include <doca_rdma.h>
#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_pe.h>
#include "rdma_common.h"
#include "bpe_tokenizer.h"

DOCA_LOG_REGISTER(DPU_CLIENT::CORE);

/* Global BPE tokenizer context */
static BPEContext g_bpe_ctx;
static int g_bpe_initialized = 0;
#define TEST_STRING "Hello from BF3 DPU to A100X GPU via RDMA!"
#define SLEEP_IN_NANOS (10 * 1000)

struct dpu_resources {
	struct doca_dev *dev;
	struct doca_rdma *rdma;
	struct doca_ctx *rdma_ctx;
	struct doca_pe *pe;
};

static void write_cb(struct doca_rdma_task_write *task, union doca_data task_user_data, union doca_data ctx_user_data) {
	*(int *)task_user_data.ptr = 1;
	fprintf(stderr, "Write Callback: Success\n");
}
static void error_cb(struct doca_rdma_task_write *task, union doca_data task_user_data, union doca_data ctx_user_data) {
	*(int *)task_user_data.ptr = -1;
	fprintf(stderr, "Write Callback: Error\n");
}

static int connect_to_server(const char *ip, int port) {
	int sock;
	struct sockaddr_in serv_addr;
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) return -1;
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) return -1;
	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) return -1;
	return sock;
}

static doca_error_t check_dev_support(struct doca_devinfo *devinfo) {
	return doca_rdma_cap_task_write_is_supported((const struct doca_devinfo *)devinfo);
}

static doca_error_t open_doca_device(const char *name, struct doca_dev **dev) {
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	doca_error_t result;
	int i;
	result = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) return result;
	
	fprintf(stderr, "Scanning %d devices for '%s'...\n", nb_devs, name);
	for (i = 0; i < nb_devs; i++) {
		char ib_name[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
		result = doca_devinfo_get_ibdev_name(dev_list[i], ib_name, DOCA_DEVINFO_IBDEV_NAME_SIZE);
		if (result != DOCA_SUCCESS) continue;
		
		if (strncmp(name, ib_name, DOCA_DEVINFO_IBDEV_NAME_SIZE) == 0) {
			result = check_dev_support(dev_list[i]);
			if (result == DOCA_SUCCESS) {
				fprintf(stderr, "    Match (%s)! Opening...\n", ib_name);
				result = doca_dev_open(dev_list[i], dev);
				doca_devinfo_destroy_list(dev_list);
				return result;
			} else {
				fprintf(stderr, "    Match (%s) but Cap Check Failed: %s\n", ib_name, doca_error_get_descr(result));
			}
		}
	}
	doca_devinfo_destroy_list(dev_list);
	return DOCA_ERROR_NOT_FOUND;
}

static doca_error_t init_dpu_resources(struct rdma_config *cfg, struct dpu_resources *res) {
	doca_error_t result;
	result = open_doca_device(cfg->device_name, &res->dev);
	if (result != DOCA_SUCCESS) return result;
	result = doca_rdma_create(res->dev, &res->rdma);
	if (result != DOCA_SUCCESS) return result;
	res->rdma_ctx = doca_rdma_as_ctx(res->rdma);
	result = doca_pe_create(&res->pe);
	if (result != DOCA_SUCCESS) return result;
	result = doca_pe_connect_ctx(res->pe, res->rdma_ctx);
	if (result != DOCA_SUCCESS) return result;
	result = doca_rdma_set_permissions(res->rdma, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
	if (result != DOCA_SUCCESS) return result;
	
	// Set GID Index from Config or Default to 1 (v2)
	if (cfg->is_gid_index_set) {
		result = doca_rdma_set_gid_index(res->rdma, cfg->gid_index);
	} else {
		result = doca_rdma_set_gid_index(res->rdma, 1);
	}
	if (result != DOCA_SUCCESS) fprintf(stderr, "Warning: Failed to set GID index: %s\n", doca_error_get_descr(result));

    // Config BEFORE Start
	result = doca_rdma_task_write_set_conf(res->rdma, write_cb, error_cb, 10);
	if (result != DOCA_SUCCESS) return result;

	result = doca_ctx_start(res->rdma_ctx);
	return result;
}

doca_error_t dpu_client_start(struct rdma_config *cfg) {
	struct dpu_resources res = {0};
	doca_error_t result;
	int sock_fd;
	struct timespec ts = {0, SLEEP_IN_NANOS};
	size_t remote_conn_len;
	void *remote_conn_details = NULL;
	const void *local_conn_details = NULL;
	size_t local_conn_len;
	struct doca_rdma_connection *connection = NULL;
	size_t remote_mmap_len;
	void *remote_mmap_export = NULL;
	const void *local_mmap_export = NULL;
	size_t local_mmap_len;
	struct doca_mmap *remote_mmap = NULL;
	struct doca_mmap *local_mmap = NULL;
	char *send_buf_ptr = NULL;
	struct doca_buf_inventory *buf_inv = NULL;
	struct doca_buf *src_buf = NULL, *dst_buf = NULL;
	struct doca_rdma_task_write *write_task = NULL;
	struct doca_task *task = NULL;
	union doca_data task_user_data = {0};
	int task_finished = 0;
	// struct client_config is invalid. Use rdma_config.
	struct rdma_config *client_cfg = cfg; 

	fprintf(stderr, "Initializing Resources...\n");
	result = init_dpu_resources(cfg, &res);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	/* Initialize BPE tokenizer if not already done */
	/* Check for FORCE_HASH_TOKENIZE to skip BPE and use hash-based for fair comparison with GPU */
	const char *force_hash = getenv("FORCE_HASH_TOKENIZE");
	if (force_hash && (strcmp(force_hash, "1") == 0 || strcmp(force_hash, "true") == 0)) {
		DOCA_LOG_INFO("FORCE_HASH_TOKENIZE enabled - using hash-based tokenization (GPU-compatible)");
		g_bpe_initialized = 0;  /* Force hash mode */
	} else if (!g_bpe_initialized) {
		const char *vocab_path = getenv("VOCAB_PATH");
		const char *merges_path = getenv("MERGES_PATH");
		if (!vocab_path) vocab_path = "./vocab/vocab.json";
		if (!merges_path) merges_path = "./vocab/merges.txt";

		if (bpe_init(&g_bpe_ctx, vocab_path, merges_path) == 0) {
			g_bpe_initialized = 1;
			DOCA_LOG_INFO("BPE tokenizer initialized (vocab: %d, merges: %d)",
			              g_bpe_ctx.vocab_size, g_bpe_ctx.num_merges);
		} else {
			DOCA_LOG_WARN("BPE init failed, using hash-based tokenization");
		}
	}

	fprintf(stderr, "Connecting to %s:2000...\n", cfg->server_ip_addr);
	sock_fd = connect_to_server(cfg->server_ip_addr, 2000);
	if (sock_fd < 0) { fprintf(stderr, "Connect failed\n"); return DOCA_ERROR_CONNECTION_ABORTED; }
    
	if (recv(sock_fd, &remote_conn_len, sizeof(size_t), 0) <= 0) return DOCA_ERROR_IO_FAILED;
	fprintf(stderr, "Remote Conn Len: %zu\n", remote_conn_len);
	remote_conn_details = calloc(1, remote_conn_len);
	if (recv(sock_fd, remote_conn_details, remote_conn_len, 0) <= 0) return DOCA_ERROR_IO_FAILED;

	result = doca_rdma_export(res.rdma, &local_conn_details, &local_conn_len, &connection);
	if (result != DOCA_SUCCESS) { fprintf(stderr, "Export failed: %s\n", doca_error_get_descr(result)); return result; }
	fprintf(stderr, "Sending Local Conn Len: %zu\n", local_conn_len);
	send(sock_fd, &local_conn_len, sizeof(size_t), 0);
	send(sock_fd, local_conn_details, local_conn_len, 0);

	result = doca_rdma_connect(res.rdma, remote_conn_details, remote_conn_len, connection);
	if (result != DOCA_SUCCESS) { fprintf(stderr, "RDMA Connect failed: %s\n", doca_error_get_descr(result)); return result; }
	fprintf(stderr, "RDMA Connected\n");

	if (recv(sock_fd, &remote_mmap_len, sizeof(size_t), 0) <= 0) return DOCA_ERROR_IO_FAILED;
	fprintf(stderr, "Remote MMap Len: %zu\n", remote_mmap_len);
	remote_mmap_export = calloc(1, remote_mmap_len);
	if (recv(sock_fd, remote_mmap_export, remote_mmap_len, 0) <= 0) return DOCA_ERROR_IO_FAILED;
	
	result = doca_mmap_create_from_export(NULL, remote_mmap_export, remote_mmap_len, res.dev, &remote_mmap);
	if (result != DOCA_SUCCESS) { fprintf(stderr, "Remote MMap Create failed: %s\n", doca_error_get_descr(result)); return result; }
	// Init Local Buf (Aligned)
    // Init Local Buf (Aligned)
    char *env_size = getenv("PAYLOAD_SIZE_KB");
    int size_kb = (env_size) ? atoi(env_size) : 8; // Default 8KB
    if (size_kb <= 0) size_kb = 1;
    if (size_kb > 128) size_kb = 128; // Max 128KB due to buffer limit

    /* Batch size support */
    char *env_batch = getenv("BATCH_SIZE");
    int batch_size = (env_batch) ? atoi(env_batch) : 1; // Default batch=1
    if (batch_size <= 0) batch_size = 1;
    if (batch_size > 32) batch_size = 32; // Max 32 batches

    int payload_len = 0;

    // Timing variables
    struct timespec ts_textgen_start, ts_textgen_end;
    struct timespec ts_tokenize_start, ts_tokenize_end;
    double textgen_time = 0, tokenize_time = 0;
    
	if (client_cfg->send_text) {
        // Mode B: Send Raw Text (size_kb * 1024 bytes)
        int num_chars = size_kb * 1024;
        payload_len = num_chars;
        if (posix_memalign((void **)&send_buf_ptr, 64, payload_len) != 0) return DOCA_ERROR_NO_MEMORY;
        
        // Measure Text Generation
        clock_gettime(CLOCK_MONOTONIC, &ts_textgen_start);
        char *text_ptr = (char*)send_buf_ptr;
        for(int i=0; i<num_chars; i++) {
            text_ptr[i] = 'a' + (rand() % 26);
            if(i % 5 == 4) text_ptr[i] = ' ';
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_textgen_end);
        textgen_time = (ts_textgen_end.tv_sec - ts_textgen_start.tv_sec) * 1e6 + 
                       (ts_textgen_end.tv_nsec - ts_textgen_start.tv_nsec) / 1e3;
        
        DOCA_LOG_INFO("DPU_TEXTGEN_TIME: %.2f us", textgen_time);
        DOCA_LOG_INFO("DPU_TOKENIZE_TIME: 0.00 us");
        DOCA_LOG_INFO("DPU_TOTAL_TIME: %.2f us", textgen_time);
        DOCA_LOG_INFO("Payload: Text, %d chars, %d KB", num_chars, size_kb);
    } else {
        // Mode A: DPU Tokenization with Batch Support
        int total_chars = size_kb * 1024;
        int chars_per_batch = total_chars / batch_size;
        int max_tokens_per_batch = chars_per_batch / 4;  /* Conservative estimate */
        int total_max_tokens = batch_size * max_tokens_per_batch;

        // Step 1: Allocate and Generate Text for all batches
        char *text_buf = (char*)malloc(total_chars);

        clock_gettime(CLOCK_MONOTONIC, &ts_textgen_start);
        for(int i = 0; i < total_chars; i++) {
            text_buf[i] = 'a' + (rand() % 26);
            if(i % 5 == 4) text_buf[i] = ' ';
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_textgen_end);
        textgen_time = (ts_textgen_end.tv_sec - ts_textgen_start.tv_sec) * 1e6 +
                       (ts_textgen_end.tv_nsec - ts_textgen_start.tv_nsec) / 1e3;

        // Step 2: Allocate Token Buffer for all batches
        payload_len = total_max_tokens * sizeof(int32_t);
        if (posix_memalign((void **)&send_buf_ptr, 64, payload_len) != 0) {
            free(text_buf);
            return DOCA_ERROR_NO_MEMORY;
        }
        
        // Step 3: Batch Tokenization (Real BPE or Hash fallback)
        clock_gettime(CLOCK_MONOTONIC, &ts_tokenize_start);
        int32_t *tokens = (int32_t*)send_buf_ptr;
        int seq_len = max_tokens_per_batch;  /* Fixed sequence length per batch */
        int total_tokens = 0;
        const char *tokenizer_mode;

        for (int b = 0; b < batch_size; b++) {
            char *batch_text = text_buf + (b * chars_per_batch);
            int32_t *batch_tokens = tokens + (b * seq_len);
            int batch_token_count;

            if (g_bpe_initialized) {
                /* Use real GPT-2 BPE tokenization */
                batch_token_count = bpe_encode(&g_bpe_ctx, batch_text, chars_per_batch,
                                               batch_tokens, seq_len);
                if (batch_token_count < 0) {
                    /* BPE failed, use hash fallback */
                    batch_token_count = seq_len;
                    for(int i = 0; i < seq_len; i++) {
                        int base = i * 4;
                        uint32_t hash = 5381;
                        for(int j = 0; j < 4 && (base + j) < chars_per_batch; j++) {
                            hash = ((hash << 5) + hash) + batch_text[base + j];
                        }
                        batch_tokens[i] = hash % 50257;
                    }
                    tokenizer_mode = "Hash(fallback)";
                } else {
                    /* Pad to fixed seq_len if needed */
                    for (int i = batch_token_count; i < seq_len; i++) {
                        batch_tokens[i] = 0;  /* PAD token */
                    }
                    batch_token_count = seq_len;
                    tokenizer_mode = "BPE";
                }
            } else {
                /* Fallback: Hash-based tokenization */
                batch_token_count = seq_len;
                for(int i = 0; i < seq_len; i++) {
                    int base = i * 4;
                    uint32_t hash = 5381;
                    for(int j = 0; j < 4 && (base + j) < chars_per_batch; j++) {
                        hash = ((hash << 5) + hash) + batch_text[base + j];
                    }
                    batch_tokens[i] = hash % 50257;
                }
                tokenizer_mode = "Hash";
            }
            total_tokens += batch_token_count;
        }

        /* Update payload length: batch_size * seq_len tokens */
        payload_len = total_tokens * sizeof(int32_t);

        clock_gettime(CLOCK_MONOTONIC, &ts_tokenize_end);
        tokenize_time = (ts_tokenize_end.tv_sec - ts_tokenize_start.tv_sec) * 1e6 +
                        (ts_tokenize_end.tv_nsec - ts_tokenize_start.tv_nsec) / 1e3;

        free(text_buf);

        DOCA_LOG_INFO("DPU_TEXTGEN_TIME: %.2f us", textgen_time);
        DOCA_LOG_INFO("DPU_TOKENIZE_TIME: %.2f us", tokenize_time);
        DOCA_LOG_INFO("DPU_TOTAL_TIME: %.2f us", textgen_time + tokenize_time);
        DOCA_LOG_INFO("Payload: %s Tokens, batch=%d, seq_len=%d, total=%d tokens, %d KB",
                      tokenizer_mode, batch_size, seq_len, total_tokens, size_kb);
    }

	if (payload_len > GPU_BUF_SIZE_A) {
	    DOCA_LOG_WARN("Payload (%d) exceeds buffer (%d)! Truncating.", payload_len, GPU_BUF_SIZE_A);
        payload_len = GPU_BUF_SIZE_A;
	}
	
	result = doca_mmap_create(&local_mmap); 
	if (result != DOCA_SUCCESS) return result;
	doca_mmap_set_memrange(local_mmap, send_buf_ptr, payload_len);
    // ... continues
    doca_mmap_add_dev(local_mmap, res.dev);
    // Add RDMA permissions for Export
	doca_mmap_set_permissions(local_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE);
	doca_mmap_start(local_mmap);

	result = doca_mmap_export_rdma(local_mmap, res.dev, &local_mmap_export, &local_mmap_len);
	if (result != DOCA_SUCCESS) { fprintf(stderr, "MMap export failed\n"); return result; }
	fprintf(stderr, "Sending Local MMap Len: %zu\n", local_mmap_len);
	send(sock_fd, &local_mmap_len, sizeof(size_t), 0);
	send(sock_fd, local_mmap_export, local_mmap_len, 0);

	doca_buf_inventory_create(10, &buf_inv);
	doca_buf_inventory_start(buf_inv);
	doca_buf_inventory_buf_get_by_addr(buf_inv, local_mmap, send_buf_ptr, strlen(TEST_STRING)+1, &src_buf);
	
	void *remote_addr; size_t remote_len;
	doca_mmap_get_memrange(remote_mmap, &remote_addr, &remote_len);
	doca_buf_inventory_buf_get_by_addr(buf_inv, remote_mmap, remote_addr, strlen(TEST_STRING)+1, &dst_buf);

	task_user_data.ptr = &task_finished;
	result = doca_rdma_task_write_allocate_init(res.rdma, connection, src_buf, dst_buf, task_user_data, &write_task);
	if (result != DOCA_SUCCESS) { fprintf(stderr, "Task Alloc Failed: %s\n", doca_error_get_descr(result)); return result; }
	task = doca_rdma_task_write_as_task(write_task);
	
	fprintf(stderr, "Submitting Task...\n");
	
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	doca_task_submit(task);
	
	while (task_finished == 0) {
		doca_pe_progress(res.pe);
		// nanosleep(&ts, &ts); // Busy poll for benchmark accuracy
	}
	
	clock_gettime(CLOCK_MONOTONIC, &end);
	double elapsed_us = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
	
	if (task_finished == 1) {
	    fprintf(stderr, "RDMA Write SUCCESS. Latency: %.2f us\n", elapsed_us);
	    printf("LATENCY_RESULT: %.2f\n", elapsed_us);
	    
	    // Send Trigger to Server
	    int signal = 1;
	    send(sock_fd, &signal, sizeof(int), 0);
	}
	else fprintf(stderr, "RDMA Write FAILURE\n");
	
	close(sock_fd);
	return DOCA_SUCCESS;
}

// Callback moved to main
