/*
 * DPU Client Main for GPUDirect RDMA Test
 * Based on NVIDIA DOCA Samples
 */

#include <stdlib.h>
#include <string.h>
#include <doca_log.h>
#include <doca_argp.h>
#include "rdma_common.h"

DOCA_LOG_REGISTER(DPU_CLIENT::MAIN);

/* ARGP Callback - Handle IB device name parameter */
static doca_error_t device_address_callback(void *param, void *config)
{
	struct rdma_config *rdma_cfg = (struct rdma_config *)config;
	char *device_name = (char *)param;
	int len = strnlen(device_name, DOCA_DEVINFO_IBDEV_NAME_SIZE);
	if (len == DOCA_DEVINFO_IBDEV_NAME_SIZE) return DOCA_ERROR_INVALID_VALUE;
	strncpy(rdma_cfg->device_name, device_name, len + 1);
	return DOCA_SUCCESS;
}

/* ARGP Callback - Handle server IP parameter */
static doca_error_t server_ip_callback(void *param, void *config)
{
	struct rdma_config *rdma_cfg = (struct rdma_config *)config;
	char *server_ip = (char *)param;
	int len = strnlen(server_ip, DOCA_DEVINFO_IBDEV_NAME_SIZE);
	if (len == DOCA_DEVINFO_IBDEV_NAME_SIZE) return DOCA_ERROR_INVALID_VALUE;
	strncpy(rdma_cfg->server_ip_addr, server_ip, len + 1);
	return DOCA_SUCCESS;
}

/* ARGP Callback - Handle GID index parameter */
static doca_error_t gid_index_callback(void *param, void *config)
{
	struct rdma_config *rdma_cfg = (struct rdma_config *)config;
	rdma_cfg->gid_index = *(uint32_t *)param;
	rdma_cfg->is_gid_index_set = true;
	return DOCA_SUCCESS;
}

static doca_error_t send_text_callback(void *param, void *config)
{
	struct rdma_config *client_cfg = (struct rdma_config *)config;
	client_cfg->send_text = true;
	return DOCA_SUCCESS;
}

static doca_error_t register_rdma_params(void)
{
	doca_error_t result;
	struct doca_argp_param *dev_param, *ip_param, *gid_param, *text_param;

	/* Device Name */
	result = doca_argp_param_create(&dev_param);
	if (result != DOCA_SUCCESS) return result;
	doca_argp_param_set_short_name(dev_param, "d");
	doca_argp_param_set_long_name(dev_param, "device");
	doca_argp_param_set_arguments(dev_param, "<IB device name>");
	doca_argp_param_set_description(dev_param, "IB device name (e.g. mlx5_0)");
	doca_argp_param_set_callback(dev_param, device_address_callback);
	doca_argp_param_set_type(dev_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(dev_param);
	result = doca_argp_register_param(dev_param);
	if (result != DOCA_SUCCESS) return result;

	/* Server IP */
	result = doca_argp_param_create(&ip_param);
	if (result != DOCA_SUCCESS) return result;
	doca_argp_param_set_short_name(ip_param, "s");
	doca_argp_param_set_long_name(ip_param, "server-ip");
	doca_argp_param_set_arguments(ip_param, "<IP>");
	doca_argp_param_set_description(ip_param, "Host Server IP Address");
	doca_argp_param_set_callback(ip_param, server_ip_callback);
	doca_argp_param_set_type(ip_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(ip_param);
	result = doca_argp_register_param(ip_param);
	if (result != DOCA_SUCCESS) return result;

	/* GID Index */
	result = doca_argp_param_create(&gid_param);
	if (result != DOCA_SUCCESS) return result;
	doca_argp_param_set_short_name(gid_param, "g");
	doca_argp_param_set_long_name(gid_param, "gid-index");
	doca_argp_param_set_arguments(gid_param, "<GID Index>");
	doca_argp_param_set_description(gid_param, "GID Index for RDMA");
	doca_argp_param_set_callback(gid_param, gid_index_callback);
	doca_argp_param_set_type(gid_param, DOCA_ARGP_TYPE_INT);
	result = doca_argp_register_param(gid_param);
	/* Send Text Flag */
	result = doca_argp_param_create(&text_param);
	if (result != DOCA_SUCCESS) return result;
	doca_argp_param_set_short_name(text_param, "T");
	doca_argp_param_set_long_name(text_param, "send-text");
	doca_argp_param_set_description(text_param, "Send Raw Text (Scenario B)");
	doca_argp_param_set_callback(text_param, send_text_callback);
	doca_argp_param_set_type(text_param, DOCA_ARGP_TYPE_BOOLEAN);
	result = doca_argp_register_param(text_param);
	if (result != DOCA_SUCCESS) return result;

	return DOCA_SUCCESS;
}

/* Implemented in dpu_client_sample.c */
doca_error_t dpu_client_start(struct rdma_config *cfg);

int main(int argc, char **argv)
{
	struct rdma_config cfg = {0};
	doca_error_t result;
	struct doca_log_backend *sdk_log;

	/* Setup Logger */
	doca_log_backend_create_standard();
	doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);

	/* Parse Args */
	doca_argp_init("dpu_rdma_client", &cfg);
	register_rdma_params();
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Arg parse failed: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = dpu_client_start(&cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Client failed: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Success!");
	return EXIT_SUCCESS;
}
