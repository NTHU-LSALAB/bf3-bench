project('DPU_RDMA_CLIENT', 'C',
	version: '1.0',
	default_options: ['buildtype=debug']
)

add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c'])

deps = []
deps += dependency('doca-common')
deps += dependency('doca-argp')
deps += dependency('doca-rdma')
# deps += dependency('libdpdk')  # Not needed for DPU client

srcs = [
	'dpu_client_main.c',
	'dpu_client_sample.c',
	'common.c',
	'bpe_tokenizer.c',
]

inc_dirs = include_directories('.')

executable('dpu_rdma_client', srcs,
	dependencies : deps,
	include_directories: inc_dirs,
	install: false)
