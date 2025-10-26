all: quality
check_dirs := gjepa experiments
# Check that source code meets quality standards

quality:
	pre-commit run --all-files
	mypy --install-types --non-interactive $(check_dirs)

fix:
	pre-commit run --all-files

install_cpu:
	# proper order prevents from issues with linking libraries
	pip install torch==2.7.1 -f https://download.pytorch.org/whl/cpu
	pip install --no-build-isolation pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
	pip install torch_geometric==2.6.1
	pip install -r requirements.txt

install_gpu:
	pip install -r requirements-gpu.txt -r requirements.txt

install_macos: install_cpu
