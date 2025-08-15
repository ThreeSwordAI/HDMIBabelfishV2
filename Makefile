# Makefile for HDMIBabelfishV2 Docker
# Location: /home/babel-fish/Desktop/HDMIBabelfishV2/Makefile

# Docker image name
IMAGE_NAME = babelfishv2
TAG = latest

# Project directory (current directory)
PROJECT_DIR = $(shell pwd)

# Default target
.PHONY: help
help:
	@echo "HDMIBabelfishV2 Docker Commands:"
	@echo "  make run       - Run the Docker container"
	@echo "  make run-gpu   - Run with GPU and video device support"
	@echo "  make shell     - Open interactive shell in container"
	@echo "  make test-cuda - Test CUDA availability in container"
	@echo "  make test-all  - Run full system test"
	@echo "  make clean     - Stop all running containers"
	@echo "  make logs      - Show container logs"

# Basic run
.PHONY: run
run:
	@echo "Starting $(IMAGE_NAME) container..."
	@sudo docker run -it --rm \
		--name hdmibabelfish \
		--runtime nvidia \
		--network host \
		--volume $(PROJECT_DIR):/workspace \
		--workdir /workspace \
		$(IMAGE_NAME):$(TAG)

# Run with full GPU and video support
.PHONY: run-gpu
run-gpu:
	@echo "Starting $(IMAGE_NAME) with GPU and video support..."
	@sudo docker run -it --rm \
		--name hdmibabelfish \
		--runtime nvidia \
		--network host \
		--device /dev/video0:/dev/video0 \
		--device /dev/video1:/dev/video1 \
		--volume $(PROJECT_DIR):/workspace \
		--workdir /workspace \
		--volume /tmp/.X11-unix:/tmp/.X11-unix \
		--env DISPLAY=$(DISPLAY) \
		--env QT_X11_NO_MITSHM=1 \
		--privileged \
		$(IMAGE_NAME):$(TAG)

# Open shell in container
.PHONY: shell
shell:
	@echo "Opening shell in $(IMAGE_NAME) container..."
	@sudo docker run -it --rm \
		--name hdmibabelfish-shell \
		--runtime nvidia \
		--network host \
		--volume $(PROJECT_DIR):/workspace \
		--workdir /workspace \
		$(IMAGE_NAME):$(TAG) \
		/bin/bash

# Test CUDA in container
.PHONY: test-cuda
test-cuda:
	@echo "Testing CUDA in $(IMAGE_NAME) container..."
	@sudo docker run --rm \
		--runtime nvidia \
		--volume $(PROJECT_DIR):/workspace \
		--workdir /workspace \
		$(IMAGE_NAME):$(TAG) \
		python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Run full system test
.PHONY: test-all
test-all:
	@echo "Running full system test..."
	@sudo docker run --rm \
		--runtime nvidia \
		--volume $(PROJECT_DIR):/workspace \
		--workdir /workspace \
		$(IMAGE_NAME):$(TAG) \
		python3 jetson/CUDA_test.py

# Stop all containers
.PHONY: clean
clean:
	@echo "Stopping all HDMIBabelfish containers..."
	@sudo docke