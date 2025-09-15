# Makefile for HDMIBabelfishV2 Docker

# Docker configuration
IMAGE_NAME = babelfishv2
DOCKER_HUB = mahfuzur552/babelfishv2
TAG = latest
PROJECT_DIR = $(shell pwd)

# Default target
.PHONY: help
help:
	@echo "HDMIBabelfishV2 Docker Commands:"
	@echo "  make build        - Build the Docker image"
	@echo "  make run          - Run container with GPU support"
	@echo "  make run-cpu      - Run container with CPU only"
	@echo "  make jetson-video - Run Jetson video translation pipeline"
	@echo "  make test-gpu     - Test GPU availability in container"
	@echo "  make help         - Show this help message"

# Build the Docker image
.PHONY: build
build:
	@echo "Building $(IMAGE_NAME) Docker image..."
	@sudo docker build -t $(IMAGE_NAME):$(TAG) .

# Run container with GPU support (default)
.PHONY: run
run:
	@echo "Starting $(IMAGE_NAME) with GPU support..."
	@sudo docker run -it --runtime=nvidia \
		-e DISPLAY=$DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		--net=host \
		-v $(pwd):/workspace \
		--device=/dev/video0:/dev/video0 \
		--device=/dev/video1:/dev/video1 \
		--privileged \
		mahfuzur552/babelfishv2:latest

# Run container with CPU only
.PHONY: run-cpu
run-cpu:
	@echo "Starting $(IMAGE_NAME) with CPU only..."
	@sudo docker run -it --rm \
		--name hdmibabelfish-cpu \
		--network host \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $(PROJECT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME):$(TAG)

# Run Jetson video translation pipeline
.PHONY: jetson-video
jetson-video:
	@echo "Running Jetson video translation pipeline..."
	@sudo docker run --rm \
		--runtime nvidia \
		--network host \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $(PROJECT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME):$(TAG) \
		python3 pipeline/jetson_video.py

# Test GPU availability
.PHONY: test-gpu
test-gpu:
	@echo "Testing GPU in $(IMAGE_NAME) container..."
	@sudo docker run --rm \
		--runtime nvidia \
		-v $(PROJECT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME):$(TAG) \
		python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"