# 🎮 HDMIBabelfishV2 - Real-Time Japanese Game Translation

[![Docker Hub](https://img.shields.io/badge/Docker-mahfuzur552%2Fbabelfishv2-blue?logo=docker)](https://hub.docker.com/r/mahfuzur552/babelfishv2)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin-green?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)

Real-time AI translation system for Japanese games on NVIDIA Jetson Orin devices.

## 🛠️ System Requirements

- **NVIDIA Jetson Orin NX 16GB**
- **Docker** with NVIDIA Container Runtime
- **Ubuntu 20.04** (JetPack 5.0+)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/ThreeSwordAI/HDMIBabelfishV2.git
cd HDMIBabelfishV2
```

### 2. Pull Docker Image
```bash
sudo docker pull mahfuzur552/babelfishv2:latest
```

### 3. Run Container
```bash
sudo docker run -it --runtime=nvidia \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --net=host \
  -v $(pwd):/workspace \
  mahfuzur552/babelfishv2:latest
```

### 4. Test Setup
```bash
# Inside container
cd /workspace/docker_setup

# Test GPU
python3 gpu_testing.py

# Run translation pipeline
python3 jetson_nintendo_optimized.py
```

## 🤖 Models Used

- **YOLO**: YOLOv11 nano for text detection
- **Translation**: Marian OPUS Japanese→English
- **OCR**: pytesseract + PaddleOCR
- **Tracking**: SORT algorithm

## 📊 Performance

- **Target Device**: Jetson Orin NX 16GB
- **Current FPS**: 11.9 average
- **GPU Usage**: 0.2GB (optimizable to 8GB+)
- **Latency**: ~84ms

## 🔧 Docker Commands

```bash
# Pull latest image
sudo docker pull mahfuzur552/babelfishv2:latest

# Run with GPU support
sudo docker run -it --runtime=nvidia \
  -v $(pwd):/workspace \
  mahfuzur552/babelfishv2:latest

# Save container changes
sudo docker commit <container_id> babelfishv2_updated

# Push to Docker Hub
sudo docker tag babelfishv2_updated mahfuzur552/babelfishv2:latest
sudo docker push mahfuzur552/babelfishv2:latest
```