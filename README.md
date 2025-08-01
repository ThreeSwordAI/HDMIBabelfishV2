Creating Venv:

python3 -m venv ~/venv

To activate:

source venv/bin/activate

Also:

pip install --upgrade pip wheel

Everytime it will auto open:

echo 'source ~/Desktop/HDMIBabelfishV2/venv/bin/activate' >> ~/.bashrc

For torch:

https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

As, I am using python 3.8 so, I downloaded

JetPack 5: PyTorch v2.1.0

then:

pip install  [torch file that you downloaded].whl


pip install torchvision==0.16.0 torchaudio==2.1.0



sudo apt update
sudo apt install libopenblas-base libopenblas-dev

to see the GPU is working or not

python3 gpu_testing.py 

pip install numpy matplotlib opencv-python


pip install ultralytics


sudo apt install tesseract-ocr libtesseract-dev
sudo apt install tesseract-ocr-jpn
pip install pytesseract


pip install paddleocr
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/paddle.html


pip install transformers accelerate


pip install sentencepiece


sudo apt install libavcodec-extra

pip install scikit-image

pip install filterpy



pip install googletrans==4.0.0rc1




# HDMIBabelFishV2 Project Setup & Installation

This README details the **step-by-step installation process** for setting up on Jetson Orin NX development environment using Docker and running your `HDMIBabelfishV2` project.

---

## 1. Host Machine Preparation

1. **Update package lists & install Docker**  
```bash
sudo apt update
sudo apt install docker.io -y
```

2. **Start & enable Docker service**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

2. **Add your user to the docker group**
```bash
sudo usermod -aG docker $USER
```

2. **Reboot to apply group changes**
```bash
sudo reboot
```

## 2. Pull & Run the NVIDIA Jetson-PyTorch Container
2. **pull the Jetson-compatible PyTorch image**
```bash
sudo docker pull dustynv/l4t-pytorch:r35.3.1
```

2. **Run the container with your project folder mounted**
```bash
sudo docker run -it --rm --runtime nvidia \
  -v ~/Desktop/HDMIBabelfishV2:/workspace \
  -w /workspace \
  dustynv/l4t-pytorch:r35.3.1
```

-v: mounts your project at /workspace inside the container
-w: sets the working directory to /workspace

## 3. Inside the Docker Container
Once inside the container (root@...:/workspace):

2. **Verify CUDA support**
```bash
python3 CUDA_test.py
```

2. **Install system dependencies**
```bash
apt-get update && apt-get install -y \
  tesseract-ocr \
  libxml2-dev libxslt1-dev \
  python3-dev build-essential \
  libsm6 libxext6 libgl1-mesa-glx ffmpeg
```

2. **Install required Python libraries**
```bash
pip install transformers \
            ultralytics \
            matplotlib \
            pytesseract \
            paddleocr \
            scikit-image \
            filterpy \
            scikit-learn
```

Note: If paddleocr fails due to lxml, ensure libxml2-dev and libxslt1-dev are installed first.

## 4. Run Your Scripts
From /workspace/(proper folder):
```bash
python3 CUDA_test.py
python3 CUDA_version_Testing.py
python3 output_jetson.py
```

## 5. Save Your Custom Docker Image
List running containers (to get the Container ID):

```bash
docker ps
```
Commit your configured container with a custom name:
```bash
docker commit [container-ID] hdmibabelfishv2docker
```
Relaunch your custom image:
```bash
sudo docker run -it --rm --runtime nvidia \
  -v ~/Desktop/HDMIBabelfishV2:/workspace \
  -w /workspace \
  hdmibabelfishv2docker
```