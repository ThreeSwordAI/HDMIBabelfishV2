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