#!/bin/bash

echo "🚀 Setting up comprehensive environment for all steps..."

# Clean up any existing conflicting installations
echo "🧹 Cleaning up existing installations..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless transformers ultralytics matplotlib pandas numpy scipy scikit-image filterpy

# Remove problematic cv2 directories
rm -rf /usr/local/lib/python3.8/dist-packages/cv2*

# Set up Python path to prioritize pip installations
export PYTHONPATH=/usr/local/lib/python3.8/dist-packages:$PYTHONPATH

# Install core dependencies in specific order to avoid conflicts
echo "📦 Installing core dependencies..."

# 1. Install numpy first (foundation for everything)
pip install --no-cache-dir numpy==1.24.4

# 2. Install scipy and scientific stack
pip install --no-cache-dir scipy==1.10.1

# 3. Install computer vision dependencies
pip install --no-cache-dir opencv-python==4.5.5.64

# 4. Install matplotlib and plotting
pip install --no-cache-dir matplotlib==3.7.5

# 5. Install pandas for data handling
pip install --no-cache-dir pandas==2.0.3

# 6. Install YOLO and tracking dependencies
pip install --no-cache-dir psutil py-cpuinfo ultralytics-thop
pip install --no-cache-dir scikit-image==0.21.0
pip install --no-cache-dir filterpy==1.4.5
pip install --no-cache-dir ultralytics==8.0.196

# 7. Install NLP dependencies
pip install --no-cache-dir tokenizers==0.13.3
pip install --no-cache-dir sentencepiece==0.1.99
pip install --no-cache-dir protobuf==3.20.3
pip install --no-cache-dir transformers==4.31.0
pip install --no-cache-dir nltk==3.8.1

# 8. Download NLTK data needed for BLEU scores
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    print('✅ NLTK data downloaded')
except:
    print('⚠️ NLTK download failed, continuing...')
"

# Set environment variables for better compatibility
echo "🔧 Setting environment variables..."
export YOLO_CONFIG_DIR=/tmp
export PYTHONPATH=/usr/local/lib/python3.8/dist-packages:$PYTHONPATH

# Test all imports
echo "🧪 Testing all imports..."
python3 -c "
import sys
print('Python path:', sys.path[:3])

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('✅ CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('❌ PyTorch error:', e)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except Exception as e:
    print('❌ OpenCV error:', e)

try:
    from ultralytics import YOLO
    print('✅ Ultralytics imported')
except Exception as e:
    print('❌ Ultralytics error:', e)

try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except Exception as e:
    print('❌ NumPy error:', e)

try:
    from transformers import MarianMTModel
    print('✅ Transformers imported')
except Exception as e:
    print('❌ Transformers error:', e)

try:
    from sort.sort import Sort
    print('✅ SORT imported')
except Exception as e:
    print('❌ SORT error:', e)

try:
    import nltk
    print('✅ NLTK imported')
except Exception as e:
    print('❌ NLTK error:', e)
"

echo "✅ Environment setup complete!"
echo "💡 Run 'source setup_environment.sh' to set environment variables"
echo "🎯 All steps should now work without conflicts"