#!/bin/bash

echo "🎮 Installing Nintendo Output Dependencies..."

# Set environment variables to prevent conflicts
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/usr/local/lib/python3.8/dist-packages:$PYTHONPATH
export YOLO_CONFIG_DIR=/tmp

# Install system dependencies for Tesseract OCR
echo "📦 Installing Tesseract OCR system packages..."
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-jpn

# Install Python OCR libraries
echo "🔤 Installing Python OCR libraries..."

# Install PaddleOCR (recommended for Jetson)
pip install paddlepaddle-gpu==2.4.2 -f https://www.paddlepaddle.org.cn/whl/jetson/stable.html
pip install paddleocr==2.7.0.3

# Install pytesseract as backup
pip install pytesseract==0.3.10

# Install any other missing dependencies
pip install pillow>=8.0.0

# Test OCR installations
echo "🧪 Testing OCR installations..."

python3 -c "
try:
    import pytesseract
    print('✅ pytesseract imported successfully')
    # Test if tesseract binary is accessible
    import subprocess
    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ Tesseract OCR engine working')
    else:
        print('❌ Tesseract OCR engine not found')
except Exception as e:
    print(f'❌ pytesseract error: {e}')

try:
    from paddleocr import PaddleOCR
    print('✅ PaddleOCR imported successfully')
    # Test basic PaddleOCR functionality
    ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)
    print('✅ PaddleOCR initialized successfully')
except Exception as e:
    print(f'❌ PaddleOCR error: {e}')
"

echo "✅ Nintendo output dependencies installation complete!"
echo ""
echo "📝 Installation Summary:"
echo "   - Tesseract OCR engine: System package"
echo "   - Japanese language pack: tesseract-ocr-jpn" 
echo "   - pytesseract: Python wrapper for Tesseract"
echo "   - PaddleOCR: AI-powered OCR (recommended for Jetson)"
echo ""
echo "🎯 Ready to test output_nintendo.py!"