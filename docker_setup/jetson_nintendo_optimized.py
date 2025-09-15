#!/usr/bin/env python3
"""
Test script to verify all output_nintendo.py dependencies work
"""

import sys
import os

def test_nintendo_dependencies():
    """Test all dependencies needed for output_nintendo.py"""
    print("🎮 Testing Nintendo Output Dependencies...")
    print("=" * 50)
    
    tests = [
        # Existing dependencies (should still work)
        ("PyTorch + CUDA", "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("OpenCV", "import cv2; print(f'OpenCV {cv2.__version__}')"),
        ("Ultralytics YOLO", "from ultralytics import YOLO; print('YOLO ✅')"),
        ("Transformers", "from transformers import MarianMTModel, MarianTokenizer; print('Transformers ✅')"),
        ("SORT Tracking", "sys.path.append('/workspace/packages'); from sort.sort import Sort; print('SORT ✅')"),
        ("NLTK", "import nltk; print('NLTK ✅')"),
        ("Pandas", "import pandas as pd; print(f'Pandas {pd.__version__}')"),
        ("NumPy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("Matplotlib", "import matplotlib.pyplot as plt; print('Matplotlib ✅')"),
        
        # New OCR dependencies
        ("pytesseract", "import pytesseract; print('pytesseract ✅')"),
        ("PaddleOCR", "from paddleocr import PaddleOCR; print('PaddleOCR ✅')"),
        
        # System dependencies
        ("Tesseract Binary", """
import subprocess
result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
if result.returncode == 0:
    print('Tesseract OCR engine ✅')
else:
    raise Exception('Tesseract not found')
"""),
    ]
    
    success_count = 0
    failed_tests = []
    
    for name, test_code in tests:
        try:
            exec(test_code)
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed_tests.append(name)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{len(tests)} successful")
    
    if failed_tests:
        print("❌ Failed tests:")
        for test in failed_tests:
            print(f"   - {test}")
        return False
    else:
        print("🎉 All dependencies working!")
        return True

def test_nintendo_file_paths():
    """Test that required file paths exist"""
    print("\n📁 Testing file paths for output_nintendo.py...")
    
    paths_to_check = [
        ("/workspace/data/test_video/JapanRPG_TestSequence.mov", "Test video file"),
        ("/workspace/packages/sort/sort.py", "SORT module"),
        ("/workspace/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt", "YOLO model"),
        ("/workspace/translation/models/marian_opus_ja_en", "Translation model"),
    ]
    
    missing_paths = []
    for path, description in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
        else:
            print(f"⚠️  {description}: MISSING - {path}")
            missing_paths.append((path, description))
    
    return len(missing_paths) == 0

def test_ocr_functionality():
    """Test OCR functionality with a simple test"""
    print("\n🔤 Testing OCR functionality...")
    
    try:
        # Test PaddleOCR (preferred)
        from paddleocr import PaddleOCR
        import numpy as np
        
        # Create a simple test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)
        print("✅ PaddleOCR initialized successfully")
        
        # Test basic OCR (without actual text, should return empty)
        result = ocr.ocr(test_image)
        print("✅ PaddleOCR test completed")
        
    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        return False
    
    try:
        # Test pytesseract
        import pytesseract
        import cv2
        import numpy as np
        
        # Create a simple test image with text
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Test OCR
        text = pytesseract.image_to_string(gray).strip()
        print(f"✅ pytesseract test completed (detected: '{text}')")
        
    except Exception as e:
        print(f"❌ pytesseract test failed: {e}")
        return False
    
    return True

def main():
    print("🎮 Nintendo Output Dependencies Verification")
    print("=" * 60)
    
    # Test all dependencies
    deps_ok = test_nintendo_dependencies()
    
    # Test file paths
    paths_ok = test_nintendo_file_paths()
    
    # Test OCR functionality
    ocr_ok = test_ocr_functionality()
    
    # Final assessment
    print("\n" + "=" * 60)
    if deps_ok and paths_ok and ocr_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ output_nintendo.py should work with your video file!")
        print("\n💡 Next steps:")
        print("   1. Copy output_nintendo.py to your docker container")
        print("   2. Modify it to use your video file instead of Nintendo capture")
        print("   3. Run the modified version")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        if not deps_ok:
            print("   - Install missing dependencies")
        if not paths_ok:
            print("   - Check file paths")
        if not ocr_ok:
            print("   - Fix OCR installation")
    
    return deps_ok and paths_ok and ocr_ok

if __name__ == "__main__":
    main()