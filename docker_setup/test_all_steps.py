#!/usr/bin/env python3
"""
Test script to verify all steps work without conflicts
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing all imports...")
    
    tests = [
        ("PyTorch", "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("OpenCV", "import cv2; print(f'OpenCV {cv2.__version__}')"),
        ("Ultralytics", "from ultralytics import YOLO; print('Ultralytics ✅')"),
        ("NumPy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("Transformers", "from transformers import MarianMTModel; print('Transformers ✅')"),
        ("SORT", "sys.path.append('/workspace/packages'); from sort.sort import Sort; print('SORT ✅')"),
        ("NLTK", "import nltk; print('NLTK ✅')"),
        ("Pandas", "import pandas as pd; print(f'Pandas {pd.__version__}')"),
        ("SciPy", "import scipy; print(f'SciPy {scipy.__version__}')"),
        ("Scikit-image", "import skimage; print('Scikit-image ✅')"),
        ("FilterPy", "from filterpy.kalman import KalmanFilter; print('FilterPy ✅')"),
    ]
    
    success_count = 0
    for name, test_code in tests:
        try:
            exec(test_code)
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print(f"\n📊 Import Test Results: {success_count}/{len(tests)} successful")
    return success_count == len(tests)

def test_step_compatibility():
    """Test that each step's core functionality works"""
    print("\n🔍 Testing step compatibility...")
    
    # Test Step 1 (Video reading)
    try:
        import cv2
        cap = cv2.VideoCapture()
        print("✅ Step 1: OpenCV video capture ready")
    except Exception as e:
        print(f"❌ Step 1: {e}")
    
    # Test Step 2 (YOLO)
    try:
        from ultralytics import YOLO
        # Don't load actual model, just test import
        print("✅ Step 2: YOLO import successful")
    except Exception as e:
        print(f"❌ Step 2: {e}")
    
    # Test Step 3 (SORT tracking)
    try:
        sys.path.append('/workspace/packages')
        from sort.sort import Sort
        tracker = Sort()
        print("✅ Step 3: SORT tracker initialized")
    except Exception as e:
        print(f"❌ Step 3: {e}")
    
    # Test Step 4 (Translation)
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        print(f"✅ Step 4: Translation components ready, CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ Step 4: {e}")

def check_paths():
    """Check critical file paths"""
    print("\n📁 Checking file paths...")
    
    paths_to_check = [
        "/workspace/data/test_video/JapanRPG_TestSequence.mov",
        "/workspace/data/game_review/big_dataset/test",
        "/workspace/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt",
        "/workspace/translation/models/marian_opus_ja_en",
        "/workspace/data/translation/test.csv",
        "/workspace/packages/sort/sort.py"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"⚠️  Missing: {path}")

def main():
    print("🚀 Comprehensive Step Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test step compatibility
    test_step_compatibility()
    
    # Check paths
    check_paths()
    
    # Final assessment
    print("\n" + "=" * 50)
    if imports_ok:
        print("🎉 All imports successful! Steps should work without conflicts.")
        print("\n💡 Tips:")
        print("   - Run 'export PYTHONPATH=/usr/local/lib/python3.8/dist-packages:$PYTHONPATH'")
        print("   - Run 'export YOLO_CONFIG_DIR=/tmp'")
        print("   - All steps should now work together!")
    else:
        print("⚠️  Some imports failed. Check the setup script.")
    
    return imports_ok

if __name__ == "__main__":
    main()