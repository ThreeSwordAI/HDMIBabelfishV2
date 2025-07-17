import cv2
import time
import threading
import queue
import numpy as np
import platform
import torch
import json
import sys
from pathlib import Path

# Add packages directory to Python path
sys.path.append('../packages')

from ultralytics import YOLO
from sort.sort import Sort  # Updated import path
from transformers import MarianTokenizer, MarianMTModel
import os

# OCR backend selection
try:
    import pytesseract
    OCR_BACKEND = 'tesseract'
except ImportError:
    from paddleocr import PaddleOCR
    OCR_BACKEND = 'paddle'

import matplotlib.pyplot as plt

# -------- NINTENDO SWITCH CONFIG --------
# Video source options:
# Option 1: HDMI Capture (HD60 S+ - RECOMMENDED FOR NINTENDO SWITCH)
USE_HD60S_CAPTURE = True      # Set to True when using HD60 S+ 
HD60S_DEVICE_ID = 1          # This will be auto-detected from your working test
HD60S_RESOLUTION = (1920, 1080)  # Nintendo Switch outputs 1080p
HD60S_FPS = 30               # Standard Nintendo Switch output

# Option 2: File input (for testing)
VIDEO_PATH = "../data/test_video/JapanRPG_TestSequence.mov"

# Option 3: Webcam (fallback)
# VIDEO_PATH = 0  # Default camera

MODEL_PATH = "../experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"  # YOLO model path
MARIAN_MODEL_PATH = "../translation/models/marian_opus_ja_en"  # Path to translation model
WINDOW_NAME = "Nintendo Switch Real-Time Translation - HD60 S+"
TARGET_FPS = 25  # Optimized for Nintendo Switch
STABILITY_MS = 800  # INCREASED from 400ms - Fix for issue 2

BASE_DIR = "output/video"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "nintendo_switch_translation.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_nintendo_switch.png")
FPS_GRAPH_PDF = os.path.join(BASE_DIR, "fps_nintendo_switch.pdf")

# Create output directory if it doesn't exist
os.makedirs(BASE_DIR, exist_ok=True)

# CUDA Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# OCR setup
if OCR_BACKEND == 'tesseract':
    OCR_LANG = "jpn"
    TESS_CONFIG = "--psm 6"
else:
    paddle_ocr = PaddleOCR(lang='japanese', use_gpu=torch.cuda.is_available())

# Global translator - will be pre-loaded
translator_model = None
translator_tokenizer = None
translation_cache = {}

class CUDAMarianTranslator:
    """CUDA-optimized Marian translation class"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load Marian model with CUDA optimizations"""
        print("="*50)
        print("LOADING OFFLINE MARIAN TRANSLATION MODEL")
        print("="*50)
        
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                print(f"❌ Model path not found: {self.model_path}")
                print("Please ensure the marian_opus_ja_en model is downloaded")
                return False
            
            # Load tokenizer
            print("📥 Loading Marian tokenizer...")
            self.tokenizer = MarianTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            # Load model
            print("📥 Loading Marian model...")
            self.model = MarianMTModel.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                local_files_only=True
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                # Enable optimizations for newer PyTorch versions
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("✓ Torch compilation enabled")
                except:
                    print("⚠ Torch compilation not available, using standard model")
            
            self.model.eval()
            
            # Test translation
            test_result = self.translate("テスト")
            print(f"✓ Test translation: 'テスト' -> '{test_result}'")
            print(f"✓ Marian model loaded successfully on {self.device}")
            
            # Print model info
            model_size = sum(p.numel() for p in self.model.parameters())
            print(f"✓ Model parameters: {model_size:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading Marian model: {e}")
            return False
    
    def translate(self, text, max_length=100):
        """Translate text using CUDA-optimized Marian inference"""
        if not text.strip():
            return ""
        
        if text in translation_cache:
            return translation_cache[text]
        
        try:
            # Tokenize with Marian tokenizer
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.device.type == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with optimizations
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast('cuda'):  # FIXED deprecation warning
                        generated_tokens = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            num_beams=4,  # Good balance for Marian
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                else:
                    generated_tokens = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            # Decode
            result = self.tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            translation_cache[text] = result
            return result
            
        except Exception as e:
            print(f"❌ Translation error for '{text}': {e}")
            error_result = f"[Error: {text}]"
            translation_cache[text] = error_result
            return error_result

def initialize_translation():
    """Initialize translation with offline Marian model"""
    global translator_model
    
    # Check if model exists
    model_path = Path(MARIAN_MODEL_PATH)
    if not model_path.exists():
        print("❌ Marian model not found!")
        print(f"Expected path: {model_path.absolute()}")
        print("Please run 'python ../translation/download_models.py' first")
        return False
    
    translator_model = CUDAMarianTranslator(MARIAN_MODEL_PATH)
    return translator_model.load_model()

def get_translation(text):
    """Get translation using CUDA Marian translator"""
    if translator_model is None:
        return f"[No translator: {text}]"
    return translator_model.translate(text)

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts - Fix for issue 2"""
    if not text1 or not text2:
        return 0.0
    
    # Remove spaces and normalize
    text1 = text1.replace(' ', '').replace('\n', '').replace('-', '').replace('_', '').replace('*', '')
    text2 = text2.replace(' ', '').replace('\n', '').replace('-', '').replace('_', '').replace('*', '')
    
    if text1 == text2:
        return 1.0
    
    # Calculate character-level similarity
    longer = max(len(text1), len(text2))
    if longer == 0:
        return 1.0
    
    # Count matching characters at same positions
    matches = sum(1 for i in range(min(len(text1), len(text2))) if text1[i] == text2[i])
    
    # Add bonus for similar length
    length_similarity = 1.0 - abs(len(text1) - len(text2)) / longer
    
    # Combine position matches with length similarity
    similarity = (matches / longer) * 0.7 + length_similarity * 0.3
    
    return similarity

def wrap_text_to_fit_box(text, box_width, font_scale=0.8, thickness=2):
    """Wrap text to fit within the detection box width - Fix for issue 1"""
    if not text:
        return []
    
    # Calculate character width for the font
    char_size = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    char_width = char_size[0]
    
    # Calculate how many characters can fit in the box width (with some padding)
    usable_width = box_width - 16  # 8px padding on each side
    chars_per_line = max(1, int(usable_width / char_width))
    
    # Split text into words
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding this word would exceed the line length
        test_line = current_line + (" " if current_line else "") + word
        
        if len(test_line) <= chars_per_line:
            current_line = test_line
        else:
            # Current line is full, start a new line
            if current_line:
                lines.append(current_line)
            
            # If the word itself is too long, split it
            if len(word) > chars_per_line:
                # Split long word across multiple lines
                while len(word) > chars_per_line:
                    lines.append(word[:chars_per_line])
                    word = word[chars_per_line:]
                current_line = word
            else:
                current_line = word
    
    # Add the last line if it has content
    if current_line:
        lines.append(current_line)
    
    return lines

def draw_multiline_text_in_box(frame, text, box, avg_color):
    """Draw multi-line text within the detection box boundaries - Fix for issue 1"""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Text styling
    font_scale = 0.8
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text metrics
    char_size = cv2.getTextSize("A", font, font_scale, thickness)[0]
    line_height = char_size[1] + 6  # Add some line spacing
    
    # Wrap text to fit in box
    lines = wrap_text_to_fit_box(text, box_width, font_scale, thickness)
    
    # Calculate total text height
    total_text_height = len(lines) * line_height
    
    # Adjust font size if text doesn't fit vertically
    if total_text_height > box_height - 16:  # 16px for top/bottom padding
        # Reduce font scale to fit
        scale_factor = (box_height - 16) / total_text_height
        font_scale = max(0.4, font_scale * scale_factor)  # Minimum font scale
        thickness = max(1, int(thickness * scale_factor))
        
        # Recalculate with new font size
        char_size = cv2.getTextSize("A", font, font_scale, thickness)[0]
        line_height = char_size[1] + 4
        lines = wrap_text_to_fit_box(text, box_width, font_scale, thickness)
        total_text_height = len(lines) * line_height
    
    # Draw background overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), avg_color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Calculate starting position for centered text
    start_y = y1 + max(8, (box_height - total_text_height) // 2) + line_height
    
    # Draw each line of text
    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            continue
            
        # Calculate y position for this line
        text_y = start_y + (i * line_height)
        
        # Make sure we don't draw outside the box
        if text_y > y2 - 8:
            # Add "..." to indicate more text
            if i > 0:  # Only if we've drawn at least one line
                prev_y = start_y + ((i-1) * line_height)
                cv2.putText(frame, "...", (x1 + 8, prev_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            break
        
        # Calculate text width for centering (optional)
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = x1 + 8  # Left align with padding
        
        # Draw text shadow for better readability
        cv2.putText(frame, line, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def find_working_hd60s_device():
    """Auto-detect working HD60 S+ device ID with enhanced detection"""
    print("🔍 Auto-detecting HD60 S+ device...")
    
    for device_id in [HD60S_DEVICE_ID, 0, 1, 2, 3, 4]:
        print(f"   Testing device ID: {device_id}")
        
        # Try multiple backends for better compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        
        for backend in backends:
            cap = cv2.VideoCapture(device_id, backend)
            
            if cap.isOpened():
                # Give camera time to initialize
                for i in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        break
                    time.sleep(0.1)
                
                if ret and test_frame is not None:
                    # More robust content detection
                    mean_color = test_frame.mean()
                    std_color = test_frame.std()
                    
                    # Check for actual video content (not solid grey/black)
                    if (10 < mean_color < 240) and (std_color > 5):
                        print(f"✅ Found working HD60 S+ on device {device_id} (backend: {backend})")
                        print(f"   Frame stats: mean={mean_color:.1f}, std={std_color:.1f}")
                        cap.release()
                        return device_id
                    else:
                        print(f"   Device {device_id}: suspicious frame (mean: {mean_color:.1f}, std: {std_color:.1f})")
                else:
                    print(f"   Device {device_id}: cannot capture frames")
            else:
                print(f"   Device {device_id}: cannot open with backend {backend}")
            
            cap.release()
    
    return None

def setup_hd60s_capture():
    """Setup HD60 S+ capture with enhanced compatibility for Nintendo Switch"""
    print("🎮 Setting up HD60 S+ capture for Nintendo Switch...")
    
    # First, check if any processes are using the device
    print("🔍 Checking for device conflicts...")
    print("💡 IMPORTANT: Make sure Elgato Game Capture software is COMPLETELY closed!")
    print("   (Check Task Manager for any Elgato processes)")
    
    # Auto-detect working device
    working_device_id = find_working_hd60s_device()
    
    if working_device_id is None:
        print("❌ No working HD60 S+ found!")
        print("💡 Troubleshooting steps:")
        print("   1. CLOSE Elgato Game Capture software completely")
        print("   2. Check Windows Task Manager:")
        print("      - End any 'Elgato' processes")
        print("      - End any 'GameCapture' processes")
        print("   3. Disconnect HD60 S+ USB cable")
        print("   4. Wait 5 seconds")
        print("   5. Reconnect HD60 S+ USB cable")
        print("   6. Try a different USB port")
        print("   7. Restart this script")
        return None
    
    # Try multiple backends for setup
    cap = None
    backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
    
    for backend in backends_to_try:
        print(f"🔧 Attempting setup with backend: {backend}")
        cap = cv2.VideoCapture(working_device_id, backend)
        
        if cap.isOpened():
            # Test capture immediately
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                mean_color = test_frame.mean()
                std_color = test_frame.std()
                if (10 < mean_color < 240) and (std_color > 5):
                    print(f"✅ Successfully opened with backend: {backend}")
                    break
            cap.release()
        cap = None
    
    if cap is None:
        print("❌ Could not setup HD60 S+ with any backend")
        return None
    
    print(f"🔧 Configuring HD60 S+ (Device {working_device_id}) for Nintendo Switch...")
    
    # Nintendo Switch settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HD60S_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD60S_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, HD60S_FPS)
    
    # Optimize for gaming capture
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    
    # Try MJPG codec for better performance
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        print("✅ Using MJPG codec")
    except:
        print("⚠ Using default codec")
    
    # Verify actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ HD60 S+ configured for Nintendo Switch:")
    print(f"   Device ID: {working_device_id}")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    
    # Final comprehensive test
    print("🧪 Final HD60 S+ test...")
    for i in range(10):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            mean_color = test_frame.mean()
            std_color = test_frame.std()
            if (10 < mean_color < 240) and (std_color > 5):
                print(f"✅ HD60 S+ test frame {i+1}: {test_frame.shape} (mean: {mean_color:.1f}, std: {std_color:.1f})")
                print("✅ HD60 S+ ready for Nintendo Switch capture!")
                return cap
        time.sleep(0.1)
    
    print("❌ HD60 S+ test failed - still getting grey/invalid frames")
    print("💡 The device may still be in use by another application")
    cap.release()
    return None

def setup_video_source():
    """Setup video source with HD60 S+ auto-detection"""
    
    if USE_HD60S_CAPTURE:
        print("🎮 Using HD60 S+ for Nintendo Switch capture")
        cap = setup_hd60s_capture()
        if cap is not None:
            return cap
        else:
            print("⚠ HD60 S+ setup failed, falling back to file input")
    
    # Fallback to file input
    print("📁 Using file input as fallback")
    video_path_to_use = VIDEO_PATH
    
    # Check if video file exists (for file input)
    if isinstance(VIDEO_PATH, str):
        video_path = Path(video_path_to_use)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            
            # Try alternative video paths
            alternative_videos = [
                "../data/test_video/sample.mp4",
                "../data/sample.mp4",
                "sample.mp4"
            ]
            
            found_video = False
            for alt_video in alternative_videos:
                if Path(alt_video).exists():
                    video_path_to_use = alt_video
                    print(f"✓ Found video at: {alt_video}")
                    found_video = True
                    break
            
            if not found_video:
                print("⚠ No video file found, trying camera...")
                video_path_to_use = 0
    
    cap = cv2.VideoCapture(video_path_to_use)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_path_to_use}")
        if video_path_to_use != 0:
            print("Trying camera input...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✓ Using camera input")
                return cap
        return None
    
    return cap

# Initialize everything BEFORE starting video processing
print("STEP 1: Loading YOLO model...")

# Check if YOLO model exists, provide alternatives
yolo_model_path = Path(MODEL_PATH)
if not yolo_model_path.exists():
    print(f"❌ YOLO model not found: {yolo_model_path}")
    
    # Try alternative paths
    alternative_paths = [
        "../experiments/yolo/best.pt",
        "../weights/best.pt", 
        "../models/yolo_best.pt",
        "yolo11n.pt"  # Download default YOLOv11 nano
    ]
    
    found_model = False
    for alt_path in alternative_paths:
        if Path(alt_path).exists():
            MODEL_PATH = alt_path
            print(f"✓ Found YOLO model at: {alt_path}")
            found_model = True
            break
    
    if not found_model:
        print("⚠ No local YOLO model found, downloading YOLOv11 nano...")
        MODEL_PATH = "yolo11n.pt"  # This will auto-download

try:
    yolo = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        # Move YOLO to GPU if possible
        yolo.to(device)
    print("✓ YOLO model loaded!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    print("Please ensure you have a trained YOLO model or internet connection for auto-download")
    exit(1)

print("\nSTEP 2: Initializing translation...")
if not initialize_translation():
    print("❌ Failed to initialize translation")
    exit(1)

print("\nSTEP 3: Setting up tracking...")
tracker = Sort()
print("✓ Tracking initialized!")

# Thread-safe pipeline queues and state
frame_q = queue.Queue(maxsize=150)  # Larger queue for CUDA
ocr_q = queue.Queue(maxsize=150)
ostates = {}            # tid -> {'box','start','pending'}
overlay_states = {}     # tid -> {'box','text','avg', 'original_text'}
active_tids = set()
overlay_lock = threading.Lock()

# FIXED CUDA-optimized detection thread - Main fix for issue 2
def detect_thread():
    print("🚀 CUDA Detection thread started")
    while True:
        item = frame_q.get()
        if item is None: 
            print("Detection thread stopping")
            break
        idx, frame = item
        h, w = frame.shape[:2]
        
        # Larger input size for better accuracy with CUDA
        small = cv2.resize(frame, (640, 640))
        
        try:
            # YOLO inference with CUDA
            res = yolo.predict(
                source=small, 
                conf=0.25,  # Slightly lower confidence for more detections
                verbose=False,
                device=device
            )
            
            dets = []
            for box in getattr(res[0], 'boxes', []):
                x1,y1,x2,y2 = box.xyxy[0].cpu().int().tolist()
                # scale back to original size
                x1 = int(x1 * w/640); x2 = int(x2 * w/640)
                y1 = int(y1 * h/640); y2 = int(y2 * h/640)
                dets.append([x1,y1,x2,y2, float(box.conf[0].cpu())])
            
            dets = np.array(dets) if dets else np.zeros((0,5))
            tracks = tracker.update(dets)
            now = time.time()*1000
            current = set()
            
            for x1,y1,x2,y2,tid in tracks:
                tid = int(tid)
                current.add(tid)
                box = (int(x1),int(y1),int(x2),int(y2))
                st = ostates.get(tid)
                
                # MAIN FIX FOR ISSUE 2: Fixed timer reset logic
                if st:
                    # Calculate position change
                    position_change = np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2]))
                    
                    if position_change < 15:  # Increased tolerance for small movements
                        # Small movement - keep existing timer, just update box position
                        st['box'] = box  # Update position but preserve timer
                        if now - st['start'] >= STABILITY_MS and not st['pending']:
                            print(f"📦 Detected stable text box {tid} at {box} (stable for {(now - st['start']):.0f}ms)")
                            ocr_q.put((tid, box, frame.copy()))
                            st['pending'] = True
                    else:
                        # Large movement - likely new text, reset timer
                        print(f"🔄 Text box {tid} moved significantly ({position_change:.1f}px) - resetting timer")
                        ostates[tid] = {'box': box, 'start': now, 'pending': False}
                else:
                    # New track
                    print(f"🆕 New text box {tid} detected at {box}")
                    ostates[tid] = {'box': box, 'start': now, 'pending': False}
            
            # update active tids
            with overlay_lock:
                active_tids.clear()
                active_tids.update(current)
            
            # clean up ended tracks
            gone = set(ostates.keys()) - current
            for tid in gone:
                if tid in ostates:
                    print(f"🗑️ Removing ended track {tid}")
                ostates.pop(tid, None)
                
        except Exception as e:
            print(f"Detection error: {e}")
            
        frame_q.task_done()

# ENHANCED OCR & translation worker - Fix for issue 2
def ocr_worker(worker_id):
    print(f"🚀 CUDA OCR worker {worker_id} started")
    while True:
        item = ocr_q.get()
        if item is None: 
            print(f"OCR worker {worker_id} stopping")
            break
        tid, box, frame = item
        x1,y1,x2,y2 = box
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            ocr_q.task_done()
            continue
            
        try:
            # OCR with preprocessing
            if OCR_BACKEND == 'tesseract':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Enhanced preprocessing for better OCR
                gray = cv2.GaussianBlur(gray, (1, 1), 0)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                txt = pytesseract.image_to_string(gray, lang=OCR_LANG,
                                                 config=TESS_CONFIG).strip()
            else:
                res = paddle_ocr.ocr(roi)
                if res and res[0]:
                    txt = ' '.join([line[1][0] for line in res[0] if line])
                else:
                    txt = ""
            
            print(f"📝 OCR box {tid}: '{txt}'")
            
            if txt:
                # FIX FOR ISSUE 2: Check if we already have a similar translation for this track
                with overlay_lock:
                    existing_data = overlay_states.get(tid)
                    if existing_data:
                        # Calculate text similarity to avoid re-translating similar text
                        existing_text = existing_data.get('original_text', '')
                        similarity = calculate_text_similarity(txt, existing_text)
                        
                        if similarity > 0.85:  # 85% similar - reuse translation
                            print(f"🔄 Text box {tid}: Similar text detected ({similarity:.2f}), reusing translation")
                            existing_data['box'] = box  # Update box position
                            ocr_q.task_done()
                            continue
                
                # CUDA-accelerated translation
                eng = get_translation(txt)
                print(f"🌐 Translation box {tid}: '{txt}' -> '{eng}'")
                avg = tuple(map(int, cv2.mean(roi)[:3]))
                
                with overlay_lock:
                    overlay_states[tid] = {
                        'box': box, 
                        'text': eng, 
                        'avg': avg,
                        'original_text': txt  # Store original text for similarity checking
                    }
                    
        except Exception as e:
            print(f"❌ OCR/Translation error for box {tid}: {e}")
            
        ocr_q.task_done()

def enhanced_nintendo_switch_hud(frame, fps_disp, overlay_count, device):
    """Enhanced HUD specifically for Nintendo Switch gaming"""
    
    # Main info
    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Translations: {overlay_count}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Nintendo Switch Translation", (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Device: {device}", (10,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    # HD60 S+ info
    cv2.putText(frame, f"HD60 S+ Capture", (10,190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        cv2.putText(frame, f"GPU Memory: {memory_used:.1f}GB", (10,230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    
    return frame

# Main display loop with Nintendo Switch optimizations
def run():
    print("\n" + "="*50)
    print("🎮 STARTING NINTENDO SWITCH REAL-TIME TRANSLATION")
    print("="*50)
    
    # Setup video source
    cap = setup_video_source()
    if cap is None:
        print("❌ Could not setup video source")
        return
        
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, round(fps_in / TARGET_FPS))
    print(f"📹 Input FPS: {fps_in}, Target FPS: {TARGET_FPS}, Skip: {skip}")
    
    # start threads
    dt = threading.Thread(target=detect_thread, daemon=True)
    dt.start()
    
    fps_list, time_list = [], []
    start_time = time.time()
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    
    # Start more OCR workers for CUDA
    workers = []
    for i in range(4):  # More workers for CUDA
        worker = threading.Thread(target=ocr_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)

    idx = 0
    last = time.time()
    
    print("🎬 Processing Nintendo Switch frames... (Press 'q' to quit)")
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("📺 End of video reached")
            break
            
        idx += 1
        if idx % skip == 0 and not frame_q.full():
            frame_q.put((idx, frame.copy()))
            
        # remove overlays for inactive ids
        with overlay_lock:
            for tid in list(overlay_states.keys()):
                if tid not in active_tids:
                    overlay_states.pop(tid, None)
                    
        # FIXED: draw overlays with multi-line text rendering - Fix for issue 1
        overlay_count = 0
        with overlay_lock:
            for tid, data in overlay_states.items():
                x1,y1,x2,y2 = data['box']
                avg = data['avg']
                text = data['text']
                
                # Use the new multi-line text rendering function
                frame = draw_multiline_text_in_box(frame, text, (x1,y1,x2,y2), avg)
                overlay_count += 1
                            
        # compute & display FPS
        now = time.time()
        fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
        
        fps_list.append(fps_disp)
        time_list.append(now-start_time)
        
        # Enhanced HUD for Nintendo Switch gaming
        frame = enhanced_nintendo_switch_hud(frame, fps_disp, overlay_count, device)
        
        last = now
        # show
        writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            print("👤 User quit")
            break
        
        # More precise timing for CUDA
        target_time = 1.0/TARGET_FPS
        elapsed = time.time() - now
        if elapsed < target_time:
            time.sleep(target_time - elapsed)

    print("🧹 Cleaning up...")
    # cleanup
    cap.release()
    writer.release()
    frame_q.put(None)
    dt.join()
    for _ in workers: 
        ocr_q.put(None)
    for w in workers: 
        w.join()
    cv2.destroyAllWindows()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("📊 Generating FPS graph...")
    plt.figure(figsize=(12, 6))
    plt.plot(time_list, fps_list, marker='o', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('Nintendo Switch Translation - FPS over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FPS_GRAPH_PNG, dpi=300)
    plt.savefig(FPS_GRAPH_PDF)
    print(f"✓ FPS graph saved to {FPS_GRAPH_PNG}")
    
    # Performance summary
    avg_fps = np.mean(fps_list)
    min_fps = np.min(fps_list)
    max_fps = np.max(fps_list)
    print(f"\n📊 Performance Summary:")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Min FPS: {min_fps:.1f}")
    print(f"   Max FPS: {max_fps:.1f}")
    print(f"\n🎮 Nintendo Switch translation complete!")


if __name__ == '__main__':
    run()