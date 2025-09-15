import cv2
import time
import threading
import queue
import numpy as np
import torch
import sys
from pathlib import Path

# Add tracking module to path
sys.path.append('/workspace/pipeline/models/tracking')

from ultralytics import YOLO
from sort import Sort
from transformers import MarianTokenizer, MarianMTModel
import os

# OCR backend selection
try:
    import pytesseract
    OCR_BACKEND = 'tesseract'
    print("Using pytesseract for OCR")
except ImportError:
    try:
        from paddleocr import PaddleOCR
        OCR_BACKEND = 'paddle'
        print("Using PaddleOCR for OCR")
    except ImportError:
        print("No OCR backend available")
        exit(1)

import matplotlib.pyplot as plt

# -------- NINTENDO SWITCH CAPTURE CONFIG --------
# Video source options:
# Option 1: HDMI Capture (HD60 S+ - RECOMMENDED FOR NINTENDO SWITCH)
USE_HD60S_CAPTURE = True      # Set to True when using HD60 S+ 
HD60S_DEVICE_ID = 1          # This will be auto-detected from your working test
HD60S_RESOLUTION = (1920, 1080)  # Nintendo Switch outputs 1080p
HD60S_FPS = 30               # Standard Nintendo Switch output

# Option 2: File input (for testing)
VIDEO_PATH = "/workspace/data/test_video/JapanRPG_TestSequence.mov"

# Model paths using Jetson structure
MODEL_PATH = "/workspace/pipeline/models/object_detection/yolo11n_text_detection.pt"
MARIAN_MODEL_PATH = "/workspace/pipeline/models/translation/marian_opus_ja_en_finetuned"
WINDOW_NAME = "Nintendo Switch Real-Time Translation - Jetson Orin"

# BALANCED SETTINGS FOR RELIABLE TRANSLATION
TARGET_FPS = 25              # Realistic target for Nintendo Switch
STABILITY_MS = 200           # Reduced for faster response
YOLO_INPUT_SIZE = 640        # Good accuracy
MAX_WORKERS = 6              # Reasonable number of workers
BATCH_SIZE = 4               # Smaller batches for reliability
QUEUE_SIZE = 150             # Moderate queue size

# Frame dropping configuration
MAX_FRAME_DELAY = 50         # Drop frames if queue gets too full
ENABLE_FRAME_DROPPING = True

# Output paths - Jetson directory structure
BASE_DIR = "/workspace/pipeline/results/jetson_nintendo"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "nintendo_switch_translation.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_nintendo_switch.png")

os.makedirs(BASE_DIR, exist_ok=True)

# CUDA Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU optimizations
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# OCR setup
if OCR_BACKEND == 'tesseract':
    OCR_LANG = "jpn"
    TESS_CONFIG = "--psm 6 -c tessedit_char_blacklist=|"
else:
    paddle_ocr = PaddleOCR(
        lang='japan', 
        use_gpu=True,
        show_log=False,
        use_mp=False,
        total_process_num=1
    )

# Global variables
translation_cache = {}
MAX_CACHE_SIZE = 1000
translator_model = None

class FixedMarianTranslator:
    """Fixed synchronous translation - no complex async"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load Marian model with simple optimization"""
        print("="*50)
        print("LOADING FIXED TRANSLATION MODEL")
        print("="*50)
        
        try:
            if not self.model_path.exists():
                print(f"Model path not found: {self.model_path}")
                return False
            
            print("Loading Marian tokenizer...")
            self.tokenizer = MarianTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            print("Loading Marian model...")
            self.model = MarianMTModel.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                local_files_only=True
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                self.model.half()
                
            self.model.eval()
            
            # Test translation
            test_result = self.translate("ãƒ†ã‚¹ãƒˆ")
            print(f"Test translation: 'ãƒ†ã‚¹ãƒˆ' -> '{test_result}'")
            print("Translation model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Error loading Marian model: {e}")
            return False
    
    def translate(self, text):
        """Simple synchronous translation"""
        if not text.strip():
            return ""
        
        # Check cache first
        if text in translation_cache:
            return translation_cache[text]
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            if self.device.type == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    generated_tokens = self.model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=2,  # Simple beam search
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # Cache management
            if len(translation_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entries
                oldest_keys = list(translation_cache.keys())[:50]
                for key in oldest_keys:
                    translation_cache.pop(key, None)
                
            translation_cache[text] = result
            return result
            
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            error_result = f"[Error: {text}]"
            translation_cache[text] = error_result
            return error_result

def initialize_translation():
    """Initialize simple translation system"""
    global translator_model
    
    model_path = Path(MARIAN_MODEL_PATH)
    if not model_path.exists():
        print("Marian model not found!")
        print(f"Expected path: {model_path.absolute()}")
        return False
    
    translator_model = FixedMarianTranslator(MARIAN_MODEL_PATH)
    return translator_model.load_model()

def get_translation(text):
    """Get translation directly"""
    if translator_model is None:
        return f"[No translator: {text}]"
    return translator_model.translate(text)

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts"""
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

def get_screen_size():
    """Get screen dimensions for window scaling"""
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        # Fallback to common resolution
        return 1920, 1080

def calculate_display_size(frame_width, frame_height, max_screen_ratio=0.8):
    """Calculate optimal display size to fit screen"""
    screen_width, screen_height = get_screen_size()
    
    # Use 80% of screen size by default
    max_display_width = int(screen_width * max_screen_ratio)
    max_display_height = int(screen_height * max_screen_ratio)
    
    # Calculate scaling factor to fit within screen
    width_ratio = max_display_width / frame_width
    height_ratio = max_display_height / frame_height
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)
    
    print(f"Screen size: {screen_width}x{screen_height}")
    print(f"Frame size: {frame_width}x{frame_height}")
    print(f"Display size: {display_width}x{display_height} (scale: {scale_factor:.2f})")
    
    return display_width, display_height, scale_factor

def find_working_hd60s_device():
    """Auto-detect working HD60 S+ device ID with enhanced detection"""
    print("Auto-detecting HD60 S+ device...")
    
    for device_id in [HD60S_DEVICE_ID, 0, 1, 2, 3, 4]:
        print(f"   Testing device ID: {device_id}")
        
        # Try multiple backends for better compatibility
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
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
                        print(f"Found working HD60 S+ on device {device_id} (backend: {backend})")
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
    """Setup HD60 S+ capture for Nintendo Switch on Jetson"""
    print("Setting up HD60 S+ capture for Nintendo Switch...")
    
    # Auto-detect working device
    working_device_id = find_working_hd60s_device()
    
    if working_device_id is None:
        print("No working HD60 S+ found!")
        print("Troubleshooting steps:")
        print("   1. Check USB connection")
        print("   2. Verify HD60 S+ is recognized: lsusb | grep -i elgato")
        print("   3. Try different USB port")
        print("   4. Check /dev/video* devices: ls /dev/video*")
        return None
    
    # Try V4L2 backend for Linux/Jetson
    cap = cv2.VideoCapture(working_device_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("Could not setup HD60 S+ with V4L2, trying default backend...")
        cap = cv2.VideoCapture(working_device_id)
        
        if not cap.isOpened():
            print("Could not setup HD60 S+ with any backend")
            return None
    
    print(f"Configuring HD60 S+ (Device {working_device_id}) for Nintendo Switch...")
    
    # Nintendo Switch settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HD60S_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD60S_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, HD60S_FPS)
    
    # Optimize for gaming capture
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    # Verify actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"HD60 S+ configured for Nintendo Switch:")
    print(f"   Device ID: {working_device_id}")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    
    # Final test
    print("Final HD60 S+ test...")
    for i in range(10):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            mean_color = test_frame.mean()
            std_color = test_frame.std()
            if (10 < mean_color < 240) and (std_color > 5):
                print(f"HD60 S+ test frame {i+1}: {test_frame.shape} (mean: {mean_color:.1f}, std: {std_color:.1f})")
                print("HD60 S+ ready for Nintendo Switch capture!")
                return cap
        time.sleep(0.1)
    
    print("HD60 S+ test failed - still getting grey/invalid frames")
    cap.release()
    return None

def setup_video_source():
    """Setup video source with HD60 S+ auto-detection"""
    
    if USE_HD60S_CAPTURE:
        print("Using HD60 S+ for Nintendo Switch capture")
        cap = setup_hd60s_capture()
        if cap is not None:
            return cap
        else:
            print("HD60 S+ setup failed, falling back to file input")
    
    # Fallback to file input
    print("Using file input as fallback")
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    return cap

# Initialize models
print("JETSON NINTENDO SWITCH INITIALIZATION")
print("="*50)

print("STEP 1: Loading YOLO model...")
yolo_model_path = Path(MODEL_PATH)
if not yolo_model_path.exists():
    print(f"YOLO model not found: {yolo_model_path}")
    exit(1)

try:
    yolo = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        yolo.to(device)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

print("\nSTEP 2: Initializing translation...")
if not initialize_translation():
    print("Failed to initialize translation")
    exit(1)

print("\nSTEP 3: Setting up tracking...")
tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
print("SORT tracking initialized")

# Queues and state
frame_q = queue.Queue(maxsize=QUEUE_SIZE)
ocr_q = queue.Queue(maxsize=QUEUE_SIZE)
ostates = {}
overlay_states = {}
active_tids = set()
overlay_lock = threading.Lock()

# Statistics
dropped_frames = 0
total_frames = 0
ocr_count = 0
translation_count = 0

def fixed_detect_thread():
    """Fixed detection thread with debug output"""
    print("Detection thread started")
    global dropped_frames
    
    while True:
        item = frame_q.get()
        if item is None:
            print("Detection thread stopping")
            break
        idx, frame = item
        h, w = frame.shape[:2]
        
        # Check queue depth for frame dropping
        if ENABLE_FRAME_DROPPING and frame_q.qsize() > MAX_FRAME_DELAY:
            dropped_frames += 1
            frame_q.task_done()
            continue
        
        small = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        
        try:
            res = yolo.predict(
                source=small, 
                conf=0.25,
                verbose=False,
                device=device,
                half=True
            )
            
            dets = []
            for box in getattr(res[0], 'boxes', []):
                x1,y1,x2,y2 = box.xyxy[0].cpu().int().tolist()
                scale_x, scale_y = w/YOLO_INPUT_SIZE, h/YOLO_INPUT_SIZE
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
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
                
                if st:
                    position_change = np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2]))
                    
                    if position_change < 30:
                        st['box'] = box
                        if now - st['start'] >= STABILITY_MS and not st['pending']:
                            print(f"Sending box {tid} to OCR")
                            if not ocr_q.full():
                                ocr_q.put((tid, box, frame.copy()))
                                st['pending'] = True
                    else:
                        print(f"Box {tid} moved, resetting timer")
                        ostates[tid] = {'box': box, 'start': now, 'pending': False}
                else:
                    print(f"New box {tid} detected")
                    ostates[tid] = {'box': box, 'start': now, 'pending': False}
            
            with overlay_lock:
                active_tids.clear()
                active_tids.update(current)
            
            # Cleanup
            gone = set(ostates.keys()) - current
            for tid in gone:
                if tid in ostates:
                    ostates.pop(tid, None)
                    
        except Exception as e:
            print(f"Detection error: {e}")
            
        frame_q.task_done()

def fixed_ocr_worker(worker_id):
    """Fixed OCR worker with direct translation"""
    print(f"OCR worker {worker_id} started")
    global ocr_count, translation_count
    
    while True:
        try:
            item = ocr_q.get(timeout=1)
            if item is None:
                print(f"OCR worker {worker_id} stopping")
                break
            
            tid, box, frame = item
            x1,y1,x2,y2 = box
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                ocr_q.task_done()
                continue
            
            # OCR processing
            if OCR_BACKEND == 'tesseract':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Upscale small text
                if gray.shape[0] < 50 or gray.shape[1] < 50:
                    scale = 3.0
                    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                txt = pytesseract.image_to_string(gray, lang=OCR_LANG, config=TESS_CONFIG).strip()
            else:
                res = paddle_ocr.ocr(roi, det=True, rec=True, cls=False)
                if res and res[0]:
                    txt = ' '.join([line[1][0] for line in res[0] if line and len(line) > 1])
                else:
                    txt = ""
            
            ocr_count += 1
            print(f"OCR {worker_id}: Box {tid} -> '{txt}'")
            
            if txt and len(txt) > 1:
                # Check for existing translations
                with overlay_lock:
                    existing_data = overlay_states.get(tid)
                    if existing_data and existing_data.get('original_text') == txt:
                        existing_data['box'] = box
                        ocr_q.task_done()
                        continue
                    
                    # Check similarity with existing text
                    if existing_data:
                        existing_text = existing_data.get('original_text', '')
                        similarity = calculate_text_similarity(txt, existing_text)
                        
                        if similarity > 0.85:  # 85% similar - reuse translation
                            print(f"Text box {tid}: Similar text detected ({similarity:.2f}), reusing translation")
                            existing_data['box'] = box
                            ocr_q.task_done()
                            continue
                
                # Direct translation (synchronous)
                print(f"Translating: '{txt}'")
                eng = get_translation(txt)
                translation_count += 1
                print(f"Translation result: '{eng}'")
                
                avg = tuple(map(int, cv2.mean(roi)[:3]))
                
                with overlay_lock:
                    overlay_states[tid] = {
                        'box': box, 
                        'text': eng, 
                        'avg': avg,
                        'original_text': txt
                    }
                    print(f"Added translation for box {tid}")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"OCR worker {worker_id} error: {e}")
            
        ocr_q.task_done()

def draw_simple_text(frame, text, box, avg_color):
    """Multi-line text rendering within detection box"""
    x1, y1, x2, y2 = box
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), avg_color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = int(font_scale * 30)  # Approximate line height
    padding = 8
    
    # Calculate available space
    box_width = x2 - x1 - (2 * padding)
    box_height = y2 - y1 - (2 * padding)
    max_lines = max(1, box_height // line_height)
    
    # Estimate characters per line based on box width
    # Average character width is approximately font_scale * 12
    chars_per_line = max(10, box_width // int(font_scale * 12))
    
    # Split text into lines that fit within the box
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding this word would exceed the line length
        test_line = current_line + (" " if current_line else "") + word
        if len(test_line) <= chars_per_line:
            current_line = test_line
        else:
            # Start new line if we have room
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Word is too long, truncate it
                current_line = word[:chars_per_line-3] + "..."
            
            # Stop if we've reached maximum lines
            if len(lines) >= max_lines:
                break
    
    # Add the last line if there's room
    if current_line and len(lines) < max_lines:
        lines.append(current_line)
    
    # If we have too many lines, truncate the last one with "..."
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            last_line = lines[-1]
            if len(last_line) > 3:
                lines[-1] = last_line[:-3] + "..."
    
    # Draw each line
    for i, line in enumerate(lines):
        text_x = x1 + padding
        text_y = y1 + padding + (i + 1) * line_height
        
        # Make sure we don't draw outside the box
        if text_y <= y2 - padding:
            cv2.putText(frame, line, (text_x, text_y), font, 
                       font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def nintendo_hud(frame, fps_disp, overlay_count, queue_depths, stats):
    """Nintendo Switch HUD - clean interface"""
    # Performance metrics
    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Translations: {overlay_count}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Nintendo Switch", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    # Queue depths
    cv2.putText(frame, f"Frame Q: {queue_depths['frame']}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"OCR Q: {queue_depths['ocr']}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # Stats
    cv2.putText(frame, f"OCR Count: {stats['ocr']}", (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Trans Count: {stats['translation']}", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # Dropped frames
    if dropped_frames > 0:
        cv2.putText(frame, f"Dropped: {dropped_frames}", (10,270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    return frame

def run():
    """Main processing loop for Nintendo Switch"""
    print("\n" + "="*50)
    print("STARTING NINTENDO SWITCH REAL-TIME TRANSLATION")
    print("="*50)
    
    global total_frames, dropped_frames, ocr_count, translation_count
    
    cap = setup_video_source()
    if cap is None:
        print("Could not setup video source")
        return
        
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(VIDEO_PATH, str) else float('inf')
    skip = max(1, round(fps_in / TARGET_FPS))
    
    print(f"Input: {fps_in}fps, Target: {TARGET_FPS}fps, Skip: {skip}")
    
    # Calculate display size to fit screen
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_width, display_height, scale_factor = calculate_display_size(w, h)
    
    # Start threads
    dt = threading.Thread(target=fixed_detect_thread, daemon=True)
    dt.start()
    
    fps_list, time_list = [], []
    start_time = time.time()
    
    # Video writer (original size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    
    # Start OCR workers
    workers = []
    for i in range(MAX_WORKERS):
        worker = threading.Thread(target=fixed_ocr_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)

    last = time.time()
    
    print("Processing Nintendo Switch frames... (Press 'q' to quit)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video/stream reached")
                break
                
            total_frames += 1
            
            # Frame skipping for target FPS
            if total_frames % skip == 0:
                try:
                    frame_q.put((total_frames, frame.copy()), block=False)
                except queue.Full:
                    dropped_frames += 1
                
            # Update overlays
            with overlay_lock:
                for tid in list(overlay_states.keys()):
                    if tid not in active_tids:
                        overlay_states.pop(tid, None)
                        
            # Draw translations
            overlay_count = 0
            with overlay_lock:
                for tid, data in overlay_states.items():
                    x1,y1,x2,y2 = data['box']
                    avg = data['avg']
                    text = data['text']
                    
                    frame = draw_simple_text(frame, text, (x1,y1,x2,y2), avg)
                    overlay_count += 1
                                
            # FPS calculation
            now = time.time()
            fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
            
            fps_list.append(fps_disp)
            time_list.append(now-start_time)
            
            # HUD with debug info
            queue_depths = {
                'frame': frame_q.qsize(),
                'ocr': ocr_q.qsize()
            }
            stats = {
                'ocr': ocr_count,
                'translation': translation_count
            }
            frame = nintendo_hud(frame, fps_disp, overlay_count, queue_depths, stats)
            
            last = now
            
            # Output - save original size, display scaled size
            writer.write(frame)
            
            # Scale frame for display if needed
            if scale_factor < 1.0:
                display_frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow(WINDOW_NAME, display_frame)
            else:
                cv2.imshow(WINDOW_NAME, frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit")
                break
            
            # Progress update - GPU info printed to terminal
            if total_frames % 200 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                if total_video_frames != float('inf'):
                    progress = (total_frames / total_video_frames) * 100
                    print(f"Progress: {progress:.1f}% - FPS: {fps_disp:.1f} - GPU: {gpu_mem:.1f}GB - OCR: {ocr_count} - Trans: {translation_count} - Active: {overlay_count}")
                else:
                    print(f"Frames: {total_frames} - FPS: {fps_disp:.1f} - GPU: {gpu_mem:.1f}GB - OCR: {ocr_count} - Trans: {translation_count} - Active: {overlay_count}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        
        # Cleanup
        cap.release()
        writer.release()
        frame_q.put(None)
        dt.join()
        for _ in workers:
            ocr_q.put(None)
        for w in workers:
            w.join()
        cv2.destroyAllWindows()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Performance report
        if fps_list:
            avg_fps = np.mean(fps_list)
            min_fps = np.min(fps_list)
            max_fps = np.max(fps_list)
            gpu_mem_final = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            drop_rate = (dropped_frames / total_frames) * 100 if total_frames > 0 else 0
            
            print(f"\nNintendo Switch Translation Performance Summary:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Min FPS: {min_fps:.1f}")
            print(f"   Max FPS: {max_fps:.1f}")
            print(f"   Total Frames: {total_frames}")
            print(f"   Dropped Frames: {dropped_frames} ({drop_rate:.1f}%)")
            print(f"   OCR Operations: {ocr_count}")
            print(f"   Translations: {translation_count}")
            print(f"   Final GPU Memory: {gpu_mem_final:.1f}GB")
            print(f"   Cache Size: {len(translation_cache)}")
            print(f"\nNintendo Switch translation complete!")
            print(f"Output saved: {OUTPUT_VIDEO}")

if __name__ == '__main__':
    run()