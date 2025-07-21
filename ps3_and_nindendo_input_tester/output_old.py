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

# -------- CONFIG --------
# Video source options:
# Option 1: HDMI Capture (HD60 S+ - RECOMMENDED FOR PS3)
USE_HD60S_CAPTURE = True      # Set to True when using HD60 S+ 
HD60S_DEVICE_ID = 1          # Device ID from your simple test (usually 1 or 2)
HD60S_RESOLUTION = (1920, 1080)  # HD60 S+ resolution
HD60S_FPS = 30               # Capture FPS for USB 2.0

# Option 2: File input (for testing)
VIDEO_PATH = "../data/test_video/JapanRPG_TestSequence.mov"

# Option 3: Webcam (fallback)
# VIDEO_PATH = 0  # Default camera

MODEL_PATH = "../experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"  # YOLO model path
MARIAN_MODEL_PATH = "../translation/models/marian_opus_ja_en"  # Path to translation model
WINDOW_NAME = "PS3 Real-Time Translation - HD60 S+"
TARGET_FPS = 25  # Slightly lower for USB 2.0 stability
STABILITY_MS = 400  # Slightly higher for better text stability

BASE_DIR = "output/video"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_cuda_marian.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_over_time_cuda_marian.png")
FPS_GRAPH_PDF = os.path.join(BASE_DIR, "fps_over_time_cuda_marian.pdf")

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
                    with torch.cuda.amp.autocast():  # Mixed precision
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
overlay_states = {}     # tid -> {'box','text','avg'}
active_tids = set()
overlay_lock = threading.Lock()

# CUDA-optimized detection thread
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
                if st and np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2])) < 8:
                    if now - st['start'] >= STABILITY_MS and not st['pending']:
                        print(f"📦 Detected stable text box {tid} at {box}")
                        ocr_q.put((tid, box, frame.copy()))
                        st['pending'] = True
                else:
                    ostates[tid] = {'box': box, 'start': now, 'pending': False}
            
            # update active tids
            with overlay_lock:
                active_tids.clear()
                active_tids.update(current)
            
            # clean up ended tracks
            gone = set(ostates.keys()) - current
            for tid in gone:
                ostates.pop(tid, None)
                
        except Exception as e:
            print(f"Detection error: {e}")
            
        frame_q.task_done()

# OCR & translation worker with CUDA optimization
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
                # CUDA-accelerated translation
                eng = get_translation(txt)
                print(f"🌐 Translation box {tid}: '{txt}' -> '{eng}'")
                avg = tuple(map(int, cv2.mean(roi)[:3]))
                with overlay_lock:
                    overlay_states[tid] = {'box': box, 'text': eng, 'avg': avg}
                    
        except Exception as e:
            print(f"❌ OCR/Translation error for box {tid}: {e}")
            
        ocr_q.task_done()

# Main display loop with CUDA optimizations
def setup_hd60s_capture():
    """Setup HD60 S+ capture with DirectShow backend"""
    print("🎮 Setting up HD60 S+ capture for PS3...")
    print(f"   Device ID: {HD60S_DEVICE_ID}")
    print(f"   Target Resolution: {HD60S_RESOLUTION}")
    print(f"   Target FPS: {HD60S_FPS}")
    
    # Use DirectShow backend (essential for HD60 S+ on Windows)
    cap = cv2.VideoCapture(HD60S_DEVICE_ID, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ Could not open HD60 S+ on device {HD60S_DEVICE_ID}")
        print("💡 Solutions:")
        print("   1. Close Elgato Game Capture software")
        print("   2. Try different device ID (1, 2, 3)")
        print("   3. Check USB connection")
        return None
    
    # Configure HD60 S+ settings
    print("🔧 Configuring HD60 S+ settings...")
    
    # Set resolution (USB 2.0 optimized)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HD60S_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD60S_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, HD60S_FPS)
    
    # Optimize for gaming capture
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    
    # Try to set MJPG codec for better USB 2.0 performance
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        print("✅ Using MJPG codec")
    except:
        print("⚠ Using default codec")
    
    # Verify actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ HD60 S+ configured:")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    
    # Test frame capture
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ Could not capture test frame from HD60 S+")
        cap.release()
        return None
    
    print(f"✅ HD60 S+ test frame: {test_frame.shape}")
    print("✅ HD60 S+ ready for PS3 capture!")
    
    return cap

def setup_video_source():
    """Setup video source with HD60 S+ priority"""
    
    if USE_HD60S_CAPTURE:
        print("🎮 Using HD60 S+ for PS3 capture")
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

def run():
    print("\n" + "="*50)
    print("🚀 STARTING PS3 REAL-TIME TRANSLATION")
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
    
    print("🎬 Processing video frames... (Press 'q' to quit)")
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
                    
        # draw overlays with better styling
        overlay_count = 0
        with overlay_lock:
            for tid, data in overlay_states.items():
                x1,y1,x2,y2 = data['box']
                avg = data['avg']
                # Better overlay styling
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), avg, -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                # Better text rendering
                text = data['text']
                font_scale = 0.8
                thickness = 2
                
                # Text background for better readability
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                cv2.rectangle(frame, (x1, y2-text_size[1]-8), (x1+text_size[0]+8, y2), (0,0,0), -1)
                
                cv2.putText(frame, text, (x1+4, y2-4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255,255,255), thickness, cv2.LINE_AA)
                overlay_count += 1
                            
        # compute & display FPS
        now = time.time()
        fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
        
        fps_list.append(fps_disp)
        time_list.append(now-start_time)
        
        # Enhanced HUD for PS3 gaming
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Translations: {overlay_count}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"HD60 S+ PS3 Capture", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Device: {device}", (10,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        # USB connection info
        if USE_HD60S_CAPTURE:
            cv2.putText(frame, f"USB 2.0 Mode", (10,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            cv2.putText(frame, f"GPU Memory: {memory_used:.1f}GB", (10,230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        
        # Gaming region indicators (helpful for debugging)
        if overlay_count == 0:  # Only show when no translations active
            height, width = frame.shape[:2]
            
            # Subtitle region indicator
            subtitle_y = int(height * 0.75)
            cv2.line(frame, (0, subtitle_y), (width, subtitle_y), (0, 255, 0), 1)
            cv2.putText(frame, "Subtitle Zone", (width - 150, subtitle_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # UI region indicator  
            ui_y = int(height * 0.2)
            cv2.line(frame, (0, ui_y), (width, ui_y), (255, 0, 0), 1)
            cv2.putText(frame, "UI Zone", (width - 100, ui_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
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
    plt.title('CUDA Pipeline - FPS over Time')
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


if __name__ == '__main__':
    run()