import cv2
import time
import threading
import queue
import numpy as np
import torch
import json
import sys
from pathlib import Path

# Add packages directory to Python path
sys.path.append('/workspace/packages')

from ultralytics import YOLO
from sort.sort import Sort
from transformers import MarianTokenizer, MarianMTModel
import os

# OCR backend selection - Optimized for Jetson
try:
    import pytesseract
    OCR_BACKEND = 'tesseract'
    print("✅ Using pytesseract for OCR")
except ImportError:
    try:
        from paddleocr import PaddleOCR
        OCR_BACKEND = 'paddle'
        print("✅ Using PaddleOCR for OCR")
    except ImportError:
        print("❌ No OCR backend available")
        exit(1)

import matplotlib.pyplot as plt

# -------- JETSON ORIN OPTIMIZED CONFIG --------
# Video source: Using test video file instead of Nintendo capture
USE_HD60S_CAPTURE = False     # DISABLED - using video file
VIDEO_PATH = "/workspace/data/test_video/JapanRPG_TestSequence.mov"

MODEL_PATH = "/workspace/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
MARIAN_MODEL_PATH = "/workspace/translation/models/marian_opus_ja_en"
WINDOW_NAME = "Jetson Orin Real-Time Translation - Test Video"

# JETSON ORIN OPTIMIZED SETTINGS
TARGET_FPS = 15              # Lower FPS for Jetson Orin (vs 25 on laptop)
STABILITY_MS = 1000          # Longer stability time for better OCR results
YOLO_INPUT_SIZE = 416        # Smaller YOLO input for Jetson (vs 640)
MAX_WORKERS = 2              # Fewer OCR workers for Jetson (vs 4)
BATCH_SIZE = 2               # Smaller translation batch size
QUEUE_SIZE = 50              # Smaller queues to save memory

BASE_DIR = "/workspace/docker_setup/yolo_output"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "jetson_orin_translation.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_jetson_orin.png")
FPS_GRAPH_PDF = os.path.join(BASE_DIR, "fps_jetson_orin.pdf")

# Create output directory if it doesn't exist
os.makedirs(BASE_DIR, exist_ok=True)

# CUDA Configuration with Jetson optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Jetson Orin memory optimizations
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

# OCR setup - Optimized for Jetson
if OCR_BACKEND == 'tesseract':
    OCR_LANG = "jpn"
    TESS_CONFIG = "--psm 6 -c tessedit_char_blacklist=|"  # Improved config
else:
    # PaddleOCR with Jetson optimizations
    paddle_ocr = PaddleOCR(
        lang='japan', 
        use_gpu=torch.cuda.is_available(),
        show_log=False,
        use_mp=False,  # Disable multiprocessing on Jetson
        total_process_num=1  # Single process for stability
    )

# Global translator with memory optimization
translator_model = None
translator_tokenizer = None
translation_cache = {}
MAX_CACHE_SIZE = 500  # Limit cache size for Jetson

class JetsonOptimizedMarianTranslator:
    """Jetson Orin optimized Marian translation class"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load Marian model with Jetson optimizations"""
        print("="*50)
        print("LOADING JETSON-OPTIMIZED MARIAN TRANSLATION MODEL")
        print("="*50)
        
        try:
            if not self.model_path.exists():
                print(f"❌ Model path not found: {self.model_path}")
                return False
            
            print("🔥 Loading Marian tokenizer...")
            self.tokenizer = MarianTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            print("🔥 Loading Marian model with Jetson optimizations...")
            self.model = MarianMTModel.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,  # Always use FP16 on Jetson
                local_files_only=True
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                # Jetson-specific optimizations
                self.model.half()  # Ensure FP16
                
            self.model.eval()
            
            # Test translation
            test_result = self.translate("テスト")
            print(f"✅ Test translation: 'テスト' -> '{test_result}'")
            print(f"✅ Jetson-optimized Marian model loaded on {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading Marian model: {e}")
            return False
    
    def translate(self, text, max_length=80):  # Shorter max length for Jetson
        """Translate text with Jetson memory optimizations"""
        if not text.strip():
            return ""
        
        # Check cache first
        if text in translation_cache:
            return translation_cache[text]
        
        try:
            # Tokenize with length limits for Jetson
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256  # Shorter for Jetson
            )
            
            if self.device.type == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with Jetson optimizations
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=2,  # Fewer beams for Jetson (vs 4)
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True  # Enable caching
                )
            
            result = self.tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            # Cache management for Jetson memory
            if len(translation_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entries
                oldest_keys = list(translation_cache.keys())[:50]
                for key in oldest_keys:
                    translation_cache.pop(key, None)
                
            translation_cache[text] = result
            return result
            
        except Exception as e:
            print(f"❌ Translation error for '{text}': {e}")
            error_result = f"[Error: {text}]"
            translation_cache[text] = error_result
            return error_result

def initialize_translation():
    """Initialize translation with Jetson optimizations"""
    global translator_model
    
    model_path = Path(MARIAN_MODEL_PATH)
    if not model_path.exists():
        print("❌ Marian model not found!")
        print(f"Expected path: {model_path.absolute()}")
        return False
    
    translator_model = JetsonOptimizedMarianTranslator(MARIAN_MODEL_PATH)
    return translator_model.load_model()

def get_translation(text):
    """Get translation with fallback"""
    if translator_model is None:
        return f"[No translator: {text}]"
    return translator_model.translate(text)

def calculate_text_similarity(text1, text2):
    """Fast text similarity for Jetson"""
    if not text1 or not text2:
        return 0.0
    
    # Simplified similarity for performance
    text1 = text1.replace(' ', '').lower()
    text2 = text2.replace(' ', '').lower()
    
    if text1 == text2:
        return 1.0
    
    # Quick character-based similarity
    longer = max(len(text1), len(text2))
    if longer == 0:
        return 1.0
    
    matches = sum(1 for i in range(min(len(text1), len(text2))) if text1[i] == text2[i])
    return matches / longer

def wrap_text_to_fit_box(text, box_width, font_scale=0.7, thickness=1):
    """Optimized text wrapping for Jetson"""
    if not text:
        return []
    
    char_size = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    char_width = char_size[0]
    usable_width = box_width - 12
    chars_per_line = max(1, int(usable_width / char_width))
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if len(test_line) <= chars_per_line:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word if len(word) <= chars_per_line else word[:chars_per_line]
    
    if current_line:
        lines.append(current_line)
    
    return lines[:3]  # Limit to 3 lines for Jetson

def draw_optimized_text_in_box(frame, text, box, avg_color):
    """Jetson-optimized text rendering"""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Smaller font for Jetson performance
    font_scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    lines = wrap_text_to_fit_box(text, box_width, font_scale, thickness)
    if not lines:
        return frame
    
    # Simplified overlay for performance
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), avg_color, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw text lines
    line_height = 20
    start_y = y1 + 15
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        text_y = start_y + (i * line_height)
        if text_y > y2 - 10:
            break
            
        text_x = x1 + 6
        
        # Single text draw (no shadow for performance)
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def setup_video_source():
    """Setup video source for Jetson"""
    print("📹 Setting up video source for Jetson Orin...")
    
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {VIDEO_PATH}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video loaded: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"🎯 Target processing: {TARGET_FPS}fps on Jetson Orin")
    
    return cap

# Initialize models
print("🚀 JETSON ORIN INITIALIZATION")
print("="*50)

print("STEP 1: Loading YOLO model...")
yolo_model_path = Path(MODEL_PATH)
if not yolo_model_path.exists():
    print(f"❌ YOLO model not found: {yolo_model_path}")
    exit(1)

try:
    yolo = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        yolo.to(device)
    print("✅ YOLO model loaded on Jetson!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit(1)

print("\nSTEP 2: Initializing translation...")
if not initialize_translation():
    print("❌ Failed to initialize translation")
    exit(1)

print("\nSTEP 3: Setting up tracking...")
tracker = Sort()
print("✅ SORT tracking initialized!")

# Jetson-optimized queues and threading
frame_q = queue.Queue(maxsize=QUEUE_SIZE)
ocr_q = queue.Queue(maxsize=QUEUE_SIZE)
ostates = {}
overlay_states = {}
active_tids = set()
overlay_lock = threading.Lock()

def jetson_detect_thread():
    """Jetson-optimized detection thread"""
    print("🚀 Jetson detection thread started")
    while True:
        item = frame_q.get()
        if item is None:
            print("Detection thread stopping")
            break
        idx, frame = item
        h, w = frame.shape[:2]
        
        # Smaller input size for Jetson
        small = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        
        try:
            # YOLO inference with Jetson settings
            res = yolo.predict(
                source=small, 
                conf=0.3,  # Higher confidence for fewer false positives
                verbose=False,
                device=device,
                half=True  # FP16 inference on Jetson
            )
            
            dets = []
            for box in getattr(res[0], 'boxes', []):
                x1,y1,x2,y2 = box.xyxy[0].cpu().int().tolist()
                # Scale back to original size
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
                    
                    if position_change < 20:  # More tolerance for Jetson
                        st['box'] = box
                        if now - st['start'] >= STABILITY_MS and not st['pending']:
                            print(f"📦 Stable text box {tid} ready for OCR")
                            if not ocr_q.full():  # Check queue capacity
                                ocr_q.put((tid, box, frame.copy()))
                                st['pending'] = True
                    else:
                        print(f"🔄 Text box {tid} moved - resetting")
                        ostates[tid] = {'box': box, 'start': now, 'pending': False}
                else:
                    print(f"🆕 New text box {tid}")
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

def jetson_ocr_worker(worker_id):
    """Jetson-optimized OCR worker"""
    print(f"🚀 Jetson OCR worker {worker_id} started")
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
            # OCR with Jetson optimizations
            if OCR_BACKEND == 'tesseract':
                # Optimized preprocessing for Jetson
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # Resize for better OCR if too small
                if gray.shape[0] < 30 or gray.shape[1] < 30:
                    scale = 2.0
                    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                txt = pytesseract.image_to_string(gray, lang=OCR_LANG, config=TESS_CONFIG).strip()
            else:
                res = paddle_ocr.ocr(roi, det=True, rec=True, cls=False)
                if res and res[0]:
                    txt = ' '.join([line[1][0] for line in res[0] if line and len(line) > 1])
                else:
                    txt = ""
            
            print(f"🔍 OCR box {tid}: '{txt}'")
            
            if txt and len(txt) > 1:  # Minimum length check
                # Check similarity with existing translations
                with overlay_lock:
                    existing_data = overlay_states.get(tid)
                    if existing_data:
                        existing_text = existing_data.get('original_text', '')
                        similarity = calculate_text_similarity(txt, existing_text)
                        
                        if similarity > 0.8:  # Reuse similar translations
                            print(f"🔄 Reusing translation for box {tid}")
                            existing_data['box'] = box
                            ocr_q.task_done()
                            continue
                
                # Translate with Jetson optimizations
                eng = get_translation(txt)
                print(f"🌍 Translation {tid}: '{txt}' -> '{eng}'")
                
                # Simplified color calculation
                avg = tuple(map(int, cv2.mean(roi)[:3]))
                
                with overlay_lock:
                    overlay_states[tid] = {
                        'box': box, 
                        'text': eng, 
                        'avg': avg,
                        'original_text': txt
                    }
                    
        except Exception as e:
            print(f"❌ OCR error for box {tid}: {e}")
            
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and worker_id == 0:
            torch.cuda.empty_cache()
            
        ocr_q.task_done()

def jetson_hud(frame, fps_disp, overlay_count, frame_num, total_frames):
    """Jetson-optimized HUD"""
    progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
    
    # Simplified HUD for performance
    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Translations: {overlay_count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Jetson Orin", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Progress: {progress:.1f}%", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        cv2.putText(frame, f"GPU: {memory_used:.1f}GB", (10,125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
    
    return frame

def run():
    """Main processing loop optimized for Jetson Orin"""
    print("\n" + "="*50)
    print("🎮 STARTING JETSON ORIN REAL-TIME TRANSLATION")
    print("="*50)
    
    cap = setup_video_source()
    if cap is None:
        print("❌ Could not setup video source")
        return
        
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, round(fps_in / TARGET_FPS))
    print(f"📹 Input: {fps_in}fps, Target: {TARGET_FPS}fps, Skip: {skip}")
    
    # Start threads
    dt = threading.Thread(target=jetson_detect_thread, daemon=True)
    dt.start()
    
    fps_list, time_list = [], []
    start_time = time.time()
    
    # Video writer setup
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    
    # Start OCR workers (fewer for Jetson)
    workers = []
    for i in range(MAX_WORKERS):
        worker = threading.Thread(target=jetson_ocr_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)

    idx = 0
    last = time.time()
    processed_frames = 0
    
    print("🎬 Processing video frames... (Press 'q' to quit)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📺 End of video reached")
                break
                
            idx += 1
            
            # Frame skipping for target FPS
            if idx % skip == 0 and not frame_q.full():
                frame_q.put((idx, frame.copy()))
                processed_frames += 1
                
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
                    
                    frame = draw_optimized_text_in_box(frame, text, (x1,y1,x2,y2), avg)
                    overlay_count += 1
                                
            # FPS calculation
            now = time.time()
            fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
            
            fps_list.append(fps_disp)
            time_list.append(now-start_time)
            
            # HUD
            frame = jetson_hud(frame, fps_disp, overlay_count, idx, total_frames)
            
            last = now
            
            # Output
            writer.write(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👤 User quit")
                break
            
            # Timing control
            target_time = 1.0/TARGET_FPS
            elapsed = time.time() - now
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
                
            # Progress update
            if idx % (skip * 30) == 0:  # Every 30 processed frames
                progress = (idx / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% - FPS: {fps_disp:.1f} - Translations: {overlay_count}")

    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
    finally:
        print("🧹 Cleaning up...")
        
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
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate performance report
        if fps_list:
            print("📊 Generating performance report...")
            plt.figure(figsize=(10, 6))
            plt.plot(time_list, fps_list, marker='o', markersize=1)
            plt.xlabel('Time (s)')
            plt.ylabel('FPS')
            plt.title('Jetson Orin Translation Performance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FPS_GRAPH_PNG, dpi=150)  # Lower DPI for Jetson
            plt.savefig(FPS_GRAPH_PDF)
            print(f"✅ Performance graphs saved")
            
            # Performance summary
            avg_fps = np.mean(fps_list)
            min_fps = np.min(fps_list)
            max_fps = np.max(fps_list)
            print(f"\n📊 Jetson Orin Performance Summary:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Min FPS: {min_fps:.1f}")
            print(f"   Max FPS: {max_fps:.1f}")
            print(f"   Processed Frames: {processed_frames}")
            print(f"   Cache Size: {len(translation_cache)}")
            print(f"\n🎮 Jetson Orin translation complete!")
            print(f"📁 Output saved: {OUTPUT_VIDEO}")

if __name__ == '__main__':
    run()