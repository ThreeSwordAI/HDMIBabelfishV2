import cv2
import time
import threading
import queue
import numpy as np
import torch
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from collections import defaultdict
import json

# Add tracking module to path
sys.path.append('models/tracking')

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

# ======================================================================
# CONFIGURATION
# ======================================================================
VIDEO_PATH = "../data/test_video/JapanRPG_TestSequence.mov"
MODEL_PATH = "../experiments/transfer_learning/YOLOv11/Target/FT_model_ultralytics_nano/runs/detect/train/weights/best.pt"
MARIAN_MODEL_PATH = "../translation/models/marian_opus_ft_ja_en_full_dataset"
WINDOW_NAME = "Pipeline Evaluation - Jetson Orin"

# Pipeline settings
TARGET_FPS = 25
STABILITY_MS = 200
YOLO_INPUT_SIZE = 640
MAX_WORKERS = 6
BATCH_SIZE = 4
QUEUE_SIZE = 150
MAX_FRAME_DELAY = 50
ENABLE_FRAME_DROPPING = True

# Output directory
BASE_DIR = "results2"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "pipeline_evaluation.mp4")
RESULTS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# Create directories
for directory in [BASE_DIR, RESULTS_DIR, PLOTS_DIR, TABLES_DIR, SCREENSHOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# CUDA Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
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

# ======================================================================
# METRICS COLLECTION
# ======================================================================
class MetricsCollector:
    """Collects detailed metrics for pipeline evaluation"""
    
    def __init__(self):
        self.frame_metrics = []
        self.detection_times = []
        self.tracking_times = []
        self.ocr_times = []
        self.translation_times = []
        self.total_latencies = []
        self.fps_values = []
        self.gpu_memory = []
        self.queue_depths = {'frame': [], 'ocr': []}
        self.timestamps = []
        self.dropped_frames = 0
        self.total_frames = 0
        self.ocr_count = 0
        self.translation_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.active_overlays = []
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def record_frame(self, frame_id, fps, total_latency, detection_time, 
                     tracking_time, queue_frame, queue_ocr, gpu_mem, active_count):
        """Record per-frame metrics"""
        with self.lock:
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            self.fps_values.append(fps)
            self.total_latencies.append(total_latency)
            self.detection_times.append(detection_time)
            self.tracking_times.append(tracking_time)
            self.gpu_memory.append(gpu_mem)
            self.queue_depths['frame'].append(queue_frame)
            self.queue_depths['ocr'].append(queue_ocr)
            self.active_overlays.append(active_count)
            
            self.frame_metrics.append({
                'frame_id': frame_id,
                'timestamp': current_time,
                'fps': fps,
                'total_latency_ms': total_latency * 1000,
                'detection_time_ms': detection_time * 1000,
                'tracking_time_ms': tracking_time * 1000,
                'queue_frame_depth': queue_frame,
                'queue_ocr_depth': queue_ocr,
                'gpu_memory_gb': gpu_mem,
                'active_overlays': active_count
            })
    
    def record_ocr(self, ocr_time):
        """Record OCR operation time"""
        with self.lock:
            self.ocr_times.append(ocr_time)
            self.ocr_count += 1
    
    def record_translation(self, translation_time, cache_hit=False):
        """Record translation operation time"""
        with self.lock:
            self.translation_times.append(translation_time)
            self.translation_count += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def increment_dropped_frames(self):
        """Increment dropped frame counter"""
        with self.lock:
            self.dropped_frames += 1
    
    def increment_total_frames(self):
        """Increment total frame counter"""
        with self.lock:
            self.total_frames += 1
    
    def get_summary_statistics(self):
        """Calculate summary statistics"""
        with self.lock:
            total_runtime = time.time() - self.start_time
            
            stats = {
                'total_runtime_seconds': total_runtime,
                'total_frames_processed': self.total_frames,
                'frames_analyzed': len(self.frame_metrics),
                'dropped_frames': self.dropped_frames,
                'frame_drop_rate_percent': (self.dropped_frames / max(self.total_frames, 1)) * 100,
                
                # FPS metrics
                'average_fps': np.mean(self.fps_values) if self.fps_values else 0,
                'min_fps': np.min(self.fps_values) if self.fps_values else 0,
                'max_fps': np.max(self.fps_values) if self.fps_values else 0,
                'std_fps': np.std(self.fps_values) if self.fps_values else 0,
                
                # Latency metrics (in ms)
                'average_total_latency_ms': np.mean([l*1000 for l in self.total_latencies]) if self.total_latencies else 0,
                'min_total_latency_ms': np.min([l*1000 for l in self.total_latencies]) if self.total_latencies else 0,
                'max_total_latency_ms': np.max([l*1000 for l in self.total_latencies]) if self.total_latencies else 0,
                'std_total_latency_ms': np.std([l*1000 for l in self.total_latencies]) if self.total_latencies else 0,
                
                # Component latencies (in ms)
                'average_detection_latency_ms': np.mean([d*1000 for d in self.detection_times]) if self.detection_times else 0,
                'average_tracking_latency_ms': np.mean([t*1000 for t in self.tracking_times]) if self.tracking_times else 0,
                'average_ocr_latency_ms': np.mean([o*1000 for o in self.ocr_times]) if self.ocr_times else 0,
                'average_translation_latency_ms': np.mean([t*1000 for t in self.translation_times]) if self.translation_times else 0,
                
                # OCR and Translation counts
                'total_ocr_operations': self.ocr_count,
                'total_translations': self.translation_count,
                'translation_cache_hits': self.cache_hits,
                'translation_cache_misses': self.cache_misses,
                'cache_hit_rate_percent': (self.cache_hits / max(self.cache_hits + self.cache_misses, 1)) * 100,
                
                # GPU metrics
                'average_gpu_memory_gb': np.mean(self.gpu_memory) if self.gpu_memory else 0,
                'max_gpu_memory_gb': np.max(self.gpu_memory) if self.gpu_memory else 0,
                
                # Queue metrics
                'average_frame_queue_depth': np.mean(self.queue_depths['frame']) if self.queue_depths['frame'] else 0,
                'max_frame_queue_depth': np.max(self.queue_depths['frame']) if self.queue_depths['frame'] else 0,
                'average_ocr_queue_depth': np.mean(self.queue_depths['ocr']) if self.queue_depths['ocr'] else 0,
                'max_ocr_queue_depth': np.max(self.queue_depths['ocr']) if self.queue_depths['ocr'] else 0,
                
                # Active overlays
                'average_active_overlays': np.mean(self.active_overlays) if self.active_overlays else 0,
                'max_active_overlays': np.max(self.active_overlays) if self.active_overlays else 0,
            }
            
            return stats
    
    def save_detailed_csv(self, filepath):
        """Save detailed per-frame metrics to CSV"""
        df = pd.DataFrame(self.frame_metrics)
        df.to_csv(filepath, index=False)
        print(f"Detailed metrics saved to: {filepath}")
    
    def save_summary_csv(self, filepath):
        """Save summary statistics to CSV"""
        stats = self.get_summary_statistics()
        df = pd.DataFrame([stats])
        df.to_csv(filepath, index=False)
        print(f"Summary statistics saved to: {filepath}")

# Global metrics collector
metrics = MetricsCollector()

# ======================================================================
# TRANSLATION MODEL
# ======================================================================
translation_cache = {}
MAX_CACHE_SIZE = 1000
translator_model = None

class FixedMarianTranslator:
    """Translation model with metrics tracking"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load Marian model"""
        print("="*50)
        print("LOADING TRANSLATION MODEL")
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
            
            test_result = self.translate("テスト")
            print(f"Test translation: 'テスト' -> '{test_result}'")
            print("Translation model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Error loading Marian model: {e}")
            return False
    
    def translate(self, text):
        """Translate with timing"""
        if not text.strip():
            return ""
        
        start_time = time.time()
        cache_hit = text in translation_cache
        
        if cache_hit:
            result = translation_cache[text]
            metrics.record_translation(time.time() - start_time, cache_hit=True)
            return result
        
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
                        num_beams=2,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            if len(translation_cache) >= MAX_CACHE_SIZE:
                oldest_keys = list(translation_cache.keys())[:50]
                for key in oldest_keys:
                    translation_cache.pop(key, None)
                
            translation_cache[text] = result
            metrics.record_translation(time.time() - start_time, cache_hit=False)
            return result
            
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            error_result = f"[Error: {text}]"
            translation_cache[text] = error_result
            metrics.record_translation(time.time() - start_time, cache_hit=False)
            return error_result

def initialize_translation():
    """Initialize translation system"""
    global translator_model
    
    model_path = Path(MARIAN_MODEL_PATH)
    if not model_path.exists():
        print("Marian model not found!")
        return False
    
    translator_model = FixedMarianTranslator(MARIAN_MODEL_PATH)
    return translator_model.load_model()

def get_translation(text):
    """Get translation"""
    if translator_model is None:
        return f"[No translator: {text}]"
    return translator_model.translate(text)

# ======================================================================
# VIDEO SETUP
# ======================================================================
def setup_video_source():
    """Setup video source"""
    print("Setting up video source...")
    
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

# ======================================================================
# INITIALIZE MODELS
# ======================================================================
print("PIPELINE EVALUATION INITIALIZATION")
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

# ======================================================================
# PIPELINE STATE
# ======================================================================
frame_q = queue.Queue(maxsize=QUEUE_SIZE)
ocr_q = queue.Queue(maxsize=QUEUE_SIZE)
ostates = {}
overlay_states = {}
active_tids = set()
overlay_lock = threading.Lock()

# Timing data storage (frame_id -> timing_dict)
frame_timing = {}
timing_lock = threading.Lock()

# Screenshot capture
screenshot_frames = []
screenshot_lock = threading.Lock()

# ======================================================================
# DETECTION THREAD
# ======================================================================
def detection_thread():
    """Detection thread with timing"""
    print("Detection thread started")
    
    while True:
        item = frame_q.get()
        if item is None:
            print("Detection thread stopping")
            break
        
        idx, frame = item
        h, w = frame.shape[:2]
        
        if ENABLE_FRAME_DROPPING and frame_q.qsize() > MAX_FRAME_DELAY:
            metrics.increment_dropped_frames()
            frame_q.task_done()
            continue
        
        # Time detection
        detection_start = time.time()
        small = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        
        try:
            res = yolo.predict(
                source=small, 
                conf=0.25,
                verbose=False,
                device=device,
                half=True
            )
            
            detection_time = time.time() - detection_start
            
            # Time tracking
            tracking_start = time.time()
            
            dets = []
            for box in getattr(res[0], 'boxes', []):
                x1,y1,x2,y2 = box.xyxy[0].cpu().int().tolist()
                scale_x, scale_y = w/YOLO_INPUT_SIZE, h/YOLO_INPUT_SIZE
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                dets.append([x1,y1,x2,y2, float(box.conf[0].cpu())])
            
            dets = np.array(dets) if dets else np.zeros((0,5))
            tracks = tracker.update(dets)
            
            tracking_time = time.time() - tracking_start
            
            # Store timing data
            with timing_lock:
                frame_timing[idx] = {
                    'detection_time': detection_time,
                    'tracking_time': tracking_time
                }
            
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
                            if not ocr_q.full():
                                ocr_q.put((tid, box, frame.copy()))
                                st['pending'] = True
                    else:
                        ostates[tid] = {'box': box, 'start': now, 'pending': False}
                else:
                    ostates[tid] = {'box': box, 'start': now, 'pending': False}
            
            with overlay_lock:
                active_tids.clear()
                active_tids.update(current)
            
            gone = set(ostates.keys()) - current
            for tid in gone:
                if tid in ostates:
                    ostates.pop(tid, None)
                    
        except Exception as e:
            print(f"Detection error: {e}")
            
        frame_q.task_done()

# ======================================================================
# OCR WORKER THREAD
# ======================================================================
def ocr_worker(worker_id):
    """OCR worker with timing"""
    print(f"OCR worker {worker_id} started")
    
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
            
            # Time OCR
            ocr_start = time.time()
            
            if OCR_BACKEND == 'tesseract':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
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
            
            ocr_time = time.time() - ocr_start
            metrics.record_ocr(ocr_time)
            
            if txt and len(txt) > 1:
                with overlay_lock:
                    existing_data = overlay_states.get(tid)
                    if existing_data and existing_data.get('original_text') == txt:
                        existing_data['box'] = box
                        ocr_q.task_done()
                        continue
                
                # Translate
                eng = get_translation(txt)
                avg = tuple(map(int, cv2.mean(roi)[:3]))
                
                with overlay_lock:
                    overlay_states[tid] = {
                        'box': box, 
                        'text': eng, 
                        'avg': avg,
                        'original_text': txt
                    }
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"OCR worker {worker_id} error: {e}")
            
        ocr_q.task_done()

# ======================================================================
# RENDERING
# ======================================================================
def draw_text_overlay(frame, text, box, avg_color):
    """Draw text overlay"""
    x1, y1, x2, y2 = box
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), avg_color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = int(font_scale * 30)
    padding = 8
    
    box_width = x2 - x1 - (2 * padding)
    box_height = y2 - y1 - (2 * padding)
    max_lines = max(1, box_height // line_height)
    chars_per_line = max(10, box_width // int(font_scale * 12))
    
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
                current_line = word
            else:
                current_line = word[:chars_per_line-3] + "..."
            
            if len(lines) >= max_lines:
                break
    
    if current_line and len(lines) < max_lines:
        lines.append(current_line)
    
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            last_line = lines[-1]
            if len(last_line) > 3:
                lines[-1] = last_line[:-3] + "..."
    
    for i, line in enumerate(lines):
        text_x = x1 + padding
        text_y = y1 + padding + (i + 1) * line_height
        
        if text_y <= y2 - padding:
            cv2.putText(frame, line, (text_x, text_y), font, 
                       font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def draw_hud(frame, fps_disp, overlay_count):
    """Draw HUD"""
    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Translations: {overlay_count}", (10,70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Pipeline Evaluation", (10,110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    return frame

# ======================================================================
# VISUALIZATION FUNCTIONS
# ======================================================================
def create_visualizations():
    """Create all visualization plots"""
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    stats = metrics.get_summary_statistics()
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Latency Breakdown Pie Chart
    create_latency_pie_chart(stats)
    
    # 2. FPS Over Time
    create_fps_over_time()
    
    # 3. Latency Histogram
    create_latency_histogram()
    
    # 4. Queue Depth Over Time
    create_queue_depth_plot()
    
    # 5. Component Timing Stacked Area
    create_stacked_timing_plot()
    
    # 6. GPU Memory Usage
    create_gpu_memory_plot()
    
    # 7. Performance Summary Dashboard
    create_summary_dashboard(stats)
    
    print("All visualizations created successfully!")

def create_latency_pie_chart(stats):
    """Create latency breakdown pie chart"""
    print("Creating latency breakdown pie chart...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    components = ['Detection', 'Tracking', 'OCR', 'Translation']
    latencies = [
        stats['average_detection_latency_ms'],
        stats['average_tracking_latency_ms'],
        stats['average_ocr_latency_ms'],
        stats['average_translation_latency_ms']
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    explode = (0.1, 0, 0, 0)
    
    wedges, texts, autotexts = ax.pie(latencies, labels=components, autopct='%1.1f%%',
                                        startangle=90, colors=colors, explode=explode,
                                        shadow=True)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title('Pipeline Latency Breakdown\n(Average Time per Component)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with values
    legend_labels = [f'{comp}: {lat:.1f} ms' for comp, lat in zip(components, latencies)]
    ax.legend(legend_labels, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save in both formats
    plt.savefig(os.path.join(PLOTS_DIR, 'latency_breakdown_pie.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'latency_breakdown_pie.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Latency pie chart saved")

def create_fps_over_time():
    """Create FPS over time line graph"""
    print("Creating FPS over time plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(metrics.timestamps, metrics.fps_values, linewidth=2, color='#2ecc71', alpha=0.8)
    ax.axhline(y=np.mean(metrics.fps_values), color='#e74c3c', linestyle='--', 
              linewidth=2, label=f'Average: {np.mean(metrics.fps_values):.1f} FPS')
    ax.axhline(y=TARGET_FPS, color='#3498db', linestyle='--', 
              linewidth=2, label=f'Target: {TARGET_FPS} FPS')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax.set_title('Real-Time Pipeline Performance (FPS Over Time)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fps_over_time.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'fps_over_time.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ FPS over time plot saved")

def create_latency_histogram():
    """Create latency distribution histogram"""
    print("Creating latency histogram...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latencies_ms = [l * 1000 for l in metrics.total_latencies]
    
    n, bins, patches = ax.hist(latencies_ms, bins=30, color='#3498db', 
                               alpha=0.7, edgecolor='black')
    
    # Color bars based on latency threshold
    for i, patch in enumerate(patches):
        if bins[i] < 100:
            patch.set_facecolor('#2ecc71')  # Green for <100ms
        elif bins[i] < 150:
            patch.set_facecolor('#f39c12')  # Orange for 100-150ms
        else:
            patch.set_facecolor('#e74c3c')  # Red for >150ms
    
    ax.axvline(x=np.mean(latencies_ms), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(latencies_ms):.1f} ms')
    ax.axvline(x=100, color='green', linestyle='--', 
              linewidth=2, label='Target: <100 ms')
    
    ax.set_xlabel('Total Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Per-Frame Latency Distribution', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'latency_histogram.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'latency_histogram.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Latency histogram saved")

def create_queue_depth_plot():
    """Create queue depth over time plot"""
    print("Creating queue depth plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Frame queue
    ax1.plot(metrics.timestamps, metrics.queue_depths['frame'], 
            linewidth=2, color='#3498db', label='Frame Queue')
    ax1.axhline(y=QUEUE_SIZE, color='red', linestyle='--', 
               linewidth=2, label=f'Max Size: {QUEUE_SIZE}')
    ax1.set_ylabel('Queue Depth', fontsize=12, fontweight='bold')
    ax1.set_title('Frame Queue Depth Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # OCR queue
    ax2.plot(metrics.timestamps, metrics.queue_depths['ocr'], 
            linewidth=2, color='#e74c3c', label='OCR Queue')
    ax2.axhline(y=QUEUE_SIZE, color='red', linestyle='--', 
               linewidth=2, label=f'Max Size: {QUEUE_SIZE}')
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Queue Depth', fontsize=12, fontweight='bold')
    ax2.set_title('OCR Queue Depth Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'queue_depths.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'queue_depths.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Queue depth plot saved")

def create_stacked_timing_plot():
    """Create stacked area chart of component timings"""
    print("Creating stacked timing plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    detection_ms = [d * 1000 for d in metrics.detection_times]
    tracking_ms = [t * 1000 for t in metrics.tracking_times]
    
    # Align OCR/translation times with detection times
    # (they happen asynchronously, so we'll show average per frame)
    avg_ocr_ms = np.mean([o * 1000 for o in metrics.ocr_times]) if metrics.ocr_times else 0
    avg_trans_ms = np.mean([t * 1000 for t in metrics.translation_times]) if metrics.translation_times else 0
    
    ocr_ms = [avg_ocr_ms] * len(detection_ms)
    trans_ms = [avg_trans_ms] * len(detection_ms)
    
    times = metrics.timestamps[:len(detection_ms)]
    
    ax.fill_between(times, 0, detection_ms, 
                    label='Detection', alpha=0.7, color='#3498db')
    ax.fill_between(times, detection_ms, 
                    [d+t for d, t in zip(detection_ms, tracking_ms)],
                    label='Tracking', alpha=0.7, color='#2ecc71')
    ax.fill_between(times, 
                    [d+t for d, t in zip(detection_ms, tracking_ms)],
                    [d+t+o for d, t, o in zip(detection_ms, tracking_ms, ocr_ms)],
                    label='OCR (avg)', alpha=0.7, color='#e74c3c')
    ax.fill_between(times,
                    [d+t+o for d, t, o in zip(detection_ms, tracking_ms, ocr_ms)],
                    [d+t+o+tr for d, t, o, tr in zip(detection_ms, tracking_ms, ocr_ms, trans_ms)],
                    label='Translation (avg)', alpha=0.7, color='#f39c12')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Component Timing Breakdown (Stacked View)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'stacked_timing.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'stacked_timing.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Stacked timing plot saved")

def create_gpu_memory_plot():
    """Create GPU memory usage over time plot"""
    print("Creating GPU memory plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(metrics.timestamps, metrics.gpu_memory, 
           linewidth=2, color='#9b59b6', alpha=0.8)
    ax.fill_between(metrics.timestamps, 0, metrics.gpu_memory, 
                    alpha=0.3, color='#9b59b6')
    ax.axhline(y=np.mean(metrics.gpu_memory), color='#e74c3c', 
              linestyle='--', linewidth=2,
              label=f'Average: {np.mean(metrics.gpu_memory):.2f} GB')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Memory Usage Over Time', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gpu_memory.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'gpu_memory.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ GPU memory plot saved")

def create_summary_dashboard(stats):
    """Create summary dashboard with key metrics"""
    print("Creating summary dashboard...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Pipeline Performance Summary Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 1. FPS gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, f"{stats['average_fps']:.1f}", 
            ha='center', va='center', fontsize=48, fontweight='bold', color='#2ecc71')
    ax1.text(0.5, 0.2, 'Average FPS', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Latency gauge
    ax2 = fig.add_subplot(gs[0, 1])
    latency_color = '#2ecc71' if stats['average_total_latency_ms'] < 100 else '#e74c3c'
    ax2.text(0.5, 0.5, f"{stats['average_total_latency_ms']:.1f}", 
            ha='center', va='center', fontsize=48, fontweight='bold', color=latency_color)
    ax2.text(0.5, 0.2, 'Avg Latency (ms)', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Cache hit rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.5, f"{stats['cache_hit_rate_percent']:.1f}%", 
            ha='center', va='center', fontsize=48, fontweight='bold', color='#3498db')
    ax3.text(0.5, 0.2, 'Cache Hit Rate', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. Component breakdown bar chart
    ax4 = fig.add_subplot(gs[1, :])
    components = ['Detection', 'Tracking', 'OCR', 'Translation']
    latencies = [
        stats['average_detection_latency_ms'],
        stats['average_tracking_latency_ms'],
        stats['average_ocr_latency_ms'],
        stats['average_translation_latency_ms']
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax4.barh(components, latencies, color=colors, alpha=0.8)
    ax4.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Component Latency Breakdown', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, latencies)):
        ax4.text(value + 1, i, f'{value:.1f} ms', 
                va='center', fontsize=10, fontweight='bold')
    
    # 5. Key statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    table_data = [
        ['Total Frames', f"{stats['total_frames_processed']:,}"],
        ['Frames Analyzed', f"{stats['frames_analyzed']:,}"],
        ['Dropped Frames', f"{stats['dropped_frames']:,} ({stats['frame_drop_rate_percent']:.2f}%)"],
        ['OCR Operations', f"{stats['total_ocr_operations']:,}"],
        ['Translations', f"{stats['total_translations']:,}"],
        ['Avg GPU Memory', f"{stats['average_gpu_memory_gb']:.2f} GB"],
        ['Max GPU Memory', f"{stats['max_gpu_memory_gb']:.2f} GB"],
        ['Avg Active Overlays', f"{stats['average_active_overlays']:.1f}"],
    ]
    
    table = ax5.table(cellText=table_data, 
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(table_data) + 1):
        if i == 0:
            table[(i, 0)].set_facecolor('#34495e')
            table[(i, 1)].set_facecolor('#34495e')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            table[(i, 1)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.savefig(os.path.join(PLOTS_DIR, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'summary_dashboard.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Summary dashboard saved")

# ======================================================================
# MAIN PIPELINE LOOP
# ======================================================================
def run_pipeline():
    """Run the full pipeline with metrics collection"""
    print("\n" + "="*50)
    print("STARTING PIPELINE EVALUATION")
    print("="*50)
    
    cap = setup_video_source()
    if cap is None:
        return
    
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, round(fps_in / TARGET_FPS))
    
    print(f"Input: {fps_in}fps, Target: {TARGET_FPS}fps, Skip: {skip}")
    
    # Start threads
    dt = threading.Thread(target=detection_thread, daemon=True)
    dt.start()
    
    workers = []
    for i in range(MAX_WORKERS):
        worker = threading.Thread(target=ocr_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)
    
    # Video writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    
    last_time = time.time()
    detection_time = 0
    tracking_time = 0
    
    # Screenshot capture intervals
    screenshot_interval = total_video_frames // 10  # Capture 10 screenshots
    
    print("Processing video... (Press 'q' to quit)")
    
    try:
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            metrics.increment_total_frames()
            frame_id = metrics.total_frames
            
            # Frame skipping
            if frame_id % skip == 0:
                try:
                    frame_q.put((frame_id, frame.copy()), block=False)
                except queue.Full:
                    metrics.increment_dropped_frames()
            
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
                    frame = draw_text_overlay(frame, text, (x1,y1,x2,y2), avg)
                    overlay_count += 1
            
            # FPS calculation
            now = time.time()
            fps_disp = 1.0 / (now - last_time) if now != last_time else TARGET_FPS
            
            # Get timing info from stored dictionary
            detection_time = 0
            tracking_time = 0
            with timing_lock:
                if frame_id in frame_timing:
                    timing_data = frame_timing[frame_id]
                    detection_time = timing_data.get('detection_time', 0)
                    tracking_time = timing_data.get('tracking_time', 0)
                    # Clean up old entries to prevent memory buildup
                    if len(frame_timing) > 1000:
                        old_keys = [k for k in frame_timing.keys() if k < frame_id - 500]
                        for k in old_keys:
                            frame_timing.pop(k, None)
            
            total_latency = now - frame_start
            
            # Record metrics
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            metrics.record_frame(
                frame_id, fps_disp, total_latency, detection_time, tracking_time,
                frame_q.qsize(), ocr_q.qsize(), gpu_mem, overlay_count
            )
            
            # Draw HUD
            frame = draw_hud(frame, fps_disp, overlay_count)
            
            # Capture screenshots
            if frame_id % screenshot_interval == 0 and len(screenshot_frames) < 10:
                screenshot_path = os.path.join(SCREENSHOTS_DIR, f'frame_{frame_id:06d}.png')
                cv2.imwrite(screenshot_path, frame)
                screenshot_frames.append(screenshot_path)
                print(f"Screenshot captured: frame {frame_id}")
            
            last_time = now
            
            # Output
            writer.write(frame)
            cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit")
                break
            
            # Progress update
            if frame_id % 200 == 0:
                progress = (frame_id / total_video_frames) * 100
                print(f"Progress: {progress:.1f}% - FPS: {fps_disp:.1f} - "
                      f"GPU: {gpu_mem:.2f}GB - Active: {overlay_count}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        print("Cleaning up...")
        
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
        
        print("Pipeline execution complete!")

# ======================================================================
# MAIN EXECUTION
# ======================================================================
def main():
    """Main execution function"""
    print("="*70)
    print("JETSON PIPELINE END-TO-END EVALUATION")
    print("="*70)
    
    # Run pipeline
    run_pipeline()
    
    # Save metrics
    print("\n" + "="*50)
    print("SAVING METRICS")
    print("="*50)
    
    detailed_csv = os.path.join(TABLES_DIR, 'detailed_metrics.csv')
    summary_csv = os.path.join(TABLES_DIR, 'summary_statistics.csv')
    
    metrics.save_detailed_csv(detailed_csv)
    metrics.save_summary_csv(summary_csv)
    
    # Create visualizations
    create_visualizations()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    stats = metrics.get_summary_statistics()
    
    print(f"\n PERFORMANCE SUMMARY:")
    print(f"   Average FPS: {stats['average_fps']:.2f}")
    print(f"   Average Latency: {stats['average_total_latency_ms']:.2f} ms")
    print(f"   Detection Latency: {stats['average_detection_latency_ms']:.2f} ms")
    print(f"   Tracking Latency: {stats['average_tracking_latency_ms']:.2f} ms")
    print(f"   OCR Latency: {stats['average_ocr_latency_ms']:.2f} ms")
    print(f"   Translation Latency: {stats['average_translation_latency_ms']:.2f} ms")
    print(f"   Frame Drop Rate: {stats['frame_drop_rate_percent']:.2f}%")
    print(f"   Cache Hit Rate: {stats['cache_hit_rate_percent']:.2f}%")
    print(f"   Average GPU Memory: {stats['average_gpu_memory_gb']:.2f} GB")
    
    print(f"\n OUTPUT LOCATIONS:")
    print(f"   Video: {OUTPUT_VIDEO}")
    print(f"   Tables: {TABLES_DIR}")
    print(f"   Plots: {PLOTS_DIR}")
    print(f"   Screenshots: {SCREENSHOTS_DIR}")
    
    print("\n All results saved successfully!")

if __name__ == '__main__':
    main()