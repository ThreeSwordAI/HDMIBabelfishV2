import cv2
import time
import threading
import queue
import numpy as np
import torch
import sys
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

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

# -------- CONFIGURATION --------
VIDEO_PATH = "../data/test_video/JapanRPG_TestSequence.mov"

# Model paths
MODEL_PATH = "../experiments/transfer_learning/YOLOv11/Target/FT_model_ultralytics_nano/runs/detect/train/weights/best.pt"
MARIAN_MODEL_PATH = "../translation/models/marian_opus_ft_ja_en_full_dataset"
WINDOW_NAME = "Jetson Orin Real-Time Translation"

# Settings
TARGET_FPS = 25
STABILITY_MS = 200  # Critical heuristic for thesis
YOLO_INPUT_SIZE = 640
MAX_WORKERS = 6
BATCH_SIZE = 4
QUEUE_SIZE = 150

# Frame dropping
MAX_FRAME_DELAY = 50
ENABLE_FRAME_DROPPING = True

# Output paths
BASE_DIR = "results/jetson_video"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "jetson_translation.mp4")
TRACKING_OUTPUT_DIR = os.path.join(BASE_DIR, "output_tracking")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(TRACKING_OUTPUT_DIR, exist_ok=True)

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

# Global variables
translation_cache = {}
MAX_CACHE_SIZE = 1000
translator_model = None

# ============================================================================
# TRACKING METRICS COLLECTION (FOR THESIS)
# ============================================================================

tracking_metrics = {
    'frame_timestamps': [],
    'track_ids_per_frame': [],
    'id_switches': [],
    'track_lifetimes': {},  # track_id -> (birth_time, death_time)
    'track_positions': defaultdict(list),  # track_id -> [(frame, x, y, w, h), ...]
    'stable_boxes': 0,
    'unstable_boxes': 0,
    'fps_samples': [],
    'detection_times': [],
    'ocr_times': []
}

track_birth_times = {}
track_last_seen = {}
previous_tracks = {}

# Track colors for visualization
TRACK_COLORS = [
    (255, 100, 100), (100, 255, 100), (255, 255, 100), (255, 100, 255),
    (100, 255, 255), (100, 100, 255), (255, 150, 100), (150, 255, 100),
    (255, 100, 150), (100, 150, 255)
]

def get_track_color(track_id):
    """Get consistent color for track ID"""
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def detect_id_switches(current_tracks, previous_tracks, frame_num, iou_threshold=0.5):
    """Detect ID switches based on IoU overlap"""
    switches = []
    
    for curr_id, curr_box in current_tracks.items():
        best_iou = 0
        best_prev_id = None
        
        for prev_id, prev_box in previous_tracks.items():
            iou = compute_iou(curr_box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_prev_id = prev_id
        
        if best_iou > iou_threshold and best_prev_id != curr_id:
            switches.append({
                'frame': frame_num,
                'old_id': int(best_prev_id),
                'new_id': int(curr_id),
                'iou': float(best_iou)
            })
    
    return switches

# ============================================================================
# TRANSLATION MODEL
# ============================================================================

class FixedMarianTranslator:
    """Fixed synchronous translation"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load Marian model with optimization"""
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
            
            # Test translation
            test_result = self.translate("テスト")
            print(f"Test translation: 'テスト' -> '{test_result}'")
            print("Translation model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Error loading Marian model: {e}")
            return False
    
    def translate(self, text):
        """Simple synchronous translation"""
        if not text.strip():
            return ""
        
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
            return result
            
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            error_result = f"[Error: {text}]"
            translation_cache[text] = error_result
            return error_result

def initialize_translation():
    """Initialize translation system"""
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
    
    text1 = text1.replace(' ', '').replace('\n', '').replace('-', '').replace('_', '').replace('*', '')
    text2 = text2.replace(' ', '').replace('\n', '').replace('-', '').replace('_', '').replace('*', '')
    
    if text1 == text2:
        return 1.0
    
    longer = max(len(text1), len(text2))
    if longer == 0:
        return 1.0
    
    matches = sum(1 for i in range(min(len(text1), len(text2))) if text1[i] == text2[i])
    length_similarity = 1.0 - abs(len(text1) - len(text2)) / longer
    similarity = (matches / longer) * 0.7 + length_similarity * 0.3
    
    return similarity

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

# ============================================================================
# INITIALIZE MODELS
# ============================================================================

print("JETSON VIDEO INITIALIZATION")
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

# Frames to save for visualization
selected_frames_for_vis = []
selected_frame_numbers = []
SAVE_FRAME_NUMBERS = [50, 100, 150]  # For Figure 6.2

# ============================================================================
# DETECTION THREAD
# ============================================================================

def fixed_detect_thread():
    """Detection thread with tracking metrics collection"""
    print("Detection thread started")
    global dropped_frames, previous_tracks
    
    while True:
        item = frame_q.get()
        if item is None:
            print("Detection thread stopping")
            break
        idx, frame, current_time = item
        h, w = frame.shape[:2]
        
        if ENABLE_FRAME_DROPPING and frame_q.qsize() > MAX_FRAME_DELAY:
            dropped_frames += 1
            frame_q.task_done()
            continue
        
        # YOLO detection
        detect_start = time.time()
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
            
            detect_time = time.time() - detect_start
            tracking_metrics['detection_times'].append(detect_time)
            
            now = time.time()*1000
            current = set()
            current_tracks_dict = {}
            track_ids = []
            
            for x1,y1,x2,y2,tid in tracks:
                tid = int(tid)
                current.add(tid)
                track_ids.append(tid)
                box = (int(x1),int(y1),int(x2),int(y2))
                current_tracks_dict[tid] = [int(x1), int(y1), int(x2), int(y2)]
                
                # Record track position
                tracking_metrics['track_positions'][tid].append((idx, int(x1), int(y1), int(x2), int(y2)))
                
                # Track birth/last seen times
                if tid not in track_birth_times:
                    track_birth_times[tid] = current_time
                track_last_seen[tid] = current_time
                
                st = ostates.get(tid)
                
                if st:
                    position_change = np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2]))
                    
                    if position_change < 30:
                        st['box'] = box
                        if now - st['start'] >= STABILITY_MS and not st['pending']:
                            if not ocr_q.full():
                                ocr_q.put((tid, box, frame.copy()))
                                st['pending'] = True
                                if not st.get('stable_counted', False):
                                    tracking_metrics['stable_boxes'] += 1
                                    st['stable_counted'] = True
                    else:
                        tracking_metrics['unstable_boxes'] += 1
                        ostates[tid] = {'box': box, 'start': now, 'pending': False, 'stable_counted': False}
                else:
                    ostates[tid] = {'box': box, 'start': now, 'pending': False, 'stable_counted': False}
            
            # Detect ID switches
            if previous_tracks:
                switches = detect_id_switches(current_tracks_dict, previous_tracks, idx)
                tracking_metrics['id_switches'].extend(switches)
            
            previous_tracks = current_tracks_dict.copy()
            
            # Record frame metrics
            tracking_metrics['track_ids_per_frame'].append(track_ids)
            
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

# ============================================================================
# OCR WORKER THREAD
# ============================================================================

def fixed_ocr_worker(worker_id):
    """OCR worker with translation"""
    print(f"OCR worker {worker_id} started")
    global ocr_count, translation_count
    
    while True:
        try:
            item = ocr_q.get(timeout=1)
            if item is None:
                print(f"OCR worker {worker_id} stopping")
                break
            
            ocr_start = time.time()
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
            tracking_metrics['ocr_times'].append(ocr_time)
            ocr_count += 1
            
            if txt and len(txt) > 1:
                # Check for existing translations
                with overlay_lock:
                    existing_data = overlay_states.get(tid)
                    if existing_data and existing_data.get('original_text') == txt:
                        existing_data['box'] = box
                        ocr_q.task_done()
                        continue
                    
                    if existing_data:
                        existing_text = existing_data.get('original_text', '')
                        similarity = calculate_text_similarity(txt, existing_text)
                        
                        if similarity > 0.85:
                            existing_data['box'] = box
                            ocr_q.task_done()
                            continue
                
                # Translation
                eng = get_translation(txt)
                translation_count += 1
                
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

# ============================================================================
# DRAWING FUNCTIONS
# ============================================================================

def draw_simple_text(frame, text, box, avg_color):
    """Multi-line text rendering within detection box"""
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

def draw_tracking_visualization(frame, current_tracks):
    """Draw colored bounding boxes with track IDs for visualization"""
    vis_frame = frame.copy()
    
    for tid, box in current_tracks.items():
        x1, y1, x2, y2 = box
        color = get_track_color(tid)
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw ID label
        label = f"ID: {tid}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Background for text
        cv2.rectangle(vis_frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(vis_frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return vis_frame

def fixed_hud(frame, fps_disp, overlay_count, queue_depths, stats):
    """HUD with debug info"""
    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Translations: {overlay_count}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Frame Q: {queue_depths['frame']}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"OCR Q: {queue_depths['ocr']}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"OCR Count: {stats['ocr']}", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Trans Count: {stats['translation']}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    if dropped_frames > 0:
        cv2.putText(frame, f"Dropped: {dropped_frames}", (10,230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    return frame

# ============================================================================
# TRACKING VISUALIZATION GENERATION (FOR THESIS)
# ============================================================================

def generate_table_6_2():
    """Generate Table 6.2: Tracking Performance Summary"""
    print("\n" + "="*50)
    print("Generating Table 6.2: Tracking Performance Summary...")
    print("="*50)
    
    total_frames = len(tracking_metrics['track_ids_per_frame'])
    avg_fps = np.mean(tracking_metrics['fps_samples']) if tracking_metrics['fps_samples'] else 0
    
    num_switches = len(tracking_metrics['id_switches'])
    switches_per_1000 = (num_switches / total_frames * 1000) if total_frames > 0 else 0
    
    lifetimes = [track_last_seen[tid] - track_birth_times[tid] 
                 for tid in track_birth_times if tid in track_last_seen]
    avg_lifetime = np.mean(lifetimes) if lifetimes else 0
    median_lifetime = np.median(lifetimes) if lifetimes else 0
    
    total_detections = tracking_metrics['stable_boxes'] + tracking_metrics['unstable_boxes']
    ocr_with_stability = tracking_metrics['stable_boxes']
    ocr_without_stability = total_detections
    ocr_reduction = ((ocr_without_stability - ocr_with_stability) / ocr_without_stability * 100) if ocr_without_stability > 0 else 0
    
    table_data = {
        'Metric': [
            'Average FPS',
            'Avg. ID Switches per 1000 frames',
            'Avg. Track Lifetime',
            'Stability Trigger Threshold',
            'Reduction in OCR Calls'
        ],
        'Description': [
            'Throughput during tracking-only test',
            'Reassignment of bounding box IDs',
            'Mean persistence of text box ID',
            'Heuristic delay before OCR call',
            'Compared to per-frame OCR'
        ],
        'Value / Comment': [
            f'~{avg_fps:.0f} FPS',
            f'< {switches_per_1000:.0f}' if switches_per_1000 < 10 else f'{switches_per_1000:.1f}',
            f'{avg_lifetime:.1f}–{median_lifetime:.1f} s',
            f'{STABILITY_MS} ms',
            f'≈ {ocr_reduction:.0f}% fewer'
        ]
    }
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = os.path.join(TRACKING_OUTPUT_DIR, "table_6_2_tracking_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path}")
    
    # Create formatted table image
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.32, 0.43, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', size=12)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Style data rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(1)
            
            if j == 2:
                cell.set_text_props(weight='bold')
    
    plt.title('Table 6.2: Tracking Performance Summary', 
              fontsize=15, weight='bold', pad=25)
    
    # Save PNG and PDF
    png_path = os.path.join(TRACKING_OUTPUT_DIR, "table_6_2_tracking_metrics.png")
    pdf_path = os.path.join(TRACKING_OUTPUT_DIR, "table_6_2_tracking_metrics.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PNG: {png_path}")
    print(f"  ✓ Saved PDF: {pdf_path}")
    
    plt.close()
    
    return df

def generate_figure_6_2():
    """Generate Figure 6.2: Temporal Association of Text Boxes"""
    print("\nGenerating Figure 6.2: Temporal Consistency Visualization...")
    
    if len(selected_frames_for_vis) < 3:
        print("  Warning: Not enough frames saved for temporal visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    
    for idx, (ax, frame, frame_num) in enumerate(zip(axes, selected_frames_for_vis, selected_frame_numbers)):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        ax.imshow(frame_rgb)
        ax.set_title(f'Frame {frame_num}', fontsize=13, weight='bold', pad=10)
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(2)
    
    # Add arrows
    fig.text(0.335, 0.5, '→', fontsize=60, ha='center', va='center', 
            weight='bold', color='#FF4444', transform=fig.transFigure)
    fig.text(0.665, 0.5, '→', fontsize=60, ha='center', va='center', 
            weight='bold', color='#FF4444', transform=fig.transFigure)
    
    title_text = 'Figure 6.2: Temporal Association of Text Boxes Across Frames'
    caption_text = ('Same color indicates consistent track ID assigned by SORT. '
                   'Only boxes stable for >200 ms are forwarded to OCR.')
    
    fig.text(0.5, 0.98, title_text, 
            ha='center', fontsize=15, weight='bold', transform=fig.transFigure)
    fig.text(0.5, 0.94, caption_text,
            ha='center', fontsize=11, style='italic', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save
    png_path = os.path.join(TRACKING_OUTPUT_DIR, "figure_6_2_temporal_consistency.png")
    pdf_path = os.path.join(TRACKING_OUTPUT_DIR, "figure_6_2_temporal_consistency.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PNG: {png_path}")
    print(f"  ✓ Saved PDF: {pdf_path}")
    
    plt.close()

def generate_supplementary_figures():
    """Generate supplementary tracking analysis figures"""
    print("\nGenerating supplementary figures...")
    
    # FPS Distribution
    if tracking_metrics['fps_samples']:
        fig, ax = plt.subplots(figsize=(10, 6))
        fps_data = tracking_metrics['fps_samples']
        ax.hist(fps_data, bins=50, color='#4472C4', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(fps_data), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(fps_data):.1f} FPS')
        ax.axvline(np.median(fps_data), color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(fps_data):.1f} FPS')
        ax.set_xlabel('FPS', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax.set_title('Tracking FPS Distribution', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        png_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_fps_distribution.png")
        pdf_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_fps_distribution.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved FPS distribution")
        plt.close()
    
    # Track Lifetime Distribution
    lifetimes = [track_last_seen[tid] - track_birth_times[tid] 
                 for tid in track_birth_times if tid in track_last_seen]
    
    if lifetimes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lifetimes, bins=30, color='#70AD47', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(lifetimes), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(lifetimes):.2f} s')
        ax.axvline(np.median(lifetimes), color='blue', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(lifetimes):.2f} s')
        ax.set_xlabel('Track Lifetime (seconds)', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax.set_title('Track Lifetime Distribution', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        png_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_track_lifetime.png")
        pdf_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_track_lifetime.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved track lifetime distribution")
        plt.close()
    
    # Active Tracks Timeline
    if tracking_metrics['track_ids_per_frame']:
        fig, ax = plt.subplots(figsize=(12, 6))
        num_tracks_per_frame = [len(track_ids) for track_ids in tracking_metrics['track_ids_per_frame']]
        frames = list(range(len(num_tracks_per_frame)))
        
        ax.plot(frames, num_tracks_per_frame, color='#4472C4', linewidth=1, alpha=0.7)
        ax.fill_between(frames, num_tracks_per_frame, alpha=0.3, color='#4472C4')
        
        ax.set_xlabel('Frame Number', fontsize=12, weight='bold')
        ax.set_ylabel('Number of Active Tracks', fontsize=12, weight='bold')
        ax.set_title('Active Tracks Over Time', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        png_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_active_tracks.png")
        pdf_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_active_tracks.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved active tracks timeline")
        plt.close()
    
    # Stability Impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    stable = tracking_metrics['stable_boxes']
    unstable = tracking_metrics['unstable_boxes']
    
    categories = [f'Stable\n(>{STABILITY_MS}ms)', f'Unstable\n(<{STABILITY_MS}ms)']
    values = [stable, unstable]
    colors = ['#70AD47', '#FFC000']
    
    bars1 = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Detections', fontsize=12, weight='bold')
    ax1.set_title('Box Stability Distribution', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax1.text(i, v + max(values)*0.02, str(v), ha='center', va='bottom', fontsize=11, weight='bold')
    
    total = stable + unstable
    ocr_with_stability = stable
    ocr_without_stability = total
    
    categories2 = ['Without\nStability', f'With\nStability\n({STABILITY_MS}ms)']
    values2 = [ocr_without_stability, ocr_with_stability]
    colors2 = ['#C5504D', '#4472C4']
    
    bars2 = ax2.bar(categories2, values2, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('OCR Calls', fontsize=12, weight='bold')
    ax2.set_title('OCR Call Reduction Impact', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values2):
        ax2.text(i, v + max(values2)*0.02, str(v), ha='center', va='bottom', fontsize=11, weight='bold')
    
    reduction = ((ocr_without_stability - ocr_with_stability) / ocr_without_stability * 100) if ocr_without_stability > 0 else 0
    ax2.text(0.5, max(values2)*0.5, f'↓ {reduction:.0f}%\nreduction', 
             ha='center', va='center', fontsize=16, weight='bold', color='green',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2))
    
    plt.suptitle(f'Impact of {STABILITY_MS}ms Stability Heuristic on OCR Efficiency', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    png_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_stability_impact.png")
    pdf_path = os.path.join(TRACKING_OUTPUT_DIR, "supplementary_stability_impact.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved stability impact visualization")
    plt.close()

def save_tracking_metrics():
    """Save complete tracking metrics as JSON"""
    print("\nSaving tracking metrics JSON...")
    
    lifetimes = [track_last_seen[tid] - track_birth_times[tid] 
                 for tid in track_birth_times if tid in track_last_seen]
    
    total_frames = len(tracking_metrics['track_ids_per_frame'])
    avg_fps = np.mean(tracking_metrics['fps_samples']) if tracking_metrics['fps_samples'] else 0
    num_switches = len(tracking_metrics['id_switches'])
    switches_per_1000 = (num_switches / total_frames * 1000) if total_frames > 0 else 0
    
    metrics_export = {
        'tracking_performance': {
            'average_fps': float(avg_fps),
            'min_fps': float(np.min(tracking_metrics['fps_samples'])) if tracking_metrics['fps_samples'] else 0,
            'max_fps': float(np.max(tracking_metrics['fps_samples'])) if tracking_metrics['fps_samples'] else 0,
        },
        'temporal_stability': {
            'total_id_switches': num_switches,
            'id_switches_per_1000_frames': float(switches_per_1000),
            'average_track_lifetime_seconds': float(np.mean(lifetimes)) if lifetimes else 0,
            'median_track_lifetime_seconds': float(np.median(lifetimes)) if lifetimes else 0,
            'unique_tracks': len(track_birth_times)
        },
        'stability_heuristic': {
            'threshold_ms': STABILITY_MS,
            'stable_boxes': tracking_metrics['stable_boxes'],
            'unstable_boxes': tracking_metrics['unstable_boxes'],
            'ocr_reduction_percentage': ((tracking_metrics['stable_boxes'] + tracking_metrics['unstable_boxes'] - tracking_metrics['stable_boxes']) / 
                                        (tracking_metrics['stable_boxes'] + tracking_metrics['unstable_boxes']) * 100) 
                                        if (tracking_metrics['stable_boxes'] + tracking_metrics['unstable_boxes']) > 0 else 0
        },
        'dataset_info': {
            'total_frames': total_frames
        }
    }
    
    json_path = os.path.join(TRACKING_OUTPUT_DIR, "tracking_metrics_complete.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_export, f, indent=2)
    
    print(f"  ✓ Saved JSON: {json_path}")
    
    return metrics_export

def generate_tracking_report(metrics):
    """Generate comprehensive tracking evaluation report"""
    print("\nGenerating tracking evaluation report...")
    
    report = []
    report.append("="*70)
    report.append("TRACKING EVALUATION REPORT")
    report.append("Thesis Section 6.4: Object Tracking and Temporal Consistency")
    report.append("="*70)
    report.append("")
    report.append("## KEY METRICS ##")
    report.append(f"Average FPS: {metrics['tracking_performance']['average_fps']:.1f}")
    report.append(f"ID Switches per 1000 frames: {metrics['temporal_stability']['id_switches_per_1000_frames']:.1f}")
    report.append(f"Average Track Lifetime: {metrics['temporal_stability']['average_track_lifetime_seconds']:.2f}s")
    report.append(f"Stability Threshold: {metrics['stability_heuristic']['threshold_ms']}ms")
    report.append(f"OCR Reduction: {metrics['stability_heuristic']['ocr_reduction_percentage']:.1f}%")
    report.append("")
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    report_path = os.path.join(TRACKING_OUTPUT_DIR, "tracking_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved report: {report_path}")
    print("\n" + report_text)

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def run():
    """Main processing loop with tracking metrics collection"""
    print("\n" + "="*50)
    print("STARTING TRANSLATION PIPELINE WITH TRACKING EVALUATION")
    print("="*50)
    
    global total_frames, dropped_frames, ocr_count, translation_count
    
    cap = setup_video_source()
    if cap is None:
        print("Could not setup video source")
        return
        
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, round(fps_in / TARGET_FPS))
    
    print(f"Input: {fps_in}fps, Target: {TARGET_FPS}fps, Skip: {skip}")
    
    # Start detection thread
    dt = threading.Thread(target=fixed_detect_thread, daemon=True)
    dt.start()
    
    fps_list, time_list = [], []
    start_time = time.time()
    
    # Video writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    
    # Start OCR workers
    workers = []
    for i in range(MAX_WORKERS):
        worker = threading.Thread(target=fixed_ocr_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)

    last = time.time()
    
    print("Processing video... (Press 'q' to quit)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
                
            total_frames += 1
            current_time = time.time()
            
            # Save frames for Figure 6.2 visualization
            if total_frames in SAVE_FRAME_NUMBERS:
                # Get current tracks for visualization
                current_tracks_dict = {}
                for tid in list(previous_tracks.keys()):
                    if tid in previous_tracks:
                        current_tracks_dict[tid] = previous_tracks[tid]
                
                vis_frame = draw_tracking_visualization(frame, current_tracks_dict)
                selected_frames_for_vis.append(vis_frame)
                selected_frame_numbers.append(total_frames)
            
            # Frame skipping for target FPS
            if total_frames % skip == 0:
                try:
                    frame_q.put((total_frames, frame.copy(), current_time), block=False)
                except queue.Full:
                    dropped_frames += 1
            
            # Record timestamp
            tracking_metrics['frame_timestamps'].append(current_time - start_time)
                
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
            tracking_metrics['fps_samples'].append(fps_disp)
            
            # HUD
            queue_depths = {
                'frame': frame_q.qsize(),
                'ocr': ocr_q.qsize()
            }
            stats = {
                'ocr': ocr_count,
                'translation': translation_count
            }
            frame = fixed_hud(frame, fps_disp, overlay_count, queue_depths, stats)
            
            last = now
            
            # Output
            writer.write(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit")
                break
            
            # Progress update
            if total_frames % 200 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                progress = (total_frames / total_video_frames) * 100
                drop_rate = (dropped_frames / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% - FPS: {fps_disp:.1f} - GPU: {gpu_mem:.1f}GB - Drop: {drop_rate:.1f}% - OCR: {ocr_count} - Trans: {translation_count} - Active: {overlay_count}")

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
        
        # Calculate track lifetimes
        for tid in track_birth_times:
            if tid in track_last_seen:
                lifetime = track_last_seen[tid] - track_birth_times[tid]
                tracking_metrics['track_lifetimes'][tid] = lifetime
        
        # Performance report
        if fps_list:
            avg_fps = np.mean(fps_list)
            min_fps = np.min(fps_list)
            max_fps = np.max(fps_list)
            gpu_mem_final = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            drop_rate = (dropped_frames / total_frames) * 100 if total_frames > 0 else 0
            
            print(f"\n" + "="*50)
            print("TRANSLATION PERFORMANCE SUMMARY")
            print("="*50)
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Min FPS: {min_fps:.1f}")
            print(f"   Max FPS: {max_fps:.1f}")
            print(f"   Total Frames: {total_frames}")
            print(f"   Dropped Frames: {dropped_frames} ({drop_rate:.1f}%)")
            print(f"   OCR Operations: {ocr_count}")
            print(f"   Translations: {translation_count}")
            print(f"   Final GPU Memory: {gpu_mem_final:.1f}GB")
            print(f"   Cache Size: {len(translation_cache)}")
            print(f"\nVideo saved: {OUTPUT_VIDEO}")
        
        # Generate tracking evaluation outputs
        print("\n" + "="*50)
        print("GENERATING TRACKING EVALUATION OUTPUTS FOR THESIS")
        print("="*50)
        
        # Generate all figures and tables
        table_df = generate_table_6_2()
        generate_figure_6_2()
        generate_supplementary_figures()
        metrics = save_tracking_metrics()
        generate_tracking_report(metrics)
        
        print("\n" + "="*50)
        print("✅ TRACKING EVALUATION COMPLETE!")
        print("="*50)
        print(f"All outputs saved to: {TRACKING_OUTPUT_DIR}")
        print("\nGenerated files:")
        print("  1. table_6_2_tracking_metrics.csv/png/pdf")
        print("  2. figure_6_2_temporal_consistency.png/pdf")
        print("  3. supplementary_fps_distribution.png/pdf")
        print("  4. supplementary_track_lifetime.png/pdf")
        print("  5. supplementary_active_tracks.png/pdf")
        print("  6. supplementary_stability_impact.png/pdf")
        print("  7. tracking_metrics_complete.json")
        print("  8. tracking_evaluation_report.txt")
        print("\n✅ Ready for thesis Section 6.4!")

if __name__ == '__main__':
    run()