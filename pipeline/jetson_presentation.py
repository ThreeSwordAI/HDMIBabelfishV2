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

# -------- FIXED REAL-TIME CONFIG --------
VIDEO_PATH = "/workspace/data/test_video/For_Presentation.mp4"

# Model paths using new structure
MODEL_PATH = "/workspace/pipeline/models/object_detection/yolo11n_text_detection.pt"
MARIAN_MODEL_PATH = "/workspace/pipeline/models/translation/marian_opus_ja_en_finetuned"
WINDOW_NAME = "Jetson Orin Real-Time Translation"

# BALANCED SETTINGS FOR RELIABLE TRANSLATION
TARGET_FPS = 25              # Realistic target
STABILITY_MS = 200           # Reduced for faster response
YOLO_INPUT_SIZE = 640        # Good accuracy
MAX_WORKERS = 6              # Reasonable number of workers
BATCH_SIZE = 4               # Smaller batches for reliability
QUEUE_SIZE = 150             # Moderate queue size

# Frame dropping configuration
MAX_FRAME_DELAY = 50         # Drop frames if queue gets too full
ENABLE_FRAME_DROPPING = True

# Output paths - UPDATED to new directory
BASE_DIR = "/workspace/pipeline/results/jetson_video"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "jetson_translation.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_translation.png")

# IMAGE SAVING CONFIGURATION
SAVE_IMAGES_PER_SECOND = 5   # Save 5-6 frames per second

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

# Global variables for tracking detected boxes
detected_boxes_data = {}  # Store detection data for saving cropped images
detected_boxes_lock = threading.Lock()

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

def setup_presentation_folders():
    """Create folder structure for presentation materials"""
    presentation_dir = os.path.join(BASE_DIR, "for_presentation")
    
    folders = {
        'input_frames': os.path.join(presentation_dir, "1_input_frames"),
        'detection_boxes': os.path.join(presentation_dir, "2_detection_boxes"),
        'cropped_boxes': os.path.join(presentation_dir, "3_cropped_boxes"),
        'output_frames': os.path.join(presentation_dir, "4_output_frames"),
        'output_video': os.path.join(presentation_dir, "5_output_video")
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    print("\nPresentation folders created:")
    for name, path in folders.items():
        print(f"  {name}: {path}")
    
    return folders

def should_save_frame(frame_number, fps, save_per_second=SAVE_IMAGES_PER_SECOND):
    """Determine if frame should be saved based on interval"""
    save_every = max(1, int(fps / save_per_second))
    return frame_number % save_every == 0

def save_input_frame(frame, frame_number, folder):
    """Save input frame"""
    filename = f"input_frame_{frame_number:06d}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)

def save_detection_boxes_frame(frame, frame_number, folder):
    """Save frame with detection boxes drawn"""
    filename = f"detection_frame_{frame_number:06d}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)

def save_cropped_boxes(frame, frame_number, boxes, folder):
    """Save cropped regions of detected boxes"""
    for idx, (tid, box) in enumerate(boxes):
        x1, y1, x2, y2 = box
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            filename = f"cropped_frame_{frame_number:06d}_box_{tid}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, cropped)

def save_output_frame(frame, frame_number, folder):
    """Save output frame with translations"""
    filename = f"output_frame_{frame_number:06d}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)

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

# Initialize models
print("FIXED JETSON INITIALIZATION")
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
            
            # Store detection data for saving
            current_boxes = []
            
            for x1,y1,x2,y2,tid in tracks:
                tid = int(tid)
                current.add(tid)
                box = (int(x1),int(y1),int(x2),int(y2))
                current_boxes.append((tid, box))
                
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
            
            # Store boxes for this frame
            with detected_boxes_lock:
                detected_boxes_data[idx] = current_boxes
            
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
    global ocr_count, translation_count
    
    while True:
        try:
            item = ocr_q.get(timeout=1)
            if item is None:
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
            
            if txt and len(txt) > 1:
                # Check for existing translations
                with overlay_lock:
                    existing_data = overlay_states.get(tid)
                    if existing_data and existing_data.get('original_text') == txt:
                        existing_data['box'] = box
                        ocr_q.task_done()
                        continue
                
                # Direct translation (synchronous)
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
    line_height = int(font_scale * 30)
    padding = 8
    
    # Calculate available space
    box_width = x2 - x1 - (2 * padding)
    box_height = y2 - y1 - (2 * padding)
    max_lines = max(1, box_height // line_height)
    
    # Estimate characters per line based on box width
    chars_per_line = max(10, box_width // int(font_scale * 12))
    
    # Split text into lines that fit within the box
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
    
    # Draw each line
    for i, line in enumerate(lines):
        text_x = x1 + padding
        text_y = y1 + padding + (i + 1) * line_height
        
        if text_y <= y2 - padding:
            cv2.putText(frame, line, (text_x, text_y), font, 
                       font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def draw_detection_boxes(frame, boxes):
    """Draw detection boxes on frame (for saving detection visualization)"""
    frame_with_boxes = frame.copy()
    for tid, box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_with_boxes, f"ID:{tid}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_with_boxes

def run():
    """Fixed processing loop with image saving"""
    print("\n" + "="*50)
    print("STARTING FIXED TRANSLATION PIPELINE")
    print("WITH PRESENTATION IMAGE SAVING")
    print("="*50)
    
    global total_frames, dropped_frames, ocr_count, translation_count
    
    # Setup presentation folders
    folders = setup_presentation_folders()
    
    cap = setup_video_source()
    if cap is None:
        print("Could not setup video source")
        return
        
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, round(fps_in / TARGET_FPS))
    
    print(f"Input: {fps_in}fps, Target: {TARGET_FPS}fps, Skip: {skip}")
    print(f"Saving {SAVE_IMAGES_PER_SECOND} frames per second to presentation folders")
    
    # Start threads
    dt = threading.Thread(target=fixed_detect_thread, daemon=True)
    dt.start()
    
    fps_list, time_list = [], []
    start_time = time.time()
    
    # Video writer for main output
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    
    # Video writer for presentation folder
    presentation_video_path = os.path.join(folders['output_video'], "translated_video.mp4")
    presentation_writer = cv2.VideoWriter(presentation_video_path, fourcc, TARGET_FPS, (w, h))
    
    # Start OCR workers
    workers = []
    for i in range(MAX_WORKERS):
        worker = threading.Thread(target=fixed_ocr_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)

    last = time.time()
    saved_frame_count = 0
    
    print("Processing video... (Press 'q' to quit)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            original_frame = frame.copy()
            total_frames += 1
            
            # Determine if we should save this frame
            should_save = should_save_frame(total_frames, fps_in, SAVE_IMAGES_PER_SECOND)
            
            # Save input frame
            if should_save:
                save_input_frame(original_frame, total_frames, folders['input_frames'])
                saved_frame_count += 1
            
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
            
            # Get detection boxes for this frame
            current_boxes = []
            with detected_boxes_lock:
                if total_frames in detected_boxes_data:
                    current_boxes = detected_boxes_data[total_frames]
            
            # Save detection boxes frame
            if should_save and len(current_boxes) > 0:
                detection_frame = draw_detection_boxes(original_frame, current_boxes)
                save_detection_boxes_frame(detection_frame, total_frames, folders['detection_boxes'])
                
                # Save cropped boxes
                save_cropped_boxes(original_frame, total_frames, current_boxes, folders['cropped_boxes'])
            
            # Draw translations (NO HUD - clean output)
            overlay_count = 0
            with overlay_lock:
                for tid, data in overlay_states.items():
                    x1,y1,x2,y2 = data['box']
                    avg = data['avg']
                    text = data['text']
                    
                    frame = draw_simple_text(frame, text, (x1,y1,x2,y2), avg)
                    overlay_count += 1
            
            # Save output frame
            if should_save:
                save_output_frame(frame, total_frames, folders['output_frames'])
                                
            # FPS calculation
            now = time.time()
            fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
            
            fps_list.append(fps_disp)
            time_list.append(now-start_time)
            
            last = now
            
            # Output - write to both video files
            writer.write(frame)
            presentation_writer.write(frame)
            
            # Display (clean - no HUD)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit")
                break
            
            # Progress update to terminal only
            if total_frames % 200 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                progress = (total_frames / total_video_frames) * 100
                drop_rate = (dropped_frames / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% - FPS: {fps_disp:.1f} - GPU: {gpu_mem:.1f}GB - "
                      f"Saved: {saved_frame_count} frames - Active: {overlay_count}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        
        # Cleanup
        cap.release()
        writer.release()
        presentation_writer.release()
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
            
            print(f"\n{'='*70}")
            print("VIDEO PROCESSING COMPLETE!")
            print(f"{'='*70}")
            print(f"\nPerformance Summary:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Min FPS: {min_fps:.1f}")
            print(f"   Max FPS: {max_fps:.1f}")
            print(f"   Total Frames: {total_frames}")
            print(f"   Dropped Frames: {dropped_frames} ({drop_rate:.1f}%)")
            print(f"   OCR Operations: {ocr_count}")
            print(f"   Translations: {translation_count}")
            print(f"   Final GPU Memory: {gpu_mem_final:.1f}GB")
            print(f"   Cache Size: {len(translation_cache)}")
            
            print(f"\nSaved Files:")
            print(f"   Input frames: {saved_frame_count} images in {folders['input_frames']}")
            print(f"   Detection boxes: {folders['detection_boxes']}")
            print(f"   Cropped boxes: {folders['cropped_boxes']}")
            print(f"   Output frames: {saved_frame_count} images in {folders['output_frames']}")
            print(f"   Main video: {OUTPUT_VIDEO}")
            print(f"   Presentation video: {presentation_video_path}")
            print(f"\n{'='*70}")

if __name__ == '__main__':
    run()