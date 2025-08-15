import cv2
import time
import threading
import queue
import numpy as np
import torch
import os
import gc
from pathlib import Path

# Add packages directory to Python path
import sys
sys.path.append('../packages')

from ultralytics import YOLO
from sort.sort import Sort
from transformers import MarianTokenizer, MarianMTModel
import matplotlib.pyplot as plt

# -------- CONFIG --------
VIDEO_PATH = "../data/test_video/JapanRPG_TestSequence.mov"
MODEL_PATH = "../experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
MARIAN_MODEL_PATH = "../translation/models/marian_opus_ja_en"
OUTPUT_VIDEO = "output/video_output.mp4"
FPS_GRAPH_PNG = "output/fps_output.png"
FPS_GRAPH_PDF = "output/fps_output.pdf"

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# CUDA Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Initialize Marian model for translation
class MarianTranslator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load Marian model and tokenizer"""
        print("="*50)
        print("LOADING MARIAN TRANSLATION MODEL")
        print("="*50)
        
        try:
            # Load tokenizer and model
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.model = MarianMTModel.from_pretrained(self.model_path)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
                print(f"✓ Marian model loaded on {self.device}")
            self.model.eval()
            return True
        except Exception as e:
            print(f"❌ Error loading Marian model: {e}")
            return False
    
    def translate(self, text):
        """Translate text using Marian model"""
        if not text.strip():
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform translation
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            # Decode translated text
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return f"[Error: {text}]"

# Initialize Marian model for translation
translator = MarianTranslator(MARIAN_MODEL_PATH)
if not translator.load_model():
    print("❌ Marian translation model failed to load.")
    exit(1)

# Load YOLO model
yolo = YOLO(MODEL_PATH)
yolo.to(device)

# OCR setup - Use Tesseract or PaddleOCR depending on availability
try:
    import pytesseract
    OCR_BACKEND = 'tesseract'
    OCR_LANG = "jpn"
    TESS_CONFIG = "--psm 6"
except ImportError:
    from paddleocr import PaddleOCR
    OCR_BACKEND = 'paddle'
    paddle_ocr = PaddleOCR(lang='japanese', use_gpu=torch.cuda.is_available())

# Initialize Sort tracker
tracker = Sort()

# -------- VIDEO PIPELINE --------

frame_q = queue.Queue(maxsize=10)  # Queue for frames
ocr_q = queue.Queue(maxsize=10)    # Queue for OCR tasks
ostates = {}            # Tracking states for each object
overlay_states = {}     # States for overlays
active_tids = set()     # Active tracked ids
overlay_lock = threading.Lock()

# Helper functions

def memory_cleanup():
    """Aggressive memory cleanup for CPU/GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def setup_video_source():
    """Setup video source"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_PATH}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video loaded:")
    print(f"   📐 Resolution: {width}x{height}")
    print(f"   🎯 FPS: {fps}")
    print(f"   📊 Total frames: {total_frames}")
    
    return cap

def draw_overlay(frame, text, box):
    """Draw overlays with detected bounding boxes and translated text"""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle around detected object
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# -------- THREAD FUNCTIONS --------

def cpu_detection_thread():
    """Thread for YOLO object detection"""
    while True:
        item = frame_q.get()
        if item is None:
            break
        
        idx, frame = item
        h, w = frame.shape[:2]
        
        # YOLO detection
        results = yolo.predict(source=frame, conf=0.3, device=device)
        dets = []
        for box in getattr(results[0], 'boxes', []):
            x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
            conf = float(box.conf[0].cpu())
            dets.append([x1, y1, x2, y2, conf])
        
        dets_array = np.array(dets) if dets else np.zeros((0, 5))
        tracks = tracker.update(dets_array)
        
        current = set()
        for track in tracks:
            x1, y1, x2, y2, tid = track
            tid = int(tid)
            current.add(tid)
            box = (int(x1), int(y1), int(x2), int(y2))
            if tid not in ostates:
                ostates[tid] = {'box': box, 'start': time.time()}
        
        with overlay_lock:
            active_tids.clear()
            active_tids.update(current)
        
        frame_q.task_done()
        memory_cleanup()

def cpu_ocr_worker():
    """OCR worker thread"""
    while True:
        item = ocr_q.get()
        if item is None:
            break
        
        tid, box, frame = item
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            ocr_q.task_done()
            continue
        
        # OCR processing (Tesseract or PaddleOCR)
        if OCR_BACKEND == 'tesseract':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray, lang=OCR_LANG, config=TESS_CONFIG).strip()
        else:
            results = paddle_ocr.ocr(roi)
            text = ' '.join([line[1][0] for line in results[0]]) if results else ""

        if text:
            # Translate the OCR result using Marian
            translated_text = translator.translate(text)
            with overlay_lock:
                overlay_states[tid] = {'box': box, 'text': translated_text, 'original': text}
        
        ocr_q.task_done()

def run_pipeline():
    """Main pipeline to process the video frames"""
    cap = setup_video_source()
    if cap is None:
        print("❌ Could not setup video source")
        return
    
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))
    
    # Start threads for detection and OCR
    detection_thread = threading.Thread(target=cpu_detection_thread, daemon=True)
    detection_thread.start()
    
    ocr_worker_thread = threading.Thread(target=cpu_ocr_worker, daemon=True)
    ocr_worker_thread.start()
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("📺 End of video reached")
            break
        
        # Add frame to the queue for detection
        if not frame_q.full():
            frame_q.put((idx, frame))
        
        # Handle overlays
        with overlay_lock:
            for tid, data in overlay_states.items():
                frame = draw_overlay(frame, data['text'], data['box'])
        
        writer.write(frame)
        
        idx += 1
    
    # Cleanup
    cap.release()
    writer.release()
    frame_q.put(None)
    ocr_q.put(None)
    detection_thread.join()
    ocr_worker_thread.join()

    print("🎬 Video processing completed!")

if __name__ == '__main__':
    run_pipeline()



