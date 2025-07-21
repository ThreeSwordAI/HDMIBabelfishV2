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

# Try different import paths for sort
try:
    from sort.sort import Sort
except ImportError:
    try:
        from models.sort import Sort
    except ImportError:
        sys.path.append('../packages')
        from sort.sort import Sort

from ultralytics import YOLO
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
VIDEO_PATH = "../data/test_video/JapanRPG_TestSequence.mov"
USE_CAPTURE_DEVICE = True
CAPTURE_DEVICE_ID = 0
CAPTURE_RESOLUTION = (1280, 720)
CAPTURE_FPS = 15

MODEL_PATH = "../experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
WINDOW_NAME = "PS3 Real-Time Translation - Jetson"
TARGET_FPS = 15
STABILITY_MS = 375  # Same as your working code

BASE_DIR = "output/video"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_jetson.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_over_time_jetson.png")
FPS_GRAPH_PDF = os.path.join(BASE_DIR, "fps_over_time_jetson.pdf")

# Create output directory if it doesn't exist
os.makedirs(BASE_DIR, exist_ok=True)

# CPU Configuration for Jetson
device = torch.device("cpu")
print(f"🚀 Using device: {device}")
print(f"Platform: {platform.machine()}")

# OCR setup
if OCR_BACKEND == 'tesseract':
    OCR_LANG = "jpn"
    TESS_CONFIG = "--psm 6"
else:
    paddle_ocr = PaddleOCR(lang='japanese', use_gpu=False)

# Global translator - will be pre-loaded
translator = None
translation_cache = {}

def initialize_translation():
    """Initialize translation with better error handling - same pattern as your working code"""
    global translator
    
    print("="*50)
    print("INITIALIZING TRANSLATION MODEL")
    print("="*50)
    
    try:
        # Use same approach as your working on_cpu.py
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Disable distributed features that are causing issues
        import torch.distributed
        if hasattr(torch.distributed, 'is_available') and torch.distributed.is_available():
            print("Disabling distributed training features...")
            torch.distributed.destroy_process_group = lambda: None
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            torch_dtype=torch.float32,  # Use float32 instead of auto
            low_cpu_mem_usage=True
        )
        
        # Force CPU usage
        model = model.to("cpu")
        model.eval()  # Set to evaluation mode
        
        print("Creating translation pipeline...")
        translator = pipeline(
            "translation", 
            model=model, 
            tokenizer=tokenizer,
            src_lang="jpn_Jpan", 
            tgt_lang="eng_Latn",
            device=-1,  # Force CPU
            batch_size=1,
            max_length=512
        )
        
        # Test translation
        test_result = translator("テスト")[0]['translation_text']
        print(f"Transformers test: 'テスト' -> '{test_result}'")
        print("✓ Transformers initialized successfully!")
        return "transformers"
        
    except Exception as e:
        print(f"✗ All translation methods failed: {e}")
        print("Using mock translator...")
        translator = "mock"
        return "mock"

def get_translation(text):
    """Get translation using the initialized translator - same pattern as your working code"""
    global translator, translation_cache
    
    if not text.strip():
        return ""
        
    if text in translation_cache:
        return translation_cache[text]
    
    try:
        if translator == "mock":
            result = f"[MOCK TRANSLATION: {text}]"
        else:  # Transformers pipeline
            result = translator(text)[0]['translation_text']
            
        translation_cache[text] = result
        return result
        
    except Exception as e:
        print(f"Translation error for '{text}': {e}")
        error_result = f"[Error: {text}]"
        translation_cache[text] = error_result
        return error_result

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
    print("✓ YOLO model loaded!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    print("Please ensure you have a trained YOLO model or internet connection for auto-download")
    exit(1)

print("\nSTEP 2: Initializing translation...")
translation_method = initialize_translation()
print(f"✓ Translation method: {translation_method}")

print("\nSTEP 3: Setting up tracking...")
tracker = Sort()
print("✓ Tracking initialized!")

# Thread-safe pipeline queues and state
frame_q = queue.Queue(maxsize=100)
ocr_q = queue.Queue(maxsize=100)
ostates = {}            # tid -> {'box','start','pending'}
overlay_states = {}     # tid -> {'box','text','avg'}
active_tids = set()
overlay_lock = threading.Lock()

# Detection & stability thread - simplified like your working code
def detect_thread():
    print("Detection thread started")
    while True:
        item = frame_q.get()
        if item is None: 
            print("Detection thread stopping")
            break
        idx, frame = item
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 320))  # Same as your working code
        
        try:
            res = yolo.predict(source=small, conf=0.3, verbose=False)
            dets = []
            for box in getattr(res[0], 'boxes', []):
                x1,y1,x2,y2 = box.xyxy[0].cpu().int().tolist()
                # scale back
                x1 = int(x1 * w/320); x2 = int(x2 * w/320)
                y1 = int(y1 * h/320); y2 = int(y2 * h/320)
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
                if st and np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2])) < 5:
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

# OCR & translation worker - simplified like your working code
def ocr_worker(worker_id):
    print(f"OCR worker {worker_id} started")
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
            # OCR - same as your working code
            if OCR_BACKEND == 'tesseract':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
                eng = get_translation(txt)
                print(f"🌐 Translation box {tid}: '{txt}' -> '{eng}'")
                avg = tuple(map(int, cv2.mean(roi)[:3]))
                with overlay_lock:
                    overlay_states[tid] = {'box': box, 'text': eng, 'avg': avg}
                    
        except Exception as e:
            print(f"❌ OCR/Translation error for box {tid}: {e}")
            
        ocr_q.task_done()

def setup_capture_device():
    """Setup USB capture device for Jetson"""
    print("🎮 Setting up USB capture device on Jetson...")
    print(f"   Device ID: {CAPTURE_DEVICE_ID}")
    print(f"   Target Resolution: {CAPTURE_RESOLUTION}")
    print(f"   Target FPS: {CAPTURE_FPS}")
    
    # Try V4L2 backend first (common on Linux/Jetson)
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        try:
            print(f"🔧 Trying backend: {backend}")
            cap = cv2.VideoCapture(CAPTURE_DEVICE_ID, backend)
            
            if cap.isOpened():
                print(f"✅ Opened capture device with backend {backend}")
                break
            else:
                cap.release()
                cap = None
        except Exception as e:
            print(f"⚠ Backend {backend} failed: {e}")
            continue
    
    if cap is None or not cap.isOpened():
        print(f"❌ Could not open capture device {CAPTURE_DEVICE_ID}")
        print("💡 Solutions:")
        print("   1. Check if capture device is connected via USB")
        print("   2. Try different device IDs (0, 1, 2, 3)")
        print("   3. Check USB permissions: sudo chmod 666 /dev/video*")
        print("   4. List available devices: v4l2-ctl --list-devices")
        return None
    
    # Configure capture settings for Jetson
    print("🔧 Configuring capture device settings...")
    
    # Set resolution (lower for Jetson performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    
    # Optimize for Jetson
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    # Verify actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Capture device configured:")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    
    # Test frame capture
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ Could not capture test frame from device")
        cap.release()
        return None
    
    print(f"✅ Test frame captured: {test_frame.shape}")
    print("✅ Capture device ready on Jetson!")
    
    return cap

def setup_video_source():
    """Setup video source with USB capture device priority"""
    
    if USE_CAPTURE_DEVICE:
        print("🎮 Using USB capture device")
        cap = setup_capture_device()
        if cap is not None:
            return cap
        else:
            print("⚠ USB capture setup failed, falling back to file input")
    
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

# Main display loop - simplified like your working code
def run():
    print("\n" + "="*50)
    print("🚀 STARTING PS3 REAL-TIME TRANSLATION ON JETSON")
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
    
    # Start OCR workers - fewer workers like your working code
    workers = []
    for i in range(2):
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
                    
        # draw overlays - simplified like your working code
        overlay_count = 0
        with overlay_lock:
            for tid, data in overlay_states.items():
                x1,y1,x2,y2 = data['box']
                avg = data['avg']
                # translucent box - same as your working code
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), avg, -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                # text - simple like your working code
                text = data['text']
                font_scale = 0.6
                thickness = 1
                cv2.putText(frame, text, (x1+4, y2-4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255,255,255), thickness, cv2.LINE_AA)
                overlay_count += 1
                            
        # compute & display FPS
        now = time.time()
        fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
        
        fps_list.append(fps_disp)
        time_list.append(now-start_time)
        
        # HUD - same as your working code
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Translations: {overlay_count}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Method: {translation_method}", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        last = now
        # show
        writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            print("👤 User quit")
            break
        
        # Timing - same as your working code
        time.sleep(max(0, (1.0/TARGET_FPS) - (time.time() - now)))

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
    
    print("📊 Generating FPS graph...")
    plt.figure()
    plt.plot(time_list, fps_list, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('Jetson CPU Pipeline - FPS over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FPS_GRAPH_PNG)
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