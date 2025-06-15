import cv2
import time
import threading
import queue
import numpy as np
import platform
from ultralytics import YOLO
from models.sort import Sort
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
VIDEO_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/data/test_video/JapanRPG_TestSequence.mov"
MODEL_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
WINDOW_NAME = "RealTime Translate"
TARGET_FPS = 25
STABILITY_MS = 375  # 0.375s for stability

BASE_DIR = "/home/babel-fish/Desktop/HDMIBabelfishV2/jetson/output/video"
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_with_fps.mp4")
FPS_GRAPH_PNG = os.path.join(BASE_DIR, "fps_over_time.png")
FPS_GRAPH_PDF = os.path.join(BASE_DIR, "fps_over_time.pdf")

# Create output directory if it doesn't exist
os.makedirs(BASE_DIR, exist_ok=True)

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
    """Initialize translation with better error handling"""
    global translator
    
    print("="*50)
    print("INITIALIZING TRANSLATION MODEL")
    print("="*50)
    
    try:
        # Try Google Translate first (much more reliable)
        try:
            from googletrans import Translator as GoogleTranslator
            translator = GoogleTranslator()
            
            # Test translation
            test_result = translator.translate("テスト", src='ja', dest='en')
            print(f"Google Translate test: 'テスト' -> '{test_result.text}'")
            print("✓ Google Translate initialized successfully!")
            return "google"
            
        except ImportError:
            print("Google Translate not available, trying Transformers...")
            
        except Exception as e:
            print(f"Google Translate failed: {e}")
            print("Falling back to Transformers...")
        
        # Fallback to transformers with fixes
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
    """Get translation using the initialized translator"""
    global translator, translation_cache
    
    if not text.strip():
        return ""
        
    if text in translation_cache:
        return translation_cache[text]
    
    try:
        if translator == "mock":
            result = f"[MOCK TRANSLATION: {text}]"
        elif hasattr(translator, 'translate'):  # Google Translate
            result = translator.translate(text, src='ja', dest='en').text
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
yolo = YOLO(MODEL_PATH)
print("✓ YOLO model loaded!")

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

# Detection & stability thread
def detect_thread():
    print("Detection thread started")
    while True:
        item = frame_q.get()
        if item is None: 
            print("Detection thread stopping")
            break
        idx, frame = item
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 320))
        
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

# OCR & translation worker
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
            # OCR
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

# Main display loop
def run():
    print("\n" + "="*50)
    print("STARTING VIDEO PROCESSING")
    print("="*50)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_PATH}")
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
    
    # Start OCR workers
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
                    
        # draw overlays
        overlay_count = 0
        with overlay_lock:
            for tid, data in overlay_states.items():
                x1,y1,x2,y2 = data['box']
                avg = data['avg']
                # translucent box
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), avg, -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                # text
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
    plt.title('FPS over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FPS_GRAPH_PNG)
    plt.savefig(FPS_GRAPH_PDF)
    print(f"✓ FPS graph saved to {FPS_GRAPH_PNG}")


if __name__ == '__main__':
    run()