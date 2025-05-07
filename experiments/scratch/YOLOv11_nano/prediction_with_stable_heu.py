import cv2
import time
import threading
import queue
import numpy as np
import platform
from ultralytics import YOLO
from sort import Sort
# OCR backend selection
try:
    import pytesseract
    OCR_BACKEND = 'tesseract'
except ImportError:
    from paddleocr import PaddleOCR
    OCR_BACKEND = 'paddle'
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# ---------- CONFIGURATION ----------
VIDEO_PATH = r"F:\FAU\Thesis\HDMIBabelfishV2\data\test_video\JapanRPG_TestSequence.mov"
MODEL_PATH = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv11_nano\runs\detect\train\weights\best.pt"
OUTPUT_WINDOW = "RealTime Translate"
TARGET_FPS = 25
STABILITY_MS = 375    # reduced from 500ms

# OCR & translation setup
if OCR_BACKEND == 'tesseract':
    if platform.system() == 'Windows':
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_LANG = "jpn"
    TESS_CONFIG = "--psm 6"
else:
    paddle_ocr = PaddleOCR(lang='japanese', use_gpu=torch.cuda.is_available())

# Translation model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
if torch.cuda.is_available():
    model = model.to("cuda")
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="jpn_Jpan",
    tgt_lang="eng_Latn",
    device=0 if torch.cuda.is_available() else -1,
)
translation_cache = {}

# Load YOLO model
yolo = YOLO(MODEL_PATH)
if torch.cuda.is_available():
    yolo.to("cuda")
    yolo.half()
# SORT tracker
tracker = Sort()

# Thread-safe shared state
frame_q = queue.Queue(maxsize=100)
ostates = {}        # tid -> {'box', 'start', 'pending'}
overlay_states = {} # tid -> {'box', 'text', 'avg'}
overlay_lock = threading.Lock()
ocrc_queue = queue.Queue(maxsize=100)

# Font loader
def get_font(sz=18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except:
        return ImageFont.load_default()

# Detection & stability check thread
def detect_thread():
    while True:
        idx, frame = frame_q.get()
        if frame is None:
            break
        # downsample for speed
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 320))
        res = yolo.predict(source=small, conf=0.3, half=torch.cuda.is_available(), verbose=False)
        dets = []
        for box in getattr(res[0], 'boxes', []):
            x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
            # scale back
            x1 = int(x1 * w/320); x2 = int(x2 * w/320)
            y1 = int(y1 * h/320); y2 = int(y2 * h/320)
            conf = float(box.conf[0].cpu())
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets) if dets else np.zeros((0,5))
        tracks = tracker.update(dets)
        now = time.time() * 1000
        current = set()
        for x1,y1,x2,y2,tid in tracks:
            tid = int(tid)
            current.add(tid)
            box = (int(x1),int(y1),int(x2),int(y2))
            st = ostates.get(tid)
            if st and np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2])) < 5:
                if now - st['start'] >= STABILITY_MS and not st['pending']:
                    ocrc_queue.put((tid, box, frame.copy()))
                    st['pending'] = True
            else:
                ostates[tid] = {'box': box, 'start': now, 'pending': False}
        # remove ended tracks
        gone = set(ostates.keys()) - current
        for tid in gone:
            ostates.pop(tid, None)
            with overlay_lock:
                overlay_states.pop(tid, None)
        frame_q.task_done()

# OCR & translation worker
def ocr_worker():
    while True:
        item = ocrc_queue.get()
        if item is None:
            break
        tid, box, frame = item
        x1,y1,x2,y2 = box
        roi = frame[y1:y2, x1:x2]
        # OCR
        if OCR_BACKEND == 'tesseract':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            txt = pytesseract.image_to_string(gray, lang=OCR_LANG, config=TESS_CONFIG).strip()
        else:
            res = paddle_ocr.ocr(roi)
            txt = ' '.join([line[1][0] for line in res])
        if txt:
            eng = translation_cache.get(txt)
            if not eng:
                eng = translator(txt)[0]['translation_text']
                translation_cache[txt] = eng
            avg = tuple(map(int, cv2.mean(roi)[:3]))
            with overlay_lock:
                overlay_states[tid] = {'box': box, 'text': eng, 'avg': avg}
        ocrc_queue.task_done()

# Main display loop
def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, round(fps_in / TARGET_FPS))
    # start threads
    dt = threading.Thread(target=detect_thread, daemon=True)
    dt.start()
    workers = [threading.Thread(target=ocr_worker, daemon=True) for _ in range(2)]
    for w in workers: w.start()

    idx = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        # queue frame for detection
        if idx % skip == 0 and not frame_q.full():
            frame_q.put((idx, frame.copy()))
        # draw persistent overlays
        with overlay_lock:
            for data in overlay_states.values():
                x1,y1,x2,y2 = data['box']
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), data['avg'], -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                font = get_font(18)
                draw.text((x1+4, y1+4), data['text'], font=font, fill=(255,255,255))
                frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        # compute & display FPS
        now = time.time()
        fps_disp = 1.0 / (now - last_time) if now != last_time else TARGET_FPS
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        last_time = now
        # show frame
        cv2.imshow(OUTPUT_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # throttle to target
        time.sleep(max(0, (1.0/TARGET_FPS) - (time.time() - now)))

    # cleanup
    cap.release()
    frame_q.put((None, None))
    dt.join()
    for _ in workers:
        ocrc_queue.put(None)
    for w in workers:
        w.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
