import os
import cv2
import time
import threading
import numpy as np
import pytesseract
import psutil
import GPUtil
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from sort import Sort

# -------- Configuration --------
video_path     = r"F:\FAU\Thesis\HDMIBabelfishV2\data\test_video\JapanRPG_TestSequence.mov"
model_path     = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv11_nano\runs\detect\train\weights\best.pt"
stability_time = 0.5  # seconds of box stability before OCR+translate

# OCR setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ocr_lang       = "jpn"
tess_config    = "--psm 6"

# Font helper
def get_font(size):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\msgothic.ttc", size)
    except IOError:
        return ImageFont.load_default()

# Translation pipeline
tokenizer         = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
if torch.cuda.is_available():
    translation_model = translation_model.to("cuda")
translator = pipeline(
    "translation",
    model=translation_model,
    tokenizer=tokenizer,
    src_lang="jpn_Jpan",
    tgt_lang="eng_Latn",
    max_length=200,
    device=0 if torch.cuda.is_available() else -1
)

# Load YOLO
print("Loading YOLO model...")
yolo = YOLO(model_path)
if torch.cuda.is_available():
    yolo.to("cuda")
    yolo.half()

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {video_path}")

# Tracker + state
tracker            = Sort()
track_state        = {}
translation_cache  = {}

# Metrics accumulators
sum_cpu            = sum_ram = sum_gpu_load = sum_gpu_mem = 0.0
samples            = 0

# FPS & timing
fps_count          = 0
fps                = 0.0
fps_timer          = time.time()
global_start       = time.time()
frame_idx          = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    orig = frame.copy()

    # 1) YOLO detection
    results = yolo.predict(
        source=frame,
        imgsz=320,
        conf=0.25,
        half=torch.cuda.is_available(),
        verbose=False
    )
    dets = []
    if results and results[0].boxes:
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 1.0
            dets.append([x1, y1, x2, y2, conf])

    # Ensure shape (N,5)
    dets = np.array(dets, dtype=float)
    dets = dets.reshape(-1,5) if dets.size else np.zeros((0,5), dtype=float)

    # 2) Tracking
    tracks = tracker.update(dets)

    # 3) Stability heuristic + background OCR/translate
    now = time.time()
    for trk in tracks:
        x1, y1, x2, y2, tid = map(int, trk)
        state = track_state.get(tid)
        if state:
            px1, py1, px2, py2 = state['box']
            if abs(px1-x1)<5 and abs(py1-y1)<5 and abs(px2-x2)<5 and abs(py2-y2)<5:
                if now - state['start_time'] >= stability_time and state['overlay'] is None:
                    def worker(track_id, box):
                        bx1, by1, bx2, by2 = box
                        roi = orig[by1:by2, bx1:bx2]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        txt = pytesseract.image_to_string(gray, lang=ocr_lang, config=tess_config).strip()
                        if not txt:
                            return
                        eng = translation_cache.get(txt) or translator(txt)[0]['translation_text']
                        translation_cache[txt] = eng
                        avg_color = tuple(map(int, cv2.mean(roi)[:3]))
                        overlay = orig.copy()
                        cv2.rectangle(overlay, (bx1,by1), (bx2,by2), avg_color, -1)
                        pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil)
                        font = get_font(20)
                        draw.text((bx1+5,by1+5), eng, font=font, fill=(255,255,255))
                        track_state[track_id]['overlay'] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    threading.Thread(target=worker, args=(tid, (x1,y1,x2,y2)), daemon=True).start()
                state['box'] = (x1, y1, x2, y2)
            else:
                track_state[tid] = {'box':(x1,y1,x2,y2), 'start_time':now, 'overlay':None}
        else:
            track_state[tid] = {'box':(x1,y1,x2,y2), 'start_time':now, 'overlay':None}

        ov = track_state[tid]['overlay']
        if ov is not None:
            frame = ov
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # 4) Resource & FPS measurement
    fps_count += 1
    if time.time() - fps_timer >= 1.0:
        fps = fps_count / (time.time() - fps_timer)
        fps_count = 0
        fps_timer = time.time()

    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = gpu.load * 100
        gpu_mem  = gpu.memoryUtil * 100
    else:
        gpu_load = gpu_mem = 0.0

    # accumulate
    sum_cpu       += cpu_percent
    sum_ram       += ram_percent
    sum_gpu_load  += gpu_load
    sum_gpu_mem   += gpu_mem
    samples       += 1

    # overlay metrics
    metrics = f"FPS:{fps:.1f} CPU:{cpu_percent:.0f}% RAM:{ram_percent:.0f}% GPU:{gpu_load:.0f}% MEM:{gpu_mem:.0f}%"
    cv2.putText(frame, metrics, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 5) Display only
    cv2.imshow("Real-Time Translation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()

# Summary
total_time   = time.time() - global_start
avg_fps      = frame_idx / total_time if total_time>0 else 0
avg_cpu      = sum_cpu / samples if samples>0 else 0
avg_ram      = sum_ram / samples if samples>0 else 0
avg_gpu_load = sum_gpu_load / samples if samples>0 else 0
avg_gpu_mem  = sum_gpu_mem / samples if samples>0 else 0

resource_dir  = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv11_nano\resource"
os.makedirs(resource_dir, exist_ok=True)
with open(os.path.join(resource_dir, "resource_calculation.txt"), "w") as f:
    f.write(f"Total frames processed: {frame_idx}\n")
    f.write(f"Total run time (s): {total_time:.2f}\n")
    f.write(f"Average FPS: {avg_fps:.2f}\n")
    f.write(f"Average CPU usage (%): {avg_cpu:.2f}\n")
    f.write(f"Average RAM usage (%): {avg_ram:.2f}\n")
    f.write(f"Average GPU load (%): {avg_gpu_load:.2f}\n")
    f.write(f"Average GPU memory (%): {avg_gpu_mem:.2f}\n")
