#!/usr/bin/env python3
import sys
import os

# ─── Make sure 'models/' is on the import path ───────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_PATH     = "/home/babel-fish/Desktop/HDMIBabelfishV2/data/test_video/JapanRPG_TestSequence.mov"
MODEL_PATH     = "/home/babel-fish/Desktop/HDMIBabelfishV2/jetson/models/yoloV11_nano_scratch.pt"
WINDOW_NAME    = "RealTime Translate"
TARGET_FPS     = 25
STABILITY_MS   = 375  # milliseconds

BASE_DIR       = "/home/babel-fish/Desktop/HDMIBabelfishV2/jetson/output/video"
OUTPUT_VIDEO   = os.path.join(BASE_DIR, "output_with_fps.mp4")
FPS_GRAPH_PNG  = os.path.join(BASE_DIR, "fps_over_time.png")
FPS_GRAPH_PDF  = os.path.join(BASE_DIR, "fps_over_time.pdf")

# Ensure output directory exists
os.makedirs(BASE_DIR, exist_ok=True)

# ─── OCR SETUP ────────────────────────────────────────────────────────────────
if OCR_BACKEND == 'tesseract':
    OCR_LANG   = "jpn"
    TESS_CONFIG = "--psm 6"
else:
    paddle_ocr = PaddleOCR(lang='japanese', use_gpu=torch.cuda.is_available())

# ─── TRANSLATION SETUP ───────────────────────────────────────────────────────
tokenizer  = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model      = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
if torch.cuda.is_available():
    model = model.to("cuda")
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="jpn_Jpan",
    tgt_lang="eng_Latn",
    device=0 if torch.cuda.is_available() else -1
)
translation_cache = {}

# ─── YOLO + SORT SETUP ───────────────────────────────────────────────────────
yolo = YOLO(MODEL_PATH)
if torch.cuda.is_available():
    yolo.to("cuda")
    yolo.half()

tracker = Sort()

# ─── THREAD-SAFE QUEUES & STATE ──────────────────────────────────────────────
frame_q        = queue.Queue(maxsize=100)
ocr_q          = queue.Queue(maxsize=100)
ostates        = {}  # tid -> {'box','start','pending'}
overlay_states = {}  # tid -> {'box','text','avg'}
active_tids    = set()
overlay_lock   = threading.Lock()

# ─── DETECTION + STABILITY THREAD ────────────────────────────────────────────
def detect_thread():
    while True:
        idx, frame = frame_q.get()
        if frame is None:
            break

        h, w   = frame.shape[:2]
        small  = cv2.resize(frame, (320, 320))
        res    = yolo.predict(source=small, conf=0.3,
                              half=torch.cuda.is_available(),
                              verbose=False)
        dets   = []
        for box in getattr(res[0], 'boxes', []):
            x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
            # scale back
            x1 = int(x1 * w/320); x2 = int(x2 * w/320)
            y1 = int(y1 * h/320); y2 = int(y2 * h/320)
            dets.append([x1, y1, x2, y2, float(box.conf[0].cpu())])

        dets   = np.array(dets) if dets else np.zeros((0,5))
        tracks = tracker.update(dets)
        now    = time.time() * 1000
        current = set()

        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            current.add(tid)
            box = (x1, y1, x2, y2)

            st = ostates.get(tid)
            if st and np.linalg.norm(np.array(st['box'][:2]) - np.array(box[:2])) < 5:
                if now - st['start'] >= STABILITY_MS and not st['pending']:
                    ocr_q.put((tid, box, frame.copy()))
                    st['pending'] = True
            else:
                ostates[tid] = {'box': box, 'start': now, 'pending': False}

        # update active tids
        with overlay_lock:
            active_tids.clear()
            active_tids.update(current)

        # clean up ended tracks
        for tid in set(ostates) - current:
            ostates.pop(tid, None)

        frame_q.task_done()

# ─── OCR + TRANSLATION THREAD ─────────────────────────────────────────────────
def ocr_worker():
    while True:
        item = ocr_q.get()
        if item is None:
            break

        tid, box, frame = item
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]

        # OCR
        if OCR_BACKEND == 'tesseract':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            txt  = pytesseract.image_to_string(gray,
                                               lang=OCR_LANG,
                                               config=TESS_CONFIG).strip()
        else:
            res  = paddle_ocr.ocr(roi)
            txt  = ' '.join([line[1][0] for line in res])

        if txt:
            eng = translation_cache.get(txt)
            if not eng:
                eng = translator(txt)[0]['translation_text']
                translation_cache[txt] = eng

            avg = tuple(map(int, cv2.mean(roi)[:3]))
            with overlay_lock:
                overlay_states[tid] = {'box': box, 'text': eng, 'avg': avg}

        ocr_q.task_done()

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
def run():
    cap     = cv2.VideoCapture(VIDEO_PATH)
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip    = max(1, round(fps_in / TARGET_FPS))

    # start detection thread
    dt = threading.Thread(target=detect_thread, daemon=True)
    dt.start()

    fps_list, time_list = [], []
    start_time = time.time()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             TARGET_FPS,
                             (w, h))

    # start OCR threads
    workers = [threading.Thread(target=ocr_worker, daemon=True) for _ in range(2)]
    for w in workers: w.start()

    idx, last = 0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        if idx % skip == 0 and not frame_q.full():
            frame_q.put((idx, frame.copy()))

        # remove overlays for finished IDs
        with overlay_lock:
            for tid in list(overlay_states):
                if tid not in active_tids:
                    overlay_states.pop(tid, None)

        # draw overlays
        with overlay_lock:
            for data in overlay_states.values():
                x1, y1, x2, y2 = data['box']
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), data['avg'], -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(frame,
                            data['text'],
                            (x1+4, y2-4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255,255,255),
                            1,
                            cv2.LINE_AA)

        # compute & display FPS
        now      = time.time()
        fps_disp = 1.0 / (now - last) if now != last else TARGET_FPS
        last     = now
        fps_list.append(fps_disp)
        time_list.append(now - start_time)

        # write & show
        writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(max(0, (1.0 / TARGET_FPS) - (time.time() - now)))

    # ─── CLEANUP ──────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    frame_q.put((None, None))
    dt.join()
    for _ in workers:
        ocr_q.put(None)
    for w in workers:
        w.join()
    cv2.destroyAllWindows()

    # ─── SAVE FPS PLOT ────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(time_list, fps_list, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FPS_GRAPH_PNG)
    plt.savefig(FPS_GRAPH_PDF)

if __name__ == '__main__':
    run()
