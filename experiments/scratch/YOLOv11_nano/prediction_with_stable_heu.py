import os
import cv2
import time
import threading
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from sort import Sort

# -------- Configuration --------
video_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\test_video\JapanRPG_TestSequence.mov"
model_path = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv11_nano\runs\detect\train\weights\best.pt"
output_folder = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv11_nano\prediction_with_stable_heu"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "output_video_heu.avi")

# OCR setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ocr_lang = "jpn"
tess_config = "--psm 6"

# Font helper
def get_font(size):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\msgothic.ttc", size)
    except IOError:
        return ImageFont.load_default()

# Translation pipeline
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
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

# Video I/O
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize SORT tracker
tracker = Sort()
# track_id -> {'box':(x1,y1,x2,y2), 'start_time':float, 'overlay':ndarray or None}
track_state = {}
translation_cache = {}
stability_time = 0.5  # seconds

frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    orig = frame.copy()

    # YOLO detection on every frame
    results = yolo.predict(source=frame, imgsz=320, conf=0.25,
                           half=torch.cuda.is_available(), verbose=False)
    dets = []
    if results and results[0].boxes:
        for b in results[0].boxes:
            coords = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 1.0
            dets.append([x1, y1, x2, y2, conf])

    # Ensure dets is always (N,5)
    dets = np.array(dets, dtype=float)
    if dets.size == 0:
        dets = np.zeros((0, 5), dtype=float)
    else:
        dets = dets.reshape(-1, 5)

    # Update tracker
    tracks = tracker.update(dets)

    # Process each track
    for trk in tracks:
        x1, y1, x2, y2, tid = map(int, trk)
        now = time.time()
        state = track_state.get(tid)

        if state:
            px1, py1, px2, py2 = state['box']
            # Check if box is roughly the same
            if (abs(px1 - x1) < 5 and abs(py1 - y1) < 5 and
                abs(px2 - x2) < 5 and abs(py2 - y2) < 5):
                # Box stable?
                if now - state['start_time'] >= stability_time and state['overlay'] is None:
                    # Launch background OCR+translate
                    def worker(track_id, box):
                        bx1, by1, bx2, by2 = box
                        roi = orig[by1:by2, bx1:bx2]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        txt = pytesseract.image_to_string(gray, lang=ocr_lang, config=tess_config).strip()
                        if not txt:
                            return
                        if txt in translation_cache:
                            eng = translation_cache[txt]
                        else:
                            eng = translator(txt)[0]['translation_text']
                            translation_cache[txt] = eng
                        avg_color = tuple(map(int, cv2.mean(roi)[:3]))
                        overlay = orig.copy()
                        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), avg_color, -1)
                        pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil)
                        font = get_font(20)
                        draw.text((bx1+5, by1+5), eng, font=font, fill=(255,255,255))
                        track_state[track_id]['overlay'] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                    threading.Thread(target=worker, args=(tid, (x1, y1, x2, y2)), daemon=True).start()
                # Update stored box
                state['box'] = (x1, y1, x2, y2)
            else:
                # Box moved: reset
                track_state[tid] = {'box': (x1, y1, x2, y2),
                                    'start_time': now,
                                    'overlay': None}
        else:
            # New track
            track_state[tid] = {'box': (x1, y1, x2, y2),
                                'start_time': now,
                                'overlay': None}

        # Draw overlay if ready, else draw bounding box
        ov = track_state[tid]['overlay']
        if ov is not None:
            frame = ov
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Always show & save
    out.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed {frame_idx} frames in {time.time() - start_time:.2f}s")
