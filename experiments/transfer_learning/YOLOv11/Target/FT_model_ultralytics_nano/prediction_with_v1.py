# Using Deepseek 

import os
import cv2
import time
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from multiprocessing import freeze_support

# -------- Configuration and Setup --------
video_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\test_video\JapanRPG_TestSequence.mov"
model_path = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv11\Target\FT_model_ultralytics_nano\runs\detect\train\weights\best.pt"
output_folder = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv11\Target\FT_model_ultralytics_nano\prediction_with_v1"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_video_path = os.path.join(output_folder, "output_video.avi")

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ocr_lang = "jpn"
tess_config = '--psm 6'

# Font file (using a Windows font that supports Japanese; adjust if necessary)
font_path = r"C:\Windows\Fonts\msgothic.ttc"


# Translation pipeline using the Facebook NLLB-200 distilled model
translator_model_id = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(translator_model_id)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_id)
if torch.cuda.is_available():
    translation_model = translation_model.to("cuda")
translator = pipeline("translation", 
                        model=translation_model, 
                        tokenizer=tokenizer,
                        src_lang="jpn_Jpan", 
                        tgt_lang="eng_Latn", 
                        max_length=200,
                        device=0 if torch.cuda.is_available() else -1)

# Load the fine-tuned YOLOv8 model
print("Loading fine-tuned YOLOv11 nano model from:", model_path)
yolo_model = YOLO(model_path)
if torch.cuda.is_available():
    yolo_model.to("cuda")

# -------- Video Capture and Writer Setup --------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file:", video_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Cache translations to speed up processing
translation_cache = {}

# -------- Processing Loop --------
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model.predict(source=frame, imgsz=416, conf=0.25, verbose=False)
    if results and results[0].boxes and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            box_coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box_coords)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Optional: draw the detection box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract ROI for OCR
            roi = frame[y1:y2, x1:x2]
            try:
                japanese_text = pytesseract.image_to_string(roi, lang=ocr_lang, config=tess_config).strip()
            except Exception as e:
                japanese_text = ""
            if not japanese_text:
                continue

            print(f"Frame {frame_count}: Detected Japanese Text: {japanese_text}")
            if japanese_text in translation_cache:
                english_text = translation_cache[japanese_text]
            else:
                try:
                    translation_result = translator(japanese_text)
                    english_text = translation_result[0]['translation_text'] if translation_result else ""
                except Exception as e:
                    english_text = ""
                translation_cache[japanese_text] = english_text
            if not english_text:
                continue
            print(f"Frame {frame_count}: Translated Text: {english_text}")

            # Fill the box area with average background color
            avg_color = cv2.mean(roi)[:3]
            avg_color = tuple(map(int, avg_color))
            frame[y1:y2, x1:x2] = avg_color

            # Overlay translated text using PIL for flexible text rendering
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            font_size = 20  # initial font size
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()
            # Get text bounding box
            bbox = draw.textbbox((0, 0), english_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Reduce font size until text fits within the box with some margin
            while (text_width > (x2 - x1) * 0.9 or text_height > (y2 - y1) * 0.9) and font_size > 5:
                font_size -= 1
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except IOError:
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), english_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            # Center the text in the box
            text_x = x1 + ((x2 - x1) - text_width) // 2
            text_y = y1 + ((y2 - y1) - text_height) // 2
            # Draw text shadow for readability
            draw.text((text_x + 1, text_y + 1), english_text, font=font, fill=(0, 0, 0))
            draw.text((text_x, text_y), english_text, font=font, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # End for each detected box

    cv2.imshow("Translated Video", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
elapsed = time.time() - start_time
print(f"Processing complete. Saved output video at: {output_video_path}")
print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count / elapsed:.2f} FPS).")
