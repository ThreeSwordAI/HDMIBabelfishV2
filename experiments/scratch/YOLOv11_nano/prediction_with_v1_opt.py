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
output_video_path = os.path.join(output_folder, "output_video_opt.avi")

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ocr_lang = "jpn"
tess_config = '--psm 6'

# Font settings (preload fonts)
font_path = r"C:\Windows\Fonts\msgothic.ttc"
font_cache = {}
def get_font(size):
    if size not in font_cache:
        try:
            font_cache[size] = ImageFont.truetype(font_path, size)
        except IOError:
            font_cache[size] = ImageFont.load_default()
    return font_cache[size]

# Translation pipeline
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

# Load YOLO model with optimizations
print("Loading YOLO model...")
yolo_model = YOLO(model_path)
if torch.cuda.is_available():
    yolo_model.to("cuda")
    yolo_model.half()  # Enable FP16

# Video setup
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file:", video_path)
    exit()
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Performance settings
translation_cache = {}
process_every_n_frame = 2  # Process every 2nd frame (1=no skip)
show_live_preview = False  # Disable to save time

# -------- Processing Loop --------
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Skip frames to reduce workload
    if frame_count % process_every_n_frame != 0:
        frame_count += 1
        continue
    
    # Resize frame for faster processing (optional)
    # processed_frame = cv2.resize(frame, (width//2, height//2))
    # results = yolo_model.predict(processed_frame, imgsz=320, ...)
    
    # YOLO detection with FP16 and smaller imgsz
    results = yolo_model.predict(
        source=frame,
        imgsz=320,
        conf=0.25,
        verbose=False,
        half=True if torch.cuda.is_available() else False
    )
    
    if results and results[0].boxes:
        # Batch process all texts in the frame
        text_batch = []
        box_data = []
        
        for box in results[0].boxes:
            box_coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box_coords)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract ROI and preprocess for OCR
            roi = frame[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            try:
                text = pytesseract.image_to_string(roi_gray, lang=ocr_lang, config=tess_config).strip()
            except:
                text = ""
            if text:
                text_batch.append(text)
                box_data.append((x1, y1, x2, y2, roi))
        
        # Batch translation
        if text_batch:
            translations = {}
            for text in text_batch:
                if text in translation_cache:
                    translations[text] = translation_cache[text]
                else:
                    try:
                        translated = translator(text)[0]['translation_text']
                        translations[text] = translated
                        translation_cache[text] = translated
                    except:
                        translations[text] = ""
            
            # Apply translations to boxes
            for (x1, y1, x2, y2, roi), text in zip(box_data, text_batch):
                english_text = translations[text]
                if not english_text:
                    continue
                
                # Fill ROI with average color
                avg_color = tuple(map(int, cv2.mean(roi)[:3]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), avg_color, -1)
                
                # Render text
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)
                font_size = 20
                while font_size > 5:
                    font = get_font(font_size)
                    bbox = draw.textbbox((0, 0), english_text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    if text_w < (x2 - x1)*0.9 and text_h < (y2 - y1)*0.9:
                        break
                    font_size -= 1
                
                text_x = x1 + (x2 - x1 - text_w) // 2
                text_y = y1 + (y2 - y1 - text_h) // 2
                draw.text((text_x+1, text_y+1), english_text, font=font, fill=(0,0,0))
                draw.text((text_x, text_y), english_text, font=font, fill=(255,255,255))
                frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Write frame
    out.write(frame)
    if show_live_preview:
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
    
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames in {time.time()-start_time:.2f}s")