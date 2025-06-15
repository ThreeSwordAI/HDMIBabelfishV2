#!/usr/bin/env python3
import cv2
import torch
from ultralytics import YOLO

# 1) Video and model paths
VIDEO_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/data/test_video/JapanRPG_TestSequence.mov"
MODEL_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/jetson/models/yoloV11_nano_scratch.pt"
#MODEL_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
# 2) Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# 3) Load YOLO model
model = YOLO(MODEL_PATH)
if torch.cuda.is_available():
    model.to("cuda")
    model.half()            # use FP16 for speed & memory

# 4) Run inference + display
cv2.namedWindow("YOLO Test", cv2.WINDOW_AUTOSIZE)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference (no_grad to reduce memory churn)
    with torch.no_grad():
        results = model.predict(frame,
                                conf=0.3,
                                half=torch.cuda.is_available(),
                                verbose=False)

    # Draw boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("YOLO Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
