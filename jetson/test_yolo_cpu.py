#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

# Paths
VIDEO_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/data/test_video/JapanRPG_TestSequence.mov"
MODEL_PATH = "/home/babel-fish/Desktop/HDMIBabelfishV2/jetson/models/yoloV11_nano_scratch.pt"

# 1) Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# 2) Load model on CPU
model = YOLO(MODEL_PATH)
model.cpu()  # ensure the model is on CPU

# 3) Run inference + display
cv2.namedWindow("YOLO CPU Test", cv2.WINDOW_AUTOSIZE)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inference on CPU
    results = model.predict(frame,
                            conf=0.3,
                            device='cpu',  # override to CPU
                            verbose=False)

    # draw boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("YOLO CPU Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

