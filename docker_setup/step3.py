import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Add the path to the 'packages' directory to import sort module
sys.path.append('/workspace/packages')  # Ensure the correct path for sort module
from sort.sort import Sort  # Correctly import the Sort class from the sort module

# Load YOLO model (replace with your trained model)
MODEL_PATH = "/workspace/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

# Initialize SORT tracker
tracker = Sort()  # Correct instantiation of the SORT tracker

# Define video path (or use camera feed)
VIDEO_PATH = "/workspace/data/test_video/JapanRPG_TestSequence.mov"

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Could not open video")
    exit()

# Get video properties (frame width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Original FPS (likely 60)
print(f"Original FPS: {fps}")

# Set FPS to 15 for output video (even if input is 60 FPS)
target_fps = 15
frame_time = 1 / target_fps  # Time between frames for 15 FPS

# Set up VideoWriter to save the output video with tracking
output_video_path = "/workspace/docker_setup/yolo_output/tracked_output_15fps.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))

# Variable to control frame skipping
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("📺 End of video reached")
        break
    
    # Process every 4th frame to get 15 FPS from 60 FPS (skip the rest)
    if frame_counter % int(fps / target_fps) == 0:
        # Run YOLO object detection on the frame
        results = yolo_model(frame)  # Perform inference

        # Get the detected bounding boxes (xyxy), confidences, and class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_labels = results[0].boxes.cls.cpu().numpy()  # Class labels
        
        # Prepare bounding boxes for tracking (format: [x1, y1, x2, y2, confidence])
        detections = []
        for i, box in enumerate(boxes):
            if len(box) == 4:  # Ensure the box has the correct format
                x1, y1, x2, y2 = box
                confidence = confidences[i]
                detections.append([x1, y1, x2, y2, confidence])

        # Convert detections to a numpy array
        detections = np.array(detections)

        # Check if detections array is not empty
        if detections.shape[0] > 0:
            # Apply SORT tracker
            tracked_objects = tracker.update(detections)  # Track objects
        else:
            tracked_objects = []  # No detections, skip tracking

        # Draw bounding boxes and tracked IDs on the frame
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw tracking ID
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the frame with bounding boxes and IDs
        out.write(frame)
    
    # Display the frame with bounding boxes and tracking info
    cv2.imshow("Video with Tracking", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1  # Increment the frame counter

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Tracking completed. Video saved to {output_video_path}")

