import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# Add the path to the 'packages' directory to import sort module
sys.path.append('/workspace/packages')  # Ensure the correct path for sort module
from sort.sort import Sort  # Correctly import the Sort class from the sort module

# Load YOLO model (replace with your trained model)
MODEL_PATH = "/workspace/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

# Initialize SORT tracker
tracker = Sort()  # Correct instantiation of the SORT tracker

# Define video path (or use camera feed)
VIDEO_PATH = "/workspace/data/test_video/For_Presentation.mp4"

# Setup output directories
BASE_OUTPUT_DIR = "/workspace/docker_setup/yolo_output"
PRESENTATION_DIR = os.path.join(BASE_OUTPUT_DIR, "for_presentation")

# Create folder structure
folders = {
    'input_frames': os.path.join(PRESENTATION_DIR, "1_input_frames"),
    'detection_boxes': os.path.join(PRESENTATION_DIR, "2_detection_boxes"),
    'cropped_boxes': os.path.join(PRESENTATION_DIR, "3_cropped_boxes"),
    'output_video': os.path.join(PRESENTATION_DIR, "4_output_video")
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

print("\nPresentation folders created:")
for name, path in folders.items():
    print(f"  {name}: {path}")

# Image saving configuration
SAVE_IMAGES_PER_SECOND = 5  # Save 5-6 frames per second

def should_save_frame(frame_number, fps, save_per_second=SAVE_IMAGES_PER_SECOND):
    """Determine if frame should be saved based on interval"""
    save_every = max(1, int(fps / save_per_second))
    return frame_number % save_every == 0

def save_input_frame(frame, frame_number, folder):
    """Save input frame"""
    filename = f"input_frame_{frame_number:06d}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)

def save_detection_frame(frame, frame_number, folder):
    """Save frame with detection boxes"""
    filename = f"detection_frame_{frame_number:06d}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)

def save_cropped_boxes(frame, frame_number, tracked_objects, folder):
    """Save cropped regions of detected boxes"""
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            filename = f"cropped_frame_{frame_number:06d}_id_{int(track_id)}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, cropped)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Could not open video")
    exit()

# Get video properties (frame width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Original FPS (likely 60)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo Properties:")
print(f"  Original FPS: {fps}")
print(f"  Resolution: {frame_width}x{frame_height}")
print(f"  Total Frames: {total_frames}")

# Set FPS to 15 for output video (even if input is 60 FPS)
target_fps = 15
frame_time = 1 / target_fps  # Time between frames for 15 FPS

print(f"  Target FPS: {target_fps}")
print(f"  Saving {SAVE_IMAGES_PER_SECOND} images per second")

# Set up VideoWriter to save the output video with tracking
output_video_path = os.path.join(folders['output_video'], "tracked_output_15fps.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))

# Variable to control frame skipping
frame_counter = 0
processed_frame_counter = 0
saved_image_counter = 0

print("\nProcessing video... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n📺 End of video reached")
        break
    
    frame_counter += 1
    original_frame = frame.copy()
    
    # Determine if we should save this frame
    should_save = should_save_frame(frame_counter, fps, SAVE_IMAGES_PER_SECOND)
    
    # Save input frame (original, before processing)
    if should_save:
        save_input_frame(original_frame, frame_counter, folders['input_frames'])
        saved_image_counter += 1
    
    # Process every Nth frame to get target FPS from original FPS
    if frame_counter % int(fps / target_fps) == 0:
        processed_frame_counter += 1
        
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

        # Create a frame with detection boxes for visualization
        detection_frame = original_frame.copy()
        
        # Draw bounding boxes and tracked IDs on the frame
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            
            # Draw bounding box on both frames
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.rectangle(detection_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw tracking ID
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(detection_frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save detection frame with boxes
        if should_save and len(tracked_objects) > 0:
            save_detection_frame(detection_frame, frame_counter, folders['detection_boxes'])
            
            # Save cropped boxes
            save_cropped_boxes(original_frame, frame_counter, tracked_objects, folders['cropped_boxes'])
        
        # Save the processed frame to output video
        out.write(frame)
    
    # Display the frame with bounding boxes and tracking info
    cv2.imshow("Video with Tracking", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nUser quit")
        break

    # Progress update every 100 frames
    if frame_counter % 100 == 0:
        progress = (frame_counter / total_frames) * 100
        print(f"Progress: {progress:.1f}% - Frame {frame_counter}/{total_frames} - "
              f"Processed: {processed_frame_counter} - Saved: {saved_image_counter} images")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Final summary
print("\n" + "="*70)
print("PROCESSING COMPLETE!")
print("="*70)
print(f"\nStatistics:")
print(f"  Total frames processed: {frame_counter}")
print(f"  Frames written to video: {processed_frame_counter}")
print(f"  Images saved: {saved_image_counter}")

print(f"\nOutput locations:")
print(f"  Input frames: {folders['input_frames']} ({saved_image_counter} images)")
print(f"  Detection frames: {folders['detection_boxes']}")
print(f"  Cropped boxes: {folders['cropped_boxes']}")
print(f"  Output video: {output_video_path}")

print(f"\n✅ Tracking completed successfully!")
print("="*70)