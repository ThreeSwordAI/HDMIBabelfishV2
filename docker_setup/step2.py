import os
import random
import cv2
from ultralytics import YOLO
import shutil

# Define directories and file paths
IMAGE_DIR = "/workspace/data/game_review/big_dataset/test"
OUTPUT_DIR = "/workspace/docker_setup/yolo_output"
MODEL_PATH = "/workspace/experiments/scratch/YOLOv11_nano/runs/detect/train/weights/best.pt"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize YOLO model
yolo_model = YOLO(MODEL_PATH)

# Get list of all image files in the test directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]

# Select 10 random images
random_images = random.sample(image_files, 10)

# Iterate over the selected random images
for img_name in random_images:
    img_path = os.path.join(IMAGE_DIR, img_name)

    # Read the image
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"❌ Failed to load image: {img_name}")
        continue
    
    # Run YOLO model inference
    results = yolo_model(img_path)
    
    # Get the detection results (boxes, confidences, class labels)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2]
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    
    # Draw the boxes and labels on the image
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        confidence = confidences[i]
        label = int(class_labels[i])
        
        # Draw rectangle around detected object
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label and confidence text
        cv2.putText(image, f"{label} {confidence:.2f}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the annotated image to output directory
    output_img_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(output_img_path, image)
    
    # Save the corresponding annotations as a .txt file
    annotation_file = os.path.join(OUTPUT_DIR, img_name.replace('.jpg', '.txt'))
    with open(annotation_file, 'w') as f:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            label = int(class_labels[i])
            
            # Write annotation in YOLO format: <class_id> <x_center> <y_center> <width> <height>
            # Normalize coordinates relative to the image size
            img_height, img_width = image.shape[:2]
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            f.write(f"{label} {x_center} {y_center} {width} {height} {confidence:.2f}\n")
    
    print(f"✅ Processed {img_name} and saved the results.")

print(f"✔️ All 10 random images processed and saved in {OUTPUT_DIR}")
