import os
import time
import random
import xml.etree.ElementTree as ET
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from multiprocessing import freeze_support

# ------------------------------------------------------------------------------
# Part 1: Convert Pascal VOC (XML) Annotations to YOLO Format
# ------------------------------------------------------------------------------
def convert_voc_to_yolo(ann_dir, img_dir):
    for ann in os.listdir(ann_dir):
        if not ann.endswith('.xml'):
            continue
        image_file = ann.replace('.xml', '.jpg')
        image_path = os.path.join(img_dir, image_file)
        if not os.path.exists(image_path):
            # Skip if the image file is missing.
            continue
        xml_path = os.path.join(ann_dir, ann)
        tree = ET.parse(xml_path)
        root = tree.getroot()


        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)


        txt_file = os.path.join(ann_dir, ann.replace('.xml', '.txt'))
        lines = []
        for obj in root.findall('object'):
            cls = 0
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            # Convert to YOLO normalized coordinates.
            x_center = ((xmin + xmax) / 2.0) / img_width
            y_center = ((ymin + ymax) / 2.0) / img_height
            box_width = (xmax - xmin) / img_width
            box_height = (ymax - ymin) / img_height
            line = f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            lines.append(line)
        with open(txt_file, 'w') as f:
            f.write("\n".join(lines))


# ------------------------------------------------------------------------------
# Part 2: Compute Intersection over Union (IoU)
# ------------------------------------------------------------------------------
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


# ------------------------------------------------------------------------------
# Part 3: Evaluate the YOLOv8 Model on Test Data
# ------------------------------------------------------------------------------
def evaluate_model(model, test_dir):
    results = []
    for file in os.listdir(test_dir):
        if file.endswith('.jpg'):
            image_path = os.path.join(test_dir, file)
            xml_path = os.path.join(test_dir, file.replace('.jpg', '.xml'))
            if not os.path.exists(xml_path):
                continue  # Skip if XML is missing.
            image = cv2.imread(image_path)
            if image is None:
                continue


            tree = ET.parse(xml_path)
            root = tree.getroot()
            gt_box = None
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                gt_box = [xmin, ymin, xmax, ymax]
                break
            if gt_box is None:
                continue

            start_time = time.time()
            pred_results = model.predict(source=image, imgsz=640, conf=0.25)
            inference_time = time.time() - start_time


            pred_box = [0, 0, 0, 0]
            if len(pred_results) > 0 and len(pred_results[0].boxes) > 0:
                box_tensor = pred_results[0].boxes.xyxy[0].cpu().numpy()
                pred_box = [int(x) for x in box_tensor]
            iou = compute_iou(pred_box, gt_box)
            results.append({'image': file, 'avg_iou': iou, 'inference_time': inference_time})
    return results


# ------------------------------------------------------------------------------
# Part 4: Save Prediction Images for a Random Sample of Test Data
# ------------------------------------------------------------------------------
def save_prediction_images(model, test_dir, prediction_dir, count=15):
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    if len(image_files) < count:
        count = len(image_files)
    sampled_images = random.sample(image_files, count)
    
    print(f"Running predictions on {count} random images and saving to: {prediction_dir}")
    model.predict(source=sampled_images, imgsz=640, conf=0.25, save=True, project=prediction_dir, name="predictions", exist_ok=True)


# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main():
    train_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\Bangladeshi_License_Plate\train"
    test_dir  = r"F:\FAU\Thesis\HDMIBabelfishV2\data\Bangladeshi_License_Plate\test"
    prediction_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Source\Scratch_model\prediction"
    
    print("Converting VOC annotations to YOLO format for training data...")
    convert_voc_to_yolo(train_dir, train_dir)
    print("Converting VOC annotations to YOLO format for test data...")
    convert_voc_to_yolo(test_dir, test_dir)
    
    dataset_yaml_content = f"""
                            train: {train_dir}
                            val: {test_dir}
                            nc: 1
                            names: ['license-plate']
                            """
    dataset_yaml_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\Bangladeshi_License_Plate\dataset.yaml"
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    print(f"Dataset YAML file created at: {dataset_yaml_path}")
    
    # Load a pretrained YOLOv8 model.
    # model = YOLO('yolov8s.pt')
    # Uncomment the following line to train from scratch:
    model = YOLO('yolov8s.yaml')

    print("Starting training...")
    model.train(data=dataset_yaml_path, epochs=30, imgsz=640, batch=16)
    
    export_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Source\Scratch_model"
    export_result = model.export(format="torchscript", save_dir=export_dir)
    print(f"Training complete. Exported model details:\n{export_result}")
    
    print("Starting evaluation on test dataset...")
    eval_results = evaluate_model(model, test_dir)
    
    results_csv = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Source\Scratch_model\evaluation_results.csv"
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['image', 'avg_iou', 'inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in eval_results:
            writer.writerow(res)
    print(f"Evaluation complete. Results saved in: {results_csv}")
    
    save_prediction_images(model, test_dir, prediction_dir, count=15)
    print("Prediction images saved.")


if __name__ == '__main__':
    freeze_support()
    main()
