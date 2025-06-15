import os
import time
import random
import cv2
import csv
import shutil
import numpy as np
from ultralytics import YOLO
from multiprocessing import freeze_support


def remap_labels(label_dir):
    for fname in os.listdir(label_dir):
        if fname.lower().endswith('.txt') and fname != "classes.txt":
            file_path = os.path.join(label_dir, fname)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                parts[0] = "0"
                new_lines.append(" ".join(parts))
            with open(file_path, 'w') as f:
                f.write("\n".join(new_lines))



def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate_model(model, test_dir):
    results = []
    for file in os.listdir(test_dir):
        if file.endswith('.jpg'):
            image_path = os.path.join(test_dir, file)
            label_path = os.path.join(test_dir, file.replace('.jpg', '.txt'))
            if not os.path.exists(label_path):
                continue
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w, _ = image.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue
            parts = lines[0].strip().split()
            if len(parts) < 5:
                continue
            
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h
            gt_box = [int(x_center - box_w / 2),
                      int(y_center - box_h / 2),
                      int(x_center + box_w / 2),
                      int(y_center + box_h / 2)]
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



def save_prediction_images(model, test_dir, prediction_dir, count=15):
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    if len(image_files) < count:
        count = len(image_files)
    sampled_images = random.sample(image_files, count)
    print(f"Running predictions on {count} random images and saving to: {prediction_dir}")
    model.predict(source=sampled_images, imgsz=640, conf=0.25, save=True, project=prediction_dir, name="predictions", exist_ok=True)

def create_val_split(train_dir, val_dir, split_ratio=0.2):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    image_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.jpg')]
    num_val = int(len(image_files) * split_ratio)
    val_images = random.sample(image_files, num_val)
    for img in val_images:
        src_img = os.path.join(train_dir, img)
        dst_img = os.path.join(val_dir, img)
        shutil.copy(src_img, dst_img)

        label_file = img.replace('.jpg', '.txt')
        src_label = os.path.join(train_dir, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(val_dir, label_file)
            shutil.copy(src_label, dst_label)
    print(f"Created validation split in: {val_dir} ({num_val} images)")


# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main():
    base_dataset = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\big_dataset"
    train_dir = os.path.join(base_dataset, "train")
    test_dir = os.path.join(base_dataset, "test")
    val_dir = os.path.join(base_dataset, "val")
    

    print("Remapping label files in train directory...")
    remap_labels(train_dir)

    create_val_split(train_dir, val_dir, split_ratio=0.2)
    

    dataset_yaml_content = f"""
                            train: {train_dir}
                            val: {val_dir}
                            nc: 1
                            names: ['text-box']
                            """
                            
    dataset_yaml_path = os.path.join(base_dataset, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    print(f"Dataset YAML file created at: {dataset_yaml_path}")
    
    # Load a pretrained YOLOv8 model (fine-tuning).
    model = YOLO('yolov8s.pt')
    # Uncomment the following line to train from scratch:
    # model = YOLO('yolov8s.yaml')
    
    print("Starting training on new dataset...")
    model.train(data=dataset_yaml_path, epochs=10, imgsz=640, batch=16)
    
  
    target_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics"
    export_result = model.export(format="torchscript", save_dir=target_dir)
    print(f"Training complete. Exported model details:\n{export_result}")
    
    print("Starting evaluation on test dataset...")
    eval_results = evaluate_model(model, test_dir)
    results_csv = os.path.join(target_dir, "evaluation_results.csv")
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['image', 'avg_iou', 'inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in eval_results:
            writer.writerow(res)
    print(f"Evaluation complete. Results saved in: {results_csv}")
    

    prediction_dir = os.path.join(target_dir, "predictions")
    save_prediction_images(model, test_dir, prediction_dir, count=15)
    print("Prediction images saved.")

if __name__ == '__main__':
    freeze_support()
    main()
