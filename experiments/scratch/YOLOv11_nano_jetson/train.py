import os
import time
import random
import shutil
import cv2
import csv
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

def remap_labels(label_dir):
    """Remap all labels to class 0 (text-box)"""
    print(f"🏷️ Remapping labels in {label_dir}...")
    count = 0
    for fname in os.listdir(label_dir):
        if fname.lower().endswith('.txt') and fname != "classes.txt":
            file_path = os.path.join(label_dir, fname)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"  # Set all to class 0
                        new_lines.append(" ".join(parts))
                if new_lines:
                    with open(file_path, 'w') as f:
                        f.write("\n".join(new_lines))
                    count += 1
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
    print(f"✅ Remapped {count} label files")

def create_val_split(train_dir, val_dir, split_ratio=0.2):
    """Create validation split from training data"""
    print(f"📂 Creating validation split ({split_ratio*100}%)...")
    
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)
    
    image_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.jpg')]
    if not image_files:
        print("❌ No images found in training directory")
        return 0
    
    num_val = int(len(image_files) * split_ratio)
    val_images = random.sample(image_files, num_val)
    
    copied_count = 0
    for img in val_images:
        try:
            src_img = os.path.join(train_dir, img)
            dst_img = os.path.join(val_dir, img)
            shutil.copy(src_img, dst_img)
            
            # Copy corresponding label file
            label_file = img.replace('.jpg', '.txt')
            src_label = os.path.join(train_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(val_dir, label_file)
                shutil.copy(src_label, dst_label)
            copied_count += 1
        except Exception as e:
            print(f"⚠️ Error copying {img}: {e}")
    
    print(f"✅ Created validation split: {copied_count} images copied to {val_dir}")
    return copied_count

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU)"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute areas of both boxes
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)
    return iou

def evaluate_model(model, test_dir):
    """Evaluate model on test dataset with Docker-compatible OpenCV"""
    print("📊 Evaluating model on test dataset...")
    results = []
    
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    print(f"   Found {len(test_files)} test images")
    
    for i, file in enumerate(test_files):
        if (i + 1) % 10 == 0:
            print(f"   Processing {i+1}/{len(test_files)}...")
        
        try:
            image_path = os.path.join(test_dir, file)
            label_path = os.path.join(test_dir, file.replace('.jpg', '.txt'))
            
            if not os.path.exists(label_path):
                continue
            
            # Use PIL instead of OpenCV for Docker compatibility
            from PIL import Image
            image_pil = Image.open(image_path)
            w, h = image_pil.size
            
            # Convert PIL to numpy array for YOLO
            image_np = np.array(image_pil)
            
            # Read ground truth
            with open(label_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue
                
            parts = lines[0].strip().split()
            if len(parts) < 5:
                continue
            
            # Convert YOLO normalized format to absolute pixel coordinates
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h
            gt_box = [
                int(x_center - box_w/2),
                int(y_center - box_h/2),
                int(x_center + box_w/2),
                int(y_center + box_h/2)
            ]
            
            # Predict with timing
            start = time.time()
            pred_results = model.predict(source=image_np, imgsz=640, conf=0.25, verbose=False)
            inference_time = time.time() - start
            
            # Extract prediction box
            pred_box = [0, 0, 0, 0]
            if pred_results and pred_results[0].boxes and len(pred_results[0].boxes) > 0:
                box_tensor = pred_results[0].boxes.xyxy[0].cpu().numpy()
                pred_box = [int(x) for x in box_tensor]
            
            iou = compute_iou(pred_box, gt_box)
            results.append({
                'image': file, 
                'avg_iou': iou, 
                'inference_time': inference_time
            })
            
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
            continue
    
    print(f"✅ Evaluation completed on {len(results)} images")
    return results

def save_prediction_images(model, test_dir, prediction_dir, count=15):
    """Save prediction images with Docker-compatible paths"""
    try:
        image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if f.lower().endswith('.jpg')]
        
        if len(image_files) < count:
            count = len(image_files)
        
        sampled_images = random.sample(image_files, count)
        print(f"🖼️ Running predictions on {count} random images...")
        print(f"   Saving to: {prediction_dir}")
        
        # Create prediction directory
        os.makedirs(prediction_dir, exist_ok=True)
        
        # Run predictions
        model.predict(
            source=sampled_images, 
            imgsz=640,  # Use 640 instead of 416 for better results
            conf=0.25, 
            save=True, 
            project=prediction_dir, 
            name="predictions", 
            exist_ok=True
        )
        
        print(f"✅ Prediction images saved to: {prediction_dir}")
        
    except Exception as e:
        print(f"⚠️ Error saving predictions: {e}")

def check_dataset_structure(base_dataset):
    """Check and report dataset structure"""
    print("\n📊 DATASET STRUCTURE:")
    
    train_dir = os.path.join(base_dataset, "train")
    val_dir = os.path.join(base_dataset, "val")
    test_dir = os.path.join(base_dataset, "test")
    
    total_files = 0
    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if os.path.exists(split_dir):
            images = [f for f in os.listdir(split_dir) if f.lower().endswith('.jpg')]
            labels = [f for f in os.listdir(split_dir) 
                     if f.lower().endswith('.txt') and f != 'classes.txt']
            print(f"   {split_name:5}: {len(images):4} images, {len(labels):4} labels")
            total_files += len(images)
        else:
            print(f"   {split_name:5}: Directory not found")
    
    return total_files

def main():
    """Main execution function optimized for Jetson Docker"""
    print("="*60)
    print("🐳 JETSON DOCKER YOLO TRAINING")
    print("🎯 Japanese Text Detection Model")
    print("="*60)
    
    # Jetson Docker paths
    base_dataset = "../data/game_review/big_dataset"
    train_dir = os.path.join(base_dataset, "train")
    test_dir = os.path.join(base_dataset, "test")
    val_dir = os.path.join(base_dataset, "val")
    
    # Check if dataset exists
    if not os.path.exists(base_dataset):
        print(f"❌ Dataset not found: {base_dataset}")
        print("Please ensure the dataset is mounted/copied to Docker")
        return
    
    print(f"✅ Dataset found: {base_dataset}")
    
    # Step 1: Dataset preparation
    print("\n1️⃣ DATASET PREPARATION:")
    
    # Check existing structure
    total_files = check_dataset_structure(base_dataset)
    if total_files == 0:
        print("❌ No training data found!")
        return
    
    # Remap labels to single class
    print("\n🏷️ Remapping labels to single class...")
    remap_labels(train_dir)
    
    # Create validation split if needed
    if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
        print("\n📂 Creating validation split...")
        val_count = create_val_split(train_dir, val_dir, split_ratio=0.2)
        if val_count == 0:
            print("❌ Failed to create validation split")
            return
    
    # Final dataset check
    check_dataset_structure(base_dataset)
    
    # Create dataset YAML
    print("\n📝 Creating dataset configuration...")
    dataset_yaml_content = f"""# Jetson Docker Japanese Text Detection Dataset
path: {os.path.abspath(base_dataset)}
train: {os.path.abspath(train_dir)}
val: {os.path.abspath(val_dir)}

# Classes
nc: 1
names: ['text-box']
"""
    
    dataset_yaml_path = os.path.join(base_dataset, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    print(f"✅ Dataset YAML created: {dataset_yaml_path}")
    
    # Step 2: Model setup
    print("\n2️⃣ MODEL SETUP:")
    
    # FIXED: Use pre-trained model instead of empty YAML
    print("📥 Loading YOLOv11 nano with pre-trained weights...")
    model = YOLO('yolo11n.pt')  # Download pre-trained weights
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model device: {model.device}")
    print(f"🎯 Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Step 3: Training configuration
    print("\n3️⃣ TRAINING CONFIGURATION:")
    
    # Jetson Docker optimized parameters
    training_config = {
        'data': dataset_yaml_path,
        'epochs': 50,              # Reasonable for Jetson
        'imgsz': 640,              # Standard YOLO size (not 416)
        'batch': 4,                # Small batch for Jetson memory
        'patience': 15,            # Early stopping
        'save': True,
        'plots': True,
        'val': True,
        'verbose': True,
        'workers': 2,              # Limited workers for Jetson
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'project': '../experiments/jetson_docker_training',
        'name': 'japanese_text_detection'
    }
    
    print("⚙️ Training parameters:")
    for key, value in training_config.items():
        print(f"   {key:12}: {value}")
    
    # Step 4: Training
    print("\n4️⃣ STARTING TRAINING:")
    print("🚀 This will take 1-3 hours on Jetson...")
    print("💡 You can monitor progress in the terminal")
    
    try:
        training_start = time.time()
        
        # Start training
        results = model.train(**training_config)
        
        training_time = time.time() - training_start
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"⏱️ Training time: {training_time/3600:.1f} hours")
        print(f"📁 Best model: {model.trainer.best}")
        print(f"📊 Training results: {model.trainer.save_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 5: Copy model to expected location
    print("\n5️⃣ COPYING MODEL TO EXPECTED LOCATION:")
    
    expected_dir = "../experiments/scratch/YOLOv11_nano/runs/detect/train/weights"
    os.makedirs(expected_dir, exist_ok=True)
    
    src_model = model.trainer.best
    dst_model = os.path.join(expected_dir, "best.pt")
    
    try:
        shutil.copy2(src_model, dst_model)
        print(f"✅ Model copied to: {dst_model}")
        
        # Verify the copied model
        model_data = torch.load(dst_model, map_location='cpu')
        epoch = model_data.get('epoch', 'Unknown')
        fitness = model_data.get('best_fitness', 'Unknown')
        print(f"📊 Model verification - Epoch: {epoch}, Fitness: {fitness}")
        
    except Exception as e:
        print(f"⚠️ Error copying model: {e}")
        dst_model = src_model  # Use original location
    
    # Step 6: Evaluation
    print("\n6️⃣ MODEL EVALUATION:")
    
    try:
        # Load the best model for evaluation
        best_model = YOLO(dst_model)
        
        # Evaluate on test set
        eval_results = evaluate_model(best_model, test_dir)
        
        if eval_results:
            # Save evaluation results
            results_dir = os.path.dirname(dst_model)
            results_csv = os.path.join(results_dir, "evaluation_results.csv")
            
            with open(results_csv, 'w', newline='') as csvfile:
                fieldnames = ['image', 'avg_iou', 'inference_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for res in eval_results:
                    writer.writerow(res)
            
            # Calculate metrics
            avg_iou = np.mean([r['avg_iou'] for r in eval_results])
            avg_inference = np.mean([r['inference_time'] for r in eval_results])
            
            print(f"✅ Evaluation completed:")
            print(f"   📊 Average IoU: {avg_iou:.3f}")
            print(f"   ⚡ Average inference: {avg_inference:.3f}s")
            print(f"   📄 Results saved: {results_csv}")
        else:
            print("⚠️ No evaluation results generated")
            
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
    
    # Step 7: Generate prediction images
    print("\n7️⃣ GENERATING PREDICTION SAMPLES:")
    
    try:
        prediction_dir = os.path.join(os.path.dirname(dst_model), "predictions")
        save_prediction_images(best_model, test_dir, prediction_dir, count=15)
    except Exception as e:
        print(f"⚠️ Prediction generation failed: {e}")
    
    # Final summary
    print(f"\n🎉 JETSON DOCKER TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Best model: {dst_model}")
    print(f"📊 Training results: {model.trainer.save_dir}")
    if 'results_csv' in locals():
        print(f"📄 Evaluation: {results_csv}")
    if 'prediction_dir' in locals():
        print(f"🖼️ Predictions: {prediction_dir}")
    print("✅ Model ready for Japanese text detection!")

if __name__ == '__main__':
    main()