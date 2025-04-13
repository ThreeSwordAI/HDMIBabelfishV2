import os
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    video_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\test_video\JapanRPG_TestSequence.mov"
    
    # Path to the fine-tuned model that detects only text-box.
    model_path = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics\runs\detect\train\weights\best.pt"
    

    # model_path = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics\yolo11n.pt"
    

    model = YOLO(model_path)
    

    project_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics"
    
    print("Starting video prediction on text-box detections...")
    model.predict(
        source=video_path,
        imgsz=640,
        conf=0.25,
        # If extra filtering is needed, force only text-box detections by uncommenting below:
        # classes=[0],
        save=True,
        project=project_dir,
        name="predictions",
        exist_ok=True
    )
    
    print("Video prediction complete. Please check the 'predictions' folder under:")
    print(project_dir)

if __name__ == '__main__':
    freeze_support()  
    main()