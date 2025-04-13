import os
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    video_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\test_video\JapanRPG_TestSequence.mov"
    

    model_path = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics\yolo11n.pt"
    

    model = YOLO(model_path)
    
    project_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics"
    
    model.predict(
        source=video_path,
        imgsz=640,
        conf=0.25,
        save=True,
        project=project_dir,
        name="predictions",
        exist_ok=True
        # If filtering is needed, add: classes=[0]
    )
    
    print("Video prediction complete. Check the predictions folder in the target directory.")

if __name__ == '__main__':
    freeze_support()  
    main()

