import cv2
import os

video_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\video\GamesForLearninJapanese.mp4"
output_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\video_to_image"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {video_path}")


fps = cap.get(cv2.CAP_PROP_FPS)  
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration_sec = int(frame_count / fps)

print(f"Video FPS: {fps}")
print(f"Total frame count: {frame_count}")
print(f"Approximate duration (seconds): {duration_sec}")


for sec in range(duration_sec):

    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if ret:
        image_name = f"{sec + 1}.jpg"
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, frame)
        if (sec+1) % 100 == 0:
            print(f"Processed {sec + 1} seconds...")
    else:
        print(f"Frame at {sec} sec could not be read. Exiting loop.")
        break


cap.release()

print("Extraction complete.")
