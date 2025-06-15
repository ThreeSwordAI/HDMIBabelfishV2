import cv2
import numpy as np
import os

# ---------------------------
# STEP 1: Determine Crop Coordinates
# ---------------------------
box_example_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\examples\50_box.jpg"
cropped_example_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\examples\50_cropped.jpg"
original_example_path = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\examples\50.jpg"

# Load the example box image (which contains the red box)
img_box = cv2.imread(box_example_path)
if img_box is None:
    raise Exception(f"Failed to load {box_example_path}")


hsv = cv2.cvtColor(img_box, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2


contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise Exception("No red region found in the example box image.")


largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
print(f"Detected raw crop coordinates: x={x}, y={y}, w={w}, h={h}")

# ---------------------------
# STEP 2: Adjust Based on Example Cropped Image
# ---------------------------
border_thickness = 2  


crop_x = x + border_thickness
crop_y = y + border_thickness
crop_w = w - 2 * border_thickness
crop_h = h - 2 * border_thickness

print(f"Adjusted crop coordinates: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")


img_cropped_example = cv2.imread(cropped_example_path)
if img_cropped_example is None:
    print(f"Warning: Could not load example cropped image from {cropped_example_path}")
else:
    ex_h, ex_w = img_cropped_example.shape[:2]
    print(f"Example cropped image dimensions: width={ex_w}, height={ex_h}")

# ---------------------------
# STEP 3: Process All Images in the Input Folder
# ---------------------------
input_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\video_to_image"
output_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\cropped_image"


os.makedirs(output_dir, exist_ok=True)


image_files = sorted(
    [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')],
    key=lambda fname: int(os.path.splitext(fname)[0])
)

total_images = len(image_files)
print(f"Total images to process: {total_images}")


for idx, filename in enumerate(image_files):
    input_path = os.path.join(input_dir, filename)
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to read {input_path}. Skipping.")
        continue


    cropped_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]


    output_filename = f"{idx+1}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, cropped_img)


    if (idx + 1) % 100 == 0:
        print(f"Processed {idx+1}/{total_images} images...")

print("Cropping complete!")
