import os
import glob
import xml.etree.ElementTree as ET
import yaml
from ultralytics import YOLO

DATA_DIR = os.path.join("F:/FAU/Thesis/HDMIBabelfishV2/data/Bangladeshi_License_Plate")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_DIR = os.path.join(TRAIN_DIR, "labels")
os.makedirs(LABELS_DIR, exist_ok=True)

def convert_xml_to_yolo(xml_file, labels_dir):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return False
    size = root.find("size")
    if size is None:
        return False
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip()
        cls_id = 0
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width  = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)
    if not lines:
        return False
    base = os.path.splitext(os.path.basename(xml_file))[0]
    txt_file = os.path.join(labels_dir, base + ".txt")
    with open(txt_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return True

img_extensions = [".jpg", ".jpeg", ".png"]
xml_files = glob.glob(os.path.join(TRAIN_DIR, "*.xml"))
for xml_file in xml_files:
    base = os.path.splitext(os.path.basename(xml_file))[0]
    image_exists = any(os.path.exists(os.path.join(TRAIN_DIR, base + ext)) for ext in img_extensions)
    if image_exists:
        if convert_xml_to_yolo(xml_file, LABELS_DIR):
            print(f"Converted {xml_file} successfully.")
        else:
            print(f"Failed to convert {xml_file}.")
    else:
        print(f"Ignoring {xml_file} as corresponding image does not exist.")

dataset_yaml = {
    "names": ["license-plate"],
    "nc": 1,
    "path": "F:/FAU/Thesis/HDMIBabelfishV2/data/Bangladeshi_License_Plate",
    "train": "train",
    "val": "train",
    "test": "test"
}
dataset_yaml_path = os.path.join(DATA_DIR, "dataset.yaml")
with open(dataset_yaml_path, "w") as yaml_file:
    yaml.dump(dataset_yaml, yaml_file)
print(f"Dataset YAML created at {dataset_yaml_path}")

PROJECT_DIR = os.path.abspath(".")
os.makedirs(PROJECT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

model.train(
    data=dataset_yaml_path, 
    epochs=50,
    imgsz=640,
    project=PROJECT_DIR,
    name="trained_model_source",
    device="cuda:0"
)


print("Training completed. Checkpoints and logs are saved under:")
print(PROJECT_DIR)


