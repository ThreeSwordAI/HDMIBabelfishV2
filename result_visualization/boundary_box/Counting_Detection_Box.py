import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# --- CONFIGURATION ----------------------------------------------------------

# evaluation CSVs + model display names
models = [
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv5\Source\Scratch_model\evaluation_results.csv",
        "model": "YOLOv5 Scratch"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv8\evaluation_results.csv",
        "model": "YOLOv8 Scratch"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv11_nano\evaluation_results.csv",
        "model": "YOLOv11 Nano Scratch"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv5\Target\FT_model_Scratch\evaluation_results.csv",
        "model": "YOLOv5 FT on Scratch"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_CarLicense\evaluation_results.csv",
        "model": "YOLOv8 FT CarLicense"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_two_level\evaluation_results.csv",
        "model": "YOLOv8 FT Two-Level"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics\evaluation_results.csv",
        "model": "YOLOv8 FT Ultralytics"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv8\Target\FT_model_ultralytics_nano\evaluation_results.csv",
        "model": "YOLOv8 FT Ultralytics Nano"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv11\Target\FT_model_ultralytics_nano\evaluation_results.csv",
        "model": "YOLOv11 FT Ultralytics Nano"
    }
]

# where your test images + labels live
test_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\big_dataset\test"

# ----------------------------------------------------------------------------


total_gt_boxes = 0
for fname in os.listdir(test_dir):
    if fname.lower().endswith(".txt"):
        with open(os.path.join(test_dir, fname), "r") as f:
            total_gt_boxes += sum(1 for _ in f)


results_list = []
for item in models:
    model_name = item["model"]

    weight_path = item["path"].replace(
        os.path.join(*item["path"].split(os.sep)[-2:]),  
        os.path.join("runs", "detect", "train", "weights", "best.pt")
    )

    model = YOLO(weight_path)
    preds = model.predict(source=test_dir, verbose=False)

    detected = sum(len(r.boxes) for r in preds)
    results_list.append({"Model": model_name, "Detected Boxes": detected})


df = pd.DataFrame(results_list)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df,
    x="Model",
    y="Detected Boxes",
    palette="deep"
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title(
    f"Detected Boxes per Model  (Total Ground-Truth Boxes: {total_gt_boxes})",
    fontsize=16
)
ax.set_xlabel("Model", fontsize=14)
ax.set_ylabel("Detected Boxes", fontsize=14)
plt.tight_layout()


plt.savefig("detected_boxes_comparison.png", dpi=300)
plt.savefig("detected_boxes_comparison.pdf")
plt.show()
