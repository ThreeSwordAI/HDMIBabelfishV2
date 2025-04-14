import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


csv_files = [
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\scratch\YOLOv8\evaluation_results.csv",
        "model": "YOLOv8 Scratch"
    },
    {
        "path": r"F:\FAU\Thesis\HDMIBabelfishV2\experiments\transfer_learning\YOLOv5\Target\FT_model_Scratch\evaluation_results.csv",
        "model": "YOLOv5 FT (Scratch)"
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
    }
]


summary_data = []

for item in csv_files:
    df = pd.read_csv(item["path"])

    avg_iou = df["avg_iou"].mean()
    avg_inference = df["inference_time"].mean()

    summary_data.append({
        "Model": item["model"],
        "IoU": avg_iou,
        "Inference Time": avg_inference
    })


summary_df = pd.DataFrame(summary_data)
print(summary_df)


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.scatterplot(data=summary_df, x="IoU", y="Inference Time", hue="Model", s=100, palette="deep")
ax.set_xlabel("IoU", fontsize=14)
ax.set_ylabel("Inference Time (sec)", fontsize=14)
ax.set_title("Model Comparison: IoU vs. Inference Time", fontsize=16)
plt.legend(title="Model", loc="upper left")
plt.tight_layout()


plt.savefig("model_comparison.png", dpi=300)
plt.savefig("model_comparison.pdf")
plt.show()
