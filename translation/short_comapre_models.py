import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import random

# Path to the models and data
MODEL_DIR = 'models'
COMPARE_DIR = 'compare'
DATA_PATH = '../data/translation/test.csv'

# Load dataset and take 100 random samples
df = pd.read_csv(DATA_PATH)
#df_sampled = df.sample(n=100, random_state=42)  # Randomly take 100 samples
df_sampled = df.copy()  # Use the entire dataset for comparison

english_texts = df_sampled['eng'].tolist()
japanese_texts = df_sampled['jp'].tolist()

# Define the models
models = {
    "marian_opus_ja_en": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "color": 'blue',  # Model color for plotting
        "marker": 'o'  # Circular marker
    },
    "marian_jap_en": {
        "model_name": "Helsinki-NLP/opus-mt-jap-en",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "color": 'green',  # Model color for plotting
        "marker": 's'  # Square marker
    },
    "nllb_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "color": 'red',  # Model color for plotting
        "marker": '^'  # Triangle marker
    }
}

def translate_and_benchmark(model_key, model_config, english_texts, japanese_texts, ngram_order=2):
    """Translate and benchmark models based on BLEU score and inference time"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available
    tokenizer = model_config["tokenizer_class"].from_pretrained(model_config["model_name"])
    model = model_config["model_class"].from_pretrained(model_config["model_name"]).to(device)  # Move model to GPU if available

    bleu_scores = []
    inference_times = []
    translations = []

    smoothing_function = SmoothingFunction().method1  # Smoothing function to handle 0 counts

    for i in range(len(english_texts)):
        input_text = japanese_texts[i]

        start_time = time.time()

        # Translation
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)  # Move inputs to GPU
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=100, num_beams=2)
        
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Calculate inference time
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Calculate BLEU score with n-gram order and smoothing
        reference = [english_texts[i].split()]
        candidate = output_text.split()
        bleu = sentence_bleu(reference, candidate, weights=[1/ngram_order]*ngram_order, smoothing_function=smoothing_function)
        bleu_scores.append(bleu)
        
        translations.append((input_text, output_text))  # Save the input and output for later

    # Calculate average BLEU score and inference time
    avg_bleu = np.mean(bleu_scores)
    avg_inference_time = np.mean(inference_times)

    return avg_bleu, avg_inference_time, translations

def plot_results(results):
    """Plot the results of BLEU scores and inference times"""
    model_names = list(results.keys())
    bleu_scores = []
    inference_times = []
    colors = []
    markers = []
    
    for model_key in model_names:
        bleu_scores.append(results[model_key]["bleu"])
        inference_times.append(results[model_key]["time"])
        colors.append(results[model_key]["color"])
        markers.append(results[model_key]["marker"])

    # Create the plot
    plt.figure(figsize=(10, 6))

    for i, model_key in enumerate(model_names):
        plt.scatter(
            bleu_scores[i], 
            inference_times[i],
            color=colors[i],
            marker=markers[i],
            label=model_key
        )

    plt.xlabel('Average BLEU Score')
    plt.ylabel('Average Inference Time (seconds)')
    plt.title("Model Comparison: BLEU Scores vs. Inference Time")
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig(f"{COMPARE_DIR}/model_comparison.png")
    plt.savefig(f"{COMPARE_DIR}/model_comparison.pdf")
    plt.show()

def save_results(results):
    """Save the results in a CSV file"""
    all_results = []
    
    for model_key, model_data in results.items():
        avg_bleu = model_data['bleu']
        avg_inference_time = model_data['time']
        for i in range(len(english_texts)):
            all_results.append({
                'model': model_key,
                'input': japanese_texts[i],
                'output': model_data['translations'][i][1],
                'bleu_score': avg_bleu,
                'inference_time': avg_inference_time
            })
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"{COMPARE_DIR}/model_comparison_results.csv", index=False)
    print(f"Results saved to {COMPARE_DIR}/model_comparison_results.csv")

def main():
    results = {}

    # Benchmark each model
    for model_key, model_config in models.items():
        print(f"Benchmarking model: {model_key}")
        avg_bleu, avg_inference_time, translations = translate_and_benchmark(
            model_key, model_config, english_texts, japanese_texts, ngram_order=3  # Try 3-gram order
        )
        results[model_key] = {
            "bleu": avg_bleu,
            "time": avg_inference_time,
            "translations": translations,
            "color": model_config["color"],
            "marker": model_config["marker"]
        }

    # Plot the results
    plot_results(results)
    
    # Save results to CSV
    save_results(results)

if __name__ == "__main__":
    main()
