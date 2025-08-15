import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

# Check CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Define the model path and test data path
MARIAN_MODEL_PATH = "/workspace/translation/models/marian_opus_ja_en"
TEST_DATA_PATH = "/workspace/data/translation/test.csv"

print("📂 Loading MarianMT model and tokenizer...")
# Load MarianMT model and tokenizer from the local directory
tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL_PATH)
model = MarianMTModel.from_pretrained(MARIAN_MODEL_PATH)

# Move model to GPU if available
model = model.to(device)
print(f"✅ Model loaded on {device}")

# Enable inference mode for better performance
model.eval()

# Load the test data
print("📄 Loading test data...")
test_data = pd.read_csv(TEST_DATA_PATH)
print(f"📊 Loaded {len(test_data)} test samples")

# Function to translate Japanese text to English
def translate_text(text):
    print(f"🔤 Translating: {text[:50]}..." if len(text) > 50 else f"🔤 Translating: {text}")
    
    # Tokenize the input text and move to device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Use torch.no_grad() for inference to save memory and speed up
    with torch.no_grad():
        # Get the translated tokens
        translated = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,  # Use beam search for better quality
            early_stopping=True,
            do_sample=False  # Deterministic output
        )
    
    # Decode the translated tokens to text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    print(f"✅ Result: {translated_text}")
    return translated_text

# Function to calculate BLEU score
def calculate_bleu_score(reference, hypothesis):
    reference = [reference.split()]  # BLEU expects a list of list of tokens
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method4)

# Function to translate in batches for better GPU utilization
def translate_batch(texts, batch_size=8):
    """Translate multiple texts in batches for better GPU utilization"""
    translations = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"🚀 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translations
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode all translations in the batch
        batch_translations = [
            tokenizer.decode(trans, skip_special_tokens=True) 
            for trans in translated
        ]
        translations.extend(batch_translations)
        
        # Optional: Clear cache to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return translations

# Evaluate the translation model
print("\n🚀 Starting translation evaluation...")
bleu_scores = []
results = []

# Extract all Japanese sentences for batch processing
jp_sentences = test_data['jp'].tolist()

# Choose processing method based on dataset size
if len(jp_sentences) > 10:
    print("📦 Using batch processing for efficiency...")
    predictions = translate_batch(jp_sentences, batch_size=4)  # Adjust batch size based on GPU memory
else:
    print("🔄 Processing individual sentences...")
    predictions = [translate_text(jp_sentence) for jp_sentence in jp_sentences]

# Calculate BLEU scores and classify results
for idx, (row, prediction) in enumerate(zip(test_data.iterrows(), predictions)):
    row_data = row[1]  # Get the actual row data
    jp_sentence = row_data['jp']
    eng_ground_truth = row_data['eng']
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(eng_ground_truth, prediction)
    bleu_scores.append(bleu_score)
    
    # Classify the result
    if bleu_score >= 0.75:
        result = "Good"
    elif bleu_score >= 0.50:
        result = "Moderate"
    else:
        result = "Bad"
    
    results.append({
        "Prediction Number": idx,
        "Japanese Sentence": jp_sentence,
        "Ground Truth (English)": eng_ground_truth,
        "Predicted Translation": prediction,
        "BLEU Score": bleu_score,
        "Result": result
    })

# Create a DataFrame with results
results_df = pd.DataFrame(results)

# Print the overall result and classification
good_count = len([r for r in results if r["Result"] == "Good"])
moderate_count = len([r for r in results if r["Result"] == "Moderate"])
bad_count = len([r for r in results if r["Result"] == "Bad"])

print(f"\n📊 Overall Evaluation Results:")
print(f"✅ Good Translations: {good_count}")
print(f"⚠️  Moderate Translations: {moderate_count}")
print(f"❌ Bad Translations: {bad_count}")
print(f"📈 Average BLEU Score: {sum(bleu_scores)/len(bleu_scores):.4f}")

# Display detailed results
print("\n📋 Detailed Results:")
print(results_df)

# Save the results to a CSV file
output_path = "/workspace/docker_setup/yolo_output/translation_results_gpu.csv"
results_df.to_csv(output_path, index=False)
print(f"💾 Results saved to: {output_path}")

# Print GPU memory usage if available
if torch.cuda.is_available():
    print(f"\n🔥 GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")