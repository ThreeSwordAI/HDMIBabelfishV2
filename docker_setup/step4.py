import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

# Define the model path and test data path
MARIAN_MODEL_PATH = "/workspace/translation/models/marian_opus_ja_en"
TEST_DATA_PATH = "/workspace/data/translation/test.csv"

# Load MarianMT model and tokenizer from the local directory
tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL_PATH)
model = MarianMTModel.from_pretrained(MARIAN_MODEL_PATH)

# Load the test data
test_data = pd.read_csv(TEST_DATA_PATH)

# Function to translate Japanese text to English
def translate_text(text):
    print(f"Translating: {text}")  # Debugging line to see the text being translated
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Print the inputs to check the tokenized form
    print(f"Tokenized inputs: {inputs}")
    
    # Get the translated tokens
    translated = model.generate(**inputs)
    
    # Decode the translated tokens to text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    print(f"Translated text: {translated_text}")  # Debugging line to check the output

    return translated_text

# Function to calculate BLEU score
def calculate_bleu_score(reference, hypothesis):
    reference = [reference.split()]  # BLEU expects a list of list of tokens
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method4)

# Evaluate the translation model
bleu_scores = []
results = []

for idx, row in test_data.iterrows():
    # Get the Japanese sentence and English ground truth
    jp_sentence = row['jp']
    eng_ground_truth = row['eng']
    
    # Translate the Japanese sentence
    prediction = translate_text(jp_sentence)
    
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

print(f"\nOverall Evaluation Results:")
print(f"Good Translations: {good_count}")
print(f"Moderate Translations: {moderate_count}")
print(f"Bad Translations: {bad_count}")

# Display detailed results
print("\nDetailed Results:")
print(results_df)

# Save the results to a CSV file (optional)
results_df.to_csv("/workspace/docker_setup/yolo_output/translation_results.csv", index=False)

