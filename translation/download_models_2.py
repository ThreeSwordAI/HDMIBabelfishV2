import os
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer

# Define the models to download
models_to_download = {
    "mbart_large": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
    },
    "mt5_base": {
        "model_name": "google/mt5-base",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
    },
    "t5_base": {
        "model_name": "t5-base",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
    },
    "nllb_200m": {
        "model_name": "facebook/nllb-200-distilled-200M",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
    },
}

# Define the directory to store the models
MODEL_DIR = 'models'

# Create the models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def download_and_save_model(model_name, model_config):
    """Download and save the model and tokenizer."""
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Check if the model already exists, if not, download and save
    if not os.path.exists(model_path):
        print(f"Downloading and saving model: {model_name}")
        
        # Load the model and tokenizer
        tokenizer = model_config["tokenizer_class"].from_pretrained(model_config["model_name"])
        model = model_config["model_class"].from_pretrained(model_config["model_name"])

        # Save the model and tokenizer
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"Model saved to {model_path}")
    else:
        print(f"Model '{model_name}' already exists in {model_path}")

# Download and save each model
for model_name, model_config in models_to_download.items():
    download_and_save_model(model_name, model_config)