import os
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    MarianTokenizer, MarianMTModel,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = Path("models")
#TRAIN_FILE = Path("../data/translation/train.csv")
TRAIN_FILE = Path("../data/translation/train_100k.csv")
BASE_MODEL_DIR = MODELS_DIR / "marian_opus_ja_en"
FINETUNED_MODEL_DIR = MODELS_DIR / "marian_opus_ja_en_finetuned"

# Base model configuration
BASE_MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"

def setup_directories():
    """Create necessary directories"""
    MODELS_DIR.mkdir(exist_ok=True)
    FINETUNED_MODEL_DIR.mkdir(exist_ok=True)
    print(f"Directories ready: {MODELS_DIR}")

def load_training_data():
    """Load training data from CSV"""
    print("Loading training data...")
    
    if not TRAIN_FILE.exists():
        print(f"Training file not found: {TRAIN_FILE}")
        return None
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(TRAIN_FILE, encoding=encoding)
                print(f"Training data loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("Could not decode training CSV file")
            return None
        
        # Validate columns
        if 'jp' not in df.columns or 'eng' not in df.columns:
            print(f"Required columns 'jp' and 'eng' not found. Available columns: {list(df.columns)}")
            return None
        
        # Clean data
        df = df.dropna(subset=['jp', 'eng'])
        df['jp'] = df['jp'].astype(str).str.strip()
        df['eng'] = df['eng'].astype(str).str.strip()
        df = df[(df['jp'].str.len() > 1) & (df['eng'].str.len() > 1)]
        
        print(f"Loaded {len(df)} training samples")
        return df
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def load_base_model():
    """Load the base Marian OPUS model"""
    print("Loading base Marian OPUS model...")
    
    try:
        # Try loading from local directory first
        if BASE_MODEL_DIR.exists() and any(BASE_MODEL_DIR.iterdir()):
            print("Loading from local directory...")
            tokenizer = MarianTokenizer.from_pretrained(str(BASE_MODEL_DIR))
            model = MarianMTModel.from_pretrained(str(BASE_MODEL_DIR))
        else:
            print("Downloading from HuggingFace...")
            tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL_NAME)
            model = MarianMTModel.from_pretrained(BASE_MODEL_NAME)
            
            # Save locally for future use
            BASE_MODEL_DIR.mkdir(exist_ok=True)
            tokenizer.save_pretrained(str(BASE_MODEL_DIR))
            model.save_pretrained(str(BASE_MODEL_DIR))
            print("Base model saved locally")
        
        print("Base Marian OPUS model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None, None

def prepare_dataset(tokenizer, train_df):
    """Prepare dataset for fine-tuning"""
    print("Preparing dataset...")
    
    def tokenize_function(examples):
        # Tokenize Japanese (source)
        model_inputs = tokenizer(
            examples['jp'],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        # Tokenize English (target)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['eng'],
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Create dataset
    dataset = Dataset.from_dict({
        'jp': train_df['jp'].tolist(),
        'eng': train_df['eng'].tolist()
    })
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print(f"Dataset prepared: {len(tokenized_dataset)} samples")
    return tokenized_dataset

def fine_tune_model(tokenizer, model, train_dataset):
    """Fine-tune the Marian model"""
    print("Starting fine-tuning...")
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
    else:
        print("Using CPU (this will be slow)")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(FINETUNED_MODEL_DIR / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=3,
        #per_device_train_batch_size=4 if cuda_available else 2,
        per_device_train_batch_size=8,
        #gradient_accumulation_steps=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to=None,
        dataloader_num_workers=0,
        fp16=cuda_available,
        push_to_hub=False,
        remove_unused_columns=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Training started...")
    train_result = trainer.train()
    
    print(f"Training completed!")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    trainer.save_model(str(FINETUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINETUNED_MODEL_DIR))
    
    print(f"Fine-tuned model saved to: {FINETUNED_MODEL_DIR}")
    
    return True

def test_finetuned_model():
    """Quick test of the fine-tuned model"""
    print("Testing fine-tuned model...")
    
    try:
        # Load fine-tuned model
        tokenizer = MarianTokenizer.from_pretrained(str(FINETUNED_MODEL_DIR))
        model = MarianMTModel.from_pretrained(str(FINETUNED_MODEL_DIR))
        
        # Test translation
        test_text = "こんにちは"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            generated = model.generate(
                **inputs, 
                max_length=50, 
                num_beams=4,
                early_stopping=True
            )
        
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Test translation: '{test_text}' -> '{result}'")
        print("Fine-tuned model is working correctly!")
        return True
        
    except Exception as e:
        print(f"Error testing fine-tuned model: {e}")
        return False

def main():
    """Main fine-tuning process"""
    print("=" * 60)
    print("MARIAN OPUS FINE-TUNING")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Load training data
    train_df = load_training_data()
    if train_df is None:
        print("Failed to load training data. Exiting...")
        return
    
    # Load base model
    tokenizer, model = load_base_model()
    if tokenizer is None or model is None:
        print("Failed to load base model. Exiting...")
        return
    
    # Prepare dataset
    train_dataset = prepare_dataset(tokenizer, train_df)
 
    # Fine-tune model
    success = fine_tune_model(tokenizer, model, train_dataset)
    if not success:
        print("Fine-tuning failed. Exiting...")
        return
   
    # Test fine-tuned model
    if test_finetuned_model():
        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Base model: Helsinki-NLP/opus-mt-ja-en")
        print(f"Fine-tuned model saved to: {FINETUNED_MODEL_DIR}")
        print(f"Model name: Marian OPUS (Fine-Tuned)")
        print("=" * 60)
    else:
        print("Fine-tuning completed but model test failed.")

if __name__ == "__main__":
    main()