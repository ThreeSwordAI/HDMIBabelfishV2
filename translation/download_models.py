"""
Download and setup translation models for comparison
Downloads multiple translation models to translation/models/
"""

import os
import sys
import torch
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
from huggingface_hub import snapshot_download
import requests
from tqdm import tqdm

# Configuration
MODELS_DIR = Path("models")
COMPARE_DIR = Path("compare")
DATA_DIR = Path("../data/translation")

# Model configurations (stable models only)
TRANSLATION_MODELS = {
    "marian_opus_ja_en": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "src_lang": "ja",
        "tgt_lang": "en",
        "description": "Marian OPUS Japanese-English (Fast, Small)"
    },
    "marian_jap_en": {
        "model_name": "Helsinki-NLP/opus-mt-jap-en", 
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "src_lang": "jap",
        "tgt_lang": "en",
        "description": "Marian Japanese-English Alternative"
    },
    "nllb_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "src_lang": "jpn_Jpan",
        "tgt_lang": "eng_Latn",
        "description": "NLLB 600M Distilled (High Quality)"
    }
}

def create_directories():
    """Create necessary directories"""
    MODELS_DIR.mkdir(exist_ok=True)
    COMPARE_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    print(f"✓ Created directories")
    print(f"  - Models: {MODELS_DIR.absolute()}")
    print(f"  - Compare: {COMPARE_DIR.absolute()}")
    print(f"  - Data: {DATA_DIR.absolute()}")

def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"✓ CUDA available: {device}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        return True
    else:
        print("⚠ CUDA not available, will use CPU")
        return False

def download_model(model_key, model_config):
    """Download a specific model"""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: {model_key.upper()}")
    print(f"Description: {model_config['description']}")
    print(f"Model: {model_config['model_name']}")
    print(f"{'='*60}")
    
    model_path = MODELS_DIR / model_key
    
    try:
        # Check if model already exists
        if model_path.exists() and any(model_path.iterdir()):
            print(f"✓ Model {model_key} already exists at {model_path}")
            return True, model_path
            
        # Create model directory
        model_path.mkdir(exist_ok=True)
        
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = model_config['tokenizer_class'].from_pretrained(
            model_config['model_name'],
            cache_dir=str(MODELS_DIR / "cache"),
            local_files_only=False
        )
        tokenizer.save_pretrained(str(model_path))
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("📥 Downloading model...")
        model = model_config['model_class'].from_pretrained(
            model_config['model_name'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir=str(MODELS_DIR / "cache"),
            local_files_only=False
        )
        model.save_pretrained(str(model_path))
        print("✓ Model downloaded")
        
        # Calculate model size
        model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*')) / (1024*1024)
        print(f"✓ Model size: {model_size_mb:.1f} MB")
        
        return True, model_path
        
    except Exception as e:
        print(f"❌ Error downloading {model_key}: {e}")
        return False, None

def test_model(model_key, model_config, model_path):
    """Test if a model can be loaded and used"""
    print(f"\n🧪 Testing {model_key}...")
    
    try:
        # Load tokenizer and model
        tokenizer = model_config['tokenizer_class'].from_pretrained(
            str(model_path),
            local_files_only=True
        )
        
        model = model_config['model_class'].from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        
        # Test translation
        test_texts = [
            "こんにちは",
            "ゲームを始めます", 
            "アイテムを使用しますか？"
        ]
        
        print("Test translations:")
        for test_text in test_texts:
            start_time = time.time()
            
            if "marian" in model_key:
                # Marian models
                inputs = tokenizer(test_text, return_tensors="pt", padding=True)
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=100, num_beams=2)
                result = tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                # NLLB/mBART models
                inputs = tokenizer(
                    test_text, 
                    return_tensors="pt", 
                    src_lang=model_config['src_lang'] if 'src_lang' in model_config else None
                )
                
                with torch.no_grad():
                    if "nllb" in model_key:
                        generated = model.generate(
                            **inputs,
                            forced_bos_token_id=tokenizer.lang_code_to_id[model_config['tgt_lang']],
                            max_length=100,
                            num_beams=2
                        )
                    else:  # mBART
                        generated = model.generate(
                            **inputs,
                            max_length=100,
                            num_beams=2,
                            forced_bos_token_id=tokenizer.lang_code_to_id[model_config['tgt_lang']]
                        )
                
                result = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            print(f"  '{test_text}' -> '{result}' ({inference_time:.3f}s)")
        
        print(f"✓ {model_key} test successful")
        return True
        
    except Exception as e:
        print(f"❌ {model_key} test failed: {e}")
        return False

def download_dataset():
    """Check if evaluation dataset exists"""
    print(f"\n{'='*60}")
    print("CHECKING EVALUATION DATASET")
    print(f"{'='*60}")
    
    # Check if dataset already exists
    dataset_file = DATA_DIR / "test.csv"
    if dataset_file.exists():
        print(f"✓ Dataset found at {dataset_file}")
        
        # Try to load and validate the dataset
        try:
            import pandas as pd
            df = pd.read_csv(dataset_file)
            
            if 'jp' in df.columns and 'eng' in df.columns:
                print(f"✓ Dataset validated: {len(df)} samples")
                print(f"✓ Columns found: {list(df.columns)}")
                print("✓ Ready for model comparison")
                return True
            else:
                print(f"⚠ Dataset columns: {list(df.columns)}")
                print("⚠ Expected columns: 'jp' and 'eng'")
                print("⚠ Please check your dataset format")
                return False
                
        except Exception as e:
            print(f"⚠ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset not found: {dataset_file}")
        print("📝 Please ensure you have:")
        print("   - test.csv file in data/translation/")
        print("   - Columns named 'jp' (Japanese) and 'eng' (English)")
        print("   - CSV format with proper encoding")
        return False

def create_model_info():
    """Create model information file"""
    model_info = {
        "models": {},
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "download_date": str(time.strftime("%Y-%m-%d %H:%M:%S"))
    }
    
    # Add model information
    for model_key, config in TRANSLATION_MODELS.items():
        model_path = MODELS_DIR / model_key
        if model_path.exists():
            model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*')) / (1024*1024)
            model_info["models"][model_key] = {
                "model_name": config["model_name"],
                "description": config["description"],
                "local_path": str(model_path),
                "size_mb": round(model_size_mb, 1),
                "src_lang": config.get("src_lang", "ja"),
                "tgt_lang": config.get("tgt_lang", "en"),
                "available": True
            }
        else:
            model_info["models"][model_key] = {
                "model_name": config["model_name"],
                "description": config["description"],
                "available": False
            }
    
    info_path = MODELS_DIR / "model_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Model info saved to {info_path}")

def main():
    """Main download process"""
    print("="*60)
    print("TRANSLATION MODELS DOWNLOADER")
    print("="*60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Check CUDA
    cuda_available = check_cuda()
    
    # Step 3: Check dataset
    print("\n3️⃣ Checking dataset...")
    if not download_dataset():
        print("⚠ Dataset check failed - you can still download models")
        print("Make sure to add test.csv to data/translation/ before running comparison")
    
    # Step 4: Download models
    successful_downloads = 0
    total_models = len(TRANSLATION_MODELS)
    
    for model_key, model_config in TRANSLATION_MODELS.items():
        print(f"\n[{successful_downloads + 1}/{total_models}] Processing {model_key}...")
        
        success, model_path = download_model(model_key, model_config)
        if success:
            # Test the model
            if test_model(model_key, model_config, model_path):
                successful_downloads += 1
            else:
                print(f"⚠ Model {model_key} downloaded but test failed")
        else:
            print(f"❌ Failed to download {model_key}")
    
    # Step 5: Create model info file
    create_model_info()
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successfully downloaded: {successful_downloads}/{total_models} models")
    print(f"📁 Models saved to: {MODELS_DIR.absolute()}")
    print(f"📊 Dataset saved to: {DATA_DIR.absolute()}")
    print("\n🚀 Next steps:")
    print("   1. Run 'python compare_models.py' to benchmark models")
    print("   2. Use results to select best model for your pipeline")
    print("="*60)
    
    if successful_downloads == 0:
        print("❌ No models downloaded successfully")
        sys.exit(1)

if __name__ == "__main__":
    main()