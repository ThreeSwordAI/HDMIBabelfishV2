import os
import sys
import time
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = Path("models")
COMPARE_DIR = Path("compare")
DATA_DIR = Path("../data/translation")
RESULTS_FILE = COMPARE_DIR / "comparison_results.json"
DATASET_FILE = DATA_DIR / "test.csv"

# Comprehensive model configurations
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
    },
    "nllb_200m": {
        "model_name": "facebook/nllb-200-distilled-200M",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "src_lang": "jpn_Jpan",
        "tgt_lang": "eng_Latn",
        "description": "NLLB 200M Distilled (Faster)"
    },
    "mbart_large": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "src_lang": "ja_XX",
        "tgt_lang": "en_XX",
        "description": "mBART Large 50 (High Quality)"
    },
    "mt5_base": {
        "model_name": "google/mt5-base",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "src_lang": "ja",
        "tgt_lang": "en",
        "description": "mT5 Base (Multilingual T5)"
    }
}

class UnifiedModelManager:
    """Unified manager for downloading and comparing translation models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        MODELS_DIR.mkdir(exist_ok=True)
        COMPARE_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        print(f"✅ Created directories")
        print(f"  - Models: {MODELS_DIR.absolute()}")
        print(f"  - Compare: {COMPARE_DIR.absolute()}")
        print(f"  - Data: {DATA_DIR.absolute()}")
        
    def check_dependencies(self):
        """Check and install required dependencies"""
        print("\n" + "="*60)
        print("DEPENDENCY CHECK")
        print("="*60)
        
        required_packages = {
            'protobuf': 'protobuf',
            'sentencepiece': 'sentencepiece', 
            'sacremoses': 'sacremoses',
            'sacrebleu': 'sacrebleu',
            'rouge_score': 'rouge-score'
        }
        
        missing_packages = []
        
        for package, pip_name in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {package}: Available")
            except ImportError:
                print(f"❌ {package}: Missing")
                missing_packages.append(pip_name)
        
        if missing_packages:
            print(f"\n⚠️ Missing {len(missing_packages)} required packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
            
            print(f"\n📥 Install with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        else:
            print("✅ All required packages are available!")
            return True
    

    def check_system(self):
        """Check system configuration"""
        print("="*60)
        print("SYSTEM CHECK")
        print("="*60)
        
        if self.cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ PyTorch version: {torch.__version__}")
        else:
            print("⚠️ CUDA not available, will use CPU")
            print(f"✅ PyTorch version: {torch.__version__}")
        
        print(f"✅ Device: {self.device}")
        return True

    def check_model_availability(self):
        """Check which models are already downloaded"""
        print("\n" + "="*60)
        print("MODEL AVAILABILITY CHECK")
        print("="*60)
        
        available_models = []
        missing_models = []
        
        for model_key in TRANSLATION_MODELS.keys():
            model_path = MODELS_DIR / model_key
            if model_path.exists() and any(model_path.iterdir()):
                available_models.append(model_key)
                size_mb = sum(f.stat().st_size for f in model_path.rglob('*')) / (1024*1024)
                print(f"✅ {model_key}: Available ({size_mb:.1f} MB)")
            else:
                missing_models.append(model_key)
                print(f"❌ {model_key}: Missing")
        
        print(f"\n📊 Summary: {len(available_models)}/{len(TRANSLATION_MODELS)} models available")
        return available_models, missing_models

    def download_model(self, model_key, model_config):
        """Download a specific model"""
        print(f"\n{'='*60}")
        print(f"DOWNLOADING: {model_key.upper()}")
        print(f"Description: {model_config['description']}")
        print(f"Model: {model_config['model_name']}")
        print(f"{'='*60}")
        
        model_path = MODELS_DIR / model_key
        
        try:
            # Create model directory
            model_path.mkdir(exist_ok=True)
            
            # Download tokenizer
            print("🔥 Downloading tokenizer...")
            tokenizer = model_config['tokenizer_class'].from_pretrained(
                model_config['model_name'],
                cache_dir=str(MODELS_DIR / "cache"),
                local_files_only=False
            )
            tokenizer.save_pretrained(str(model_path))
            print("✅ Tokenizer downloaded")
            
            # Download model
            print("🔥 Downloading model...")
            model = model_config['model_class'].from_pretrained(
                model_config['model_name'],
                torch_dtype=torch.float16 if self.cuda_available else torch.float32,
                cache_dir=str(MODELS_DIR / "cache"),
                local_files_only=False
            )
            model.save_pretrained(str(model_path))
            print("✅ Model downloaded")
            
            # Calculate model size
            model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*')) / (1024*1024)
            print(f"✅ Model size: {model_size_mb:.1f} MB")
            
            # Test the model
            if self.test_model(model_key, model_config, model_path):
                print(f"✅ {model_key} test successful")
                return True
            else:
                print(f"⚠️ {model_key} test failed")
                return False
            
        except Exception as e:
            print(f"❌ Error downloading {model_key}: {e}")
            return False

    def test_model(self, model_key, model_config, model_path):
        """Test if a model can be loaded and used"""
        print(f"🧪 Testing {model_key}...")
        
        try:
            # Load tokenizer and model
            tokenizer = model_config['tokenizer_class'].from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            model = model_config['model_class'].from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.cuda_available else torch.float32,
                local_files_only=True
            )
            
            # Quick test translation
            test_text = "こんにちは"
            
            if "marian" in model_key:
                inputs = tokenizer(test_text, return_tensors="pt", padding=True)
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=50, num_beams=2)
                result = tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                inputs = tokenizer(test_text, return_tensors="pt")
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=50, num_beams=2)
                result = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            print(f"  Test: '{test_text}' -> '{result}'")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    def download_missing_models(self, missing_models):
        """Download all missing models"""
        if not missing_models:
            print("✅ All models are already available!")
            return []
        
        print(f"\n📥 Downloading {len(missing_models)} missing models...")
        
        successful_downloads = []
        
        for i, model_key in enumerate(missing_models):
            print(f"\n[{i+1}/{len(missing_models)}] Processing {model_key}...")
            
            model_config = TRANSLATION_MODELS[model_key]
            if self.download_model(model_key, model_config):
                successful_downloads.append(model_key)
            else:
                print(f"❌ Failed to download {model_key}")
        
        print(f"\n✅ Successfully downloaded: {len(successful_downloads)}/{len(missing_models)} models")
        return successful_downloads

    def load_dataset(self):
        """Load evaluation dataset"""
        print(f"\n{'='*60}")
        print("LOADING EVALUATION DATASET")
        print(f"{'='*60}")
        
        if not DATASET_FILE.exists():
            print(f"❌ Dataset not found: {DATASET_FILE}")
            print("Creating sample dataset for testing...")
            return self.create_sample_dataset()
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(DATASET_FILE, encoding=encoding)
                    print(f"✅ Dataset loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("❌ Could not decode the CSV file")
                return self.create_sample_dataset()
            
            # Check columns
            if 'jp' not in df.columns or 'eng' not in df.columns:
                print(f"❌ Required columns not found. Found: {list(df.columns)}")
                return self.create_sample_dataset()
            
            # Clean data
            df = df.dropna(subset=['jp', 'eng'])
            df['jp'] = df['jp'].astype(str).str.strip()
            df['eng'] = df['eng'].astype(str).str.strip()
            df = df[(df['jp'].str.len() > 1) & (df['eng'].str.len() > 1)]
            
            print(f"✅ Dataset loaded: {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return self.create_sample_dataset()

    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
        print("📝 Creating sample dataset...")
        
        sample_data = [
            {"jp": "こんにちは", "eng": "Hello"},
            {"jp": "ありがとう", "eng": "Thank you"},
            {"jp": "ゲームを始めます", "eng": "Starting the game"},
            {"jp": "アイテムを使用しますか？", "eng": "Do you want to use the item?"},
            {"jp": "レベルアップしました", "eng": "Level up!"},
            {"jp": "敵を倒しました", "eng": "Defeated the enemy"},
            {"jp": "セーブしますか？", "eng": "Do you want to save?"},
            {"jp": "新しいクエストが始まります", "eng": "A new quest begins"},
            {"jp": "体力が少なくなりました", "eng": "Health is low"},
            {"jp": "魔法を覚えました", "eng": "Learned magic"},
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Save sample dataset
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_FILE, index=False, encoding='utf-8')
        
        print(f"✅ Sample dataset created: {len(df)} samples")
        print(f"💾 Saved to: {DATASET_FILE}")
        
        return df

class ModelEvaluator:
    """Class to evaluate translation models"""
    
    def __init__(self, model_key, model_config, device):
        self.model_key = model_key
        self.model_config = model_config
        self.model_path = MODELS_DIR / model_key
        self.model = None
        self.tokenizer = None
        self.device = device
        
    def load_model(self):
        """Load model and tokenizer"""
        try:
            print(f"🔥 Loading {self.model_key}...")
            
            self.tokenizer = self.model_config['tokenizer_class'].from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            self.model = self.model_config['model_class'].from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                local_files_only=True
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"✅ {self.model_key} loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {self.model_key}: {e}")
            return False
    
    def translate_batch(self, texts, max_length=100):
        """Translate a batch of texts"""
        if not self.model or not self.tokenizer:
            return []
        
        translations = []
        
        for text in texts:
            try:
                start_time = time.time()
                
                if "marian" in self.model_key:
                    # Marian models
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if self.device.type == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = self.model.generate(
                            **inputs, 
                            max_length=max_length, 
                            num_beams=4,
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                elif "nllb" in self.model_key:
                    # NLLB models
                    inputs = self.tokenizer(text, return_tensors="pt")
                    
                    if self.device.type == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        try:
                            if hasattr(self.tokenizer, 'lang_code_to_id'):
                                forced_bos_token_id = self.tokenizer.lang_code_to_id[self.model_config['tgt_lang']]
                            else:
                                forced_bos_token_id = 256047  # Fallback for eng_Latn
                        except:
                            forced_bos_token_id = 256047
                        
                        generated = self.model.generate(
                            **inputs,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=max_length,
                            num_beams=2,
                            early_stopping=True,
                            do_sample=False
                        )
                    
                    result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                
                elif "mbart" in self.model_key:
                    # mBART models
                    self.tokenizer.src_lang = self.model_config['src_lang']
                    inputs = self.tokenizer(text, return_tensors="pt")
                    
                    if self.device.type == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            num_beams=2,
                            early_stopping=True,
                            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.model_config['tgt_lang']]
                        )
                    
                    result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                
                else:
                    # Generic fallback (mT5, etc.)
                    inputs = self.tokenizer(text, return_tensors="pt")
                    
                    if self.device.type == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            num_beams=2,
                            early_stopping=True
                        )
                    
                    result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                
                inference_time = time.time() - start_time
                translations.append({
                    'translation': result,
                    'time': inference_time
                })
                
            except Exception as e:
                print(f"❌ Translation error: {e}")
                translations.append({
                    'translation': f"[ERROR: {text}]",
                    'time': 0.0
                })
        
        return translations
    
    def cleanup(self):
        """Clean up model from memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def calculate_metrics(predictions, references):
    """Calculate translation quality metrics"""
    metrics = {}
    
    try:
        # BLEU Score
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        metrics['bleu'] = bleu.score
        
        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        metrics['rouge1'] = np.mean(rouge_scores['rouge1'])
        metrics['rouge2'] = np.mean(rouge_scores['rouge2'])
        metrics['rougeL'] = np.mean(rouge_scores['rougeL'])
        
    except Exception as e:
        print(f"⚠️ Error calculating metrics: {e}")
        metrics = {'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    return metrics

def evaluate_model(model_key, model_config, test_data, device):
    """Evaluate a single model"""
    print(f"\n{'='*50}")
    print(f"EVALUATING: {model_key.upper()}")
    print(f"{'='*50}")
    
    # Check if model exists
    model_path = MODELS_DIR / model_key
    if not model_path.exists():
        print(f"❌ Model {model_key} not found. Skipping...")
        return None
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_key, model_config, device)
    if not evaluator.load_model():
        return None
    
    # Get model size
    model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*')) / (1024*1024)
    
    # Prepare test data
    japanese_texts = test_data['jp'].tolist()
    english_refs = test_data['eng'].tolist()
    
    print(f"📊 Evaluating on {len(japanese_texts)} samples...")
    
    # Translate all texts
    start_time = time.time()
    results = evaluator.translate_batch(japanese_texts)
    total_time = time.time() - start_time
    
    # Extract translations and times
    translations = [r['translation'] for r in results]
    translation_times = [r['time'] for r in results]
    
    # Calculate metrics
    print("📈 Calculating quality metrics...")
    quality_metrics = calculate_metrics(translations, english_refs)
    
    # Performance metrics
    avg_time = np.mean(translation_times)
    total_inference_time = sum(translation_times)
    throughput = len(japanese_texts) / total_inference_time if total_inference_time > 0 else 0
    
    # Compile results
    results = {
        'model_key': model_key,
        'description': model_config['description'],
        'model_size_mb': round(model_size_mb, 1),
        'quality_metrics': quality_metrics,
        'performance_metrics': {
            'avg_translation_time': round(avg_time, 4),
            'total_time': round(total_time, 2),
            'throughput_texts_per_sec': round(throughput, 2),
            'total_samples': len(japanese_texts)
        },
        'sample_translations': [
            {
                'japanese': japanese_texts[i],
                'reference': english_refs[i],
                'translation': translations[i],
                'time': round(translation_times[i], 4)
            }
            for i in range(min(5, len(japanese_texts)))
        ]
    }
    
    # Print summary
    print(f"✅ Model: {model_key}")
    print(f"  Size: {model_size_mb:.1f} MB")
    print(f"  BLEU: {quality_metrics['bleu']:.2f}")
    print(f"  ROUGE-L: {quality_metrics['rougeL']:.3f}")
    print(f"  Avg Time: {avg_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} texts/sec")
    
    # Cleanup
    evaluator.cleanup()
    
    return results

def create_comparison_graphs(all_results):
    """Create comprehensive comparison graphs with clean model names"""
    print(f"\n📊 Generating comparison graphs...")
    
    # Define proper display names for models
    model_display_names = {
        'marian_opus_ja_en': 'Marian OPUS (Standard)',
        'marian_jap_en': 'Marian OPUS (Legacy)', 
        'nllb_600m': 'NLLB 600M (High Quality)',
        'nllb_200m': 'NLLB 200M (Fast)',
        'nllb_1b': 'NLLB 1.3B (Premium)',
        'mbart_large': 'mBART Large (Multilingual)',
        'mt5_base': 'mT5 Base (Google)',
        't5_base': 'T5 Base',
        'opus-mt-ja-en': 'Marian OPUS (Standard)',
        'opus-mt-jap-en': 'Marian OPUS (Legacy)',
        'nllb-200-distilled-600M': 'NLLB 600M (High Quality)',
        'nllb-200-distilled-200M': 'NLLB 200M (Fast)',
        'nllb-200-1.3B': 'NLLB 1.3B (Premium)',
        'mbart-large-50-many-to-many-mmt': 'mBART Large (Multilingual)',
        'mt5-base': 'mT5 Base (Google)'
    }

    # Prepare data with clean names
    models_raw = [r['model_key'] for r in all_results]
    models_clean = [model_display_names.get(model, model.replace('_', ' ').title()) 
                   for model in models_raw]
    sizes = [r['model_size_mb'] for r in all_results]
    bleu_scores = [r['quality_metrics']['bleu'] for r in all_results]
    rouge_scores = [r['quality_metrics']['rougeL'] for r in all_results]
    avg_times = [r['performance_metrics']['avg_translation_time'] for r in all_results]
    throughputs = [r['performance_metrics']['throughput_texts_per_sec'] for r in all_results]
    
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available 
                  else 'seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available 
                  else 'default')
    
    # Modern color palette
    colors = {
        'model_sizes': '#3498db',      # Modern blue
        'bleu_scores': '#2ecc71',      # Modern green  
        'translation_speed': '#e74c3c', # Modern red
        'throughput': '#f39c12',       # Modern orange
        'efficiency': '#9b59b6'        # Modern purple
    }
    
    # Graph configurations with improved styling
    graph_configs = [
        {
            'data': sizes,
            'title': 'Model Size Comparison',
            'ylabel': 'Model Size (MB)',
            'color': colors['model_sizes'],
            'filename': 'model_sizes',
            'format_func': lambda x: f'{x:.0f} MB'
        },
        {
            'data': bleu_scores,
            'title': 'Translation Quality (BLEU Score)',
            'ylabel': 'BLEU Score (Higher is Better)',
            'color': colors['bleu_scores'],
            'filename': 'bleu_scores',
            'format_func': lambda x: f'{x:.1f}'
        },
        {
            'data': avg_times,
            'title': 'Translation Speed',
            'ylabel': 'Average Time per Text (seconds)',
            'color': colors['translation_speed'],
            'filename': 'translation_speed',
            'format_func': lambda x: f'{x:.3f}s'
        },
        {
            'data': throughputs,
            'title': 'Translation Throughput',
            'ylabel': 'Texts per Second (Higher is Better)',
            'color': colors['throughput'],
            'filename': 'throughput',
            'format_func': lambda x: f'{x:.1f}/s'
        }
    ]
    
    # Create individual bar charts with improved styling
    for config in graph_configs:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bars with gradient effect
        bars = ax.bar(range(len(models_clean)), config['data'], 
                     color=config['color'], alpha=0.8, 
                     edgecolor='white', linewidth=1.5,
                     capsize=4)
        
        # Styling improvements
        ax.set_xlabel('Translation Models', fontsize=14, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=14, fontweight='bold')
        ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(models_clean)))
        ax.set_xticklabels(models_clean, rotation=45, ha='right', fontsize=11)
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add value labels on bars with better positioning
        for i, (bar, value) in enumerate(zip(bars, config['data'])):
            height = bar.get_height()
            label = config['format_func'](value)
            
            # Position label above bar with some padding
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + max(config['data'])*0.02, 
                   label, ha='center', va='bottom', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Set y-axis to start from 0 and add some padding at top
        ax.set_ylim(0, max(config['data']) * 1.15)
        
        # Improve layout
        plt.tight_layout()
        
        # Save with high quality
        png_path = COMPARE_DIR / f"{config['filename']}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ {config['title']} saved: {png_path}")
    
    # Enhanced Quality vs Speed scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create scatter plot with size based on model size
    scatter = ax.scatter(avg_times, bleu_scores, 
                        s=[size*3 for size in sizes],  # Scale bubble sizes
                        c=range(len(models_clean)), 
                        cmap='viridis', alpha=0.7, 
                        edgecolors='white', linewidth=2)
    
    # Styling
    ax.set_xlabel('Average Translation Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('BLEU Score (Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Quality vs Speed Trade-off\n(Bubble size represents model size)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add model labels with better positioning
    for i, model in enumerate(models_clean):
        ax.annotate(model, (avg_times[i], bleu_scores[i]), 
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Grid and layout
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add colorbar legend
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Models', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    png_path = COMPARE_DIR / "quality_vs_speed.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Quality vs Speed scatter plot saved: {png_path}")
    
    # Enhanced Efficiency scores
    efficiency = [bleu/time if time > 0 else 0 for bleu, time in zip(bleu_scores, avg_times)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars with gradient colors based on efficiency values
    bars = ax.bar(range(len(models_clean)), efficiency, 
                 color=colors['efficiency'], alpha=0.8,
                 edgecolor='white', linewidth=1.5)
    
    # Color bars based on performance (gradient effect)
    max_eff = max(efficiency)
    for bar, eff in zip(bars, efficiency):
        # Color intensity based on efficiency
        intensity = eff / max_eff
        bar.set_color(plt.cm.RdYlGn(intensity))
    
    ax.set_xlabel('Translation Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Efficiency Score (BLEU/Time)', fontsize=14, fontweight='bold')
    ax.set_title('Overall Efficiency Ranking\n(Higher is Better)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(models_clean)))
    ax.set_xticklabels(models_clean, rotation=45, ha='right', fontsize=11)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels with ranking
    sorted_indices = sorted(range(len(efficiency)), key=lambda i: efficiency[i], reverse=True)
    rankings = {sorted_indices[i]: i+1 for i in range(len(sorted_indices))}
    
    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        rank = rankings[i]
        label = f'#{rank}\n{eff:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2, 
               bar.get_height() + max(efficiency)*0.02, 
               label, ha='center', va='bottom', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Set y-axis limits
    ax.set_ylim(0, max(efficiency) * 1.2)
    
    plt.tight_layout()
    png_path = COMPARE_DIR / "efficiency_scores.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Efficiency ranking saved: {png_path}")
    
    # Create a comprehensive summary dashboard
    create_summary_dashboard(models_clean, sizes, bleu_scores, avg_times, throughputs, efficiency)
    
    return efficiency

def create_summary_dashboard(models, sizes, bleu_scores, avg_times, throughputs, efficiency):
    """Create a comprehensive dashboard with all metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Translation Models Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'][:len(models)]
    
    # 1. Model Sizes (Top Left)
    ax1.bar(models, sizes, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
    ax1.set_title('Model Size (MB)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(sizes):
        ax1.text(i, v + max(sizes)*0.01, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Quality Scores (Top Right)
    ax2.bar(models, bleu_scores, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
    ax2.set_title('Translation Quality (BLEU)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(bleu_scores):
        ax2.text(i, v + max(bleu_scores)*0.01, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Speed (Bottom Left)
    ax3.bar(models, avg_times, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
    ax3.set_title('Translation Speed (seconds)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(avg_times):
        ax3.text(i, v + max(avg_times)*0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Efficiency (Bottom Right)
    bars = ax4.bar(models, efficiency, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
    ax4.set_title('Overall Efficiency (BLEU/Time)', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # Highlight best efficiency
    best_idx = efficiency.index(max(efficiency))
    bars[best_idx].set_color('#27ae60')
    bars[best_idx].set_alpha(1.0)
    
    for i, v in enumerate(efficiency):
        symbol = '★' if i == best_idx else ''
        ax4.text(i, v + max(efficiency)*0.01, f'{v:.1f}{symbol}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    dashboard_path = COMPARE_DIR / "performance_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Performance dashboard saved: {dashboard_path}")

def create_detailed_report(all_results):
    """Create detailed comparison report"""
    print(f"\n📄 Generating detailed report...")
    
    # Create summary table
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model_key'],
            'Description': result['description'],
            'Size (MB)': result['model_size_mb'],
            'BLEU': round(result['quality_metrics']['bleu'], 2),
            'ROUGE-L': round(result['quality_metrics']['rougeL'], 3),
            'Avg Time (s)': round(result['performance_metrics']['avg_translation_time'], 4),
            'Throughput (texts/s)': round(result['performance_metrics']['throughput_texts_per_sec'], 2),
            'Efficiency': round(result['quality_metrics']['bleu'] / result['performance_metrics']['avg_translation_time']
                              if result['performance_metrics']['avg_translation_time'] > 0 else 0, 1)
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary CSV
    csv_path = COMPARE_DIR / "model_comparison_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"✅ Summary table saved: {csv_path}")
    
    # Create detailed report
    report_path = COMPARE_DIR / "detailed_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRANSLATION MODELS COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY TABLE:\n")
        f.write("-"*80 + "\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-"*80 + "\n")
        
        for result in all_results:
            f.write(f"\n{result['model_key'].upper()}:\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"Model Size: {result['model_size_mb']} MB\n")
            f.write(f"Quality Metrics:\n")
            f.write(f"  - BLEU Score: {result['quality_metrics']['bleu']:.2f}\n")
            f.write(f"  - ROUGE-1: {result['quality_metrics']['rouge1']:.3f}\n")
            f.write(f"  - ROUGE-2: {result['quality_metrics']['rouge2']:.3f}\n")
            f.write(f"  - ROUGE-L: {result['quality_metrics']['rougeL']:.3f}\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  - Avg Translation Time: {result['performance_metrics']['avg_translation_time']:.4f}s\n")
            f.write(f"  - Total Time: {result['performance_metrics']['total_time']:.2f}s\n")
            f.write(f"  - Throughput: {result['performance_metrics']['throughput_texts_per_sec']:.2f} texts/sec\n")
            
            f.write(f"Sample Translations:\n")
            for i, sample in enumerate(result['sample_translations'][:3]):
                f.write(f"  {i+1}. JP: {sample['japanese']}\n")
                f.write(f"     REF: {sample['reference']}\n")
                f.write(f"     PRED: {sample['translation']}\n")
                f.write(f"     TIME: {sample['time']:.4f}s\n\n")
            
            f.write("-"*50 + "\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        
        # Find best models for different criteria
        best_quality = max(all_results, key=lambda x: x['quality_metrics']['bleu'])
        best_speed = min(all_results, key=lambda x: x['performance_metrics']['avg_translation_time'])
        best_efficiency = max(all_results, key=lambda x: x['quality_metrics']['bleu'] / x['performance_metrics']['avg_translation_time']
                             if x['performance_metrics']['avg_translation_time'] > 0 else 0)
        smallest_model = min(all_results, key=lambda x: x['model_size_mb'])
        
        f.write(f"Best Quality: {best_quality['model_key']} (BLEU: {best_quality['quality_metrics']['bleu']:.2f})\n")
        f.write(f"Fastest Speed: {best_speed['model_key']} (Time: {best_speed['performance_metrics']['avg_translation_time']:.4f}s)\n")
        f.write(f"Best Efficiency: {best_efficiency['model_key']} (Efficiency: {best_efficiency['quality_metrics']['bleu'] / best_efficiency['performance_metrics']['avg_translation_time']:.1f})\n")
        f.write(f"Smallest Model: {smallest_model['model_key']} (Size: {smallest_model['model_size_mb']:.1f} MB)\n")
        
        f.write(f"\nFor Real-time Gaming Applications:\n")
        f.write(f"- Recommended: {best_efficiency['model_key']} (Best balance of quality and speed)\n")
        f.write(f"- Alternative: {best_speed['model_key']} (If speed is critical)\n")
        f.write(f"- High-quality: {best_quality['model_key']} (If quality is more important than speed)\n")
    
    print(f"✅ Detailed report saved: {report_path}")
    return df_summary

def save_model_info(all_results):
    """Save model information to JSON file"""
    model_info = {
        "models": {},
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "evaluation_date": str(time.strftime("%Y-%m-%d %H:%M:%S")),
        "total_models_evaluated": len(all_results)
    }
    
    # Add results to model info
    for result in all_results:
        model_key = result['model_key']
        model_info["models"][model_key] = {
            "model_name": TRANSLATION_MODELS[model_key]["model_name"],
            "description": result["description"],
            "local_path": str(MODELS_DIR / model_key),
            "size_mb": result["model_size_mb"],
            "quality_metrics": result["quality_metrics"],
            "performance_metrics": result["performance_metrics"],
            "available": True,
            "tested": True
        }
    
    # Add info for models that weren't evaluated
    for model_key, config in TRANSLATION_MODELS.items():
        if model_key not in model_info["models"]:
            model_info["models"][model_key] = {
                "model_name": config["model_name"],
                "description": config["description"],
                "available": False,
                "tested": False
            }
    
    info_path = MODELS_DIR / "model_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Model info saved to {info_path}")

def main():
    """Main unified process"""
    print("="*80)
    print("UNIFIED TRANSLATION MODEL MANAGER")
    print("Downloads missing models + Comprehensive comparison")
    print("="*80)
    
    # Initialize manager
    manager = UnifiedModelManager()
    
    # Step 1: Check dependencies first
    #if not manager.check_dependencies():
    #    print("❌ Missing required dependencies. Please install them first.")
    #    print("Run: pip install protobuf sentencepiece sacremoses sacrebleu rouge-score")
    #    sys.exit(1)
    
    # Step 2: System check
    manager.check_system()
    
    # Step 3: Check model availability
    available_models, missing_models = manager.check_model_availability()
    
    # Step 4: Download missing models
    if missing_models:
        print(f"\n📥 Found {len(missing_models)} missing models:")
        for model in missing_models:
            print(f"  - {model}: {TRANSLATION_MODELS[model]['description']}")
        
        response = input(f"\nDownload {len(missing_models)} missing models? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            newly_downloaded = manager.download_missing_models(missing_models)
            available_models.extend(newly_downloaded)
        else:
            print("⏭️ Skipping downloads, proceeding with available models...")
    
    if not available_models:
        print("❌ No models available for comparison!")
        print("Please download models first or check your models directory.")
        sys.exit(1)
    
    # Step 5: Load dataset
    test_data = manager.load_dataset()
    if test_data is None:
        print("❌ Failed to load dataset!")
        sys.exit(1)
    
    # Step 6: Model comparison
    print(f"\n{'='*60}")
    print("STARTING MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"📋 Evaluating {len(available_models)} models:")
    for model in available_models:
        print(f"  - {model}: {TRANSLATION_MODELS[model]['description']}")
    
    # Evaluate all available models
    all_results = []
    for i, model_key in enumerate(available_models):
        print(f"\n[{i+1}/{len(available_models)}] Evaluating {model_key}...")
        
        model_config = TRANSLATION_MODELS[model_key]
        result = evaluate_model(model_key, model_config, test_data, manager.device)
        
        if result:
            all_results.append(result)
        else:
            print(f"⚠️ Skipping {model_key} due to errors")
    
    if not all_results:
        print("❌ No models evaluated successfully")
        sys.exit(1)
    
    # Step 7: Save raw results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ Raw results saved: {RESULTS_FILE}")
    
    # Step 8: Generate comparison graphs
    try:
        efficiency_scores = create_comparison_graphs(all_results)
        print("✅ Comparison graphs generated successfully")
    except Exception as e:
        print(f"⚠️ Error generating graphs: {e}")
        print("Continuing without graphs...")
    
    # Step 9: Generate detailed report
    try:
        summary_df = create_detailed_report(all_results)
        print("✅ Detailed report generated successfully")
    except Exception as e:
        print(f"⚠️ Error generating report: {e}")
        summary_df = pd.DataFrame()
    
    # Step 10: Save model info
    save_model_info(all_results)
    
    # Step 11: Final summary
    print("\n" + "="*80)
    print("UNIFIED MODEL MANAGER COMPLETE")
    print("="*80)
    print(f"✅ Evaluated {len(all_results)} models successfully")
    print(f"📊 Results saved to: {COMPARE_DIR}")
    print(f"📄 Reports saved to: {COMPARE_DIR}")
    print(f"💾 Model info saved to: {MODELS_DIR}")
    
    if len(summary_df) > 0:
        print("\n📋 Quick Summary:")
        print(summary_df[['Model', 'BLEU', 'Avg Time (s)', 'Efficiency']].to_string(index=False))
        
        # Show recommendations
        if len(all_results) > 0:
            best_efficiency = max(all_results, key=lambda x: x['quality_metrics']['bleu'] / x['performance_metrics']['avg_translation_time']
                                 if x['performance_metrics']['avg_translation_time'] > 0 else 0)
            best_speed = min(all_results, key=lambda x: x['performance_metrics']['avg_translation_time'])
            best_quality = max(all_results, key=lambda x: x['quality_metrics']['bleu'])
            
            print(f"\n🎯 RECOMMENDATIONS:")
            print(f"  📈 Best Overall: {best_efficiency['model_key']}")
            print(f"  ⚡ Fastest: {best_speed['model_key']}")  
            print(f"  🏆 Highest Quality: {best_quality['model_key']}")
    
    print("="*80)
    print("🎮 Ready for gaming translation pipeline!")
    print("Use the comparison results to select the best model for your needs.")
    print("="*80)

if __name__ == "__main__":
    main()