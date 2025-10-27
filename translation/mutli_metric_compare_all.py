import os
import time
import json
import pandas as pd
import numpy as np
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import matplotlib.pyplot as plt
import sacrebleu
from sacrebleu.metrics import CHRF
import warnings
warnings.filterwarnings("ignore")

# Try to import COMET and METEOR
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
    print("COMET available for evaluation")
except ImportError:
    COMET_AVAILABLE = False
    print("COMET not available. Install with: pip install unbabel-comet")

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    METEOR_AVAILABLE = True
    print("METEOR available for evaluation")
except ImportError:
    METEOR_AVAILABLE = False
    print("METEOR not available. Install with: pip install nltk")

# Configuration
MODELS_DIR = Path("models")
COMPARE_DIR = Path("compare")
DATA_DIR = Path("../data/translation")
TEST_FILE = DATA_DIR / "test.csv"

# Model configurations
TRANSLATION_MODELS = {
    "nllb_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "local_path": MODELS_DIR / "nllb_600m",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "type": "nllb",
        "src_lang": "jpn_Jpan",
        "tgt_lang": "eng_Latn",
        "description": "NLLB 600M (High Quality)",
        "color": "#2ecc71"
    },
    "marian_opus_ja_en": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "local_path": MODELS_DIR / "marian_opus_ja_en",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "type": "marian",
        "description": "Marian OPUS (Base)",
        "color": "#3498db"
    },
    "marian_opus_ft_ja_en_100K_Data": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "local_path": MODELS_DIR / "marian_opus_ft_ja_en_100K_Data",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "type": "marian",
        "description": "Marian OPUS (Fine-Tuned Small)",
        "color": "#e74c3c"
    },
    "marian_opus_ft_ja_en_full_dataset": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "local_path": MODELS_DIR / "marian_opus_ft_ja_en_full_dataset",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "type": "marian",
        "description": "Marian OPUS (Fine-Tuned Large)",
        "color": "#e67e22"
    },
    "marian_jap_en": {
        "model_name": "Helsinki-NLP/opus-mt-jap-en",
        "local_path": MODELS_DIR / "marian_jap_en",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "type": "marian",
        "description": "Marian OPUS (Legacy)",
        "color": "#95a5a6"
    },
    "mbart_large": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "local_path": MODELS_DIR / "mbart_large",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "type": "mbart",
        "src_lang": "ja_XX",
        "tgt_lang": "en_XX",
        "description": "mBART Large 50",
        "color": "#9b59b6"
    },
    "mt5_base": {
        "model_name": "google/mt5-base",
        "local_path": MODELS_DIR / "mt5_base",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "type": "mt5",
        "description": "mT5 Base",
        "color": "#1abc9c"
    }
}

# Japanese character detection
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3000-\u303f]")
ASCII_RE = re.compile(r"^[\x00-\x7F]+$")

class MultiMetricComparator:
    """Compare translation models using multiple metrics"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()
        self.setup_directories()
        self.comet_model = None
        if COMET_AVAILABLE:
            self.load_comet_model()
        
    def setup_directories(self):
        """Create necessary directories"""
        COMPARE_DIR.mkdir(exist_ok=True)
        print(f"Comparison directory ready: {COMPARE_DIR}")
        
    def load_comet_model(self):
        """Load COMET model for evaluation"""
        try:
            print("Loading COMET model...")
            model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(model_path)
            print("COMET model loaded successfully")
        except Exception as e:
            print(f"Failed to load COMET model: {e}")
            self.comet_model = None
        
    def is_mostly_japanese(self, text, threshold=0.05):
        """Check if text is mostly Japanese"""
        if not isinstance(text, str) or not text.strip():
            return False
        jp_chars = len(JP_CHAR_RE.findall(text))
        return (jp_chars / max(len(text), 1)) >= threshold
    
    def detect_jp_en_columns(self, df):
        """Properly detect Japanese and English columns"""
        text_cols = [c for c in df.columns if df[c].dtype == object]
        
        jp_scores = {}
        for col in text_cols:
            jp_ratio = df[col].astype(str).apply(self.is_mostly_japanese).mean()
            jp_scores[col] = jp_ratio
        
        jp_col = max(jp_scores, key=jp_scores.get)
        
        ascii_scores = {}
        for col in text_cols:
            if col != jp_col:
                ascii_ratio = df[col].astype(str).apply(
                    lambda s: bool(ASCII_RE.match(s.strip()))
                ).mean()
                ascii_scores[col] = ascii_ratio
        
        en_col = max(ascii_scores, key=ascii_scores.get)
        
        print(f"Detected columns - Japanese: '{jp_col}', English: '{en_col}'")
        return jp_col, en_col
        
    def check_model_availability(self):
        """Check which models are available locally"""
        print("Checking model availability...")
        available_models = []
        
        for model_key, model_config in TRANSLATION_MODELS.items():
            local_path = model_config.get("local_path")
            
            if local_path and local_path.exists() and any(local_path.iterdir()):
                size_mb = sum(f.stat().st_size for f in local_path.rglob('*')) / (1024*1024)
                print(f"✅ {model_key}: Available ({size_mb:.1f} MB)")
                available_models.append(model_key)
            else:
                print(f"❌ {model_key}: Not available (skipping)")
        
        print(f"\nFound {len(available_models)} available models")
        return available_models
    
    def load_test_data(self):
        """Load test data with proper column detection"""
        print(f"Loading test data from {TEST_FILE}...")
        
        if not TEST_FILE.exists():
            print(f"Test file not found: {TEST_FILE}")
            return None
        
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(TEST_FILE, encoding=encoding)
                    print(f"Test data loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("Could not decode test CSV file")
                return None
            
            print(f"Original columns: {list(df.columns)}")
            
            # Smart column detection
            candidates = [c.lower() for c in df.columns]
            if {"jp", "eng"}.issubset(set(candidates)):
                jp_col = df.columns[candidates.index("jp")]
                en_col = df.columns[candidates.index("eng")]
                print("Found exact 'jp' and 'eng' columns")
            else:
                jp_col, en_col = self.detect_jp_en_columns(df)
            
            # Clean data
            df = df.dropna(subset=[jp_col, en_col])
            df[jp_col] = df[jp_col].astype(str).str.strip()
            df[en_col] = df[en_col].astype(str).str.strip()
            df = df[(df[jp_col].str.len() > 1) & (df[en_col].str.len() > 1)]
            
            # Rename for consistency
            df = df.rename(columns={jp_col: 'jp', en_col: 'eng'})
            
            print(f"Loaded {len(df)} test samples")
            return df
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            return None
    
    def load_model(self, model_key, model_config):
        """Load model with proper path resolution"""
        try:
            print(f"Loading {model_key}...")
            
            local_path = model_config.get("local_path")
            
            if local_path and local_path.exists():
                model_path = str(local_path)
                local_files_only = True
                print(f"  Using local path: {model_path}")
            else:
                model_path = model_config["model_name"]
                local_files_only = False
                print(f"  Using HuggingFace: {model_path}")
            
            tokenizer = model_config['tokenizer_class'].from_pretrained(
                model_path,
                local_files_only=local_files_only
            )
            
            model = model_config['model_class'].from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.cuda_available else torch.float32,
                local_files_only=local_files_only
            )
            
            if self.cuda_available:
                model = model.to(self.device)
            
            model.eval()
            print(f"✅ {model_key} loaded successfully")
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ Error loading {model_key}: {e}")
            return None, None
    
    def translate_batch(self, model_key, model_config, tokenizer, model, texts):
        """Translate with proper model-specific handling"""
        translations = []
        translation_times = []
        
        model_type = model_config.get("type", "generic")
        
        for text in texts:
            try:
                start_time = time.time()
                
                if model_type == "marian":
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if self.cuda_available:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=120,
                            num_beams=4,
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    result = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                elif model_type == "nllb":
                    tokenizer.src_lang = model_config["src_lang"]
                    inputs = tokenizer(text, return_tensors="pt")
                    if self.cuda_available:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        if hasattr(tokenizer, 'lang_code_to_id'):
                            forced_bos_token_id = tokenizer.lang_code_to_id[model_config["tgt_lang"]]
                        elif hasattr(tokenizer, 'convert_tokens_to_ids'):
                            forced_bos_token_id = tokenizer.convert_tokens_to_ids(model_config["tgt_lang"])
                        else:
                            forced_bos_token_id = 256047
                        
                        generated = model.generate(
                            **inputs,
                            forced_bos_token_id=forced_bos_token_id,
                            max_new_tokens=120,
                            num_beams=4,
                            early_stopping=True,
                            do_sample=False
                        )
                    result = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                elif model_type == "mbart":
                    tokenizer.src_lang = model_config["src_lang"]
                    inputs = tokenizer(text, return_tensors="pt")
                    if self.cuda_available:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=120,
                            num_beams=4,
                            early_stopping=True,
                            forced_bos_token_id=tokenizer.lang_code_to_id[model_config["tgt_lang"]]
                        )
                    result = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                elif model_type == "mt5":
                    prefixed_text = f"translate Japanese to English: {text}"
                    inputs = tokenizer(prefixed_text, return_tensors="pt")
                    if self.cuda_available:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=120,
                            num_beams=4,
                            early_stopping=True
                        )
                    result = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                else:
                    inputs = tokenizer(text, return_tensors="pt")
                    if self.cuda_available:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=120,
                            num_beams=4,
                            early_stopping=True
                        )
                    result = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                inference_time = time.time() - start_time
                translations.append(result.strip())
                translation_times.append(inference_time)
                
            except Exception as e:
                print(f"Translation error for {model_key}: {e}")
                translations.append("[ERROR]")
                translation_times.append(0.1)
        
        return translations, translation_times
    
    def calculate_metrics(self, predictions, references, source_texts):
        """Calculate all metrics: BLEU, chrF, COMET, METEOR"""
        metrics = {}
        
        # BLEU Score
        try:
            bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="13a")
            metrics['bleu'] = bleu.score
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            metrics['bleu'] = 0.0
        
        # chrF Score
        try:
            chrf = CHRF()
            chrf_score = chrf.corpus_score(predictions, [references])
            metrics['chrf'] = chrf_score.score
        except Exception as e:
            print(f"chrF calculation error: {e}")
            metrics['chrf'] = 0.0
        
        # COMET Score
        if COMET_AVAILABLE and self.comet_model:
            try:
                data = []
                for src, pred, ref in zip(source_texts, predictions, references):
                    data.append({
                        "src": src,
                        "mt": pred,
                        "ref": ref
                    })
                
                model_output = self.comet_model.predict(data, batch_size=8, gpus=1 if self.cuda_available else 0)
                metrics['comet'] = np.mean(model_output.scores)
            except Exception as e:
                print(f"COMET calculation error: {e}")
                metrics['comet'] = 0.0
        else:
            metrics['comet'] = 0.0
        
        # METEOR Score
        if METEOR_AVAILABLE:
            try:
                meteor_scores = []
                for pred, ref in zip(predictions, references):
                    # METEOR expects tokenized input
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()
                    score = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores.append(score)
                metrics['meteor'] = np.mean(meteor_scores)
            except Exception as e:
                print(f"METEOR calculation error: {e}")
                metrics['meteor'] = 0.0
        else:
            metrics['meteor'] = 0.0
        
        return metrics
    
    def evaluate_model(self, model_key, model_config, test_data):
        """Evaluate a single model with all metrics"""
        print(f"\nEvaluating {model_key}...")
        
        # Load model
        tokenizer, model = self.load_model(model_key, model_config)
        if tokenizer is None or model is None:
            return None
        
        # Get model size
        local_path = model_config.get("local_path")
        if local_path and local_path.exists():
            model_size_mb = sum(f.stat().st_size for f in local_path.rglob('*')) / (1024*1024)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = (total_params * 2) / (1024*1024)
        
        # Prepare test data
        japanese_texts = test_data['jp'].tolist()
        english_refs = test_data['eng'].tolist()
        
        print(f"Testing on {len(japanese_texts)} samples...")
        
        # Translate
        translations, translation_times = self.translate_batch(
            model_key, model_config, tokenizer, model, japanese_texts
        )
        
        # Calculate all metrics
        print(f"Calculating metrics for {model_key}...")
        metrics = self.calculate_metrics(translations, english_refs, japanese_texts)
        avg_time = np.mean(translation_times)
        
        # Show sample translations
        print(f"Sample translations for {model_key}:")
        for i in range(min(2, len(translations))):
            print(f"  JP: {japanese_texts[i]}")
            print(f"  REF: {english_refs[i]}")
            print(f"  PRED: {translations[i]}")
        print()
        
        # Cleanup
        del model, tokenizer
        if self.cuda_available:
            torch.cuda.empty_cache()
        
        result = {
            'model_key': model_key,
            'description': model_config['description'],
            'model_size_mb': model_size_mb,
            'avg_time': avg_time,
            'color': model_config['color'],
            **metrics  # Include all metric scores
        }
        
        print(f"✅ {model_key}: BLEU={metrics['bleu']:.2f}, chrF={metrics['chrf']:.2f}, "
              f"COMET={metrics['comet']:.3f}, METEOR={metrics['meteor']:.3f}, "
              f"Time={avg_time:.4f}s, Size={model_size_mb:.1f}MB")
        
        return result
    
    def create_metric_plots(self, results):
        """Create four separate time vs score plots for each metric"""
        print("\nCreating metric comparison plots...")
        
        metrics_info = {
            'bleu': {'title': 'BLEU Score', 'higher_better': True},
            'chrf': {'title': 'chrF Score', 'higher_better': True},
            'comet': {'title': 'COMET Score', 'higher_better': True},
            'meteor': {'title': 'METEOR Score', 'higher_better': True}
        }
        
        plot_paths = {}
        
        for metric, info in metrics_info.items():
            if all(r[metric] == 0.0 for r in results):
                print(f"Skipping {metric} plot - no valid scores")
                continue
                
            # Extract data
            models = [r['description'] for r in results]
            times = [r['avg_time'] for r in results]
            scores = [r[metric] for r in results]
            sizes = [r['model_size_mb'] for r in results]
            colors = [r['color'] for r in results]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            scatter_sizes = [size * 3 for size in sizes]
            
            scatter = plt.scatter(times, scores, 
                                 s=scatter_sizes, 
                                 c=colors, 
                                 alpha=0.7, 
                                 edgecolors='white', 
                                 linewidth=2)
            
            # Add model labels
            for i, (model, time, score) in enumerate(zip(models, times, scores)):
                plt.annotate(model, 
                            (time, score), 
                            xytext=(8, 8), 
                            textcoords='offset points',
                            fontsize=10, 
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Styling
            plt.xlabel('Average Translation Time (seconds)', fontsize=14, fontweight='bold')
            plt.ylabel(f'{info["title"]} ({"Higher is Better" if info["higher_better"] else "Lower is Better"})', 
                      fontsize=14, fontweight='bold')
            plt.title(f'Translation Models: {info["title"]} vs Speed\n(Circle size represents model size)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Grid
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Legend in top-right corner
            legend_elements = []
            for result in results:
                legend_elements.append(plt.Line2D([0], [0], 
                                                marker='o', 
                                                color='w', 
                                                markerfacecolor=result['color'], 
                                                markersize=8, 
                                                label=f"{result['description']} ({result['model_size_mb']:.0f}MB)",
                                                alpha=0.7))
            
            plt.legend(handles=legend_elements, 
                      loc='upper right',
                      fontsize=10,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      framealpha=0.9)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = COMPARE_DIR / f"model_comparison_{metric}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.show()
            print(f"✅ {info['title']} plot saved: {plot_path}")
            plot_paths[metric] = plot_path
        
        return plot_paths
    
    def save_results_to_csv(self, results):
        """Save all results to CSV for further analysis"""
        print("\nSaving results to CSV...")
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            csv_data.append({
                'Model': result['description'],
                'Model_Key': result['model_key'],
                'Size_MB': result['model_size_mb'],
                'Avg_Time_Seconds': result['avg_time'],
                'BLEU_Score': result['bleu'],
                'chrF_Score': result['chrf'],
                'COMET_Score': result['comet'],
                'METEOR_Score': result['meteor'],
                'Color': result['color']
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = COMPARE_DIR / "multi_metric_results.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Results saved to CSV: {csv_path}")
        
        # Also save detailed results as JSON
        json_path = COMPARE_DIR / "multi_metric_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Detailed results saved to JSON: {json_path}")
        
        return csv_path, df

def main():
    """Main comparison process"""
    print("=" * 70)
    print("MULTI-METRIC TRANSLATION MODEL COMPARISON")
    print("BLEU | chrF | COMET | METEOR")
    print("=" * 70)
    
    # Initialize comparator
    comparator = MultiMetricComparator()
    
    # Check system
    if comparator.cuda_available:
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Check metric availability
    available_metrics = ['BLEU', 'chrF']
    if COMET_AVAILABLE and comparator.comet_model:
        available_metrics.append('COMET')
    if METEOR_AVAILABLE:
        available_metrics.append('METEOR')
    
    print(f"Available metrics: {', '.join(available_metrics)}")
    
    # Check available models
    available_models = comparator.check_model_availability()
    if not available_models:
        print("No models available for comparison!")
        return
    
    # Load test data
    test_data = comparator.load_test_data()
    if test_data is None:
        print("Failed to load test data!")
        return
    
    # Evaluate models
    print(f"\nEvaluating {len(available_models)} models with {len(available_metrics)} metrics...")
    results = []
    
    for i, model_key in enumerate(available_models):
        print(f"\n[{i+1}/{len(available_models)}] Processing {model_key}...")
        model_config = TRANSLATION_MODELS[model_key]
        
        result = comparator.evaluate_model(model_key, model_config, test_data)
        if result:
            results.append(result)
    
    if not results:
        print("No models evaluated successfully!")
        return
    
    # Save results to CSV
    csv_path, df = comparator.save_results_to_csv(results)
    
    # Create metric plots
    plot_paths = comparator.create_metric_plots(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("MULTI-METRIC COMPARISON COMPLETE")
    print("=" * 70)
    print(f"✅ Evaluated {len(results)} models")
    print(f"📊 Generated {len(plot_paths)} metric plots")
    print(f"📁 Results directory: {COMPARE_DIR}")
    print(f"📈 CSV results: {csv_path}")
    
    # Show summary table
    print(f"\nSummary Table:")
    print("Model".ljust(25) + "BLEU".ljust(8) + "chrF".ljust(8) + "COMET".ljust(8) + "METEOR".ljust(8) + "Time(s)".ljust(10))
    print("-" * 75)
    for result in sorted(results, key=lambda x: x['bleu'], reverse=True):
        print(f"{result['description'][:24].ljust(25)}{result['bleu']:.2f}".ljust(8) + 
              f"{result['chrf']:.2f}".ljust(8) + 
              f"{result['comet']:.3f}".ljust(8) +
              f"{result['meteor']:.3f}".ljust(8) +
              f"{result['avg_time']:.4f}".ljust(10))
    
    print(f"\n📊 Use the CSV file to create custom visualizations and analysis!")

if __name__ == "__main__":
    main()