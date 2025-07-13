"""
Compare translation models for accuracy and speed
Evaluates all downloaded models on test dataset and generates comparison graphs
"""

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

# Model configurations (updated with fixes)
TRANSLATION_MODELS = {
    "marian_opus_ja_en": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "src_lang": "ja",
        "tgt_lang": "en",
        "description": "Marian OPUS (Fast, Small)"
    },
    "marian_jap_en": {
        "model_name": "Helsinki-NLP/opus-mt-jap-en", 
        "tokenizer_class": MarianTokenizer,
        "model_class": MarianMTModel,
        "src_lang": "jap",
        "tgt_lang": "en",
        "description": "Marian Alternative"
    },
    "nllb_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
        "src_lang": "jpn_Jpan",
        "tgt_lang": "eng_Latn",
        "description": "NLLB 600M (High Quality)"
    }
}

class ModelEvaluator:
    """Class to evaluate translation models"""
    
    def __init__(self, model_key, model_config):
        self.model_key = model_key
        self.model_config = model_config
        self.model_path = MODELS_DIR / model_key
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load model and tokenizer"""
        try:
            print(f"📥 Loading {self.model_key}...")
            
            # Load tokenizer
            self.tokenizer = self.model_config['tokenizer_class'].from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            # Load model
            self.model = self.model_config['model_class'].from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                local_files_only=True
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"✓ {self.model_key} loaded on {self.device}")
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
                
                # Debug: Print first few characters of input
                if len(translations) < 3:  # Only for first 3 samples
                    print(f"  🔍 Input ({self.model_key}): '{text[:50]}...'")
                
                if "marian" in self.model_key:
                    # Marian models - simple tokenization
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if self.device.type == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = self.model.generate(
                            **inputs, 
                            max_length=max_length, 
                            num_beams=4,  # Increased beams for better quality
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                elif "nllb" in self.model_key:
                    # NLLB models - special handling with fallback
                    inputs = self.tokenizer(text, return_tensors="pt")
                    
                    if self.device.type == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        # Try different methods for language code
                        try:
                            if hasattr(self.tokenizer, 'lang_code_to_id'):
                                forced_bos_token_id = self.tokenizer.lang_code_to_id[self.model_config['tgt_lang']]
                            elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.model_config['tgt_lang'])
                            else:
                                # Fallback: find English token ID manually
                                forced_bos_token_id = 256047  # Common eng_Latn token ID for NLLB
                        except:
                            forced_bos_token_id = 256047  # Fallback
                        
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
                    # mBART models - special tokenization
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
                    # Generic fallback
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
                
                # Debug: Print result for first few samples
                if len(translations) < 3:
                    print(f"  ✅ Output ({self.model_key}): '{result}'")
                
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

def load_dataset():
    """Load evaluation dataset"""
    try:
        if not DATASET_FILE.exists():
            print(f"❌ Dataset not found: {DATASET_FILE}")
            print("Please ensure test.csv exists in data/translation/")
            return None
        
        # Load CSV file with proper encoding
        try:
            df = pd.read_csv(DATASET_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            # Try different encodings if utf-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(DATASET_FILE, encoding=encoding)
                    print(f"✓ Dataset loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("❌ Could not decode the CSV file with any common encoding")
                return None
        
        # Check if required columns exist
        if 'jp' not in df.columns or 'eng' not in df.columns:
            print(f"❌ Required columns not found. Expected 'jp' and 'eng', found: {list(df.columns)}")
            return None
        
        # Clean the data
        df = df.dropna(subset=['jp', 'eng'])  # Remove rows with missing values
        df['jp'] = df['jp'].astype(str).str.strip()  # Clean Japanese text
        df['eng'] = df['eng'].astype(str).str.strip()  # Clean English text
        
        # Remove empty or very short entries
        df = df[(df['jp'].str.len() > 1) & (df['eng'].str.len() > 1)]
        
        print(f"✓ Dataset loaded: {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:")
        for i in range(min(3, len(df))):
            print(f"  JP: {df.iloc[i]['jp']}")
            print(f"  EN: {df.iloc[i]['eng']}")
            print()
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

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
        print(f"⚠ Error calculating metrics: {e}")
        metrics = {'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    return metrics

def evaluate_model(model_key, model_config, test_data):
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
    evaluator = ModelEvaluator(model_key, model_config)
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
            for i in range(min(5, len(japanese_texts)))  # First 5 samples
        ]
    }
    
    # Print summary
    print(f"✓ Model: {model_key}")
    print(f"  Size: {model_size_mb:.1f} MB")
    print(f"  BLEU: {quality_metrics['bleu']:.2f}")
    print(f"  ROUGE-L: {quality_metrics['rougeL']:.3f}")
    print(f"  Avg Time: {avg_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} texts/sec")
    
    # Cleanup
    evaluator.cleanup()
    
    return results

def create_comparison_graphs(all_results):
    """Create separate comparison graphs"""
    print(f"\n📊 Generating comparison graphs...")
    
    # Prepare data for plotting
    models = []
    sizes = []
    bleu_scores = []
    rouge_scores = []
    avg_times = []
    throughputs = []
    descriptions = []
    
    for result in all_results:
        models.append(result['model_key'])
        sizes.append(result['model_size_mb'])
        bleu_scores.append(result['quality_metrics']['bleu'])
        rouge_scores.append(result['quality_metrics']['rougeL'])
        avg_times.append(result['performance_metrics']['avg_translation_time'])
        throughputs.append(result['performance_metrics']['throughput_texts_per_sec'])
        descriptions.append(result['description'])
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create individual graphs
    graph_configs = [
        {
            'data': sizes,
            'title': 'Model Size Comparison',
            'ylabel': 'Model Size (MB)',
            'color': 'skyblue',
            'filename': 'model_sizes'
        },
        {
            'data': bleu_scores,
            'title': 'Translation Quality (BLEU Score)',
            'ylabel': 'BLEU Score',
            'color': 'lightgreen',
            'filename': 'bleu_scores'
        },
        {
            'data': avg_times,
            'title': 'Translation Speed (Lower is Better)',
            'ylabel': 'Average Time (seconds)',
            'color': 'salmon',
            'filename': 'translation_speed'
        },
        {
            'data': throughputs,
            'title': 'Translation Throughput (Higher is Better)',
            'ylabel': 'Throughput (texts/sec)',
            'color': 'gold',
            'filename': 'throughput'
        }
    ]
    
    # Create bar charts
    for config in graph_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(models)), config['data'], 
                     color=config['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, config['data'])):
            height = bar.get_height()
            if config['filename'] == 'model_sizes':
                label = f'{value:.1f}MB'
            elif config['filename'] == 'bleu_scores':
                label = f'{value:.1f}'
            elif config['filename'] == 'translation_speed':
                label = f'{value:.3f}s'
            else:  # throughput
                label = f'{value:.1f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, height + max(config['data'])*0.01, 
                   label, ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save individual graphs
        png_path = COMPARE_DIR / f"{config['filename']}.png"
        pdf_path = COMPARE_DIR / f"{config['filename']}.pdf"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {config['title']} saved: {png_path}")
    
    # 5. Quality vs Speed Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(avg_times, bleu_scores, s=[size*2 for size in sizes], 
                        c=range(len(models)), cmap='viridis', alpha=0.7, edgecolors='black')
    ax.set_xlabel('Average Translation Time (seconds)', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Quality vs Speed Trade-off\n(Bubble size = Model size)', fontsize=14, fontweight='bold')
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (avg_times[i], bleu_scores[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    png_path = COMPARE_DIR / "quality_vs_speed.png"
    pdf_path = COMPARE_DIR / "quality_vs_speed.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Quality vs Speed scatter plot saved: {png_path}")
    
    # 6. Efficiency Score (Quality/Time ratio)
    efficiency = [bleu/time if time > 0 else 0 for bleu, time in zip(bleu_scores, avg_times)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(models)), efficiency, color='mediumpurple', alpha=0.7, 
                 edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Efficiency (BLEU/Time)', fontsize=12)
    ax.set_title('Overall Efficiency Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.01, 
               f'{eff:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    png_path = COMPARE_DIR / "efficiency_scores.png"
    pdf_path = COMPARE_DIR / "efficiency_scores.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Efficiency scores saved: {png_path}")
    
    print(f"\n✅ All 6 graphs saved as separate PNG and PDF files in: {COMPARE_DIR}")
    
    return efficiency

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
    print(f"✓ Summary table saved: {csv_path}")
    
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
    
    print(f"✓ Detailed report saved: {report_path}")
    
    return df_summary

def main():
    """Main comparison process"""
    print("="*60)
    print("TRANSLATION MODELS COMPARISON")
    print("="*60)
    
    # Check prerequisites
    if not MODELS_DIR.exists():
        print("❌ Models directory not found!")
        print("Please run 'python download_models.py' first")
        sys.exit(1)
    
    # Create compare directory
    COMPARE_DIR.mkdir(exist_ok=True)
    
    # Load dataset
    test_data = load_dataset()
    if test_data is None:
        sys.exit(1)
    
    # Check which models are available
    available_models = []
    for model_key in TRANSLATION_MODELS.keys():
        model_path = MODELS_DIR / model_key
        if model_path.exists():
            available_models.append(model_key)
    
    if not available_models:
        print("❌ No models found!")
        print("Please run 'python download_models.py' first")
        sys.exit(1)
    
    print(f"📋 Found {len(available_models)} models to evaluate:")
    for model in available_models:
        print(f"  - {model}")
    
    # Evaluate all models
    all_results = []
    for i, model_key in enumerate(available_models):
        print(f"\n[{i+1}/{len(available_models)}] Evaluating {model_key}...")
        
        model_config = TRANSLATION_MODELS[model_key]
        result = evaluate_model(model_key, model_config, test_data)
        
        if result:
            all_results.append(result)
        else:
            print(f"⚠ Skipping {model_key} due to errors")
    
    if not all_results:
        print("❌ No models evaluated successfully")
        sys.exit(1)
    
    # Save raw results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Raw results saved: {RESULTS_FILE}")
    
    # Generate comparison graphs
    efficiency_scores = create_comparison_graphs(all_results)
    
    # Generate detailed report
    summary_df = create_detailed_report(all_results)
    
    # Print final summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"✅ Evaluated {len(all_results)} models successfully")
    print(f"📊 Graphs saved to: {COMPARE_DIR}")
    print(f"📄 Reports saved to: {COMPARE_DIR}")
    print("\n📋 Quick Summary:")
    print(summary_df[['Model', 'BLEU', 'Avg Time (s)', 'Efficiency']].to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()