import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
COMPARE_DIR = Path("compare")
JSON_FILE = COMPARE_DIR / "multi_metric_results.json"
CSV_FILE = COMPARE_DIR / "multi_metric_results.csv"

def load_results_data():
    """Load results from JSON or CSV file"""
    
    # Try JSON first
    if JSON_FILE.exists():
        print(f"Loading data from {JSON_FILE}")
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    
    # Fallback to CSV
    elif CSV_FILE.exists():
        print(f"Loading data from {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        
        # Convert CSV back to the expected format
        results = []
        for _, row in df.iterrows():
            result = {
                'model_key': row['Model_Key'],
                'description': row['Model'],
                'model_size_mb': row['Size_MB'],
                'avg_time': row['Avg_Time_Seconds'],
                'color': row['Color'],
                'bleu': row['BLEU_Score'],
                'chrf': row['chrF_Score'],
                'comet': row['COMET_Score'],
                'meteor': row['METEOR_Score']
            }
            results.append(result)
        return results
    
    else:
        print("No results file found!")
        return None

def create_metric_plots(results):
    """Create four separate time vs score plots for each metric"""
    print("Creating metric comparison plots from saved data...")
    
    # Metrics information
    metrics_info = {
        'bleu': {'title': 'BLEU Score', 'ylabel': 'BLEU Score (Higher is Better)'},
        'chrf': {'title': 'chrF Score', 'ylabel': 'chrF Score (Higher is Better)'},
        'comet': {'title': 'COMET Score', 'ylabel': 'COMET Score (Higher is Better)'},
        'meteor': {'title': 'METEOR Score', 'ylabel': 'METEOR Score (Higher is Better)'}
    }
    
    plot_paths = {}
    
    for metric, info in metrics_info.items():
        # Skip if all scores are 0
        if all(r[metric] == 0.0 for r in results):
            print(f"Skipping {metric} plot - no valid scores")
            continue
        
        print(f"Creating {info['title']} plot...")
        
        # Extract data
        models = [r['description'] for r in results]
        times = [r['avg_time'] for r in results]
        scores = [r[metric] for r in results]
        sizes = [r['model_size_mb'] for r in results]
        colors = [r['color'] for r in results]
        
        # Debug: Print actual sizes
        print(f"Model sizes for {metric}:")
        for model, size in zip(models, sizes):
            print(f"  {model}: {size:.1f} MB")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Calculate proper circle sizes
        # Normalize sizes to a reasonable range (e.g., 50-500 for scatter plot)
        min_size = min(sizes)
        max_size = max(sizes)
        size_range = max_size - min_size
        
        if size_range > 0:
            # Scale to 50-500 range based on actual model sizes
            scatter_sizes = [50 + (size - min_size) / size_range * 450 for size in sizes]
        else:
            scatter_sizes = [200] * len(sizes)  # Same size if all equal
        
        # Debug: Print scatter sizes
        print(f"Scatter sizes for {metric}:")
        for model, size, scatter_size in zip(models, sizes, scatter_sizes):
            print(f"  {model}: {size:.1f} MB -> scatter size {scatter_size:.1f}")
        
        # Create scatter plot with corrected sizes
        scatter = plt.scatter(times, scores, 
                             s=scatter_sizes, 
                             c=colors, 
                             alpha=0.7, 
                             edgecolors='white', 
                             linewidth=2)
        
        # Add model labels with model size information
        for i, (model, time, score, size) in enumerate(zip(models, times, scores, sizes)):
            # Create label with actual model size
            label = f"{model}\n({size:.0f}MB)"
            
            plt.annotate(label, 
                        (time, score), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9, 
                        fontweight='bold',
                        ha='left',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
        
        # Styling
        plt.xlabel('Average Translation Time (seconds)', fontsize=14, fontweight='bold')
        plt.ylabel(info['ylabel'], fontsize=14, fontweight='bold')
        plt.title(f'Translation Models: {info["title"]} vs Speed\n(Circle size represents model size)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Grid
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set axis limits with some padding
        x_padding = (max(times) - min(times)) * 0.1
        y_padding = (max(scores) - min(scores)) * 0.1
        
        plt.xlim(min(times) - x_padding, max(times) + x_padding)
        plt.ylim(min(scores) - y_padding, max(scores) + y_padding)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = COMPARE_DIR / f"corrected_comparison_{metric}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        print(f"✅ {info['title']} plot saved: {plot_path}")
        plot_paths[metric] = plot_path
    
    return plot_paths

def create_summary_comparison(results):
    """Create a summary comparison showing all metrics"""
    print("Creating summary comparison...")
    
    # Filter out models with all zero scores
    valid_results = [r for r in results if not all(r[metric] == 0.0 for metric in ['bleu', 'chrf', 'comet', 'meteor'])]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    models = [r['description'] for r in valid_results]
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Translation Models Performance Summary\n(All Metrics)', fontsize=18, fontweight='bold')
    
    metrics = [
        ('bleu', 'BLEU Score', axes[0, 0]),
        ('chrf', 'chrF Score', axes[0, 1]),
        ('comet', 'COMET Score', axes[1, 0]),
        ('meteor', 'METEOR Score', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        scores = [r[metric] for r in valid_results]
        colors = [r['color'] for r in valid_results]
        
        if all(score == 0.0 for score in scores):
            ax.text(0.5, 0.5, f'{title}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        bars = ax.bar(range(len(models)), scores, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Highlight best performer
        best_idx = scores.index(max(scores))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    # Save plot
    summary_path = COMPARE_DIR / "corrected_summary_comparison.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()
    print(f"✅ Summary comparison saved: {summary_path}")
    
    return summary_path

def print_detailed_results(results):
    """Print detailed results table"""
    print("\n" + "="*90)
    print("DETAILED RESULTS FROM SAVED DATA")
    print("="*90)
    
    # Sort by BLEU score (descending)
    sorted_results = sorted(results, key=lambda x: x['bleu'], reverse=True)
    
    print(f"{'Model':<25} {'Size(MB)':<10} {'Time(s)':<8} {'BLEU':<6} {'chrF':<6} {'COMET':<7} {'METEOR':<7}")
    print("-"*90)
    
    for result in sorted_results:
        print(f"{result['description']:<25} "
              f"{result['model_size_mb']:<10.1f} "
              f"{result['avg_time']:<8.4f} "
              f"{result['bleu']:<6.2f} "
              f"{result['chrf']:<6.2f} "
              f"{result['comet']:<7.3f} "
              f"{result['meteor']:<7.3f}")
    
    print("\nKey Observations:")
    
    # Find best in each category
    best_bleu = max(results, key=lambda x: x['bleu'])
    best_chrf = max(results, key=lambda x: x['chrf'])
    best_comet = max(results, key=lambda x: x['comet'])
    best_meteor = max(results, key=lambda x: x['meteor'])
    fastest = min(results, key=lambda x: x['avg_time'])
    
    print(f"• Best BLEU: {best_bleu['description']} ({best_bleu['bleu']:.2f})")
    print(f"• Best chrF: {best_chrf['description']} ({best_chrf['chrf']:.2f})")
    print(f"• Best COMET: {best_comet['description']} ({best_comet['comet']:.3f})")
    print(f"• Best METEOR: {best_meteor['description']} ({best_meteor['meteor']:.3f})")
    print(f"• Fastest: {fastest['description']} ({fastest['avg_time']:.4f}s)")
    
    # Fine-tuning analysis
    finetuned = next((r for r in results if 'Fine-Tuned' in r['description']), None)
    base = next((r for r in results if 'Base' in r['description'] and 'Fine-Tuned' not in r['description']), None)
    
    if finetuned and base:
        print(f"\nFine-tuning Impact:")
        print(f"• BLEU improvement: {finetuned['bleu'] - base['bleu']:+.2f}")
        print(f"• chrF improvement: {finetuned['chrf'] - base['chrf']:+.2f}")
        print(f"• COMET improvement: {finetuned['comet'] - base['comet']:+.3f}")
        print(f"• METEOR improvement: {finetuned['meteor'] - base['meteor']:+.3f}")

def main():
    """Main function to generate graphs from saved results"""
    print("=" * 60)
    print("GENERATING CORRECTED GRAPHS FROM SAVED RESULTS")
    print("=" * 60)
    
    # Load results
    results = load_results_data()
    if not results:
        print("Failed to load results data!")
        return
    
    print(f"Loaded results for {len(results)} models")
    
    # Print detailed results
    print_detailed_results(results)
    
    # Create individual metric plots (without legend, with corrected sizes)
    plot_paths = create_metric_plots(results)
    
    # Create summary comparison
    summary_path = create_summary_comparison(results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("CORRECTED GRAPHS GENERATION COMPLETE")
    print("=" * 60)
    print(f"✅ Generated {len(plot_paths)} individual metric plots")
    print(f"✅ Generated 1 summary comparison plot")
    print(f"📁 All plots saved in: {COMPARE_DIR}")
    
    print("\nGenerated files:")
    for metric, path in plot_paths.items():
        print(f"• {path.name}")
    if summary_path:
        print(f"• {summary_path.name}")

if __name__ == "__main__":
    main()