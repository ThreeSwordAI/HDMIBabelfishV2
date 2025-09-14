import pandas as pd
import numpy as np
from pathlib import Path

def reduce_to_100k_samples(input_file, output_file, max_samples=100000):
    """Reduce training dataset to 100K samples for GPU training"""
    
    print(f"Reducing training data from {input_file}")
    print(f"Target size: {max_samples:,} samples")
    
    # Load the large dataset
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Loaded {len(df):,} samples with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("Could not load the file")
        return False
    
    print(f"Original columns: {list(df.columns)}")
    
    # Check if dataset is already small enough
    if len(df) <= max_samples:
        print(f"Dataset already small enough ({len(df):,} samples)")
        df.to_csv(output_file, index=False, encoding='utf-8')
        return True
    
    # Clean data first before sampling
    print("Cleaning data...")
    
    # Handle column names (assuming they're fixed or need fixing)
    if 'jp' not in df.columns or 'eng' not in df.columns:
        if len(df.columns) >= 3:
            # Auto-fix columns: assume first is index, second is English, third is Japanese
            df.columns = ['index', 'eng', 'jp'] + list(df.columns[3:])
            print("Fixed column names automatically")
        else:
            print("Cannot identify Japanese and English columns")
            return False
    
    # Remove empty or invalid rows
    original_size = len(df)
    df = df.dropna(subset=['jp', 'eng'])
    df['jp'] = df['jp'].astype(str).str.strip()
    df['eng'] = df['eng'].astype(str).str.strip()
    df = df[(df['jp'].str.len() > 1) & (df['eng'].str.len() > 1)]
    
    print(f"After cleaning: {len(df):,} samples (removed {original_size - len(df):,} invalid entries)")
    
    # Sample randomly for diversity
    print(f"Randomly sampling {max_samples:,} samples...")
    
    if len(df) > max_samples:
        # Use stratified sampling if possible, otherwise random
        df_sampled = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        df_sampled = df.reset_index(drop=True)
    
    # Final validation
    df_final = df_sampled.dropna(subset=['jp', 'eng'])
    df_final = df_final[(df_final['jp'].str.len() > 1) & (df_final['eng'].str.len() > 1)]
    
    # Keep only necessary columns
    if 'index' in df_final.columns:
        df_final = df_final[['jp', 'eng']].copy()
    else:
        df_final = df_final[['jp', 'eng']].copy()
    
    # Save the reduced dataset
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nDataset reduction complete!")
    print(f"Final dataset saved to: {output_file}")
    print(f"Final size: {len(df_final):,} samples")
    print(f"Reduction ratio: {len(df_final)/original_size*100:.1f}% of original")
    
    # Show sample data
    print(f"\nSample data preview:")
    print(df_final.head())
    
    # Estimate training time
    samples_per_second = 3.5  # Based on your current speed
    estimated_time_hours = len(df_final) / samples_per_second / 3600
    print(f"\nEstimated training time: {estimated_time_hours:.1f} hours")
    
    if estimated_time_hours > 10:
        print("WARNING: Still quite long for training. Consider reducing further.")
    elif estimated_time_hours < 5:
        print("Good: Training time should be manageable.")
    
    return True

def optimize_for_gpu_training():
    """Provide GPU-optimized training suggestions"""
    
    print("\nGPU TRAINING OPTIMIZATION SUGGESTIONS:")
    print("=" * 50)
    print("For 100K samples on GPU, use these training arguments:")
    print("""
    per_device_train_batch_size=8,      # Larger batch for GPU
    gradient_accumulation_steps=4,       # Effective batch = 32
    num_train_epochs=2,                  # Fewer epochs for large dataset
    learning_rate=3e-5,                  # Slightly lower LR
    warmup_steps=500,                    # More warmup for stability
    fp16=True,                          # Use mixed precision
    dataloader_num_workers=4,           # Faster data loading
    save_steps=2000,                    # Less frequent saving
    logging_steps=100,                  # Regular progress updates
    """)
    
    print("Expected training time: 3-8 hours depending on GPU")

def main():
    input_file = Path("../data/translation/train.csv")
    output_file = Path("../data/translation/train_100k.csv")
    
    print("=" * 60)
    print("CREATING 100K SAMPLE TRAINING DATASET")
    print("=" * 60)
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    # Create 100K sample dataset
    success = reduce_to_100k_samples(input_file, output_file, max_samples=100000)
    
    if success:
        optimize_for_gpu_training()
        print(f"\nUpdate your fine-tuning script to use: {output_file}")
        print("\nYou can now restart training with much faster completion time!")
    else:
        print("Failed to create reduced dataset")

if __name__ == "__main__":
    main()