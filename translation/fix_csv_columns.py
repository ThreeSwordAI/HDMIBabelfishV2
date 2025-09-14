import pandas as pd
from pathlib import Path

def fix_csv_columns(file_path):
    """Fix CSV column names to match expected format"""
    print(f"Fixing CSV column names for: {file_path}")
    
    # Try different encodings
    df = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"File loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("Could not read the CSV file with any encoding")
        return False
    
    # Show current columns
    print(f"Current columns: {list(df.columns)}")
    
    # Rename columns based on the pattern you described
    # 'Unnamed: 0', "it's suliban.", 'スリバン人です'
    # Should become: 'index', 'eng', 'jp'
    
    if len(df.columns) >= 3:
        new_column_names = ['index', 'eng', 'jp']
        df.columns = new_column_names[:len(df.columns)]
        
        print(f"New columns: {list(df.columns)}")
        
        # Save the fixed CSV
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Fixed CSV saved to: {file_path}")
        
        # Show sample data
        print("\nSample data:")
        print(df.head())
        
        return True
    else:
        print(f"Expected at least 3 columns, but found {len(df.columns)}")
        return False

def main():
    train_file = Path("../data/translation/train.csv")
    test_file = Path("../data/translation/test.csv")
    
    print("=" * 50)
    print("FIXING CSV COLUMN NAMES")
    print("=" * 50)
    
    # Fix train.csv
    if train_file.exists():
        print("\nFixing train.csv...")
        fix_csv_columns(train_file)
    else:
        print(f"Train file not found: {train_file}")
    
    # Fix test.csv if it exists
    if test_file.exists():
        print("\nFixing test.csv...")
        fix_csv_columns(test_file)
    else:
        print(f"Test file not found: {test_file}")
    
    print("\n" + "=" * 50)
    print("COLUMN FIXING COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()