"""
Filter Alternate Titles by O*NET-SOC Code patterns
Filters codes beginning with: 15-12xx, 17-20xx, and 15-20xx
"""

import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
ARCHIVE_DIR = BASE_DIR / "archive"
OUTPUT_DIR = Path(__file__).parent

# File paths
INPUT_FILE = ARCHIVE_DIR / "Alternate Titles.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "alternate_titles_filtered.csv"

# Code patterns to filter (matching beginning of code)
CODE_PATTERNS = [
    '15-12',  # 15-12xx codes
    '17-20',  # 17-20xx codes
    '15-20'   # 15-20xx codes
]

def filter_alternate_titles():
    """Filter alternate titles by O*NET-SOC code patterns"""
    print("=" * 80)
    print("Filtering Alternate Titles by O*NET-SOC Code")
    print("=" * 80)
    
    # Load the Excel file
    print(f"\n1. Loading {INPUT_FILE.name}...")
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"   [OK] Loaded {len(df):,} rows")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   [ERROR] Error loading file: {e}")
        return
    
    # Check the code column
    code_col = 'O*NET-SOC Code'
    if code_col not in df.columns:
        print(f"   [ERROR] Column '{code_col}' not found")
        return
    
    print(f"\n2. Filtering by code patterns: {CODE_PATTERNS}")
    
    # Convert codes to string and extract prefix (first 5 characters: "15-12", "17-20", etc.)
    df['code_prefix'] = df[code_col].astype(str).str[:5]
    
    # Filter rows where code prefix matches any of our patterns
    mask = df['code_prefix'].isin(CODE_PATTERNS)
    filtered_df = df[mask].copy()
    
    # Drop the temporary code_prefix column
    filtered_df = filtered_df.drop(columns=['code_prefix'])
    
    print(f"   [OK] Filtered to {len(filtered_df):,} rows")
    print(f"   Removed {len(df) - len(filtered_df):,} rows")
    
    # Show breakdown by pattern
    print(f"\n3. Breakdown by code pattern:")
    for pattern in CODE_PATTERNS:
        count = len(df[df['code_prefix'] == pattern])
        print(f"   {pattern}xx: {count:,} rows")
    
    # Show unique codes in filtered data
    print(f"\n4. Unique codes in filtered data:")
    unique_codes = filtered_df[code_col].unique()
    print(f"   Total unique codes: {len(unique_codes)}")
    print(f"   Sample codes: {sorted(unique_codes)[:10]}")
    
    # Save to CSV
    print(f"\n5. Saving filtered data...")
    filtered_df.to_csv(OUTPUT_FILE, index=False)
    print(f"   [OK] Saved to: {OUTPUT_FILE}")
    print(f"   Rows: {len(filtered_df):,}")
    print(f"   Columns: {list(filtered_df.columns)}")
    
    # Show summary
    print(f"\n" + "=" * 80)
    print("FILTERING COMPLETED")
    print("=" * 80)
    print(f"Input: {len(df):,} rows")
    print(f"Output: {len(filtered_df):,} rows")
    print(f"Filtered: {len(df) - len(filtered_df):,} rows removed")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_alternate_titles()
