"""
Task 8: Load YC Corpus (2012-2024)
Goal: Load and prepare YC corpus data for detector application

NOTE: This script expects YC corpus data to be available.
Place your YC data file in the task8 directory or update YC_DATA_FILE path below.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
YC_DATA_FILE = OUTPUT_DIR / "Yearly_Job_Posts_filtered_gt30.json"  # YC yearly job posts JSON

OUTPUT_FILE = OUTPUT_DIR / "yc_corpus_2012_2024.parquet"
LOG_FILE = OUTPUT_DIR / "task8_data_loading_log.json"

def extract_year_from_date(date_value):
    """Extract year from date value"""
    if pd.isna(date_value):
        return None
    
    try:
        # Try parsing as datetime
        if isinstance(date_value, str):
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.year
                except:
                    continue
            # If string, try to extract year (4 digits)
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', date_value)
            if year_match:
                return int(year_match.group())
        elif hasattr(date_value, 'year'):
            return date_value.year
        elif isinstance(date_value, (int, float)):
            # Assume it's already a year if it's a reasonable year value
            if 1900 <= date_value <= 2100:
                return int(date_value)
    except:
        pass
    
    return None

def classify_ai_posting(job_title, job_text):
    """Classify if posting is AI-related (simple keyword-based)"""
    if pd.isna(job_title):
        job_title = ""
    if pd.isna(job_text):
        job_text = ""
    
    text = str(job_title).lower() + " " + str(job_text).lower()
    
    # AI-related keywords
    ai_keywords = [
        'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'ai', 'ml', 'nlp', 'natural language processing', 'computer vision',
        'data science', 'data scientist', 'ml engineer', 'ai engineer',
        'tensorflow', 'pytorch', 'keras', 'transformer', 'llm', 'large language model'
    ]
    
    return any(keyword in text for keyword in ai_keywords)

def load_yc_data():
    """Load YC corpus data from JSON file"""
    print("\n1. Loading YC corpus data...")
    
    if not YC_DATA_FILE.exists():
        print(f"   [ERROR] YC data file not found: {YC_DATA_FILE}")
        print(f"   Please ensure the file exists: {YC_DATA_FILE}")
        raise FileNotFoundError(f"YC data file not found: {YC_DATA_FILE}")
    
    print(f"   File: {YC_DATA_FILE.name}")
    
    # Load JSON file
    with open(YC_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse YC structure: {'YC': {year: {'total_job_posts': N, 'comments': [...]}}}
    if 'YC' not in data:
        raise ValueError("JSON file must have 'YC' key at top level")
    
    yc_data = data['YC']
    
    # Build list of postings
    postings = []
    posting_id = 0
    
    for year_str, year_data in yc_data.items():
        year = int(year_str)
        comments = year_data.get('comments', [])
        
        for comment_text in comments:
            if comment_text and len(str(comment_text).strip()) > 0:
                postings.append({
                    'job_link': f"yc_{posting_id}",
                    'year': year,
                    'job_text': str(comment_text).strip(),
                    'job_title': 'Unknown',  # Will try to extract later if possible
                    'company': 'Unknown'  # Will try to extract later if possible
                })
                posting_id += 1
    
    df = pd.DataFrame(postings)
    print(f"   [OK] Loaded {len(df):,} postings from {len(yc_data)} years")
    
    return df

def prepare_yc_corpus(df):
    """Prepare YC corpus with required columns"""
    print("\n2. Preparing YC corpus...")
    
    # Required columns
    required_cols = ['job_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create output dataframe
    output_df = df.copy()
    
    # Extract year from date
    print("   Extracting year from date...")
    if 'year' in df.columns:
        output_df['year'] = df['year'].astype('Int64')  # Nullable integer
    elif 'posting_date' in df.columns:
        output_df['year'] = df['posting_date'].apply(extract_year_from_date)
    elif 'date' in df.columns:
        output_df['year'] = df['date'].apply(extract_year_from_date)
    else:
        print("   [WARNING] No date/year column found. Creating placeholder year column.")
        output_df['year'] = None
    
    # Filter to 2012-2024
    if 'year' in output_df.columns:
        before = len(output_df)
        output_df = output_df[(output_df['year'] >= 2012) & (output_df['year'] <= 2024)].copy()
        after = len(output_df)
        print(f"   [OK] Filtered to 2012-2024: {before:,} -> {after:,} postings")
    
    # Ensure job_text exists and is string
    if 'job_text' in output_df.columns:
        output_df['job_text'] = output_df['job_text'].astype(str)
        output_df['job_text'] = output_df['job_text'].replace('nan', '')
    
    # Add job_title if missing
    if 'job_title' not in output_df.columns:
        if 'title' in df.columns:
            output_df['job_title'] = df['title']
        else:
            output_df['job_title'] = 'Unknown'
            print("   [WARNING] No job_title column found. Using 'Unknown'")
    
    # Add company if missing
    if 'company' not in output_df.columns:
        if 'company_name' in df.columns:
            output_df['company'] = df['company_name']
        else:
            output_df['company'] = 'Unknown'
            print("   [WARNING] No company column found. Using 'Unknown'")
    
    # Add unique ID if missing
    if 'job_link' not in output_df.columns and 'id' not in output_df.columns:
        output_df['job_link'] = [f"yc_{i}" for i in range(len(output_df))]
        print("   [OK] Created job_link column")
    elif 'id' in output_df.columns and 'job_link' not in output_df.columns:
        output_df['job_link'] = output_df['id'].astype(str)
    
    # Classify AI vs non-AI (if not already present)
    if 'is_ai' not in output_df.columns:
        print("   Classifying AI vs non-AI postings...")
        output_df['is_ai'] = output_df.apply(
            lambda row: classify_ai_posting(row.get('job_title', ''), row.get('job_text', '')),
            axis=1
        )
        print(f"   [OK] Classified: {output_df['is_ai'].sum():,} AI, {(~output_df['is_ai']).sum():,} non-AI")
    
    # Select final columns
    final_cols = ['job_link', 'job_title', 'company', 'job_text', 'year']
    if 'is_ai' in output_df.columns:
        final_cols.append('is_ai')
    
    # Keep any other columns that might be useful
    for col in output_df.columns:
        if col not in final_cols and col not in ['id', 'title', 'company_name', 'date', 'posting_date']:
            final_cols.append(col)
    
    output_df = output_df[[col for col in final_cols if col in output_df.columns]].copy()
    
    print(f"   [OK] Prepared corpus with {len(output_df.columns)} columns")
    print(f"   Columns: {list(output_df.columns)}")
    
    return output_df

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 8: Load YC Corpus (2012-2024)")
    print("=" * 80)
    
    try:
        # Step 1: Load YC data
        df = load_yc_data()
        
        # Step 2: Prepare corpus
        output_df = prepare_yc_corpus(df)
        
        # Step 3: Calculate statistics
        print("\n3. Calculating statistics...")
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_postings': len(output_df),
            'years_covered': sorted(output_df['year'].dropna().unique().tolist()) if 'year' in output_df.columns else [],
            'columns': list(output_df.columns)
        }
        
        if 'year' in output_df.columns:
            year_counts = output_df['year'].value_counts().sort_index()
            stats['postings_by_year'] = {int(k): int(v) for k, v in year_counts.items()}
        
        if 'is_ai' in output_df.columns:
            stats['ai_classification'] = {
                'ai_postings': int(output_df['is_ai'].sum()),
                'non_ai_postings': int((~output_df['is_ai']).sum())
            }
        
        # Step 4: Save results
        print("\n4. Saving results...")
        output_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"   [OK] Saved corpus to {OUTPUT_FILE.name}")
        
        with open(LOG_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   [OK] Saved log to {LOG_FILE.name}")
        
        # Step 5: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("YC CORPUS LOADING COMPLETED")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings: {len(output_df):,}")
        if 'year' in output_df.columns:
            print(f"  Years: {min(output_df['year'].dropna())} - {max(output_df['year'].dropna())}")
            print(f"  Years covered: {len(stats['years_covered'])}")
        if 'is_ai' in output_df.columns:
            print(f"  AI postings: {stats['ai_classification']['ai_postings']:,}")
            print(f"  Non-AI postings: {stats['ai_classification']['non_ai_postings']:,}")
        print(f"\nOutput file:")
        print(f"  {OUTPUT_FILE}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo use this script:")
        print("1. Place your YC corpus data file in the task8 directory")
        print("2. Update YC_DATA_FILE path in this script")
        print("3. Ensure the file has at least: job_text column")
        print("4. Optional but recommended: posting_date/year, job_title, company")
    except Exception as e:
        print(f"\n[ERROR] Loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
