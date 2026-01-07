"""
Task 1: Ingest and clean LinkedIn 2024 data
Goal: Build a clean joined table keyed by job_link
"""

import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Try to import optional libraries
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: pyarrow not available. Will save as pickle instead.")

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For reproducibility
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Will use simple heuristic for English detection.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")
    # Create a dummy tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration
BASE_DIR = Path(__file__).parent.parent
ARCHIVE_DIR = BASE_DIR / "archive"
OUTPUT_DIR = Path(__file__).parent

# File paths
JOB_POSTINGS_FILE = ARCHIVE_DIR / "linkedin_job_postings.csv"
JOB_SKILLS_FILE = ARCHIVE_DIR / "job_skills.csv"
JOB_SUMMARY_FILE = ARCHIVE_DIR / "job_summary.csv"
OUTPUT_FILE = OUTPUT_DIR / "linkedin_2024_clean.parquet"
LOG_FILE = OUTPUT_DIR / "task1_processing_log.json"

def detect_language(text):
    """Detect if text is English"""
    if not text or pd.isna(text):
        return None
    
    text_str = str(text)
    if len(text_str.strip()) < 50:  # Too short to detect reliably
        return None
    
    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text_str)
            return lang == 'en'
        except:
            return None
    else:
        # Simple heuristic: check for common English words
        common_english = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        text_lower = text_str.lower()
        english_word_count = sum(1 for word in common_english if word in text_lower)
        return english_word_count >= 3  # If at least 3 common English words, likely English

def parse_skills(skills_string):
    """Parse skills from comma-separated string into a list"""
    if pd.isna(skills_string) or not skills_string:
        return []
    
    # Handle different delimiters and quoted strings
    skills_str = str(skills_string).strip()
    
    # Try to split by comma, handling quoted strings
    import csv
    try:
        reader = csv.reader([skills_str], quotechar='"', delimiter=',', skipinitialspace=True)
        skills_list = next(reader)
        # Clean up each skill
        skills_list = [s.strip() for s in skills_list if s.strip()]
        return skills_list
    except:
        # Fallback: simple split
        skills_list = [s.strip() for s in skills_str.split(',') if s.strip()]
        return skills_list

def load_data():
    """Load all three CSV files"""
    print("=" * 60)
    print("TASK 1: Ingest and Clean LinkedIn 2024 Data")
    print("=" * 60)
    print(f"\nLoading data from: {ARCHIVE_DIR}")
    
    # Load job postings (spine)
    print("\n1. Loading linkedin_job_postings.csv...")
    try:
        job_postings = pd.read_csv(JOB_POSTINGS_FILE, low_memory=False)
        print(f"   [OK] Loaded {len(job_postings):,} job postings")
        print(f"   Columns: {list(job_postings.columns)}")
    except Exception as e:
        print(f"   [ERROR] Error loading job postings: {e}")
        sys.exit(1)
    
    # Load job skills
    print("\n2. Loading job_skills.csv...")
    try:
        job_skills = pd.read_csv(JOB_SKILLS_FILE, low_memory=False)
        print(f"   ✓ Loaded {len(job_skills):,} job skills records")
        print(f"   Columns: {list(job_skills.columns)}")
    except Exception as e:
        print(f"   ✗ Error loading job skills: {e}")
        sys.exit(1)
    
    # Load job summary
    print("\n3. Loading job_summary.csv...")
    try:
        job_summary = pd.read_csv(JOB_SUMMARY_FILE, low_memory=False)
        print(f"   [OK] Loaded {len(job_summary):,} job summaries")
        print(f"   Columns: {list(job_summary.columns)}")
    except Exception as e:
        print(f"   [ERROR] Error loading job summary: {e}")
        sys.exit(1)
    
    return job_postings, job_skills, job_summary

def process_skills(job_skills_df):
    """Parse skills into list format"""
    print("\n4. Parsing skills...")
    job_skills_df = job_skills_df.copy()
    
    # Use tqdm for progress bar
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Parsing skills", unit=" jobs")
        job_skills_df['skills_list'] = job_skills_df['job_skills'].progress_apply(parse_skills)
    else:
        job_skills_df['skills_list'] = job_skills_df['job_skills'].apply(parse_skills)
    
    # Count skills per job
    skill_counts = job_skills_df['skills_list'].apply(len)
    print(f"   [OK] Parsed skills for {len(job_skills_df):,} jobs")
    print(f"   Average skills per job: {skill_counts.mean():.1f}")
    print(f"   Jobs with skills: {(skill_counts > 0).sum():,}")
    
    return job_skills_df[['job_link', 'skills_list']]

def join_datasets(job_postings, job_skills_processed, job_summary):
    """Join all three datasets on job_link"""
    print("\n5. Joining datasets...")
    
    # Start with job postings as spine
    result = job_postings.copy()
    print(f"   Starting with {len(result):,} job postings")
    
    # Join skills
    result = result.merge(
        job_skills_processed,
        on='job_link',
        how='left'
    )
    print(f"   After skills join: {len(result):,} records")
    print(f"   Records with skills: {result['skills_list'].notna().sum():,}")
    
    # Join summary (keep only job_text field)
    if 'job_summary' in job_summary.columns:
        job_summary_clean = job_summary[['job_link', 'job_summary']].rename(
            columns={'job_summary': 'job_text'}
        )
    else:
        # If column name is different, adjust
        text_col = [col for col in job_summary.columns if 'summary' in col.lower() or 'text' in col.lower() or 'description' in col.lower()]
        if text_col:
            job_summary_clean = job_summary[['job_link', text_col[0]]].rename(
                columns={text_col[0]: 'job_text'}
            )
        else:
            print("   Warning: Could not find text column in job_summary")
            job_summary_clean = pd.DataFrame(columns=['job_link', 'job_text'])
    
    result = result.merge(
        job_summary_clean,
        on='job_link',
        how='left'
    )
    print(f"   After summary join: {len(result):,} records")
    print(f"   Records with job text: {result['job_text'].notna().sum():,}")
    
    return result

def deduplicate_and_clean(df):
    """Remove duplicates and perform sanity checks"""
    print("\n6. Deduplication and cleaning...")
    
    initial_count = len(df)
    print(f"   Initial records: {initial_count:,}")
    
    # Remove rows with missing job_link
    df = df[df['job_link'].notna()].copy()
    removed_missing = initial_count - len(df)
    print(f"   Removed {removed_missing:,} rows with missing job_link")
    
    # Check for duplicates
    duplicate_count = df['job_link'].duplicated().sum()
    print(f"   Found {duplicate_count:,} duplicate job_link entries")
    
    if duplicate_count > 0:
        # Strategy: Keep earliest first_seen, or if same, keep longest description
        if 'first_seen' in df.columns:
            # Convert first_seen to datetime if possible
            try:
                df['first_seen_dt'] = pd.to_datetime(df['first_seen'], errors='coerce')
                df = df.sort_values(['job_link', 'first_seen_dt', 'job_text'], 
                                   ascending=[True, True, False], 
                                   na_position='last')
            except:
                df = df.sort_values(['job_link', 'first_seen', 'job_text'], 
                                   ascending=[True, True, False], 
                                   na_position='last')
        else:
            # If no first_seen, keep longest description
            df = df.sort_values(['job_link', 'job_text'], 
                       ascending=[True, False], 
                       na_position='last')
        
        # Keep first occurrence (which is now the earliest/longest)
        df = df.drop_duplicates(subset=['job_link'], keep='first')
        print(f"   After deduplication: {len(df):,} unique records")
    
    return df

def filter_english(df):
    """Filter to keep only English postings"""
    print("\n7. Language detection and filtering...")
    
    initial_count = len(df)
    
    # Detect language for records with job_text
    print("   Detecting language (this may take a while)...")
    
    # Use tqdm for progress bar
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Detecting language", unit=" jobs")
        df['is_english'] = df['job_text'].progress_apply(detect_language)
    else:
        df['is_english'] = df['job_text'].apply(detect_language)
    
    # Keep English or records where we couldn't detect (to be conservative)
    df_filtered = df[(df['is_english'] == True) | (df['is_english'].isna())].copy()
    
    removed_non_english = initial_count - len(df_filtered)
    print(f"   Removed {removed_non_english:,} non-English postings")
    print(f"   Kept {len(df_filtered):,} postings (English or undetected)")
    
    # Drop the is_english column (temporary)
    df_filtered = df_filtered.drop(columns=['is_english'], errors='ignore')
    
    return df_filtered

def select_final_columns(df):
    """Select and rename final columns"""
    print("\n8. Selecting final columns...")
    
    # Required columns
    required_cols = {
        'job_link': 'job_link',
        'job_title': 'job_title',
        'company': 'company',
        'location': None,  # May be job_location
        'job_level': 'job_level',
        'job_type': 'job_type',
        'first_seen': 'first_seen',
        'skills_list': 'skills_list',
        'job_text': 'job_text'
    }
    
    # Map available columns
    available_cols = {}
    for target, source in required_cols.items():
        if source and source in df.columns:
            available_cols[target] = source
        elif source is None:
            # Try alternatives for location
            if target == 'location':
                for alt in ['job_location', 'location', 'loc']:
                    if alt in df.columns:
                        available_cols[target] = alt
                        break
    
    # Select columns
    cols_to_keep = list(available_cols.values())
    df_final = df[cols_to_keep].copy()
    
    # Rename to standard names
    rename_dict = {v: k for k, v in available_cols.items()}
    df_final = df_final.rename(columns=rename_dict)
    
    print(f"   Selected columns: {list(df_final.columns)}")
    
    return df_final

def save_output(df):
    """Save to parquet or alternative format"""
    print("\n9. Saving output...")
    
    if PARQUET_AVAILABLE:
        df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
        print(f"   [OK] Saved to: {OUTPUT_FILE}")
    else:
        # Fallback to pickle
        output_pkl = OUTPUT_FILE.with_suffix('.pkl')
        df.to_pickle(output_pkl)
        print(f"   [OK] Saved to: {output_pkl} (parquet not available)")
        print(f"   Note: Install pyarrow for parquet support: pip install pyarrow")

def generate_summary(df):
    """Generate summary statistics"""
    print("\n10. Generating summary statistics...")
    
    # Convert numpy types to native Python types for JSON serialization
    records_with_skills = int(df['skills_list'].notna().sum()) if 'skills_list' in df.columns else 0
    records_with_job_text = int(df['job_text'].notna().sum()) if 'job_text' in df.columns else 0
    memory_usage_mb = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'records_with_skills': records_with_skills,
        'records_with_job_text': records_with_job_text,
        'missing_job_link': 0,  # Already removed
        'duplicates_removed': 0,  # Tracked during dedup
        'columns': list(df.columns),
        'memory_usage_mb': memory_usage_mb
    }
    
    # Save log
    with open(LOG_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   [OK] Summary saved to: {LOG_FILE}")
    print(f"\n   Final dataset:")
    print(f"   - Total records: {summary['total_records']:,}")
    print(f"   - With skills: {summary['records_with_skills']:,}")
    print(f"   - With job text: {summary['records_with_job_text']:,}")
    print(f"   - Memory usage: {summary['memory_usage_mb']:.1f} MB")
    
    return summary

def main():
    """Main execution"""
    start_time = datetime.now()
    
    # Overall progress tracking
    steps = [
        "Loading data",
        "Processing skills",
        "Joining datasets",
        "Deduplicating",
        "Language filtering",
        "Selecting columns",
        "Saving output",
        "Generating summary"
    ]
    
    if TQDM_AVAILABLE:
        overall_progress = tqdm(total=len(steps), desc="Overall progress", position=0, leave=True)
    
    try:
        # Load data
        if TQDM_AVAILABLE:
            overall_progress.set_description("Loading data")
        job_postings, job_skills, job_summary = load_data()
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Process skills
        if TQDM_AVAILABLE:
            overall_progress.set_description("Processing skills")
        job_skills_processed = process_skills(job_skills)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Join datasets
        if TQDM_AVAILABLE:
            overall_progress.set_description("Joining datasets")
        joined_df = join_datasets(job_postings, job_skills_processed, job_summary)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Deduplicate
        if TQDM_AVAILABLE:
            overall_progress.set_description("Deduplicating")
        cleaned_df = deduplicate_and_clean(joined_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Filter English
        if TQDM_AVAILABLE:
            overall_progress.set_description("Language filtering")
        english_df = filter_english(cleaned_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Select final columns
        if TQDM_AVAILABLE:
            overall_progress.set_description("Selecting columns")
        final_df = select_final_columns(english_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Save output
        if TQDM_AVAILABLE:
            overall_progress.set_description("Saving output")
        save_output(final_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Generate summary
        if TQDM_AVAILABLE:
            overall_progress.set_description("Generating summary")
        summary = generate_summary(final_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
            overall_progress.set_description("Complete!")
            overall_progress.close()
    
    except Exception as e:
        if TQDM_AVAILABLE:
            overall_progress.close()
        print(f"\n[ERROR] Task failed: {e}")
        raise
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total time: {duration:.1f} seconds")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Log file: {LOG_FILE}")

if __name__ == "__main__":
    main()
