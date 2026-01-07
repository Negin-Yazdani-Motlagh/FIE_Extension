"""
Task 2: Define comparable job families
Goal: Keep LinkedIn aligned to your computing/engineering focus and to YC.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import re

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not available. Progress bars will be disabled.")
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration
BASE_DIR = Path(__file__).parent.parent
TASK1_DIR = BASE_DIR / "task1"
OUTPUT_DIR = Path(__file__).parent

# File paths
INPUT_FILE = TASK1_DIR / "linkedin_2024_clean.parquet"
OUTPUT_FILE = OUTPUT_DIR / "linkedin_cs_subset.parquet"
LOG_FILE = OUTPUT_DIR / "task2_processing_log.json"
RULES_FILE = OUTPUT_DIR / "task2_filtering_rules.json"
COUNTS_TABLE_FILE = OUTPUT_DIR / "task2_job_family_counts.csv"
ALTERNATE_TITLES_FILE = OUTPUT_DIR / "alternate_titles_filtered.csv"

# Global variable to store loaded alternate titles
ALTERNATE_TITLES_SET = None
ALTERNATE_TITLES_PATTERN = None

def load_alternate_titles():
    """Load and normalize alternate titles from the filtered O*NET CSV"""
    global ALTERNATE_TITLES_SET, ALTERNATE_TITLES_PATTERN
    
    print("   Loading alternate titles from O*NET CSV...")
    
    try:
        df = pd.read_csv(ALTERNATE_TITLES_FILE)
        print(f"   [OK] Loaded {len(df):,} rows from {ALTERNATE_TITLES_FILE.name}")
    except Exception as e:
        print(f"   [ERROR] Could not load alternate titles: {e}")
        return set()
    
    # Collect all titles from relevant columns
    all_titles = set()
    
    # Add main titles
    for title in df['Title'].dropna().unique():
        all_titles.add(str(title).lower().strip())
    
    # Add alternate titles
    for title in df['Alternate Title'].dropna().unique():
        all_titles.add(str(title).lower().strip())
    
    # Add short titles (if not empty)
    for title in df['Short Title'].dropna().unique():
        if str(title).strip():
            all_titles.add(str(title).lower().strip())
    
    # Remove empty strings
    all_titles.discard('')
    
    print(f"   [OK] Extracted {len(all_titles):,} unique job titles for matching")
    
    # Create a regex pattern for efficient matching
    # Escape special regex characters and create word boundaries
    escaped_titles = [re.escape(title) for title in sorted(all_titles, key=len, reverse=True)]
    pattern_str = '|'.join(escaped_titles)
    ALTERNATE_TITLES_PATTERN = re.compile(r'\b(' + pattern_str + r')\b', re.IGNORECASE)
    
    ALTERNATE_TITLES_SET = all_titles
    return all_titles

# Exclusion keywords (case-insensitive matching)
EXCLUSION_KEYWORDS = [
    # Non-computing roles
    'nurse', 'nursing', 'physician', 'doctor', 'medical', 'healthcare',
    'teacher', 'teaching', 'educator', 'professor', 'instructor',
    'lawyer', 'attorney', 'legal', 'paralegal',
    'accountant', 'accounting', 'bookkeeper',
    'chef', 'cook', 'restaurant', 'food service',
    'driver', 'delivery', 'truck',
    'sales', 'retail', 'cashier', 'store',
    'janitor', 'custodian', 'cleaning', 'housekeeping',
    'construction', 'contractor', 'plumber', 'electrician',
    'mechanic', 'automotive',
    'receptionist', 'administrative assistant',
    # Non-technical roles that might match
    'data entry', 'data clerk', 'data analyst'  # Keep data analyst but exclude data entry/clerk
]

# Computing keywords - if title contains these, don't exclude even if it has exclusion terms
COMPUTING_INDICATORS = [
    'engineer', 'developer', 'programmer', 'analyst', 'scientist', 'architect',
    'software', 'data', 'system', 'network', 'security', 'cloud', 'devops',
    'informatics', 'technology', 'technical', 'tech', 'it ', 'information'
]

# Job family definitions (order matters - more specific first)
JOB_FAMILY_PATTERNS = {
    'ML/AI': [
        r'\b(ml|machine learning|deep learning|neural network|ai|artificial intelligence|nlp|natural language|computer vision|cv|llm|large language model)\b',
        r'\b(data scientist|ml engineer|ai engineer|ml researcher|ai researcher)\b'
    ],
    'Data Science': [
        r'\b(data scientist|data science|data analyst|analytics|statistician|bi analyst|business intelligence)\b',
        r'\b(data engineer|etl|data pipeline|data warehouse)\b'
    ],
    'Security': [
        r'\b(security|cybersecurity|cyber security|infosec|information security|penetration|pen tester|security engineer)\b',
        r'\b(security analyst|security architect|security consultant)\b'
    ],
    'SWE': [
        r'\b(software engineer|software developer|swe|programmer|developer|software architect)\b',
        r'\b(backend|back-end|frontend|front-end|fullstack|full-stack)\b',
        r'\b(web developer|mobile developer|ios developer|android developer)\b'
    ],
    'Product': [
        r'\b(product manager|product owner|pm|technical product|product engineer)\b',
        r'\b(product designer|ux designer|ui designer|interaction designer)\b'
    ],
    'Other': []  # Catch-all for computing roles that don't fit above
}

def normalize_title(title):
    """Normalize job title for matching"""
    if pd.isna(title):
        return ""
    return str(title).lower().strip()

def matches_inclusion(title_normalized):
    """Check if title matches any O*NET alternate title (inclusion criteria)"""
    global ALTERNATE_TITLES_SET, ALTERNATE_TITLES_PATTERN
    
    if ALTERNATE_TITLES_SET is None or ALTERNATE_TITLES_PATTERN is None:
        load_alternate_titles()
    
    # Check if the normalized title exactly matches any alternate title
    if title_normalized in ALTERNATE_TITLES_SET:
        return True
    
    # Use regex pattern to check if any alternate title appears in the job title
    # (e.g., "Senior Software Developer" contains "Software Developer")
    if ALTERNATE_TITLES_PATTERN.search(title_normalized):
        return True
    
    return False

def matches_exclusion(title_normalized):
    """Check if title matches exclusion criteria"""
    # First check if title contains computing indicators - if so, don't exclude
    for indicator in COMPUTING_INDICATORS:
        if indicator in title_normalized:
            return False  # Don't exclude computing roles even if they mention healthcare/medical/etc.
    
    # Now check exclusion keywords
    for exclusion in EXCLUSION_KEYWORDS:
        if exclusion in title_normalized:
            # Special case: exclude "data entry" and "data clerk" but keep "data analyst"
            if exclusion == 'data entry' and 'data entry' in title_normalized:
                return True
            if exclusion == 'data clerk' and 'data clerk' in title_normalized:
                return True
            # For other exclusions, check if it's the main term
            if exclusion in title_normalized and exclusion not in ['data analyst']:
                return True
    return False

def classify_job_family(title_normalized):
    """Classify job into family based on title patterns"""
    # Check in order of specificity (ML/AI, DS, Security, SWE, Product, Other)
    for family, patterns in JOB_FAMILY_PATTERNS.items():
        if family == 'Other':
            continue  # Other is catch-all, check last
        
        for pattern in patterns:
            if re.search(pattern, title_normalized, re.IGNORECASE):
                return family
    
    # If no specific match but passed inclusion, classify as Other
    return 'Other'

def apply_filters(df):
    """Apply inclusion and exclusion filters"""
    print("\n2. Applying filters...")
    
    # Load alternate titles first
    load_alternate_titles()
    
    initial_count = len(df)
    print(f"   Initial records: {initial_count:,}")
    
    # Normalize titles
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Normalizing titles", unit=" jobs")
        df['title_normalized'] = df['job_title'].progress_apply(normalize_title)
    else:
        df['title_normalized'] = df['job_title'].apply(normalize_title)
    
    # Apply inclusion filter (based on O*NET alternate titles)
    print("   Applying inclusion filter (matching against O*NET titles)...")
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Matching titles", unit=" jobs")
        df['matches_inclusion'] = df['title_normalized'].progress_apply(matches_inclusion)
    else:
        df['matches_inclusion'] = df['title_normalized'].apply(matches_inclusion)
    
    included = df[df['matches_inclusion']].copy()
    excluded_by_inclusion = initial_count - len(included)
    print(f"   After inclusion filter: {len(included):,} records")
    print(f"   Excluded (no O*NET title match): {excluded_by_inclusion:,}")
    
    # Apply exclusion filter
    print("   Applying exclusion filter...")
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Checking exclusions", unit=" jobs")
        included['matches_exclusion'] = included['title_normalized'].progress_apply(matches_exclusion)
    else:
        included['matches_exclusion'] = included['title_normalized'].apply(matches_exclusion)
    
    final_df = included[~included['matches_exclusion']].copy()
    excluded_by_exclusion = len(included) - len(final_df)
    print(f"   After exclusion filter: {len(final_df):,} records")
    print(f"   Excluded (non-computing roles): {excluded_by_exclusion:,}")
    
    return final_df

def classify_jobs(df):
    """Classify jobs into families"""
    print("\n3. Classifying jobs into families...")
    
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Classifying jobs", unit=" jobs")
        df['job_family'] = df['title_normalized'].progress_apply(classify_job_family)
    else:
        df['job_family'] = df['title_normalized'].apply(classify_job_family)
    
    # Count by family
    family_counts = df['job_family'].value_counts().sort_index()
    print(f"\n   Job family distribution:")
    for family, count in family_counts.items():
        pct = (count / len(df)) * 100
        print(f"     {family}: {count:,} ({pct:.1f}%)")
    
    return df

def generate_counts_table(df):
    """Generate counts table by job family"""
    print("\n4. Generating counts table...")
    
    counts = df['job_family'].value_counts().sort_index()
    counts_df = pd.DataFrame({
        'job_family': counts.index,
        'count': counts.values,
        'percentage': (counts.values / len(df) * 100).round(2)
    })
    
    counts_df.to_csv(COUNTS_TABLE_FILE, index=False)
    print(f"   [OK] Saved to: {COUNTS_TABLE_FILE}")
    
    return counts_df

def save_output(df):
    """Save filtered dataset"""
    print("\n5. Saving output...")
    
    # Drop temporary columns
    df_output = df.drop(columns=['title_normalized', 'matches_inclusion', 'matches_exclusion'], errors='ignore')
    
    # Save to parquet
    try:
        df_output.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
        print(f"   [OK] Saved to: {OUTPUT_FILE}")
        print(f"   Records: {len(df_output):,}")
    except Exception as e:
        print(f"   [ERROR] Could not save as parquet: {e}")
        # Fallback to pickle
        output_pkl = OUTPUT_FILE.with_suffix('.pkl')
        df_output.to_pickle(output_pkl)
        print(f"   [OK] Saved to: {output_pkl} (pickle format)")

def save_filtering_rules():
    """Save filtering rules for documentation"""
    global ALTERNATE_TITLES_SET
    
    rules = {
        'timestamp': datetime.now().isoformat(),
        'inclusion_source': str(ALTERNATE_TITLES_FILE),
        'inclusion_titles_count': len(ALTERNATE_TITLES_SET) if ALTERNATE_TITLES_SET else 0,
        'inclusion_titles_sample': sorted(list(ALTERNATE_TITLES_SET))[:50] if ALTERNATE_TITLES_SET else [],
        'exclusion_keywords': EXCLUSION_KEYWORDS,
        'computing_indicators': COMPUTING_INDICATORS,
        'job_family_patterns': {k: v for k, v in JOB_FAMILY_PATTERNS.items()},
        'description': {
            'inclusion': 'Job titles must match or contain any O*NET alternate title from the filtered CSV (computing/engineering SOC codes)',
            'exclusion': 'Job titles containing non-computing role keywords are excluded, unless they also contain computing indicators (e.g., "Healthcare Informatics Engineer" is kept because it contains "engineer" and "informatics")',
            'classification': 'Jobs are classified into families based on title patterns (order: ML/AI, DS, Security, SWE, Product, Other)'
        }
    }
    
    with open(RULES_FILE, 'w') as f:
        json.dump(rules, f, indent=2)
    
    print(f"   [OK] Filtering rules saved to: {RULES_FILE}")

def generate_summary(df, initial_count):
    """Generate summary statistics"""
    print("\n6. Generating summary statistics...")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'initial_records': int(initial_count),
        'final_records': int(len(df)),
        'records_removed': int(initial_count - len(df)),
        'removal_percentage': float((initial_count - len(df)) / initial_count * 100),
        'job_family_counts': df['job_family'].value_counts().to_dict(),
        'job_family_percentages': (df['job_family'].value_counts() / len(df) * 100).round(2).to_dict(),
        'columns': list(df.columns),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
    
    # Convert numpy types to native Python types
    summary['job_family_counts'] = {k: int(v) for k, v in summary['job_family_counts'].items()}
    summary['job_family_percentages'] = {k: float(v) for k, v in summary['job_family_percentages'].items()}
    
    with open(LOG_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   [OK] Summary saved to: {LOG_FILE}")
    
    return summary

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 2: Define Comparable Job Families")
    print("=" * 80)
    
    # Overall progress tracking
    steps = ["Loading data", "Applying filters", "Classifying jobs", "Generating table", "Saving output", "Generating summary"]
    if TQDM_AVAILABLE:
        overall_progress = tqdm(total=len(steps), desc="Overall progress", position=0, leave=True)
    
    try:
        # Step 1: Load data
        if TQDM_AVAILABLE:
            overall_progress.set_description("Loading data")
        
        print("\n1. Loading cleaned LinkedIn data...")
        try:
            df = pd.read_parquet(INPUT_FILE)
            print(f"   [OK] Loaded {len(df):,} records from {INPUT_FILE.name}")
        except Exception as e:
            print(f"   [ERROR] Error loading parquet file: {e}")
            if TQDM_AVAILABLE:
                overall_progress.close()
            return
        
        initial_count = len(df)
        
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Step 2: Apply filters
        if TQDM_AVAILABLE:
            overall_progress.set_description("Applying filters")
        filtered_df = apply_filters(df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Step 3: Classify jobs
        if TQDM_AVAILABLE:
            overall_progress.set_description("Classifying jobs")
        classified_df = classify_jobs(filtered_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Step 4: Generate counts table
        if TQDM_AVAILABLE:
            overall_progress.set_description("Generating table")
        counts_table = generate_counts_table(classified_df)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Step 5: Save output
        if TQDM_AVAILABLE:
            overall_progress.set_description("Saving output")
        save_output(classified_df)
        save_filtering_rules()
        if TQDM_AVAILABLE:
            overall_progress.update(1)
        
        # Step 6: Generate summary
        if TQDM_AVAILABLE:
            overall_progress.set_description("Generating summary")
        summary = generate_summary(classified_df, initial_count)
        if TQDM_AVAILABLE:
            overall_progress.update(1)
            overall_progress.set_description("Complete!")
            overall_progress.close()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TASK 2 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Initial records: {initial_count:,}")
        print(f"  Final records: {len(classified_df):,}")
        print(f"  Removed: {initial_count - len(classified_df):,} ({(initial_count - len(classified_df))/initial_count*100:.1f}%)")
        print(f"\nJob Family Distribution:")
        for family, count, pct in counts_table.itertuples(index=False):
            print(f"  {family}: {count:,} ({pct:.1f}%)")
        print(f"\nTotal time: {duration:.1f} seconds")
        print(f"Output file: {OUTPUT_FILE}")
        print(f"Counts table: {COUNTS_TABLE_FILE}")
        print(f"Log file: {LOG_FILE}")
        print(f"Rules file: {RULES_FILE}")
    
    except Exception as e:
        if TQDM_AVAILABLE:
            overall_progress.close()
        print(f"\n[ERROR] Task failed: {e}")
        raise

if __name__ == "__main__":
    main()
