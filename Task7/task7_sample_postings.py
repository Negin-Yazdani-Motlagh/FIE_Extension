"""
Task 7: Sample Postings for Human Coding
Goal: Create stratified sample of ~200 postings for human validation
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
BASE_DIR = Path(__file__).parent.parent
TASK5_DIR = BASE_DIR / "task5"
OUTPUT_DIR = Path(__file__).parent

# File paths
TASK5_FILE = TASK5_DIR / "task5_agreement_analysis.parquet"

OUTPUT_SAMPLING_METADATA = OUTPUT_DIR / "task7_sampling_metadata.json"

# Domain names (from Task 4/5)
DOMAIN_NAMES = [
    'Adaptability & Continuous Learning',
    'Collaboration And Team Dynamics',
    'Communication Skills',
    'Creativity And Inovation',
    'Emotional Intelligence (Eq)',
    'Problem-Solving And Critical Thinking',
    'Time Management And Organizational Skills',
    'Work Ethic And Professionalism'
]

# Agreement categories
AGREEMENT_CATEGORIES = ['both', 'tag_only', 'text_only', 'neither']

# Target sample size
TARGET_SAMPLE_SIZE = 200

# Random seed for reproducibility
RANDOM_SEED = 42

def get_agreement_category(row, domain):
    """Get agreement category for a domain"""
    tag_col = f'tag_domain_present_{domain}'
    text_col = f'text_domain_present_{domain}'
    
    tag_present = bool(row.get(tag_col, False))
    text_present = bool(row.get(text_col, False))
    
    if tag_present and text_present:
        return 'both'
    elif tag_present and not text_present:
        return 'tag_only'
    elif not tag_present and text_present:
        return 'text_only'
    else:
        return 'neither'

def create_sampling_strata(df):
    """Create sampling strata based on job_family and agreement categories"""
    print("\n2. Creating sampling strata...")
    
    # Add agreement category columns for each domain
    for domain in DOMAIN_NAMES:
        agreement_col = f'agreement_category_{domain}'
        if agreement_col not in df.columns:
            df[agreement_col] = df.apply(lambda row: get_agreement_category(row, domain), axis=1)
    
    # Create strata: job_family x agreement_category (for each domain)
    # We'll sample to ensure representation across job families and mismatch types
    strata_info = {}
    
    # Get job families
    job_families = sorted(df['job_family'].dropna().unique())
    print(f"   Found {len(job_families)} job families: {job_families}")
    
    # Count postings per job family
    family_counts = df['job_family'].value_counts()
    print("\n   Postings per job family:")
    for family, count in family_counts.items():
        print(f"     {family}: {count:,}")
    
    # Count agreement categories across all domains
    print("\n   Agreement categories (across all domains):")
    for category in AGREEMENT_CATEGORIES:
        count = 0
        for domain in DOMAIN_NAMES:
            col = f'agreement_category_{domain}'
            if col in df.columns:
                count += (df[col] == category).sum()
        print(f"     {category}: {count:,}")
    
    return df, job_families

def stratified_sample(df, job_families, target_size=200):
    """Perform stratified sampling"""
    print(f"\n3. Performing stratified sampling (target: {target_size} postings)...")
    
    sampled_rows = []
    sampling_info = {
        'target_size': target_size,
        'random_seed': RANDOM_SEED,
        'strata': {}
    }
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Strategy: Proportional sampling by job family, ensuring mismatch representation
    # 1. Calculate proportional allocation by job family
    total_postings = len(df)
    family_proportions = df['job_family'].value_counts(normalize=True)
    
    # 2. Allocate samples per family (minimum 10 per family for statistical power)
    min_per_family = 10
    samples_per_family = {}
    
    for family in job_families:
        prop = family_proportions.get(family, 0)
        allocated = max(min_per_family, int(target_size * prop))
        samples_per_family[family] = allocated
    
    # Adjust if total exceeds target
    total_allocated = sum(samples_per_family.values())
    if total_allocated > target_size:
        # Reduce proportionally
        scale = target_size / total_allocated
        for family in samples_per_family:
            samples_per_family[family] = max(min_per_family, int(samples_per_family[family] * scale))
    
    # Ensure we have at least some mismatch cases in each family
    print("\n   Sampling strategy:")
    for family in job_families:
        family_df = df[df['job_family'] == family].copy()
        n_samples = samples_per_family[family]
        
        # Within each family, try to get representation of mismatch types
        # Prioritize: tag_only, text_only, both, neither (in that order for validation)
        family_samples = []
        
        # Try to get tag_only cases (important for validation)
        tag_only = family_df[family_df.apply(
            lambda row: any(get_agreement_category(row, d) == 'tag_only' for d in DOMAIN_NAMES),
            axis=1
        )]
        if len(tag_only) > 0:
            n_tag_only = min(5, len(tag_only), n_samples // 4)
            sampled_tag_only = tag_only.sample(n=n_tag_only, random_state=RANDOM_SEED)
            family_samples.append(sampled_tag_only)
        
        # Try to get text_only cases
        text_only = family_df[family_df.apply(
            lambda row: any(get_agreement_category(row, d) == 'text_only' for d in DOMAIN_NAMES),
            axis=1
        )]
        if len(text_only) > 0:
            n_text_only = min(5, len(text_only), (n_samples - len(family_samples) * 5) // 3)
            sampled_text_only = text_only.sample(n=n_text_only, random_state=RANDOM_SEED)
            family_samples.append(sampled_text_only)
        
        # Fill remaining with random sample
        already_sampled = pd.concat(family_samples) if family_samples else pd.DataFrame()
        remaining = family_df[~family_df.index.isin(already_sampled.index)]
        n_remaining = n_samples - len(already_sampled)
        
        if n_remaining > 0 and len(remaining) > 0:
            sampled_remaining = remaining.sample(n=min(n_remaining, len(remaining)), random_state=RANDOM_SEED)
            family_samples.append(sampled_remaining)
        
        # Combine
        if family_samples:
            family_sample = pd.concat(family_samples)
            sampled_rows.append(family_sample)
            
            sampling_info['strata'][family] = {
                'allocated': n_samples,
                'actual': len(family_sample),
                'tag_only': len(tag_only) if len(tag_only) > 0 else 0,
                'text_only': len(text_only) if len(text_only) > 0 else 0
            }
            
            print(f"     {family}: {len(family_sample)} postings (target: {n_samples})")
    
    # Combine all samples
    if sampled_rows:
        sample_df = pd.concat(sampled_rows, ignore_index=True)
        # Remove duplicates (if any)
        sample_df = sample_df.drop_duplicates(subset=['job_link'], keep='first')
        
        # If we have more than target, randomly sample down
        if len(sample_df) > target_size:
            sample_df = sample_df.sample(n=target_size, random_state=RANDOM_SEED).reset_index(drop=True)
        
        print(f"\n   [OK] Final sample size: {len(sample_df):,} postings")
        
        return sample_df, sampling_info
    else:
        raise ValueError("No samples were selected!")

def prepare_sample_for_coding(df):
    """Prepare sample dataframe for human coding"""
    print("\n4. Preparing sample for coding...")
    
    # Select columns needed for coding
    columns_to_keep = [
        'job_link',
        'job_title',
        'company',
        'location',
        'job_family',
        'job_text'
    ]
    
    # Add tag and text detection columns
    for domain in DOMAIN_NAMES:
        tag_col = f'tag_domain_present_{domain}'
        text_col = f'text_domain_present_{domain}'
        agreement_col = f'agreement_category_{domain}'
        
        if tag_col in df.columns:
            columns_to_keep.append(tag_col)
        if text_col in df.columns:
            columns_to_keep.append(text_col)
        if agreement_col in df.columns:
            columns_to_keep.append(agreement_col)
    
    # Select columns
    sample_df = df[columns_to_keep].copy()
    
    # Add sampling metadata columns
    sample_df['sample_id'] = range(1, len(sample_df) + 1)
    sample_df['sampling_date'] = datetime.now().strftime('%Y-%m-%d')
    
    print(f"   [OK] Prepared {len(sample_df):,} postings with {len(sample_df.columns)} columns")
    
    return sample_df

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 7: Sample Postings for Human Coding")
    print("=" * 80)
    
    try:
        # Step 1: Load Task 5 agreement analysis
        print("\n1. Loading Task 5 agreement analysis...")
        print(f"   File: {TASK5_FILE.name}")
        df = pd.read_parquet(TASK5_FILE)
        print(f"   [OK] Loaded {len(df):,} postings")
        
        # Check required columns
        required_cols = ['job_link', 'job_title', 'company', 'job_family', 'job_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Step 2: Create sampling strata
        df, job_families = create_sampling_strata(df)
        
        # Step 3: Perform stratified sampling
        sample_df, sampling_info = stratified_sample(df, job_families, target_size=TARGET_SAMPLE_SIZE)
        
        # Step 4: Prepare sample for coding
        sample_df = prepare_sample_for_coding(sample_df)
        
        # Step 5: Save sample
        print("\n5. Saving sample...")
        sample_df.to_csv(OUTPUT_SAMPLE, index=False)
        print(f"   [OK] Saved sample to {OUTPUT_SAMPLE.name}")
        
        # Step 6: Save sampling metadata
        sampling_info['total_postings'] = len(df)
        sampling_info['sample_size'] = len(sample_df)
        sampling_info['sampling_date'] = datetime.now().isoformat()
        sampling_info['domains'] = DOMAIN_NAMES
        
        with open(OUTPUT_SAMPLING_METADATA, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        print(f"   [OK] Saved sampling metadata to {OUTPUT_SAMPLING_METADATA.name}")
        
        # Step 7: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("SAMPLING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings: {len(df):,}")
        print(f"  Sample size: {len(sample_df):,}")
        print(f"  Job families: {len(job_families)}")
        print(f"  Domains: {len(DOMAIN_NAMES)}")
        print(f"\nSample distribution by job family:")
        for family in job_families:
            count = len(sample_df[sample_df['job_family'] == family])
            print(f"  {family}: {count}")
        print(f"\nOutput files:")
        print(f"  Sample: {OUTPUT_SAMPLE}")
        print(f"  Metadata: {OUTPUT_SAMPLING_METADATA}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
