"""
Task 4: Run Text Detector v1 on LinkedIn Text
Goal: Produce comparable domain labels from job descriptions using a dictionary-based method (unstructured signal).

Features:
- Checkpoint/resume functionality: saves progress every 1000 rows and every 5 minutes
- Can resume from checkpoint if script is interrupted
"""

import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
TASK2_DIR = BASE_DIR / "task2"
TASK3_DIR = BASE_DIR / "task3"
OUTPUT_DIR = Path(__file__).parent

# File paths
INPUT_FILE = TASK2_DIR / "linkedin_cs_subset.parquet"
MAPPING_FILE = TASK3_DIR / "softskill_tag_mapping.csv"  # Use Task 3 CSV mapping instead of Excel
OUTPUT_FILE = OUTPUT_DIR / "linkedin_text_v1.parquet"
LOG_FILE = OUTPUT_DIR / "task4_processing_log.json"
CHECKPOINT_FILE = OUTPUT_DIR / "task4_checkpoint.parquet"
CHECKPOINT_INFO_FILE = OUTPUT_DIR / "task4_checkpoint_info.json"

# Checkpoint settings
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N rows
CHECKPOINT_TIME_INTERVAL = 300  # Save checkpoint every N seconds (5 minutes)

def load_mapping_from_csv(file_path):
    """Load skill-to-domain mapping from Task 3 CSV file"""
    print(f"\n1. Loading skill mapping from Task 3 CSV...")
    print(f"   File: {file_path.name}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"   [OK] Loaded {len(df):,} rows from mapping file")
        
        # Filter to only mapped skills (exclude unmapped)
        mapped_df = df[df['domain'] != 'unmapped'].copy()
        print(f"   [OK] Found {len(mapped_df):,} mapped soft skills (excluding {len(df) - len(mapped_df):,} unmapped)")
        
        # Map CSV domain names to full domain names (matching Task 3 QA file structure)
        # CSV has shorter names, QA file has full names - use QA file as source of truth
        # Based on task3_top_200_skills_qa.csv and task3_processing_log.json
        domain_name_mapping = {
            'Communication': 'Communication Skills',
            'Teamwork': 'Collaboration And Team Dynamics',
            'Problem-solving': 'Problem-Solving And Critical Thinking',
            'Time Management': 'Time Management And Organizational Skills',
            'Attention to Detail': 'Work Ethic And Professionalism',
            'Leadership': 'Collaboration And Team Dynamics',
            'Adaptability': 'Adaptability & Continuous Learning',
            'Creativity': 'Creativity And Inovation',
            'Emotional Intelligence': 'Emotional Intelligence (Eq)',
            'Work Ethic': 'Work Ethic And Professionalism',
        }
        
        # Build dictionary: full_domain_name -> list of LinkedIn skills
        soft_skill_domains = {}
        
        for _, row in mapped_df.iterrows():
            csv_domain = str(row['domain']).strip()
            skill = str(row['linkedin_skill']).strip()
            
            # Skip empty
            if not csv_domain or not skill or csv_domain == 'nan' or skill == 'nan':
                continue
            
            # Map to full domain name (use mapping if available, otherwise keep original)
            full_domain = domain_name_mapping.get(csv_domain, csv_domain)
            
            if full_domain not in soft_skill_domains:
                soft_skill_domains[full_domain] = []
            
            # Add skill if not already present (avoid duplicates)
            if skill not in soft_skill_domains[full_domain]:
                soft_skill_domains[full_domain].append(skill)
        
        # Warn if any domains weren't mapped
        unmapped_domains = set(mapped_df['domain'].unique()) - set(domain_name_mapping.keys())
        if unmapped_domains:
            print(f"   [WARNING] Found unmapped domain names in CSV: {unmapped_domains}")
            print(f"   These will be used as-is (may need manual mapping)")
        
        print(f"   [OK] Loaded mapping with {len(soft_skill_domains)} domains")
        for domain, skills in soft_skill_domains.items():
            print(f"     {domain}: {len(skills)} skills")
        
        return soft_skill_domains
        
    except Exception as e:
        print(f"   [ERROR] Error loading mapping file: {e}")
        raise

def normalize_text(text):
    """Normalize text for matching: lowercase, handle whitespace"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespace (but keep it for phrase matching)
    text = ' '.join(text.split())
    return text

def build_regex_patterns(soft_skill_domains):
    """Build regex patterns for each domain's terms"""
    domain_patterns = {}
    
    for domain, terms in soft_skill_domains.items():
        patterns = []
        for term in terms:
            # Normalize term
            normalized_term = normalize_text(term)
            if not normalized_term:
                continue
            
            # Escape special regex characters
            escaped_term = re.escape(normalized_term)
            
            # Use word boundaries for whole-word matching
            # Handle multi-word phrases
            if ' ' in escaped_term:
                # Multi-word phrase: match as phrase with word boundaries
                pattern = r'\b' + escaped_term + r'\b'
            else:
                # Single word: match with word boundaries
                pattern = r'\b' + escaped_term + r'\b'
            
            patterns.append((pattern, normalized_term))
        
        domain_patterns[domain] = patterns
    
    return domain_patterns

def detect_domains_in_text(text, domain_patterns):
    """Detect which domains are present in text and count matches"""
    if pd.isna(text) or text is None or not text:
        return {}, {}
    
    normalized_text = normalize_text(text)
    if not normalized_text:
        return {}, {}
    
    domain_present = {}
    domain_counts = {}
    
    for domain, patterns in domain_patterns.items():
        count = 0
        for pattern, original_term in patterns:
            matches = len(re.findall(pattern, normalized_text, re.IGNORECASE))
            if matches > 0:
                count += matches
        
        domain_counts[domain] = count
        domain_present[domain] = count > 0
    
    return domain_present, domain_counts

def save_checkpoint(df, current_idx, total_rows):
    """Save checkpoint to resume later"""
    try:
        # Save dataframe
        df.to_parquet(CHECKPOINT_FILE, index=False, engine='pyarrow')
        
        # Save checkpoint info
        checkpoint_info = {
            'last_processed_idx': int(current_idx),
            'total_rows': int(total_rows),
            'progress_percentage': float((current_idx + 1) / total_rows * 100) if total_rows > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'rows_remaining': int(total_rows - current_idx - 1)
        }
        
        with open(CHECKPOINT_INFO_FILE, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        return True
    except Exception as e:
        print(f"   [WARNING] Failed to save checkpoint: {e}")
        return False

def load_checkpoint():
    """Load checkpoint if it exists"""
    if not CHECKPOINT_FILE.exists():
        return None, None
    
    try:
        print(f"\n   [RESUME] Found checkpoint file: {CHECKPOINT_FILE.name}")
        df = pd.read_parquet(CHECKPOINT_FILE)
        
        if CHECKPOINT_INFO_FILE.exists():
            with open(CHECKPOINT_INFO_FILE, 'r') as f:
                checkpoint_info = json.load(f)
            
            last_idx = checkpoint_info.get('last_processed_idx', -1)
            print(f"   [RESUME] Last processed index: {last_idx:,}")
            print(f"   [RESUME] Progress: {checkpoint_info.get('progress_percentage', 0):.1f}%")
            print(f"   [RESUME] Rows remaining: {checkpoint_info.get('rows_remaining', 0):,}")
            
            return df, last_idx
        else:
            # Checkpoint exists but no info file - check which rows are done
            print(f"   [RESUME] Checkpoint found but no info file. Checking progress...")
            domain_names = [col.replace('text_domain_present_', '') for col in df.columns if col.startswith('text_domain_present_')]
            
            # Find first unprocessed row
            unprocessed_mask = df['text_length'] == 0
            if unprocessed_mask.any():
                last_idx = df[unprocessed_mask].index[0] - 1
                if last_idx < 0:
                    last_idx = -1
            else:
                last_idx = len(df) - 1
            
            return df, last_idx
            
    except Exception as e:
        print(f"   [WARNING] Failed to load checkpoint: {e}")
        return None, None

def process_job_postings(df, domain_patterns, start_idx=0):
    """Process all job postings and add domain detection columns"""
    total_rows = len(df)
    
    if start_idx > 0:
        print(f"\n3. Resuming processing from row {start_idx + 1:,} of {total_rows:,}...")
    else:
        print(f"\n3. Processing {total_rows:,} job postings...")
    
    # Initialize new columns if they don't exist
    domain_names = list(domain_patterns.keys())
    
    # Initialize presence columns (boolean) if they don't exist
    for domain in domain_names:
        col_name = f'text_domain_present_{domain}'
        if col_name not in df.columns:
            df[col_name] = False
    
    # Initialize count columns (integer) if they don't exist
    for domain in domain_names:
        col_name = f'text_domain_count_{domain}'
        if col_name not in df.columns:
            df[col_name] = 0
    
    # Initialize text_length column if it doesn't exist
    if 'text_length' not in df.columns:
        df['text_length'] = 0
    
    # Process each row starting from start_idx
    rows_to_process = list(range(start_idx + 1, total_rows)) if start_idx >= 0 else list(range(total_rows))
    
    if TQDM_AVAILABLE:
        iterator = tqdm(rows_to_process, desc="Processing postings", initial=start_idx + 1, total=total_rows)
    else:
        iterator = rows_to_process
        if start_idx > 0:
            print(f"   Resuming from row {start_idx + 1:,}...")
    
    last_checkpoint_time = datetime.now()
    processed_since_checkpoint = 0
    
    for idx in iterator:
        row = df.iloc[idx]
        job_text = row.get('job_text', '')
        
        # Calculate text length
        if pd.notna(job_text):
            df.at[idx, 'text_length'] = len(str(job_text))
        else:
            df.at[idx, 'text_length'] = 0
        
        # Detect domains
        domain_present, domain_counts = detect_domains_in_text(job_text, domain_patterns)
        
        # Update columns
        for domain in domain_names:
            df.at[idx, f'text_domain_present_{domain}'] = domain_present.get(domain, False)
            df.at[idx, f'text_domain_count_{domain}'] = domain_counts.get(domain, 0)
        
        # Save checkpoint periodically
        processed_since_checkpoint += 1
        current_time = datetime.now()
        time_since_checkpoint = (current_time - last_checkpoint_time).total_seconds()
        
        # Save checkpoint every N rows OR every N seconds
        if processed_since_checkpoint >= CHECKPOINT_INTERVAL or time_since_checkpoint >= CHECKPOINT_TIME_INTERVAL:
            if save_checkpoint(df, idx, total_rows):
                last_checkpoint_time = current_time
                processed_since_checkpoint = 0
                if TQDM_AVAILABLE:
                    iterator.set_postfix_str(f"Checkpoint saved at row {idx + 1:,}")
                else:
                    print(f"   [CHECKPOINT] Saved at row {idx + 1:,} ({((idx + 1) / total_rows * 100):.1f}%)")
    
    # Final checkpoint
    save_checkpoint(df, total_rows - 1, total_rows)
    
    print(f"   [OK] Processed {total_rows:,} postings")
    
    return df

def generate_summary(df, domain_patterns):
    """Generate summary statistics"""
    domain_names = list(domain_patterns.keys())
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_postings': len(df),
        'postings_with_text': len(df[df['text_length'] > 0]),
        'domain_detection': {}
    }
    
    for domain in domain_names:
        present_col = f'text_domain_present_{domain}'
        count_col = f'text_domain_count_{domain}'
        
        postings_with_domain = df[present_col].sum()
        total_matches = df[count_col].sum()
        avg_matches = df[count_col].mean() if len(df) > 0 else 0
        
        summary['domain_detection'][domain] = {
            'postings_with_domain': int(postings_with_domain),
            'percentage': float(postings_with_domain / len(df) * 100) if len(df) > 0 else 0,
            'total_matches': int(total_matches),
            'avg_matches_per_posting': float(avg_matches)
        }
    
    return summary

def save_outputs(df, summary):
    """Save output files"""
    print(f"\n4. Saving outputs...")
    
    # Save parquet file
    df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    print(f"   [OK] Saved to {OUTPUT_FILE.name}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Save processing log
    with open(LOG_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   [OK] Saved processing log to {LOG_FILE.name}")

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 4: Run Text Detector v1 on LinkedIn Text")
    print("=" * 80)
    print("\nCheckpoint/Resume: Saves progress every 1000 rows and every 5 minutes")
    print("If interrupted, run again to resume from checkpoint.")
    
    try:
        # Check for existing checkpoint
        checkpoint_df, start_idx = load_checkpoint()
        
        if checkpoint_df is not None and start_idx is not None:
            print(f"\n   [RESUME] Resuming from checkpoint...")
            print(f"   [RESUME] To start fresh, delete: {CHECKPOINT_FILE.name}")
            try:
                user_input = input(f"\n   Resume from checkpoint? (y/n, default=y): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # If running non-interactively, default to resume
                user_input = 'y'
            
            if user_input == 'n':
                print("   [OK] Starting fresh...")
                checkpoint_df = None
                start_idx = -1
                # Delete checkpoint files
                if CHECKPOINT_FILE.exists():
                    CHECKPOINT_FILE.unlink()
                if CHECKPOINT_INFO_FILE.exists():
                    CHECKPOINT_INFO_FILE.unlink()
            else:
                df = checkpoint_df
                print(f"   [OK] Resuming from row {start_idx + 1:,}")
        else:
            start_idx = -1
            df = None
        
        # Step 1: Load mapping from Task 3 CSV
        soft_skill_domains = load_mapping_from_csv(MAPPING_FILE)
        
        # Step 2: Build regex patterns
        print(f"\n2. Building regex patterns for {len(soft_skill_domains)} domains...")
        domain_patterns = build_regex_patterns(soft_skill_domains)
        total_patterns = sum(len(patterns) for patterns in domain_patterns.values())
        print(f"   [OK] Created {total_patterns} patterns")
        
        # Step 3: Load job postings (if not resuming)
        if df is None:
            print(f"\n3. Loading job postings...")
            df = pd.read_parquet(INPUT_FILE)
            print(f"   [OK] Loaded {len(df):,} job postings from {INPUT_FILE.name}")
        else:
            print(f"\n3. Using checkpoint data with {len(df):,} job postings")
        
        # Step 4: Process postings
        df = process_job_postings(df, domain_patterns, start_idx=start_idx)
        
        # Step 5: Generate summary
        print(f"\n4. Generating summary...")
        summary = generate_summary(df, domain_patterns)
        
        # Step 6: Save outputs
        save_outputs(df, summary)
        
        # Step 7: Clean up checkpoint files (processing complete)
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print(f"   [OK] Removed checkpoint file (processing complete)")
        if CHECKPOINT_INFO_FILE.exists():
            CHECKPOINT_INFO_FILE.unlink()
        
        # Step 8: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TASK 4 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings: {len(df):,}")
        print(f"  Postings with text: {summary['postings_with_text']:,}")
        print(f"\nDomain Detection Results:")
        for domain, stats in summary['domain_detection'].items():
            print(f"  {domain}:")
            print(f"    Postings with domain: {stats['postings_with_domain']:,} ({stats['percentage']:.1f}%)")
            print(f"    Total matches: {stats['total_matches']:,}")
            print(f"    Avg matches per posting: {stats['avg_matches_per_posting']:.2f}")
        print(f"\nOutput file: {OUTPUT_FILE}")
        print(f"Processing log: {LOG_FILE}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Processing interrupted by user.")
        print(f"   Progress saved to checkpoint: {CHECKPOINT_FILE.name}")
        print(f"   To resume, run the script again.")
        print(f"   To start fresh, delete: {CHECKPOINT_FILE.name}")
    except Exception as e:
        print(f"\n[ERROR] Task failed: {e}")
        print(f"   Progress saved to checkpoint: {CHECKPOINT_FILE.name}")
        print(f"   To resume, run the script again.")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
