"""
Task 5: Dual-Signal Agreement & Mismatch Analysis (RQ1)
Goal: Quantify how often skills are tagged vs stated in job descriptions.
This is the core of RQ1 (Validity): How well do text-based soft-skill mentions 
align with explicitly tagged skills in LinkedIn 2024?
"""

import pandas as pd
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
TASK4_DIR = BASE_DIR / "task4"
OUTPUT_DIR = Path(__file__).parent

# File paths
TASK2_FILE = TASK2_DIR / "linkedin_cs_subset.parquet"
MAPPING_FILE = TASK3_DIR / "softskill_tag_mapping.csv"
TASK4_FILE = TASK4_DIR / "linkedin_text_v1.parquet"

OUTPUT_FILE = OUTPUT_DIR / "task5_agreement_analysis.parquet"
SUMMARY_FILE = OUTPUT_DIR / "task5_mismatch_summary.json"
EXAMPLES_FILE = OUTPUT_DIR / "task5_mismatch_examples.csv"
LOG_FILE = OUTPUT_DIR / "task5_processing_log.json"

def normalize_skill(skill):
    """Normalize skill name for matching"""
    if pd.isna(skill) or skill is None:
        return ""
    return str(skill).lower().strip()

def load_mapping():
    """Load skill-to-domain mapping from Task 3"""
    print("\n1. Loading skill-to-domain mapping...")
    print(f"   File: {MAPPING_FILE.name}")
    
    df = pd.read_csv(MAPPING_FILE)
    print(f"   [OK] Loaded {len(df):,} skill mappings")
    
    # Filter to only mapped skills (exclude unmapped)
    mapped_df = df[df['domain'] != 'unmapped'].copy()
    print(f"   [OK] Found {len(mapped_df):,} mapped soft skills")
    
    # Map CSV domain names to full domain names (matching Task 4)
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
    
    # Build mapping: normalized_skill -> full_domain_name
    skill_to_domain = {}
    
    for _, row in mapped_df.iterrows():
        skill = normalize_skill(row['linkedin_skill'])
        csv_domain = str(row['domain']).strip()
        
        if not skill or not csv_domain or csv_domain == 'nan':
            continue
        
        # Map to full domain name
        full_domain = domain_name_mapping.get(csv_domain, csv_domain)
        
        # Store mapping (if multiple domains for same skill, keep the most frequent)
        if skill not in skill_to_domain:
            skill_to_domain[skill] = full_domain
    
    # Build reverse mapping: domain -> set of skills
    domain_to_skills = defaultdict(set)
    for skill, domain in skill_to_domain.items():
        domain_to_skills[domain].add(skill)
    
    print(f"   [OK] Created mapping for {len(domain_to_skills)} domains")
    for domain, skills in sorted(domain_to_skills.items()):
        print(f"     {domain}: {len(skills)} skills")
    
    return skill_to_domain, domain_to_skills

def create_tag_detection(df, domain_to_skills):
    """Create tag-based domain detection columns"""
    print("\n2. Creating tag-based domain detection...")
    
    domain_names = sorted(domain_to_skills.keys())
    
    # Initialize tag detection columns
    for domain in domain_names:
        col_name = f'tag_domain_present_{domain}'
        df[col_name] = False
    
    # Process each posting
    total_rows = len(df)
    iterator = tqdm(df.iterrows(), total=total_rows, desc="   Processing tags") if TQDM_AVAILABLE else df.iterrows()
    
    for idx, row in iterator:
        skills_list = row.get('skills_list', [])
        
        # Normalize all skills in the list
        normalized_skills = set()
        
        # Handle different types of skills_list (list, numpy array, string, None)
        if skills_list is None:
            continue
        
        # Convert numpy array to list
        try:
            import numpy as np
            if isinstance(skills_list, np.ndarray):
                skills_list = skills_list.tolist()
        except:
            pass
        
        if isinstance(skills_list, list):
            if len(skills_list) == 0:
                continue
            for skill in skills_list:
                if pd.notna(skill):  # Check individual skill
                    normalized_skill = normalize_skill(skill)
                    if normalized_skill:
                        normalized_skills.add(normalized_skill)
        elif isinstance(skills_list, str):
            # Handle string representation of list
            try:
                import ast
                skills_list_parsed = ast.literal_eval(skills_list)
                if isinstance(skills_list_parsed, list) and len(skills_list_parsed) > 0:
                    for skill in skills_list_parsed:
                        if pd.notna(skill):
                            normalized_skill = normalize_skill(skill)
                            if normalized_skill:
                                normalized_skills.add(normalized_skill)
            except:
                pass
        else:
            # Skip if not a list or string
            continue
        
        # Check which domains are present
        for domain, domain_skills in domain_to_skills.items():
            if normalized_skills & domain_skills:  # Set intersection
                df.at[idx, f'tag_domain_present_{domain}'] = True
    
    # Count postings with tags per domain
    print("\n   Tag detection summary:")
    for domain in domain_names:
        col_name = f'tag_domain_present_{domain}'
        count = df[col_name].sum()
        pct = (count / total_rows * 100) if total_rows > 0 else 0
        print(f"     {domain}: {count:,} postings ({pct:.1f}%)")
    
    return df

def create_agreement_categories(df, domain_names):
    """Create agreement category columns (both, tag_only, text_only, neither)"""
    print("\n3. Creating agreement categories...")
    
    for domain in domain_names:
        tag_col = f'tag_domain_present_{domain}'
        text_col = f'text_domain_present_{domain}'
        agreement_col = f'agreement_category_{domain}'
        
        # Create agreement category
        conditions = [
            (df[tag_col] & df[text_col]),  # Both
            (df[tag_col] & ~df[text_col]),  # Tag-only
            (~df[tag_col] & df[text_col]),  # Text-only
        ]
        choices = ['both', 'tag_only', 'text_only']
        
        df[agreement_col] = pd.Series(['neither'] * len(df))  # Default
        for condition, choice in zip(conditions, choices):
            df.loc[condition, agreement_col] = choice
    
    print("   [OK] Created agreement categories for all domains")
    return df

def calculate_statistics(df, domain_names):
    """Calculate agreement statistics overall and by job family"""
    print("\n4. Calculating agreement statistics...")
    
    total_postings = len(df)
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_postings': int(total_postings),
        'overall': {},
        'by_job_family': {}
    }
    
    # Overall statistics
    for domain in domain_names:
        agreement_col = f'agreement_category_{domain}'
        category_counts = df[agreement_col].value_counts().to_dict()
        
        both = category_counts.get('both', 0)
        tag_only = category_counts.get('tag_only', 0)
        text_only = category_counts.get('text_only', 0)
        neither = category_counts.get('neither', 0)
        
        # Agreement rate = (both + neither) / total (both methods agree)
        agreement_rate = ((both + neither) / total_postings * 100) if total_postings > 0 else 0
        
        stats['overall'][domain] = {
            'both': int(both),
            'tag_only': int(tag_only),
            'text_only': int(text_only),
            'neither': int(neither),
            'agreement_rate': float(agreement_rate),
            'both_percentage': float(both / total_postings * 100) if total_postings > 0 else 0,
            'tag_only_percentage': float(tag_only / total_postings * 100) if total_postings > 0 else 0,
            'text_only_percentage': float(text_only / total_postings * 100) if total_postings > 0 else 0,
        }
    
    # Statistics by job family
    if 'job_family' in df.columns:
        job_families = df['job_family'].unique()
        for job_family in job_families:
            if pd.isna(job_family):
                continue
            
            family_df = df[df['job_family'] == job_family]
            family_total = len(family_df)
            
            if family_total == 0:
                continue
            
            stats['by_job_family'][str(job_family)] = {}
            
            for domain in domain_names:
                agreement_col = f'agreement_category_{domain}'
                category_counts = family_df[agreement_col].value_counts().to_dict()
                
                both = category_counts.get('both', 0)
                tag_only = category_counts.get('tag_only', 0)
                text_only = category_counts.get('text_only', 0)
                neither = category_counts.get('neither', 0)
                
                agreement_rate = ((both + neither) / family_total * 100) if family_total > 0 else 0
                
                stats['by_job_family'][str(job_family)][domain] = {
                    'both': int(both),
                    'tag_only': int(tag_only),
                    'text_only': int(text_only),
                    'neither': int(neither),
                    'agreement_rate': float(agreement_rate),
                    'total_postings': int(family_total)
                }
    
    return stats

def extract_examples(df, domain_names, n_examples=10):
    """Extract qualitative examples for each mismatch type per domain"""
    print("\n5. Extracting qualitative examples...")
    
    examples = []
    
    for domain in domain_names:
        agreement_col = f'agreement_category_{domain}'
        
        for category in ['both', 'tag_only', 'text_only']:
            category_df = df[df[agreement_col] == category].copy()
            
            if len(category_df) == 0:
                continue
            
            # Sample up to n_examples
            n_sample = min(n_examples, len(category_df))
            sampled = category_df.sample(n=n_sample, random_state=42) if len(category_df) > n_sample else category_df
            
            for idx, row in sampled.iterrows():
                # Get relevant text snippet (first 500 chars)
                job_text = str(row.get('job_text', ''))[:500] if pd.notna(row.get('job_text')) else ''
                
                # Get skills list
                skills_list = row.get('skills_list', [])
                if isinstance(skills_list, str):
                    try:
                        import ast
                        skills_list = ast.literal_eval(skills_list)
                    except:
                        skills_list = []
                
                # Filter to relevant skills for this domain
                relevant_skills = []
                if isinstance(skills_list, list):
                    for skill in skills_list:
                        normalized_skill = normalize_skill(skill)
                        # Check if this skill maps to this domain (simplified check)
                        if normalized_skill and domain.lower() in normalized_skill.lower():
                            relevant_skills.append(skill)
                
                example = {
                    'domain': domain,
                    'category': category,
                    'job_link': row.get('job_link', ''),
                    'job_title': row.get('job_title', ''),
                    'company': row.get('company', ''),
                    'job_family': row.get('job_family', ''),
                    'text_snippet': job_text,
                    'relevant_skills': ', '.join(relevant_skills[:5]),  # Top 5 relevant skills
                    'tag_present': bool(row.get(f'tag_domain_present_{domain}', False)),
                    'text_present': bool(row.get(f'text_domain_present_{domain}', False)),
                }
                examples.append(example)
    
    examples_df = pd.DataFrame(examples)
    print(f"   [OK] Extracted {len(examples_df):,} examples")
    
    return examples_df

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 5: Dual-Signal Agreement & Mismatch Analysis (RQ1)")
    print("=" * 80)
    
    try:
        # Step 1: Load mapping
        skill_to_domain, domain_to_skills = load_mapping()
        domain_names = sorted(domain_to_skills.keys())
        
        # Step 2: Load Task 2 data (original postings with skills_list)
        print("\n2. Loading job postings from Task 2...")
        df_task2 = pd.read_parquet(TASK2_FILE)
        print(f"   [OK] Loaded {len(df_task2):,} job postings")
        
        # Step 3: Create tag-based detection
        df_task2 = create_tag_detection(df_task2, domain_to_skills)
        
        # Step 4: Load Task 4 text-based detection
        print("\n3. Loading text-based detection from Task 4...")
        df_task4 = pd.read_parquet(TASK4_FILE)
        print(f"   [OK] Loaded {len(df_task4):,} job postings with text detection")
        
        # Step 5: Merge Task 2 and Task 4 on job_link
        print("\n4. Merging tag and text detection...")
        df = df_task2.merge(
            df_task4[['job_link'] + [f'text_domain_present_{d}' for d in domain_names] + [f'text_domain_count_{d}' for d in domain_names] + ['text_length']],
            on='job_link',
            how='inner'
        )
        print(f"   [OK] Merged dataset: {len(df):,} postings")
        
        # Step 6: Create agreement categories
        df = create_agreement_categories(df, domain_names)
        
        # Step 7: Calculate statistics
        stats = calculate_statistics(df, domain_names)
        
        # Step 8: Extract examples
        examples_df = extract_examples(df, domain_names, n_examples=10)
        
        # Step 9: Save outputs
        print("\n5. Saving outputs...")
        
        # Save combined dataset
        df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
        print(f"   [OK] Saved agreement analysis to {OUTPUT_FILE.name}")
        print(f"        Rows: {len(df):,}, Columns: {len(df.columns)}")
        
        # Save statistics
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   [OK] Saved mismatch summary to {SUMMARY_FILE.name}")
        
        # Save examples
        examples_df.to_csv(EXAMPLES_FILE, index=False)
        print(f"   [OK] Saved examples to {EXAMPLES_FILE.name}")
        print(f"        Examples: {len(examples_df):,}")
        
        # Save processing log
        log = {
            'timestamp': datetime.now().isoformat(),
            'total_postings': int(len(df)),
            'domains_analyzed': len(domain_names),
            'total_examples': int(len(examples_df)),
            'output_files': {
                'agreement_analysis': str(OUTPUT_FILE.name),
                'mismatch_summary': str(SUMMARY_FILE.name),
                'examples': str(EXAMPLES_FILE.name)
            }
        }
        with open(LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"   [OK] Saved processing log to {LOG_FILE.name}")
        
        # Step 10: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TASK 5 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings analyzed: {len(df):,}")
        print(f"  Domains analyzed: {len(domain_names)}")
        print(f"  Examples extracted: {len(examples_df):,}")
        print(f"\nOverall Agreement Rates:")
        for domain in domain_names:
            agreement_rate = stats['overall'][domain]['agreement_rate']
            both_pct = stats['overall'][domain]['both_percentage']
            tag_only_pct = stats['overall'][domain]['tag_only_percentage']
            text_only_pct = stats['overall'][domain]['text_only_percentage']
            print(f"  {domain}:")
            print(f"    Agreement rate: {agreement_rate:.1f}%")
            print(f"    Both: {both_pct:.1f}% | Tag-only: {tag_only_pct:.1f}% | Text-only: {text_only_pct:.1f}%")
        print(f"\nOutput files:")
        print(f"  Agreement analysis: {OUTPUT_FILE}")
        print(f"  Mismatch summary: {SUMMARY_FILE}")
        print(f"  Examples: {EXAMPLES_FILE}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Task failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
