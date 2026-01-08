"""
Task 9: Run Detector v2 on LinkedIn 2024
Goal: Apply refined lexicon v2 to full LinkedIn CS subset (2024) for bridge-year comparison
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent.parent
TASK2_DIR = BASE_DIR / "task2"
TASK6_DIR = BASE_DIR / "task6"
TASK9_DIR = Path(__file__).parent

# File paths
LINKEDIN_CS_FILE = TASK2_DIR / "linkedin_cs_subset.parquet"
LEXICON_V2_FILE = TASK6_DIR / "lexicon_v2.csv"

OUTPUT_FILE = TASK9_DIR / "linkedin_2024_v2_results.parquet"
SUMMARY_FILE = TASK9_DIR / "linkedin_2024_v2_summary.json"
LOG_FILE = TASK9_DIR / "task9_detector_v2_linkedin_log.json"

# Domain names
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

def normalize_text(text):
    """Normalize text for matching: lowercase, handle whitespace"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def load_lexicon_v2():
    """Load lexicon v2 from Task 6"""
    print("\n1. Loading lexicon v2...")
    print(f"   File: {LEXICON_V2_FILE.name}")
    
    df = pd.read_csv(LEXICON_V2_FILE)
    mapped_df = df[df['domain'] != 'unmapped'].copy()
    
    # Map CSV domain names to full domain names
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
    
    # Build dictionary: full_domain_name -> list of skills
    soft_skill_domains = {}
    
    for _, row in mapped_df.iterrows():
        csv_domain = str(row['domain']).strip()
        skill = str(row['linkedin_skill']).strip()
        
        if not csv_domain or not skill or csv_domain == 'nan' or skill == 'nan':
            continue
        
        full_domain = domain_name_mapping.get(csv_domain, csv_domain)
        
        if full_domain not in soft_skill_domains:
            soft_skill_domains[full_domain] = []
        
        if skill not in soft_skill_domains[full_domain]:
            soft_skill_domains[full_domain].append(skill)
    
    print(f"   [OK] Loaded lexicon v2 with {len(soft_skill_domains)} domains")
    total_skills = sum(len(skills) for skills in soft_skill_domains.values())
    print(f"   [OK] Total skills: {total_skills}")
    
    return soft_skill_domains

def build_regex_patterns(soft_skill_domains):
    """Build regex patterns for each domain's terms"""
    domain_patterns = {}
    
    for domain, terms in soft_skill_domains.items():
        patterns = []
        for term in terms:
            normalized_term = normalize_text(term)
            if not normalized_term:
                continue
            
            escaped_term = re.escape(normalized_term)
            
            if ' ' in escaped_term:
                pattern = r'\b' + escaped_term + r'\b'
            else:
                pattern = r'\b' + escaped_term + r'\b'
            
            patterns.append((pattern, normalized_term))
        
        domain_patterns[domain] = patterns
    
    return domain_patterns

def detect_domains_in_text(text, domain_patterns):
    """Detect which domains are present in text and count matches"""
    text_normalized = normalize_text(text)
    
    domain_results = {}
    
    for domain, patterns in domain_patterns.items():
        matches = 0
        for pattern, term in patterns:
            found = len(re.findall(pattern, text_normalized, re.IGNORECASE))
            matches += found
        
        domain_results[domain] = {
            'present': matches > 0,
            'count': matches
        }
    
    return domain_results

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 9: Run Detector v2 on LinkedIn 2024")
    print("=" * 80)
    
    try:
        # Step 1: Load lexicon v2
        soft_skill_domains = load_lexicon_v2()
        
        # Step 2: Build regex patterns
        print("\n2. Building regex patterns...")
        domain_patterns = build_regex_patterns(soft_skill_domains)
        print(f"   [OK] Built patterns for {len(domain_patterns)} domains")
        
        # Step 3: Load LinkedIn CS subset
        print("\n3. Loading LinkedIn CS subset...")
        print(f"   File: {LINKEDIN_CS_FILE.name}")
        df = pd.read_parquet(LINKEDIN_CS_FILE)
        print(f"   [OK] Loaded {len(df):,} postings")
        
        # Note: LinkedIn data is from 2024 (from Task 1/2)
        # Filter if there's a year column, otherwise assume all are 2024
        if 'first_seen' in df.columns:
            # Extract year from first_seen if it's a date
            try:
                df['year'] = pd.to_datetime(df['first_seen'], errors='coerce').dt.year
                df_2024 = df[df['year'] == 2024].copy()
                if len(df_2024) > 0:
                    df = df_2024
                    print(f"   [OK] Filtered to 2024: {len(df):,} postings")
            except:
                print("   [NOTE] Could not parse year from first_seen, using all postings")
        
        # Step 4: Apply detector v2
        print("\n4. Applying detector v2...")
        domain_names = sorted(soft_skill_domains.keys())
        
        # Initialize columns
        for domain in domain_names:
            df[f'text_domain_present_{domain}_v2'] = False
            df[f'text_domain_count_{domain}_v2'] = 0
        
        # Process each posting
        for idx, row in df.iterrows():
            job_text = row.get('job_text', '')
            
            if pd.isna(job_text) or not job_text:
                continue
            
            results = detect_domains_in_text(job_text, domain_patterns)
            
            for domain in domain_names:
                if domain in results:
                    df.at[idx, f'text_domain_present_{domain}_v2'] = results[domain]['present']
                    df.at[idx, f'text_domain_count_{domain}_v2'] = results[domain]['count']
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,}/{len(df):,} postings...")
        
        print(f"   [OK] Completed detection for {len(df):,} postings")
        
        # Step 5: Calculate summary statistics
        print("\n5. Calculating summary statistics...")
        summary = {
            'timestamp': datetime.now().isoformat(),
            'lexicon_version': 'v2',
            'total_postings': len(df),
            'domains': {}
        }
        
        # Overall statistics
        for domain in domain_names:
            present_col = f'text_domain_present_{domain}_v2'
            count_col = f'text_domain_count_{domain}_v2'
            
            if present_col in df.columns:
                n_present = df[present_col].sum()
                total_matches = df[count_col].sum() if count_col in df.columns else 0
                avg_matches = total_matches / n_present if n_present > 0 else 0
                
                summary['domains'][domain] = {
                    'postings_with_domain': int(n_present),
                    'coverage_percent': round(100 * n_present / len(df), 1),
                    'total_matches': int(total_matches),
                    'avg_matches_per_posting': round(avg_matches, 2)
                }
        
        # Step 6: Save results
        print("\n6. Saving results...")
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"   [OK] Saved results to {OUTPUT_FILE.name}")
        
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Saved summary to {SUMMARY_FILE.name}")
        
        # Step 7: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("DETECTOR V2 ON LINKEDIN 2024 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings: {len(df):,}")
        print(f"  Domains detected: {len(domain_names)}")
        print(f"\nDomain detection (v2):")
        for domain in domain_names:
            if domain in summary['domains']:
                stats = summary['domains'][domain]
                print(f"  {domain}: {stats['postings_with_domain']:,} ({stats['coverage_percent']}%)")
        print(f"\nOutput files:")
        print(f"  Results: {OUTPUT_FILE}")
        print(f"  Summary: {SUMMARY_FILE}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
