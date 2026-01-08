"""
Task 10: Build Co-Occurrence Matrix
Goal: Calculate co-occurrence between soft-skill domains and technical skills
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
TASK3_DIR = BASE_DIR / "task3"
TASK9_DIR = BASE_DIR / "task9"
TASK10_DIR = Path(__file__).parent

# File paths
LINKEDIN_V2_FILE = TASK9_DIR / "linkedin_2024_v2_results.parquet"
SOFTSKILL_MAPPING = TASK3_DIR / "softskill_tag_mapping.csv"
TECHNICAL_SKILLS_FILE = TASK10_DIR / "technical_skills_list.csv"

OUTPUT_COOCCURRENCE = TASK10_DIR / "soft_technical_cooccurrence.csv"
OUTPUT_SUMMARY = TASK10_DIR / "task10_cooccurrence_summary.json"

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

def load_softskill_mapping():
    """Load soft-skill mapping to filter technical skills"""
    df = pd.read_csv(SOFTSKILL_MAPPING)
    soft_skills = set(df[df['domain'] != 'unmapped']['linkedin_skill'].str.lower().str.strip())
    return soft_skills

def extract_technical_skills_from_posting(skills_list, soft_skills_set):
    """Extract technical skills from a posting's skills_list"""
    import numpy as np
    technical_skills = set()
    
    # Handle numpy arrays first
    if isinstance(skills_list, np.ndarray):
        if skills_list.size == 0:
            return technical_skills
        skills_list = skills_list.tolist()
    
    # Handle different data types
    if skills_list is None:
        return technical_skills
    
    # Check for NaN/None (but not for arrays - already handled)
    if not isinstance(skills_list, np.ndarray):
        try:
            if pd.isna(skills_list):
                return technical_skills
        except (ValueError, TypeError):
            pass
    
    # Handle string representation of list
    if isinstance(skills_list, str):
        try:
            import ast
            skills_list = ast.literal_eval(skills_list)
        except:
            return technical_skills
    
    if not isinstance(skills_list, (list, tuple)):
        return technical_skills
    
    if len(skills_list) == 0:
        return technical_skills
    
    for skill in skills_list:
        if pd.isna(skill) or not skill:
            continue
        
        skill_normalized = str(skill).lower().strip()
        
        # Skip soft skills
        if skill_normalized in soft_skills_set:
            continue
        
        # Skip empty or very short
        if len(skill_normalized) < 2:
            continue
        
        technical_skills.add(skill_normalized)
    
    return technical_skills

def extract_soft_domains_from_posting(row):
    """Extract soft-skill domains present in a posting"""
    domains = set()
    
    for domain in DOMAIN_NAMES:
        present_col = f'text_domain_present_{domain}_v2'
        if present_col in row.index and row[present_col]:
            domains.add(domain)
    
    return domains

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 10: Build Co-Occurrence Matrix")
    print("=" * 80)
    
    try:
        # Step 1: Load soft-skill mapping
        print("\n1. Loading soft-skill mapping...")
        soft_skills_set = load_softskill_mapping()
        print(f"   [OK] Loaded {len(soft_skills_set)} soft-skill terms")
        
        # Step 2: Load LinkedIn data
        print("\n2. Loading LinkedIn 2024 data...")
        print(f"   File: {LINKEDIN_V2_FILE.name}")
        df = pd.read_parquet(LINKEDIN_V2_FILE)
        print(f"   [OK] Loaded {len(df):,} postings")
        
        # Step 3: Load technical skills list (for filtering to top skills)
        print("\n3. Loading technical skills list...")
        if TECHNICAL_SKILLS_FILE.exists():
            tech_skills_df = pd.read_csv(TECHNICAL_SKILLS_FILE)
            # Use top N technical skills (e.g., top 100) to keep co-occurrence manageable
            top_tech_skills = set(tech_skills_df.head(100)['technical_skill'].str.lower().str.strip())
            print(f"   [OK] Using top {len(top_tech_skills)} technical skills")
        else:
            print("   [WARNING] Technical skills file not found, using all technical skills")
            top_tech_skills = None
        
        # Step 4: Build co-occurrence matrix
        print("\n4. Building co-occurrence matrix...")
        cooccurrence = defaultdict(int)
        postings_processed = 0
        
        for idx, row in df.iterrows():
            # Extract soft domains
            soft_domains = extract_soft_domains_from_posting(row)
            
            # Extract technical skills
            technical_skills = extract_technical_skills_from_posting(
                row.get('skills_list', []), 
                soft_skills_set
            )
            
            # Filter to top technical skills if available
            if top_tech_skills:
                technical_skills = {s for s in technical_skills if s in top_tech_skills}
            
            # Count co-occurrences
            if soft_domains and technical_skills:
                postings_processed += 1
                for domain in soft_domains:
                    for tech_skill in technical_skills:
                        cooccurrence[(domain, tech_skill)] += 1
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,}/{len(df):,} postings...")
        
        print(f"   [OK] Found co-occurrences in {postings_processed:,} postings")
        print(f"   [OK] Total co-occurrence pairs: {len(cooccurrence):,}")
        
        # Step 5: Create co-occurrence dataframe
        print("\n5. Creating co-occurrence table...")
        cooccurrence_data = []
        for (domain, tech_skill), count in sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True):
            cooccurrence_data.append({
                'soft_domain': domain,
                'technical_skill': tech_skill,
                'cooccurrence_count': count,
                'percentage_of_postings': round(100 * count / len(df), 2)
            })
        
        cooccurrence_df = pd.DataFrame(cooccurrence_data)
        
        # Step 6: Save results
        print("\n6. Saving results...")
        cooccurrence_df.to_csv(OUTPUT_COOCCURRENCE, index=False)
        print(f"   [OK] Saved to {OUTPUT_COOCCURRENCE.name}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_postings': len(df),
            'postings_with_cooccurrence': postings_processed,
            'unique_cooccurrence_pairs': len(cooccurrence),
            'top_10_cooccurrences': [
                {
                    'soft_domain': domain,
                    'technical_skill': tech_skill,
                    'count': count
                }
                for (domain, tech_skill), count in sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
        
        with open(OUTPUT_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Saved summary to {OUTPUT_SUMMARY.name}")
        
        # Step 7: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("CO-OCCURRENCE MATRIX COMPLETED")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings: {len(df):,}")
        print(f"  Postings with co-occurrence: {postings_processed:,}")
        print(f"  Unique co-occurrence pairs: {len(cooccurrence):,}")
        print(f"\nTop 10 Co-Occurrences:")
        for i, item in enumerate(summary['top_10_cooccurrences'], 1):
            print(f"  {i}. {item['soft_domain']} + {item['technical_skill']}: {item['count']:,}")
        print(f"\nOutput files:")
        print(f"  Co-Occurrence Matrix: {OUTPUT_COOCCURRENCE}")
        print(f"  Summary: {OUTPUT_SUMMARY}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Co-occurrence building failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
