"""
Task 10: Extract Technical Skills
Goal: Filter out soft-skill tags and extract technical skills from LinkedIn postings
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# Configuration
BASE_DIR = Path(__file__).parent.parent
TASK3_DIR = BASE_DIR / "task3"
TASK9_DIR = BASE_DIR / "task9"
TASK10_DIR = Path(__file__).parent

# File paths
LINKEDIN_V2_FILE = TASK9_DIR / "linkedin_2024_v2_results.parquet"
SOFTSKILL_MAPPING = TASK3_DIR / "softskill_tag_mapping.csv"

OUTPUT_TECHNICAL_SKILLS = TASK10_DIR / "technical_skills_list.csv"
OUTPUT_SUMMARY = TASK10_DIR / "task10_technical_skills_summary.json"

def load_softskill_mapping():
    """Load soft-skill mapping to identify which skills to exclude"""
    print("\n1. Loading soft-skill mapping...")
    print(f"   File: {SOFTSKILL_MAPPING.name}")
    
    df = pd.read_csv(SOFTSKILL_MAPPING)
    # Get all soft skills (domain != 'unmapped')
    soft_skills = set(df[df['domain'] != 'unmapped']['linkedin_skill'].str.lower().str.strip())
    
    print(f"   [OK] Loaded {len(soft_skills)} soft-skill terms to exclude")
    return soft_skills

def extract_technical_skills(df, soft_skills_set):
    """Extract technical skills from postings, excluding soft skills"""
    print("\n2. Extracting technical skills...")
    
    all_technical_skills = Counter()
    postings_with_skills = 0
    
    import numpy as np
    
    for idx, row in df.iterrows():
        skills_list = row.get('skills_list', [])
        
        # Handle numpy arrays first
        if isinstance(skills_list, np.ndarray):
            if skills_list.size == 0:
                continue
            skills_list = skills_list.tolist()
        
        # Handle different data types
        if skills_list is None:
            continue
        
        # Check for NaN/None (but not for arrays - already handled)
        if not isinstance(skills_list, np.ndarray):
            try:
                if pd.isna(skills_list):
                    continue
            except (ValueError, TypeError):
                pass
        
        # Handle string representation of list
        if isinstance(skills_list, str):
            try:
                import ast
                skills_list = ast.literal_eval(skills_list)
            except:
                continue
        
        if not isinstance(skills_list, (list, tuple)):
            continue
        
        if len(skills_list) == 0:
            continue
        
        posting_skills = set()
        for skill in skills_list:
            if pd.isna(skill) or not skill:
                continue
            
            skill_normalized = str(skill).lower().strip()
            
            # Skip if it's a soft skill
            if skill_normalized in soft_skills_set:
                continue
            
            # Skip empty or very short skills
            if len(skill_normalized) < 2:
                continue
            
            posting_skills.add(skill_normalized)
        
        if posting_skills:
            postings_with_skills += 1
            for skill in posting_skills:
                all_technical_skills[skill] += 1
        
        if (idx + 1) % 10000 == 0:
            print(f"   Processed {idx + 1:,}/{len(df):,} postings...")
    
    print(f"   [OK] Extracted technical skills from {postings_with_skills:,} postings")
    print(f"   [OK] Found {len(all_technical_skills):,} unique technical skills")
    
    return all_technical_skills

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 10: Extract Technical Skills")
    print("=" * 80)
    
    try:
        # Step 1: Load soft-skill mapping
        soft_skills_set = load_softskill_mapping()
        
        # Step 2: Load LinkedIn data
        print("\n2. Loading LinkedIn 2024 data...")
        print(f"   File: {LINKEDIN_V2_FILE.name}")
        df = pd.read_parquet(LINKEDIN_V2_FILE)
        print(f"   [OK] Loaded {len(df):,} postings")
        
        # Step 3: Extract technical skills
        technical_skills_counter = extract_technical_skills(df, soft_skills_set)
        
        # Step 4: Create technical skills list
        print("\n3. Creating technical skills list...")
        technical_skills_data = []
        for skill, count in technical_skills_counter.most_common():
            technical_skills_data.append({
                'technical_skill': skill,
                'frequency': count,
                'percentage_of_postings': round(100 * count / len(df), 2)
            })
        
        technical_skills_df = pd.DataFrame(technical_skills_data)
        
        # Step 5: Save results
        print("\n4. Saving results...")
        technical_skills_df.to_csv(OUTPUT_TECHNICAL_SKILLS, index=False)
        print(f"   [OK] Saved to {OUTPUT_TECHNICAL_SKILLS.name}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_postings': len(df),
            'unique_technical_skills': len(technical_skills_counter),
            'top_10_technical_skills': [
                {'skill': skill, 'frequency': count} 
                for skill, count in technical_skills_counter.most_common(10)
            ]
        }
        
        with open(OUTPUT_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Saved summary to {OUTPUT_SUMMARY.name}")
        
        # Step 6: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TECHNICAL SKILLS EXTRACTION COMPLETED")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total postings: {len(df):,}")
        print(f"  Unique technical skills: {len(technical_skills_counter):,}")
        print(f"\nTop 10 Technical Skills:")
        for i, (skill, count) in enumerate(technical_skills_counter.most_common(10), 1):
            pct = 100 * count / len(df)
            print(f"  {i}. {skill}: {count:,} ({pct:.1f}%)")
        print(f"\nOutput files:")
        print(f"  Technical Skills List: {OUTPUT_TECHNICAL_SKILLS}")
        print(f"  Summary: {OUTPUT_SUMMARY}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
