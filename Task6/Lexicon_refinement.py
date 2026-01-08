"""
Task 6: Lexicon Refinement
Goal: Improve text detector by refining lexicon based on tag-only cases from Task 5.

Steps:
1. Identify high-frequency tag skills that text detector misses (tag-only cases)
2. Extract common phrasings from job descriptions
3. Create refined lexicon v2
4. Generate changelog documenting all changes
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

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
TASK3_DIR = BASE_DIR / "task3"
TASK5_DIR = BASE_DIR / "task5"
OUTPUT_DIR = Path(__file__).parent

# File paths
TASK5_FILE = TASK5_DIR / "task5_agreement_analysis.parquet"
MAPPING_FILE = TASK3_DIR / "softskill_tag_mapping.csv"  # Current lexicon v1

OUTPUT_LEXICON_V2 = OUTPUT_DIR / "lexicon_v2.csv"
OUTPUT_CHANGELOG = OUTPUT_DIR / "lexicon_changelog.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "task6_lexicon_refinement_summary.json"

def normalize_skill(skill):
    """Normalize skill name for matching"""
    if pd.isna(skill) or skill is None:
        return ""
    return str(skill).lower().strip()

def load_current_lexicon():
    """Load current lexicon v1 from Task 3 mapping"""
    print("\n1. Loading current lexicon (v1)...")
    print(f"   File: {MAPPING_FILE.name}")
    
    df = pd.read_csv(MAPPING_FILE)
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
    
    # Build lexicon: domain -> set of skills (current v1)
    lexicon = defaultdict(set)
    
    # Also build full mapping: normalized_skill -> full_domain (for validation)
    full_skill_to_domain = {}
    
    for _, row in mapped_df.iterrows():
        skill = normalize_skill(row['linkedin_skill'])
        csv_domain = str(row['domain']).strip()
        
        if not skill or not csv_domain or csv_domain == 'nan':
            continue
        
        full_domain = domain_name_mapping.get(csv_domain, csv_domain)
        lexicon[full_domain].add(skill)
        full_skill_to_domain[skill] = full_domain
    
    print(f"   [OK] Loaded lexicon v1 with {len(lexicon)} domains")
    total_skills = sum(len(skills) for skills in lexicon.values())
    print(f"   [OK] Total skills: {total_skills}")
    print(f"   [OK] Loaded full mapping for validation: {len(full_skill_to_domain):,} skills")
    
    return lexicon, domain_name_mapping, full_skill_to_domain

def analyze_tag_only_cases(df, domain_names):
    """Analyze tag-only cases to find missed skills"""
    print("\n2. Analyzing tag-only cases...")
    
    missed_skills_by_domain = defaultdict(Counter)
    
    for domain in domain_names:
        tag_col = f'tag_domain_present_{domain}'
        text_col = f'text_domain_present_{domain}'
        agreement_col = f'agreement_category_{domain}'
        
        # Find tag-only cases
        tag_only_df = df[df[agreement_col] == 'tag_only'].copy()
        
        if len(tag_only_df) == 0:
            print(f"   {domain}: No tag-only cases")
            continue
        
        print(f"   {domain}: {len(tag_only_df):,} tag-only postings")
        
        # Extract skills from tag-only postings
        for idx, row in tag_only_df.iterrows():
            skills_list = row.get('skills_list', [])
            
            # Handle None
            if skills_list is None:
                continue
            
            # Convert numpy array to list if needed
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
                            missed_skills_by_domain[domain][normalized_skill] += 1
    
    # Print summary
    print("\n   Missed skills summary (top 10 per domain):")
    for domain in domain_names:
        if domain in missed_skills_by_domain:
            top_skills = missed_skills_by_domain[domain].most_common(10)
            if top_skills:
                print(f"     {domain}:")
                for skill, count in top_skills:
                    print(f"       {skill}: {count:,} occurrences")
    
    return missed_skills_by_domain

def extract_common_phrasings(df, domain, missed_skills, n_samples=100):
    """Extract common phrasings from job_text for tag-only cases"""
    print(f"\n3. Extracting common phrasings for {domain}...")
    
    tag_col = f'tag_domain_present_{domain}'
    text_col = f'text_domain_present_{domain}'
    agreement_col = f'agreement_category_{domain}'
    
    # Get tag-only postings
    tag_only_df = df[df[agreement_col] == 'tag_only'].copy()
    
    if len(tag_only_df) == 0:
        return []
    
    # Sample postings for analysis
    sample_size = min(n_samples, len(tag_only_df))
    sampled_df = tag_only_df.sample(n=sample_size, random_state=42) if len(tag_only_df) > sample_size else tag_only_df
    
    # Extract phrases containing missed skills
    phrases = []
    missed_skill_set = set(missed_skills.keys())
    
    for idx, row in sampled_df.iterrows():
        job_text = str(row.get('job_text', ''))
        if pd.isna(job_text) or not job_text:
            continue
        
        job_text_lower = job_text.lower()
        
        # Look for missed skills in text
        for skill in missed_skill_set:
            if skill in job_text_lower:
                # Extract context around the skill (20 words before and after)
                pattern = re.escape(skill)
                matches = list(re.finditer(pattern, job_text_lower))
                
                for match in matches[:3]:  # Limit to 3 matches per posting
                    start = max(0, match.start() - 200)
                    end = min(len(job_text), match.end() + 200)
                    context = job_text[start:end]
                    
                    # Extract sentence or phrase
                    sentences = re.split(r'[.!?]\s+', context)
                    for sentence in sentences:
                        if skill in sentence.lower():
                            phrases.append(sentence.strip())
    
    return phrases

def generate_variants(skill):
    """Generate common variants of a skill term"""
    variants = set()
    variants.add(skill)  # Include original
    
    # Hyphen <-> space variants
    if '-' in skill:
        # "problem-solving" -> "problem solving"
        variants.add(skill.replace('-', ' '))
    if ' ' in skill:
        # "problem solving" -> "problem-solving"
        variants.add(skill.replace(' ', '-'))
        # "problem solving" -> "problemsolving"
        variants.add(skill.replace(' ', ''))
    
    # Add "skills" suffix variants
    if not skill.endswith(' skills'):
        variants.add(skill + ' skills')
    if not skill.endswith(' skill'):
        variants.add(skill + ' skill')
    
    # Remove "skills" suffix variants
    if skill.endswith(' skills'):
        base = skill[:-7].strip()
        variants.add(base)
    if skill.endswith(' skill'):
        base = skill[:-6].strip()
        variants.add(base)
    
    return variants

def create_refined_lexicon(lexicon_v1, missed_skills_by_domain, full_skill_to_domain, min_frequency=10):
    """Create refined lexicon v2 by adding variant forms and high-frequency missed skills"""
    print("\n4. Creating refined lexicon v2...")
    print("   Strategy: Add variant forms of existing skills + legitimate soft skills from Task 3")
    
    lexicon_v2 = defaultdict(set)
    changelog = []
    
    # Copy v1 lexicon to v2
    for domain, skills in lexicon_v1.items():
        lexicon_v2[domain] = skills.copy()
    
    # Strategy 1: Add variant forms of existing lexicon v1 skills
    print("   Generating variant forms of existing skills...")
    variant_count = 0
    for domain, skills in lexicon_v1.items():
        for skill in skills:
            variants = generate_variants(skill)
            for variant in variants:
                if variant != skill and variant not in lexicon_v2[domain]:
                    # Add variant if:
                    # 1. It's already mapped in Task 3 to the same domain, OR
                    # 2. It's a simple variant (hyphen/space) and not a technical skill
                    should_add = False
                    rationale = ""
                    
                    if variant in full_skill_to_domain:
                        mapped_domain = full_skill_to_domain[variant]
                        if mapped_domain == domain:
                            should_add = True
                            rationale = f'Variant form of "{skill}" - already mapped in Task 3'
                    else:
                        # Check if it's a simple variant (hyphen/space change only)
                        # and not obviously technical (no programming languages, tools, etc.)
                        is_simple_variant = (
                            variant.replace('-', ' ').replace(' ', '') == skill.replace('-', ' ').replace(' ', '') or
                            variant.replace(' ', '-') == skill.replace(' ', '-') or
                            variant.replace(' ', '') == skill.replace(' ', '')
                        )
                        
                        # Filter out technical terms
                        technical_indicators = ['python', 'java', 'sql', 'aws', 'docker', 'kubernetes', 'javascript', 
                                              'engineering', 'degree', 'bachelor', 'master', 'autocad', 'solidworks']
                        is_technical = any(indicator in variant.lower() for indicator in technical_indicators)
                        
                        if is_simple_variant and not is_technical and len(variant) < 50:
                            should_add = True
                            rationale = f'Variant form of "{skill}" - hyphen/space variation'
                    
                    if should_add:
                        lexicon_v2[domain].add(variant)
                        variant_count += 1
                        changelog.append({
                            'domain': domain,
                            'term': variant,
                            'action': 'added',
                            'frequency': 0,  # Variant, not from tag-only analysis
                            'rationale': rationale
                        })
    print(f"      Added {variant_count} variant forms")
    
    # Strategy 2: Add high-frequency missed skills that are legitimate soft skills
    print("   Adding high-frequency tag-only skills...")
    for domain, skill_counts in missed_skills_by_domain.items():
        for skill, count in skill_counts.items():
            # Only add if:
            # 1. Meets frequency threshold
            # 2. Not already in lexicon v2
            # 3. Skill is already mapped to a soft-skill domain in Task 3 (legitimate soft skill)
            # 4. The mapped domain matches the current domain we're analyzing
            if count >= min_frequency:
                if skill not in lexicon_v2[domain]:
                    # Check if skill is in full Task 3 mapping (legitimate soft skill)
                    if skill in full_skill_to_domain:
                        mapped_domain = full_skill_to_domain[skill]
                        # Only add if it maps to the same domain we're analyzing
                        if mapped_domain == domain:
                            lexicon_v2[domain].add(skill)
                            changelog.append({
                                'domain': domain,
                                'term': skill,
                                'action': 'added',
                                'frequency': int(count),
                                'rationale': f'High-frequency tag-only skill ({count} occurrences) - already mapped in Task 3'
                            })
    
    print(f"   [OK] Added {len(changelog)} new terms to lexicon")
    
    return lexicon_v2, changelog

def save_lexicon_v2(lexicon_v2, domain_name_mapping):
    """Save refined lexicon v2 to CSV"""
    print("\n5. Saving lexicon v2...")
    
    # Reverse domain mapping for CSV
    reverse_mapping = {v: k for k, v in domain_name_mapping.items()}
    
    rows = []
    for domain, skills in sorted(lexicon_v2.items()):
        csv_domain = reverse_mapping.get(domain, domain)
        for skill in sorted(skills):
            rows.append({
                'linkedin_skill': skill,
                'domain': csv_domain,
                'notes': 'Lexicon v2 - refined from tag-only analysis',
                'confidence': 'high',
                'version': 'v2'
            })
    
    df_v2 = pd.DataFrame(rows)
    df_v2.to_csv(OUTPUT_LEXICON_V2, index=False)
    print(f"   [OK] Saved lexicon v2 to {OUTPUT_LEXICON_V2.name}")
    print(f"        Total terms: {len(df_v2):,}")
    
    return df_v2

def save_changelog(changelog):
    """Save changelog to CSV"""
    print("\n6. Saving changelog...")
    
    if not changelog:
        print("   [WARNING] No changes to document")
        # Create empty changelog
        df_changelog = pd.DataFrame(columns=['domain', 'term', 'action', 'frequency', 'rationale'])
    else:
        df_changelog = pd.DataFrame(changelog)
    
    df_changelog.to_csv(OUTPUT_CHANGELOG, index=False)
    print(f"   [OK] Saved changelog to {OUTPUT_CHANGELOG.name}")
    print(f"        Total changes: {len(df_changelog):,}")
    
    return df_changelog

def generate_summary(lexicon_v1, lexicon_v2, changelog):
    """Generate summary statistics"""
    v1_total = sum(len(skills) for skills in lexicon_v1.values())
    v2_total = sum(len(skills) for skills in lexicon_v2.values())
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'lexicon_v1': {
            'total_terms': int(v1_total),
            'domains': len(lexicon_v1)
        },
        'lexicon_v2': {
            'total_terms': int(v2_total),
            'domains': len(lexicon_v2),
            'new_terms': int(v2_total - v1_total)
        },
        'changes': {
            'total_additions': len([c for c in changelog if c.get('action') == 'added']),
            'total_removals': len([c for c in changelog if c.get('action') == 'removed']),
            'by_domain': {}
        }
    }
    
    # Count changes by domain
    for change in changelog:
        domain = change['domain']
        if domain not in summary['changes']['by_domain']:
            summary['changes']['by_domain'][domain] = {'added': 0, 'removed': 0}
        summary['changes']['by_domain'][domain][change['action']] += 1
    
    return summary

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 6: Lexicon Refinement (Option A)")
    print("=" * 80)
    
    try:
        # Step 1: Load current lexicon v1 and full Task 3 mapping
        lexicon_v1, domain_name_mapping, full_skill_to_domain = load_current_lexicon()
        domain_names = sorted(lexicon_v1.keys())
        
        # Step 2: Load Task 5 agreement analysis
        print("\n2. Loading Task 5 agreement analysis...")
        df = pd.read_parquet(TASK5_FILE)
        print(f"   [OK] Loaded {len(df):,} postings")
        
        # Step 3: Analyze tag-only cases
        missed_skills_by_domain = analyze_tag_only_cases(df, domain_names)
        
        # Step 4: Create refined lexicon v2 (only legitimate soft skills)
        lexicon_v2, changelog = create_refined_lexicon(lexicon_v1, missed_skills_by_domain, full_skill_to_domain, min_frequency=10)
        
        # Step 5: Save lexicon v2
        df_v2 = save_lexicon_v2(lexicon_v2, domain_name_mapping)
        
        # Step 6: Save changelog
        df_changelog = save_changelog(changelog)
        
        # Step 7: Generate and save summary
        print("\n7. Generating summary...")
        summary = generate_summary(lexicon_v1, lexicon_v2, changelog)
        
        with open(OUTPUT_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Saved summary to {OUTPUT_SUMMARY.name}")
        
        # Step 8: Print final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TASK 6 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Lexicon v1 terms: {summary['lexicon_v1']['total_terms']:,}")
        print(f"  Lexicon v2 terms: {summary['lexicon_v2']['total_terms']:,}")
        print(f"  New terms added: {summary['lexicon_v2']['new_terms']:,}")
        print(f"  Total additions: {summary['changes']['total_additions']:,}")
        print(f"  Total removals: {summary['changes']['total_removals']:,}")
        print(f"\nChanges by domain:")
        for domain, counts in summary['changes']['by_domain'].items():
            print(f"  {domain}: +{counts['added']} added, -{counts['removed']} removed")
        print(f"\nOutput files:")
        print(f"  Lexicon v2: {OUTPUT_LEXICON_V2}")
        print(f"  Changelog: {OUTPUT_CHANGELOG}")
        print(f"  Summary: {OUTPUT_SUMMARY}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Task failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
