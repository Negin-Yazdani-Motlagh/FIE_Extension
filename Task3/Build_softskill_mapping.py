"""
Task 3: Build a "soft-skill tag mapping" from LinkedIn skills
Goal: Map LinkedIn skill tags to your soft-skill domains (structured signal).
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
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
TASK2_DIR = BASE_DIR / "task2"
OUTPUT_DIR = Path(__file__).parent

# File paths
INPUT_FILE = TASK2_DIR / "linkedin_cs_subset.parquet"
MAPPING_FILE = OUTPUT_DIR / "softskill_tag_mapping.csv"
COVERAGE_REPORT_FILE = OUTPUT_DIR / "task3_coverage_report.json"
TOP_SKILLS_QA_FILE = OUTPUT_DIR / "task3_top_200_skills_qa.csv"
LOG_FILE = OUTPUT_DIR / "task3_processing_log.json"

# Soft-skill domains and their dictionary terms (exact matching)
SOFT_SKILL_DOMAINS = {
    'Communication': [
        'communication', 'verbal communication', 'written communication', 'presentation', 'public speaking',
        'interpersonal communication', 'listening', 'articulate', 'explain', 'convey', 'express',
        'stakeholder management', 'client communication', 'verbal', 'written', 'interpersonal'
    ],
    'Teamwork': [
        'teamwork', 'collaboration', 'collaborative', 'team player', 'team-oriented', 'team-oriented',
        'cooperation', 'working with others', 'cross-functional', 'team building', 'team work',
        'team-work', 'collaborative work'
    ],
    'Leadership': [
        'leadership', 'leading', 'mentor', 'mentoring', 'coach', 'coaching',
        'manage', 'management', 'supervision', 'supervise', 'direct', 'guide',
        'influence', 'delegate', 'empower', 'inspire', 'motivate', 'people management'
    ],
    'Problem-solving': [
        'problem solving', 'problem-solving', 'analytical', 'analysis', 'critical thinking',
        'troubleshooting', 'debugging', 'resolve', 'solution', 'solve problems',
        'decision making', 'decision-making', 'judgment', 'reasoning', 'analytical thinking'
    ],
    'Adaptability': [
        'adaptability', 'adaptable', 'flexible', 'flexibility', 'agile', 'versatile',
        'change management', 'embrace change', 'resilient', 'resilience', 'adjust'
    ],
    'Creativity': [
        'creativity', 'creative', 'innovation', 'innovative', 'innovate', 'imagination',
        'think outside the box', 'brainstorm', 'ideation', 'original thinking'
    ],
    'Time Management': [
        'time management', 'organizational', 'organization', 'prioritize', 'prioritization',
        'multitask', 'multi-task', 'deadline', 'efficient', 'efficiency', 'planning',
        'project management', 'task management', 'time-management'
    ],
    'Work Ethic': [
        'work ethic', 'diligent', 'diligence', 'dedicated', 'dedication', 'commitment',
        'reliable', 'reliability', 'responsible', 'responsibility', 'accountable',
        'self-motivated', 'proactive', 'initiative', 'hardworking', 'hard-working', 'work-ethic'
    ],
    'Emotional Intelligence': [
        'emotional intelligence', 'empathy', 'empathetic', 'self-awareness', 'self awareness',
        'emotional awareness', 'social skills', 'interpersonal skills', 'people skills',
        'relationship building', 'conflict resolution', 'diplomacy', 'emotional-intelligence'
    ],
    'Attention to Detail': [
        'attention to detail', 'detail-oriented', 'detail oriented', 'meticulous',
        'precision', 'accurate', 'accuracy', 'thorough', 'thoroughness', 'quality',
        'quality assurance', 'quality control', 'attention-to-detail'
    ]
}

# Synonym mapping: LinkedIn skill variations -> standard dictionary terms
SYNONYM_MAPPING = {
    # Teamwork synonyms
    'collaboration': 'teamwork',
    'collaborative': 'teamwork',
    'cooperation': 'teamwork',
    'team player': 'teamwork',
    'team-oriented': 'teamwork',
    'team work': 'teamwork',
    'team-work': 'teamwork',
    'working with others': 'teamwork',
    'cross-functional': 'teamwork',
    'team building': 'teamwork',
    
    # Communication synonyms
    'verbal communication': 'communication',
    'written communication': 'communication',
    'interpersonal communication': 'communication',
    'client communication': 'communication',
    'stakeholder management': 'communication',
    'communication skills': 'communication',
    'technical writing': 'communication',
    'customer service': 'communication',
    
    # Leadership synonyms
    'people management': 'leadership',
    'mentoring': 'leadership',
    'coaching': 'leadership',
    'supervision': 'leadership',
    
    # Problem-solving synonyms
    'analytical thinking': 'problem solving',
    'critical thinking': 'problem solving',
    'troubleshooting': 'problem solving',
    'debugging': 'problem solving',
    'problemsolving': 'problem solving',
    'problemsolving skills': 'problem solving',
    'analytical skills': 'problem solving',
    'problem solving skills': 'problem solving',
    
    # Time Management synonyms
    'organizational': 'time management',
    'organization': 'time management',
    'prioritization': 'time management',
    'project management': 'time management',
    'task management': 'time management',
    'organizational skills': 'time management',
    
    # Work Ethic synonyms
    'diligence': 'work ethic',
    'dedication': 'work ethic',
    'commitment': 'work ethic',
    'reliability': 'work ethic',
    'responsibility': 'work ethic',
    
    # Emotional Intelligence synonyms
    'self awareness': 'emotional intelligence',
    'social skills': 'emotional intelligence',
    'interpersonal skills': 'emotional intelligence',
    'people skills': 'emotional intelligence',
    'relationship building': 'emotional intelligence',
    'conflict resolution': 'emotional intelligence',
    
    # Attention to Detail synonyms
    'detail oriented': 'attention to detail',
    'detail-oriented': 'attention to detail',
    'quality assurance': 'attention to detail',
    'quality control': 'attention to detail'
}

def normalize_skill(skill):
    """Normalize a skill string: lowercase, trim, unify variants"""
    if pd.isna(skill) or not skill:
        return ""
    
    # Convert to string and lowercase
    normalized = str(skill).lower().strip()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Common variant unifications
    variants = {
        'team work': 'teamwork',
        'team-work': 'teamwork',
        'problem solving': 'problem-solving',
        'problem-solving': 'problem-solving',
        'decision making': 'decision-making',
        'decision-making': 'decision-making',
        'time management': 'time-management',
        'work ethic': 'work-ethic',
        'emotional intelligence': 'emotional-intelligence',
        'attention to detail': 'attention-to-detail',
        'detail oriented': 'detail-oriented',
        'self motivated': 'self-motivated',
        'self-motivated': 'self-motivated',
        'cross functional': 'cross-functional',
        'multi task': 'multitask',
        'multi-task': 'multitask'
    }
    
    for variant, standard in variants.items():
        if variant in normalized:
            normalized = normalized.replace(variant, standard)
    
    return normalized

def extract_all_skills(df):
    """Extract all unique skills from the dataset and count frequencies"""
    print("\n1. Extracting all skills from job postings...")
    
    all_skills = []
    
    if TQDM_AVAILABLE:
        iterator = tqdm(df['skills_list'], desc="   Processing jobs")
    else:
        iterator = df['skills_list']
    
    for skills_list in iterator:
        # Skip None or NaN values
        if skills_list is None:
            continue
        try:
            if isinstance(skills_list, float) and pd.isna(skills_list):
                continue
        except (TypeError, ValueError):
            pass
        
        # Check if it's iterable (list, array, etc.)
        try:
            if hasattr(skills_list, '__len__') and len(skills_list) > 0:
                # It's a collection - iterate through items
                for skill in skills_list:
                    if skill is not None:
                        try:
                            if not (isinstance(skill, float) and pd.isna(skill)):
                                all_skills.append(str(skill))
                        except (TypeError, ValueError):
                            all_skills.append(str(skill))
            else:
                # Single value, not a collection
                all_skills.append(str(skills_list))
        except (TypeError, AttributeError):
            # Not iterable, treat as single skill
            all_skills.append(str(skills_list))
    
    print(f"   [OK] Extracted {len(all_skills):,} total skill mentions")
    
    # Normalize and count
    print("   Normalizing skills...")
    normalized_skills = [normalize_skill(skill) for skill in all_skills if normalize_skill(skill)]
    skill_counts = Counter(normalized_skills)
    
    print(f"   [OK] Found {len(skill_counts):,} unique normalized skills")
    
    return skill_counts

def build_dictionary_mapping():
    """Build a mapping from normalized dictionary terms to domains"""
    dictionary_map = {}
    
    for domain, terms in SOFT_SKILL_DOMAINS.items():
        for term in terms:
            # Normalize the dictionary term
            normalized_term = normalize_skill(term)
            if normalized_term:
                dictionary_map[normalized_term] = domain
    
    return dictionary_map

def build_synonym_mapping():
    """Build a normalized synonym mapping"""
    normalized_synonyms = {}
    
    for synonym, standard in SYNONYM_MAPPING.items():
        normalized_synonym = normalize_skill(synonym)
        normalized_standard = normalize_skill(standard)
        if normalized_synonym and normalized_standard:
            normalized_synonyms[normalized_synonym] = normalized_standard
    
    return normalized_synonyms

def map_skill_to_domain(skill_normalized, dictionary_map, synonym_map):
    """Map a normalized LinkedIn skill to a soft-skill domain using exact matching"""
    
    # First, check if the skill matches a dictionary term exactly
    if skill_normalized in dictionary_map:
        return dictionary_map[skill_normalized]
    
    # Second, check if it's a synonym (apply synonym mapping first, then check)
    if skill_normalized in synonym_map:
        synonym_standard = synonym_map[skill_normalized]
        if synonym_standard in dictionary_map:
            return dictionary_map[synonym_standard]
    
    # If no exact match or synonym match, it's unmapped (likely a technical skill)
    return 'unmapped'

def create_mapping(skill_counts):
    """Create the mapping file with automatic domain assignment using exact matching"""
    print("\n2. Creating skill-to-domain mapping...")
    
    # Build dictionary mapping from soft-skill terms to domains
    print("   Building dictionary mapping from soft-skill terms...")
    dictionary_map = build_dictionary_mapping()
    print(f"   [OK] Dictionary contains {len(dictionary_map):,} normalized terms")
    
    # Build normalized synonym mapping
    print("   Building synonym mapping...")
    synonym_map = build_synonym_mapping()
    print(f"   [OK] Synonym mapping contains {len(synonym_map):,} entries")
    
    mappings = []
    
    for skill, frequency in tqdm(skill_counts.items(), desc="   Mapping skills", disable=not TQDM_AVAILABLE):
        domain = map_skill_to_domain(skill, dictionary_map, synonym_map)
        
        # Determine notes based on mapping type
        if domain != 'unmapped':
            if skill in dictionary_map:
                notes = 'Exact match to dictionary term'
            elif skill in synonym_map:
                notes = f'Synonym mapping: {skill} -> {synonym_map[skill]}'
            else:
                notes = 'Mapped to domain'
        else:
            notes = 'Technical skill or not a soft skill'
        
        mapping = {
            'linkedin_skill': skill,
            'domain': domain,
            'notes': notes,
            'confidence': 'high' if domain != 'unmapped' else 'unmapped',
            'frequency': frequency
        }
        mappings.append(mapping)
    
    # Create DataFrame and sort by frequency (descending)
    mapping_df = pd.DataFrame(mappings)
    mapping_df = mapping_df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    print(f"   [OK] Created mapping for {len(mapping_df):,} skills")
    
    # Show breakdown
    domain_counts = mapping_df['domain'].value_counts()
    print(f"\n   Domain breakdown:")
    for domain, count in domain_counts.items():
        pct = (count / len(mapping_df)) * 100
        print(f"     {domain}: {count:,} skills ({pct:.1f}%)")
    
    return mapping_df

def generate_coverage_report(df, mapping_df):
    """Generate coverage report: % of postings with mapped soft-skill tags"""
    print("\n3. Generating coverage report...")
    
    # Create a set of all soft-skill skills (exclude unmapped)
    soft_skill_set = set(mapping_df[mapping_df['domain'] != 'unmapped']['linkedin_skill'])
    
    # Count postings with at least one soft-skill tag
    postings_with_soft_skills = 0
    soft_skill_counts_by_domain = Counter()
    all_soft_skills_found = Counter()
    
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="   Analyzing coverage", unit=" jobs")
        iterator = tqdm(df.iterrows(), total=len(df), desc="   Processing postings")
    else:
        iterator = df.iterrows()
    
    for idx, row in iterator:
        skills_list = row['skills_list']
        
        # Skip None or NaN values
        if skills_list is None:
            continue
        try:
            if isinstance(skills_list, float) and pd.isna(skills_list):
                continue
        except (TypeError, ValueError):
            pass
        
        # Check if it's iterable and has items
        try:
            if hasattr(skills_list, '__len__') and len(skills_list) > 0:
                # Normalize skills
                skills = []
                for skill in skills_list:
                    if skill is not None:
                        try:
                            if not (isinstance(skill, float) and pd.isna(skill)):
                                normalized = normalize_skill(skill)
                                if normalized:
                                    skills.append(normalized)
                        except (TypeError, ValueError):
                            normalized = normalize_skill(skill)
                            if normalized:
                                skills.append(normalized)
                
                has_soft_skill = False
                for skill in skills:
                    if skill in soft_skill_set:
                        has_soft_skill = True
                        all_soft_skills_found[skill] += 1
                        # Find domain for this skill
                        domain = mapping_df[mapping_df['linkedin_skill'] == skill]['domain'].iloc[0]
                        soft_skill_counts_by_domain[domain] += 1
                
                if has_soft_skill:
                    postings_with_soft_skills += 1
        except (TypeError, AttributeError):
            # Not iterable, skip
            pass
    
    total_postings = len(df)
    coverage_pct = (postings_with_soft_skills / total_postings * 100) if total_postings > 0 else 0
    
    # Top mapped tags per domain
    top_tags_by_domain = {}
    for domain in SOFT_SKILL_DOMAINS.keys():
        domain_skills = mapping_df[mapping_df['domain'] == domain].head(10)
        top_tags_by_domain[domain] = domain_skills[['linkedin_skill', 'frequency']].to_dict('records')
    
    # Top unmapped tags
    unmapped_skills = mapping_df[mapping_df['domain'] == 'unmapped'].head(20)
    top_unmapped = unmapped_skills[['linkedin_skill', 'frequency']].to_dict('records')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_postings': int(total_postings),
        'postings_with_soft_skills': int(postings_with_soft_skills),
        'coverage_percentage': float(coverage_pct),
        'soft_skill_domains': dict(soft_skill_counts_by_domain),
        'top_tags_by_domain': top_tags_by_domain,
        'top_unmapped_tags': top_unmapped,
        'total_unique_skills': int(len(mapping_df)),
        'mapped_skills': int(len(mapping_df[mapping_df['domain'] != 'unmapped'])),
        'unmapped_skills': int(len(mapping_df[mapping_df['domain'] == 'unmapped']))
    }
    
    print(f"   [OK] Coverage: {coverage_pct:.1f}% of postings have ≥1 soft-skill tag")
    print(f"   Postings with soft skills: {postings_with_soft_skills:,} / {total_postings:,}")
    
    return report

def save_top_skills_qa(mapping_df, n=200):
    """Save top N skills for manual QA review"""
    print(f"\n4. Creating QA file with top {n} skills...")
    
    top_skills = mapping_df.head(n).copy()
    
    # Add columns for manual review
    top_skills['reviewed'] = False
    top_skills['correct_domain'] = top_skills['domain']  # For manual correction
    top_skills['notes_manual'] = ''
    
    top_skills.to_csv(TOP_SKILLS_QA_FILE, index=False)
    print(f"   [OK] Saved to: {TOP_SKILLS_QA_FILE}")
    print(f"   Please review and update the mapping as needed")

def save_outputs(mapping_df, coverage_report):
    """Save all output files"""
    print("\n5. Saving outputs...")
    
    # Save mapping file
    mapping_df.to_csv(MAPPING_FILE, index=False)
    print(f"   [OK] Mapping saved to: {MAPPING_FILE}")
    
    # Save coverage report
    with open(COVERAGE_REPORT_FILE, 'w') as f:
        json.dump(coverage_report, f, indent=2)
    print(f"   [OK] Coverage report saved to: {COVERAGE_REPORT_FILE}")

def generate_summary(mapping_df, coverage_report):
    """Generate processing summary"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_unique_skills': int(len(mapping_df)),
        'mapped_soft_skills': int(len(mapping_df[mapping_df['domain'] != 'unmapped'])),
        'unmapped_skills': int(len(mapping_df[mapping_df['domain'] == 'unmapped'])),
        'coverage_percentage': coverage_report['coverage_percentage'],
        'domains': {domain: int(len(mapping_df[mapping_df['domain'] == domain])) 
                   for domain in SOFT_SKILL_DOMAINS.keys()},
        'domains_unmapped': int(len(mapping_df[mapping_df['domain'] == 'unmapped']))
    }
    
    with open(LOG_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   [OK] Summary saved to: {LOG_FILE}")

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 3: Build Soft-Skill Tag Mapping")
    print("=" * 80)
    
    try:
        # Step 1: Load data
        print("\nLoading data...")
        df = pd.read_parquet(INPUT_FILE)
        print(f"   [OK] Loaded {len(df):,} job postings from {INPUT_FILE.name}")
        
        # Step 2: Extract and normalize skills
        skill_counts = extract_all_skills(df)
        
        # Step 3: Create mapping
        mapping_df = create_mapping(skill_counts)
        
        # Step 4: Generate coverage report
        coverage_report = generate_coverage_report(df, mapping_df)
        
        # Step 5: Save top skills for QA
        save_top_skills_qa(mapping_df, n=200)
        
        # Step 6: Save outputs
        save_outputs(mapping_df, coverage_report)
        
        # Step 7: Generate summary
        generate_summary(mapping_df, coverage_report)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TASK 3 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total unique skills: {len(mapping_df):,}")
        print(f"  Mapped soft skills: {len(mapping_df[mapping_df['domain'] != 'unmapped']):,}")
        print(f"  Unmapped skills: {len(mapping_df[mapping_df['domain'] == 'unmapped']):,}")
        print(f"  Coverage: {coverage_report['coverage_percentage']:.1f}% of postings have soft-skill tags")
        print(f"\nOutput files:")
        print(f"  Mapping: {MAPPING_FILE}")
        print(f"  Coverage report: {COVERAGE_REPORT_FILE}")
        print(f"  QA file (top 200): {TOP_SKILLS_QA_FILE}")
        print(f"  Processing log: {LOG_FILE}")
        print(f"\nTotal time: {duration:.1f} seconds")
        
    except Exception as e:
        print(f"\n[ERROR] Task failed: {e}")
        raise

if __name__ == "__main__":
    main()
