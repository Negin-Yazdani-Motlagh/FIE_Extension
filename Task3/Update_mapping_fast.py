"""
Fast script to update mapping using dictionary from Excel
Uses existing skill counts from CSV to avoid slow re-extraction
Skips coverage report (7+ hours) - only updates mapping files
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
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration
OUTPUT_DIR = Path(__file__).parent
DICTIONARY_FILE = OUTPUT_DIR / "Dictionary of soft skills (9.1).xlsx"
MAPPING_FILE = OUTPUT_DIR / "softskill_tag_mapping.csv"
TOP_SKILLS_QA_FILE = OUTPUT_DIR / "task3_top_200_skills_qa.csv"
LOG_FILE = OUTPUT_DIR / "task3_processing_log.json"
COVERAGE_REPORT_FILE = OUTPUT_DIR / "task3_coverage_report.json"

# Synonym mapping (keep existing)
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
        'problemsolving': 'problem-solving',  # Handle no space/hyphen
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

def load_dictionary_from_excel(file_path):
    """Load dictionary from Excel file - handles different structures"""
    print(f"\n1. Loading dictionary from Excel...")
    print(f"   File: {file_path.name}")
    
    try:
        # Try to read all sheets
        excel_file = pd.ExcelFile(file_path)
        print(f"   [OK] Found {len(excel_file.sheet_names)} sheet(s): {excel_file.sheet_names}")
        
        # Try first sheet
        df = pd.read_excel(file_path, sheet_name=0)
        print(f"   [OK] Loaded {len(df):,} rows from first sheet")
        print(f"   Columns: {list(df.columns)}")
        
        # Detect structure
        # Option 1: Domain column + Skill/Term column
        # Option 2: Domain as header, skills in rows
        # Option 3: Skill column + Domain column
        
        domain_col = None
        skill_col = None
        
        # Check for common column names
        cols_lower = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
        
        if 'domain' in cols_lower:
            domain_col = df.columns[cols_lower.index('domain')]
        elif any('skill' in c.lower() or 'term' in c.lower() for c in df.columns):
            # Find domain column (might be first column or named differently)
            for col in df.columns:
                if col.lower() not in ['skill', 'skills', 'term', 'terms', 'soft skill', 'soft skills']:
                    domain_col = col
                    break
        
        # Find skill/term column
        for col in df.columns:
            col_lower = str(col).lower()
            if 'skill' in col_lower or 'term' in col_lower:
                skill_col = col
                break
        
        # If no clear structure, assume first column is domain, second is skill
        if domain_col is None:
            domain_col = df.columns[0]
        if skill_col is None and len(df.columns) > 1:
            skill_col = df.columns[1]
        elif skill_col is None:
            skill_col = df.columns[0]
        
        print(f"   Using domain column: '{domain_col}'")
        print(f"   Using skill column: '{skill_col}'")
        
        # Build dictionary structure
        soft_skill_domains = {}
        
        for _, row in df.iterrows():
            domain = str(row[domain_col]).strip()
            skill = str(row[skill_col]).strip()
            
            # Skip empty rows
            if pd.isna(row[domain_col]) or pd.isna(row[skill_col]) or not domain or not skill:
                continue
            
            # Normalize domain name (capitalize first letter of each word)
            domain = domain.title()
            
            if domain not in soft_skill_domains:
                soft_skill_domains[domain] = []
            
            soft_skill_domains[domain].append(skill)
        
        print(f"   [OK] Loaded dictionary with {len(soft_skill_domains)} domains")
        for domain, terms in soft_skill_domains.items():
            print(f"     {domain}: {len(terms)} terms")
        
        return soft_skill_domains
        
    except Exception as e:
        print(f"   [ERROR] Error loading dictionary: {e}")
        raise

def build_dictionary_mapping(soft_skill_domains):
    """Build a mapping from normalized dictionary terms to domains"""
    dictionary_map = {}
    # Also build a reverse mapping: word -> list of (domain, full_term) for partial matching
    word_to_domains = {}
    
    for domain, terms in soft_skill_domains.items():
        for term in terms:
            # Normalize the dictionary term
            normalized_term = normalize_skill(term)
            if normalized_term:
                dictionary_map[normalized_term] = domain
                
                # Extract words from the normalized term for partial matching
                words = normalized_term.split()
                for word in words:
                    # Remove common stop words and very short words
                    if len(word) > 2 and word not in ['the', 'and', 'or', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by']:
                        if word not in word_to_domains:
                            word_to_domains[word] = []
                        word_to_domains[word].append((domain, normalized_term))
    
    return dictionary_map, word_to_domains

def build_synonym_mapping():
    """Build a normalized synonym mapping"""
    normalized_synonyms = {}
    
    for synonym, standard in SYNONYM_MAPPING.items():
        normalized_synonym = normalize_skill(synonym)
        normalized_standard = normalize_skill(standard)
        if normalized_synonym and normalized_standard:
            normalized_synonyms[normalized_synonym] = normalized_standard
    
    return normalized_synonyms

def map_skill_to_domain(skill_normalized, dictionary_map, synonym_map, word_to_domains):
    """Map a normalized LinkedIn skill to a soft-skill domain using exact and word-based matching"""
    
    # First, check if the skill matches a dictionary term exactly
    if skill_normalized in dictionary_map:
        return dictionary_map[skill_normalized]
    
    # Second, check if it's a synonym (apply synonym mapping first, then check)
    if skill_normalized in synonym_map:
        synonym_standard = synonym_map[skill_normalized]
        if synonym_standard in dictionary_map:
            return dictionary_map[synonym_standard]
    
    # Third, try word-based matching: check if skill words appear in dictionary terms
    skill_words = skill_normalized.split()
    matched_domains = {}
    
    for word in skill_words:
        if len(word) > 2 and word in word_to_domains:
            # Found a matching word, collect all domains that contain this word
            for domain, full_term in word_to_domains[word]:
                if domain not in matched_domains:
                    matched_domains[domain] = []
                matched_domains[domain].append(full_term)
    
    # If we found matches, return the domain with the most/best matches
    if matched_domains:
        # Prefer exact word matches (e.g., "communication" matching "communication" in "verbal communication")
        # over partial matches (e.g., "comm" matching "communication")
        best_domain = None
        best_score = 0
        
        for domain, terms in matched_domains.items():
            # Score based on how many terms matched and if any term contains the full skill
            score = len(terms)
            # Bonus if the skill appears as a complete word in any term
            for term in terms:
                if skill_normalized in term or term in skill_normalized:
                    score += 10
            if score > best_score:
                best_score = score
                best_domain = domain
        
        if best_domain:
            return best_domain
    
    # Fourth, try reverse: check if any dictionary term word appears in the skill
    for word, domain_terms in word_to_domains.items():
        if word in skill_normalized and len(word) > 3:  # Only for words longer than 3 chars
            # Check if this word is a significant part of the skill
            if skill_normalized.startswith(word) or skill_normalized.endswith(word) or f' {word} ' in f' {skill_normalized} ':
                # Return the first matching domain (could be improved with scoring)
                return domain_terms[0][0]
    
    # If no match found, it's unmapped (likely a technical skill)
    return 'unmapped'

def load_existing_skill_counts():
    """Load existing skill counts from CSV to avoid re-extraction"""
    print(f"\n2. Loading existing skill counts from CSV...")
    
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")
    
    df = pd.read_csv(MAPPING_FILE)
    print(f"   [OK] Loaded {len(df):,} skills from existing mapping")
    
    # Create skill_counts dict from CSV
    skill_counts = {}
    for _, row in df.iterrows():
        skill = row['linkedin_skill']
        frequency = row['frequency']
        skill_counts[skill] = frequency
    
    print(f"   [OK] Extracted {len(skill_counts):,} unique skills with frequencies")
    return skill_counts

def create_mapping(skill_counts, soft_skill_domains):
    """Create the mapping file with automatic domain assignment using exact matching"""
    print("\n3. Creating skill-to-domain mapping...")
    
    # Build dictionary mapping from soft-skill terms to domains
    print("   Building dictionary mapping from soft-skill terms...")
    dictionary_map, word_to_domains = build_dictionary_mapping(soft_skill_domains)
    print(f"   [OK] Dictionary contains {len(dictionary_map):,} normalized terms")
    print(f"   [OK] Word index contains {len(word_to_domains):,} unique words for partial matching")
    
    # Build normalized synonym mapping
    print("   Building synonym mapping...")
    synonym_map = build_synonym_mapping()
    print(f"   [OK] Synonym mapping contains {len(synonym_map):,} entries")
    
    mappings = []
    
    for skill, frequency in tqdm(skill_counts.items(), desc="   Mapping skills", disable=not TQDM_AVAILABLE):
        skill_normalized = normalize_skill(skill)
        if not skill_normalized:
            domain = 'unmapped'
            notes = 'Technical skill or not a soft skill'
        else:
            domain = map_skill_to_domain(skill_normalized, dictionary_map, synonym_map, word_to_domains)
            
            # Determine notes based on mapping type
            if domain != 'unmapped':
                if skill_normalized in dictionary_map:
                    notes = 'Exact match to dictionary term'
                elif skill_normalized in synonym_map:
                    notes = f'Synonym mapping: {skill_normalized} -> {synonym_map[skill_normalized]}'
                else:
                    notes = 'Word-based match to dictionary term'
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

def save_outputs(mapping_df):
    """Save all output files"""
    print("\n5. Saving outputs...")
    
    # Save mapping file
    mapping_df.to_csv(MAPPING_FILE, index=False)
    print(f"   [OK] Mapping saved to: {MAPPING_FILE}")
    
    # Load old coverage report if exists (to preserve it)
    if COVERAGE_REPORT_FILE.exists():
        print(f"   [INFO] Keeping existing coverage report (not regenerated)")
    else:
        print(f"   [INFO] Coverage report not found (will be generated in full run)")

def generate_summary(mapping_df, soft_skill_domains):
    """Generate processing summary"""
    # Load old coverage report if exists
    coverage_pct = 0
    if COVERAGE_REPORT_FILE.exists():
        try:
            with open(COVERAGE_REPORT_FILE, 'r') as f:
                old_report = json.load(f)
                coverage_pct = old_report.get('coverage_percentage', 0)
        except:
            pass
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_unique_skills': int(len(mapping_df)),
        'mapped_soft_skills': int(len(mapping_df[mapping_df['domain'] != 'unmapped'])),
        'unmapped_skills': int(len(mapping_df[mapping_df['domain'] == 'unmapped'])),
        'coverage_percentage': float(coverage_pct),  # From old report
        'domains': {domain: int(len(mapping_df[mapping_df['domain'] == domain])) 
                   for domain in soft_skill_domains.keys()},
        'domains_unmapped': int(len(mapping_df[mapping_df['domain'] == 'unmapped'])),
        'note': 'Mapping updated using dictionary from Excel. Coverage report not regenerated.'
    }
    
    with open(LOG_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   [OK] Summary saved to: {LOG_FILE}")

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 3: FAST UPDATE - Regenerate Mapping with Dictionary")
    print("=" * 80)
    print("\nThis script:")
    print("  [OK] Loads dictionary from Excel")
    print("  [OK] Uses existing skill counts (no re-extraction)")
    print("  [OK] Regenerates mapping")
    print("  [SKIP] Skips coverage report (7+ hours)")
    
    try:
        # Step 1: Load dictionary from Excel
        soft_skill_domains = load_dictionary_from_excel(DICTIONARY_FILE)
        
        # Step 2: Load existing skill counts
        skill_counts = load_existing_skill_counts()
        
        # Step 3: Create mapping
        mapping_df = create_mapping(skill_counts, soft_skill_domains)
        
        # Step 4: Save top skills for QA
        save_top_skills_qa(mapping_df, n=200)
        
        # Step 5: Save outputs
        save_outputs(mapping_df)
        
        # Step 6: Generate summary
        generate_summary(mapping_df, soft_skill_domains)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("MAPPING UPDATE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total unique skills: {len(mapping_df):,}")
        print(f"  Mapped soft skills: {len(mapping_df[mapping_df['domain'] != 'unmapped']):,}")
        print(f"  Unmapped skills: {len(mapping_df[mapping_df['domain'] == 'unmapped']):,}")
        print(f"\nOutput files:")
        print(f"  Mapping: {MAPPING_FILE}")
        print(f"  QA file (top 200): {TOP_SKILLS_QA_FILE}")
        print(f"  Processing log: {LOG_FILE}")
        print(f"\nTotal time: {duration:.1f} seconds")
        print(f"\nNote: Coverage report was NOT regenerated (would take 7+ hours)")
        print(f"      To regenerate coverage report, run the full script.")
        
    except Exception as e:
        print(f"\n[ERROR] Update failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
