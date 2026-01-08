"""
Task 10: Create Curriculum Mapping Table
Goal: Map skill bundles to curricular homes and assessment strategies
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Configuration
TASK10_DIR = Path(__file__).parent

# File paths
BUNDLES_FILE = TASK10_DIR / "skill_bundles.csv"

OUTPUT_CURRICULUM_MATRIX = TASK10_DIR / "curriculum_alignment_matrix.csv"
OUTPUT_CURRICULUM_TEX = TASK10_DIR / "curriculum_alignment_matrix.tex"
OUTPUT_SUMMARY = TASK10_DIR / "task10_curriculum_mapping_summary.json"

# Curriculum mapping rules (simplified - can be expanded)
CURRICULUM_MAPPING = {
    # Technical skill patterns -> curricular home
    'python': 'Intro Programming / Data Systems',
    'java': 'Intro Programming / Software Engineering',
    'javascript': 'Web Development / Software Engineering',
    'sql': 'Data Systems / Database Courses',
    'machine learning': 'ML/AI Courses',
    'deep learning': 'ML/AI Courses',
    'data analysis': 'Data Systems / Data Science Courses',
    'aws': 'Cloud Computing / Systems Courses',
    'docker': 'Software Engineering / Systems Courses',
    'kubernetes': 'Systems Courses / Advanced SE',
    'react': 'Web Development / Software Engineering',
    'node.js': 'Web Development / Software Engineering',
    'git': 'Software Engineering (all levels)',
    'agile': 'Software Engineering / Capstone',
    'scrum': 'Software Engineering / Capstone',
}

# Assessment strategies by soft-skill domain
ASSESSMENT_STRATEGIES = {
    'Collaboration And Team Dynamics': [
        'Team rubrics',
        'Peer evaluation',
        'Group project assessments',
        'Stakeholder interviews'
    ],
    'Communication Skills': [
        'Writing assignments',
        'Presentation rubrics',
        'Code documentation reviews',
        'Technical writing assessments'
    ],
    'Problem-Solving And Critical Thinking': [
        'Algorithm design problems',
        'Debugging exercises',
        'Code review rubrics',
        'Case study analysis'
    ],
    'Work Ethic And Professionalism': [
        'Project completion tracking',
        'Code quality metrics',
        'Professional conduct rubrics',
        'Portfolio assessments'
    ],
    'Time Management And Organizational Skills': [
        'Project milestone tracking',
        'Sprint planning assessments',
        'Task breakdown evaluations',
        'Deadline adherence metrics'
    ],
    'Creativity And Inovation': [
        'Design thinking exercises',
        'Innovation project rubrics',
        'Creative problem-solving challenges',
        'Portfolio reviews'
    ],
    'Adaptability & Continuous Learning': [
        'Learning reflection journals',
        'Technology adoption assessments',
        'Self-directed learning projects',
        'Continuous improvement tracking'
    ],
    'Emotional Intelligence (Eq)': [
        'Team dynamics observations',
        'Conflict resolution exercises',
        'Stakeholder interaction assessments',
        'Self-awareness reflections'
    ]
}

def map_bundle_to_curriculum(soft_domain, technical_skill):
    """Map a bundle to curricular home"""
    technical_lower = technical_skill.lower()
    
    # Check for specific technical skill matches
    for pattern, curricular_home in CURRICULUM_MAPPING.items():
        if pattern in technical_lower:
            return curricular_home
    
    # Default mappings based on technical skill type
    if any(term in technical_lower for term in ['python', 'java', 'javascript', 'c++', 'c#']):
        return 'Intro Programming / Software Engineering'
    elif any(term in technical_lower for term in ['sql', 'database', 'data']):
        return 'Data Systems / Database Courses'
    elif any(term in technical_lower for term in ['ml', 'machine learning', 'ai', 'deep learning']):
        return 'ML/AI Courses'
    elif any(term in technical_lower for term in ['web', 'frontend', 'backend', 'fullstack']):
        return 'Web Development / Software Engineering'
    elif any(term in technical_lower for term in ['cloud', 'aws', 'azure', 'gcp']):
        return 'Cloud Computing / Systems Courses'
    elif any(term in technical_lower for term in ['security', 'cyber']):
        return 'Security Courses'
    else:
        return 'Software Engineering / Capstone'

def get_assessment_ideas(soft_domain):
    """Get assessment ideas for a soft-skill domain"""
    return ASSESSMENT_STRATEGIES.get(soft_domain, [
        'Project-based assessment',
        'Portfolio review',
        'Peer evaluation'
    ])

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 10: Create Curriculum Mapping Table")
    print("=" * 80)
    
    try:
        # Step 1: Load bundles
        print("\n1. Loading skill bundles...")
        print(f"   File: {BUNDLES_FILE.name}")
        if not BUNDLES_FILE.exists():
            print(f"   [ERROR] File not found: {BUNDLES_FILE}")
            print("   [INFO] Please run task10_extract_bundles.py first")
            return
        
        bundles_df = pd.read_csv(BUNDLES_FILE)
        print(f"   [OK] Loaded {len(bundles_df)} bundles")
        
        # Step 2: Create curriculum mapping table
        print("\n2. Creating curriculum mapping table...")
        curriculum_data = []
        
        for _, bundle_row in bundles_df.iterrows():
            soft_domain = bundle_row['soft_domain']
            technical_skill = bundle_row['technical_skill']
            
            curricular_home = map_bundle_to_curriculum(soft_domain, technical_skill)
            assessment_ideas = get_assessment_ideas(soft_domain)
            
            curriculum_data.append({
                'bundle_rank': int(bundle_row['rank']),
                'soft_domain': soft_domain,
                'technical_skill': technical_skill,
                'bundle_description': f"{soft_domain} + {technical_skill}",
                'curricular_home': curricular_home,
                'primary_assessment': assessment_ideas[0] if assessment_ideas else 'Project-based assessment',
                'additional_assessments': '; '.join(assessment_ideas[1:3]) if len(assessment_ideas) > 1 else '',
                'cooccurrence_count': int(bundle_row['cooccurrence_count']),
                'percentage_of_postings': bundle_row['percentage_of_postings'],
                'interpretation': bundle_row.get('interpretation', '')
            })
        
        curriculum_df = pd.DataFrame(curriculum_data)
        curriculum_df = curriculum_df.sort_values('bundle_rank')
        
        # Step 3: Save results
        print("\n3. Saving results...")
        curriculum_df.to_csv(OUTPUT_CURRICULUM_MATRIX, index=False)
        print(f"   [OK] Saved curriculum matrix to {OUTPUT_CURRICULUM_MATRIX.name}")
        
        # Step 4: Create LaTeX table
        print("\n4. Creating LaTeX table...")
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Curriculum Alignment Matrix: Top Skill Bundles}",
            "\\label{tab:curriculum_alignment}",
            "\\begin{tabular}{p{2cm}p{3cm}p{2.5cm}p{2cm}p{3cm}}",
            "\\toprule",
            "Bundle & Soft Domain & Technical Skill & Curricular Home & Primary Assessment \\\\",
            "\\midrule"
        ]
        
        for _, row in curriculum_df.head(10).iterrows():
            bundle_desc = row['bundle_description'].replace('&', '\\&')
            soft_domain = row['soft_domain'].replace('&', '\\&')
            tech_skill = row['technical_skill'].replace('_', '\\_')
            curricular = row['curricular_home'].replace('&', '\\&')
            assessment = row['primary_assessment'].replace('&', '\\&')
            
            latex_lines.append(
                f"{row['bundle_rank']} & {soft_domain} & {tech_skill} & {curricular} & {assessment} \\\\"
            )
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        with open(OUTPUT_CURRICULUM_TEX, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))
        print(f"   [OK] Saved LaTeX table to {OUTPUT_CURRICULUM_TEX.name}")
        
        # Step 5: Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_bundles': len(curriculum_df),
            'curricular_homes': curriculum_df['curricular_home'].value_counts().to_dict(),
            'top_bundles': curriculum_df.head(10).to_dict('records')
        }
        
        with open(OUTPUT_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   [OK] Saved summary to {OUTPUT_SUMMARY.name}")
        
        # Step 6: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("CURRICULUM MAPPING COMPLETED")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Total bundles mapped: {len(curriculum_df)}")
        print(f"\nCurricular Homes Distribution:")
        for home, count in curriculum_df['curricular_home'].value_counts().head(5).items():
            print(f"  {home}: {count}")
        print(f"\nTop 5 Bundles:")
        for _, row in curriculum_df.head(5).iterrows():
            print(f"  {row['bundle_rank']}. {row['soft_domain']} + {row['technical_skill']}")
            print(f"     -> {row['curricular_home']}")
            print(f"     -> Assessment: {row['primary_assessment']}")
        print(f"\nOutput files:")
        print(f"  Curriculum Matrix: {OUTPUT_CURRICULUM_MATRIX}")
        print(f"  LaTeX Table: {OUTPUT_CURRICULUM_TEX}")
        print(f"  Summary: {OUTPUT_SUMMARY}")
        print(f"\nTotal time: {duration:.1f} seconds")
        
    except Exception as e:
        print(f"\n[ERROR] Curriculum mapping failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
