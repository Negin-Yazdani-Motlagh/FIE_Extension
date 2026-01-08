"""
Task 7: Calculate Inter-Coder Agreement
Goal: Calculate agreement statistics between two human coders
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
CODER1_FILE = OUTPUT_DIR / "task7_coder1_labels.csv"
CODER2_FILE = OUTPUT_DIR / "task7_coder2_labels.csv"

OUTPUT_AGREEMENT = OUTPUT_DIR / "task7_intercoder_agreement.json"
OUTPUT_AGREEMENT_TABLE = OUTPUT_DIR / "task7_intercoder_agreement_table.csv"

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

def calculate_cohens_kappa(coder1_labels, coder2_labels):
    """Calculate Cohen's kappa for inter-coder agreement"""
    # Create confusion matrix
    n = len(coder1_labels)
    if n == 0:
        return None
    
    # Count agreements
    both_positive = sum((c1 == 1) & (c2 == 1) for c1, c2 in zip(coder1_labels, coder2_labels))
    both_negative = sum((c1 == 0) & (c2 == 0) for c1, c2 in zip(coder1_labels, coder2_labels))
    c1_positive_c2_negative = sum((c1 == 1) & (c2 == 0) for c1, c2 in zip(coder1_labels, coder2_labels))
    c1_negative_c2_positive = sum((c1 == 0) & (c2 == 1) for c1, c2 in zip(coder1_labels, coder2_labels))
    
    # Observed agreement
    po = (both_positive + both_negative) / n
    
    # Expected agreement (chance)
    p1_positive = sum(c1 == 1 for c1 in coder1_labels) / n
    p1_negative = 1 - p1_positive
    p2_positive = sum(c2 == 1 for c2 in coder2_labels) / n
    p2_negative = 1 - p2_positive
    
    pe = (p1_positive * p2_positive) + (p1_negative * p2_negative)
    
    # Cohen's kappa
    if pe == 1:
        kappa = 1.0  # Perfect agreement
    else:
        kappa = (po - pe) / (1 - pe)
    
    return {
        'kappa': kappa,
        'observed_agreement': po,
        'expected_agreement': pe,
        'both_positive': both_positive,
        'both_negative': both_negative,
        'c1_positive_c2_negative': c1_positive_c2_negative,
        'c1_negative_c2_positive': c1_negative_c2_positive,
        'n': n
    }

def calculate_agreement_per_domain(coder1_df, coder2_df):
    """Calculate agreement statistics per domain"""
    agreement_results = {}
    
    for domain in DOMAIN_NAMES:
        # Get labels for this domain
        c1_domain = coder1_df[coder1_df['domain'] == domain]['human_label'].values
        c2_domain = coder2_df[coder2_df['domain'] == domain]['human_label'].values
        
        # Ensure same length
        min_len = min(len(c1_domain), len(c2_domain))
        c1_domain = c1_domain[:min_len]
        c2_domain = c2_domain[:min_len]
        
        if len(c1_domain) == 0:
            continue
        
        # Calculate percent agreement
        agreements = sum(c1 == c2 for c1, c2 in zip(c1_domain, c2_domain))
        percent_agreement = agreements / len(c1_domain)
        
        # Calculate Cohen's kappa
        kappa_stats = calculate_cohens_kappa(c1_domain, c2_domain)
        
        agreement_results[domain] = {
            'percent_agreement': percent_agreement,
            'n_postings': len(c1_domain),
            'agreements': agreements,
            'disagreements': len(c1_domain) - agreements,
            **kappa_stats
        }
    
    return agreement_results

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 7: Calculate Inter-Coder Agreement")
    print("=" * 80)
    
    try:
        # Step 1: Load coder labels
        print("\n1. Loading coder labels...")
        print(f"   Coder 1: {CODER1_FILE.name}")
        coder1_df = pd.read_csv(CODER1_FILE)
        print(f"   [OK] Loaded {len(coder1_df):,} labels from coder 1")
        
        print(f"   Coder 2: {CODER2_FILE.name}")
        coder2_df = pd.read_csv(CODER2_FILE)
        print(f"   [OK] Loaded {len(coder2_df):,} labels from coder 2")
        
        # Validate structure
        required_cols = ['job_link', 'domain', 'human_label']
        for col in required_cols:
            if col not in coder1_df.columns:
                raise ValueError(f"Coder 1 missing required column: {col}")
            if col not in coder2_df.columns:
                raise ValueError(f"Coder 2 missing required column: {col}")
        
        # Step 2: Calculate overall agreement
        print("\n2. Calculating overall agreement...")
        # Merge on job_link and domain
        merged = pd.merge(
            coder1_df[['job_link', 'domain', 'human_label']],
            coder2_df[['job_link', 'domain', 'human_label']],
            on=['job_link', 'domain'],
            suffixes=('_c1', '_c2')
        )
        
        overall_agreements = sum(merged['human_label_c1'] == merged['human_label_c2'])
        overall_total = len(merged)
        overall_percent = overall_agreements / overall_total if overall_total > 0 else 0
        
        print(f"   Overall agreement: {overall_agreements}/{overall_total} ({overall_percent*100:.1f}%)")
        
        # Step 3: Calculate per-domain agreement
        print("\n3. Calculating per-domain agreement...")
        domain_agreement = calculate_agreement_per_domain(coder1_df, coder2_df)
        
        # Step 4: Create agreement table
        agreement_table_rows = []
        for domain, stats in domain_agreement.items():
            agreement_table_rows.append({
                'domain': domain,
                'n_postings': stats['n_postings'],
                'agreements': stats['agreements'],
                'disagreements': stats['disagreements'],
                'percent_agreement': round(stats['percent_agreement'] * 100, 2),
                'cohens_kappa': round(stats['kappa'], 3) if stats['kappa'] is not None else None,
                'observed_agreement': round(stats['observed_agreement'], 3),
                'expected_agreement': round(stats['expected_agreement'], 3)
            })
        
        agreement_table_df = pd.DataFrame(agreement_table_rows)
        
        # Step 5: Save results
        print("\n4. Saving results...")
        
        # Save agreement statistics (convert numpy types to Python native types)
        def convert_to_python_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        agreement_summary = {
            'timestamp': datetime.now().isoformat(),
            'overall': {
                'agreements': int(overall_agreements),
                'total': int(overall_total),
                'percent_agreement': round(overall_percent * 100, 2)
            },
            'by_domain': convert_to_python_types(domain_agreement)
        }
        
        with open(OUTPUT_AGREEMENT, 'w') as f:
            json.dump(agreement_summary, f, indent=2)
        print(f"   [OK] Saved agreement statistics to {OUTPUT_AGREEMENT.name}")
        
        # Save agreement table
        agreement_table_df.to_csv(OUTPUT_AGREEMENT_TABLE, index=False)
        print(f"   [OK] Saved agreement table to {OUTPUT_AGREEMENT_TABLE.name}")
        
        # Step 6: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("AGREEMENT CALCULATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nOverall Agreement:")
        print(f"  {overall_agreements}/{overall_total} ({overall_percent*100:.1f}%)")
        print(f"\nPer-Domain Agreement:")
        for domain, stats in domain_agreement.items():
            print(f"  {domain}:")
            print(f"    Percent Agreement: {stats['percent_agreement']*100:.1f}%")
            print(f"    Cohen's Kappa: {stats['kappa']:.3f}" if stats['kappa'] is not None else "    Cohen's Kappa: N/A")
            print(f"    Agreements: {stats['agreements']}/{stats['n_postings']}")
        print(f"\nOutput files:")
        print(f"  Agreement Statistics: {OUTPUT_AGREEMENT}")
        print(f"  Agreement Table: {OUTPUT_AGREEMENT_TABLE}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("   Please ensure coder label files exist:")
        print(f"   - {CODER1_FILE}")
        print(f"   - {CODER2_FILE}")
    except Exception as e:
        print(f"\n[ERROR] Agreement calculation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
