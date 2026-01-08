"""
Task 7: Validity Analysis - Compare Detectors vs Human Labels
Goal: Calculate accuracy, precision, recall, F1 for v1 and v2 vs human labels
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
SAMPLE_WITH_V2_FILE = OUTPUT_DIR / "task7_sample_with_v2_predictions.csv"
GOLD_STANDARD_FILE = OUTPUT_DIR / "task7_gold_standard_labels.csv"

OUTPUT_VALIDITY_TABLE = OUTPUT_DIR / "task7_validity_table.csv"
OUTPUT_VALIDITY_SUMMARY = OUTPUT_DIR / "task7_validity_summary.json"
OUTPUT_IMPROVEMENT = OUTPUT_DIR / "task7_improvement_analysis.csv"

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

def calculate_metrics(detector_labels, human_labels):
    """Calculate accuracy, precision, recall, F1"""
    # Convert to lists if needed
    if isinstance(detector_labels, pd.Series):
        detector_labels = detector_labels.values
    if isinstance(human_labels, pd.Series):
        human_labels = human_labels.values
    
    # Ensure boolean/binary
    detector_labels = [bool(int(x)) if pd.notna(x) else False for x in detector_labels]
    human_labels = [bool(int(x)) if pd.notna(x) else False for x in human_labels]
    
    # Calculate confusion matrix
    tp = sum((d == True) & (h == True) for d, h in zip(detector_labels, human_labels))
    fp = sum((d == True) & (h == False) for d, h in zip(detector_labels, human_labels))
    fn = sum((d == False) & (h == True) for d, h in zip(detector_labels, human_labels))
    tn = sum((d == False) & (h == False) for d, h in zip(detector_labels, human_labels))
    
    total = tp + fp + fn + tn
    
    if total == 0:
        return {
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0
        }
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3)
    }

def compare_detector_vs_human(sample_df, gold_standard_df, detector_version='v1'):
    """Compare detector predictions vs human labels"""
    results = {}
    
    # Determine column names based on version
    if detector_version == 'v1':
        # v1 columns don't have suffix
        detector_cols = ['job_link'] + [f'text_domain_present_{d}' for d in DOMAIN_NAMES]
    else:
        # v2 columns have _v2 suffix
        detector_cols = ['job_link'] + [f'text_domain_present_{d}_{detector_version}' for d in DOMAIN_NAMES]
    
    # Filter to only columns that exist
    available_cols = [col for col in detector_cols if col in sample_df.columns]
    if 'job_link' not in available_cols:
        raise ValueError("job_link column not found in sample_df")
    
    # Merge sample with gold standard
    merged = pd.merge(
        sample_df[available_cols],
        gold_standard_df[['job_link', 'domain', 'human_label']],
        on='job_link',
        how='inner'
    )
    
    for domain in DOMAIN_NAMES:
        # Get detector predictions for this domain
        if detector_version == 'v1':
            detector_col = f'text_domain_present_{domain}'
        else:
            detector_col = f'text_domain_present_{domain}_{detector_version}'
        
        if detector_col not in merged.columns:
            continue
        
        domain_merged = merged[merged['domain'] == domain].copy()
        
        if len(domain_merged) == 0:
            continue
        
        detector_labels = domain_merged[detector_col].values
        human_labels = domain_merged['human_label'].values
        
        metrics = calculate_metrics(detector_labels, human_labels)
        results[domain] = metrics
    
    return results

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 7: Validity Analysis - Detectors vs Human Labels")
    print("=" * 80)
    
    try:
        # Step 1: Load sample with v1 and v2 predictions
        print("\n1. Loading sample with detector predictions...")
        print(f"   File: {SAMPLE_WITH_V2_FILE.name}")
        sample_df = pd.read_csv(SAMPLE_WITH_V2_FILE)
        print(f"   [OK] Loaded {len(sample_df):,} postings")
        
        # Step 2: Load gold standard labels
        print("\n2. Loading gold standard labels...")
        print(f"   File: {GOLD_STANDARD_FILE.name}")
        gold_standard_df = pd.read_csv(GOLD_STANDARD_FILE)
        print(f"   [OK] Loaded {len(gold_standard_df):,} labels")
        
        # Validate structure
        if 'job_link' not in gold_standard_df.columns or 'domain' not in gold_standard_df.columns or 'human_label' not in gold_standard_df.columns:
            raise ValueError("Gold standard file must have columns: job_link, domain, human_label")
        
        # Step 3: Compare detector v1 vs human
        print("\n3. Comparing detector v1 vs human labels...")
        v1_results = compare_detector_vs_human(sample_df, gold_standard_df, detector_version='v1')
        
        # Step 4: Compare detector v2 vs human
        print("\n4. Comparing detector v2 vs human labels...")
        v2_results = compare_detector_vs_human(sample_df, gold_standard_df, detector_version='v2')
        
        # Step 5: Calculate improvement from v1 to v2
        print("\n5. Calculating improvement from v1 to v2...")
        improvement_rows = []
        validity_table_rows = []
        
        for domain in DOMAIN_NAMES:
            if domain not in v1_results or domain not in v2_results:
                continue
            
            v1 = v1_results[domain]
            v2 = v2_results[domain]
            
            # Calculate improvement
            f1_improvement = v2['f1'] - v1['f1']
            precision_improvement = v2['precision'] - v1['precision']
            recall_improvement = v2['recall'] - v1['recall']
            accuracy_improvement = v2['accuracy'] - v1['accuracy']
            
            improvement_rows.append({
                'domain': domain,
                'f1_v1': v1['f1'],
                'f1_v2': v2['f1'],
                'f1_improvement': round(f1_improvement, 3),
                'precision_v1': v1['precision'],
                'precision_v2': v2['precision'],
                'precision_improvement': round(precision_improvement, 3),
                'recall_v1': v1['recall'],
                'recall_v2': v2['recall'],
                'recall_improvement': round(recall_improvement, 3),
                'accuracy_v1': v1['accuracy'],
                'accuracy_v2': v2['accuracy'],
                'accuracy_improvement': round(accuracy_improvement, 3)
            })
            
            # Validity table row
            validity_table_rows.append({
                'domain': domain,
                'detector_v1_accuracy': v1['accuracy'],
                'detector_v1_precision': v1['precision'],
                'detector_v1_recall': v1['recall'],
                'detector_v1_f1': v1['f1'],
                'detector_v2_accuracy': v2['accuracy'],
                'detector_v2_precision': v2['precision'],
                'detector_v2_recall': v2['recall'],
                'detector_v2_f1': v2['f1'],
                'f1_improvement': round(f1_improvement, 3),
                'precision_improvement': round(precision_improvement, 3),
                'recall_improvement': round(recall_improvement, 3),
                'accuracy_improvement': round(accuracy_improvement, 3)
            })
        
        improvement_df = pd.DataFrame(improvement_rows)
        validity_table_df = pd.DataFrame(validity_table_rows)
        
        # Step 6: Save results
        print("\n6. Saving results...")
        
        # Save validity table
        validity_table_df.to_csv(OUTPUT_VALIDITY_TABLE, index=False)
        print(f"   [OK] Saved validity table to {OUTPUT_VALIDITY_TABLE.name}")
        
        # Save improvement analysis
        improvement_df.to_csv(OUTPUT_IMPROVEMENT, index=False)
        print(f"   [OK] Saved improvement analysis to {OUTPUT_IMPROVEMENT.name}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_postings': len(sample_df),
            'total_labels': len(gold_standard_df),
            'detector_v1': v1_results,
            'detector_v2': v2_results,
            'average_improvements': {
                'f1': round(improvement_df['f1_improvement'].mean(), 3),
                'precision': round(improvement_df['precision_improvement'].mean(), 3),
                'recall': round(improvement_df['recall_improvement'].mean(), 3),
                'accuracy': round(improvement_df['accuracy_improvement'].mean(), 3)
            }
        }
        
        with open(OUTPUT_VALIDITY_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Saved validity summary to {OUTPUT_VALIDITY_SUMMARY.name}")
        
        # Step 7: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("VALIDITY ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nAverage Improvements (v1 -> v2):")
        print(f"  F1: {summary['average_improvements']['f1']:+.3f}")
        print(f"  Precision: {summary['average_improvements']['precision']:+.3f}")
        print(f"  Recall: {summary['average_improvements']['recall']:+.3f}")
        print(f"  Accuracy: {summary['average_improvements']['accuracy']:+.3f}")
        print(f"\nPer-Domain Results:")
        for domain in DOMAIN_NAMES:
            if domain in v1_results and domain in v2_results:
                v1_f1 = v1_results[domain]['f1']
                v2_f1 = v2_results[domain]['f1']
                improvement = v2_f1 - v1_f1
                print(f"  {domain}:")
                print(f"    v1 F1: {v1_f1:.3f}, v2 F1: {v2_f1:.3f}, Improvement: {improvement:+.3f}")
        print(f"\nOutput files:")
        print(f"  Validity Table: {OUTPUT_VALIDITY_TABLE}")
        print(f"  Improvement Analysis: {OUTPUT_IMPROVEMENT}")
        print(f"  Validity Summary: {OUTPUT_VALIDITY_SUMMARY}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("   Please ensure required files exist:")
        print(f"   - {SAMPLE_WITH_V2_FILE}")
        print(f"   - {GOLD_STANDARD_FILE}")
    except Exception as e:
        print(f"\n[ERROR] Validity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
