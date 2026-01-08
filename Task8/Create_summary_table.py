"""
Task 8: Create Summary Table - Largest Changes After Calibration
Goal: Create interpretation-oriented table of largest changes
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
LARGEST_CHANGES_FILE = OUTPUT_DIR / "yc_largest_changes.csv"
TREND_COMPARISON_FILE = OUTPUT_DIR / "yc_trend_comparison.csv"

OUTPUT_TABLE = OUTPUT_DIR / "yc_largest_changes_table.csv"
OUTPUT_TABLE_TEX = OUTPUT_DIR / "yc_largest_changes_table.tex"

def calculate_trend_direction(comparison_df, domain):
    """Calculate trend direction (increasing/decreasing/stable)"""
    domain_df = comparison_df[comparison_df['domain'] == domain].sort_values('year')
    
    if len(domain_df) < 2:
        return "Insufficient data"
    
    # Calculate slope (simple linear regression)
    years = domain_df['year'].values
    v1_values = domain_df['v1_coverage_percent'].values
    v2_values = domain_df['v2_coverage_percent'].values
    
    # Simple slope calculation
    v1_slope = (v1_values[-1] - v1_values[0]) / (years[-1] - years[0]) if len(years) > 1 else 0
    v2_slope = (v2_values[-1] - v2_values[0]) / (years[-1] - years[0]) if len(years) > 1 else 0
    
    # Determine direction
    if abs(v1_slope) < 0.5:
        v1_dir = "Stable"
    elif v1_slope > 0:
        v1_dir = "Increasing"
    else:
        v1_dir = "Decreasing"
    
    if abs(v2_slope) < 0.5:
        v2_dir = "Stable"
    elif v2_slope > 0:
        v2_dir = "Increasing"
    else:
        v2_dir = "Decreasing"
    
    return f"{v1_dir} -> {v2_dir}"

def create_summary_table(largest_changes_df, comparison_df):
    """Create interpretation-oriented summary table"""
    print("\n3. Creating summary table...")
    
    table_rows = []
    
    for idx, row in largest_changes_df.iterrows():
        domain = row['domain']
        change = row['absolute_change']
        trend_direction = calculate_trend_direction(comparison_df, domain)
        
        # Create interpretation
        if abs(change) < 1:
            interpretation = f"Minimal change after calibration ({change:+.1f} pp). Detector v2 shows similar coverage to v1."
        elif change > 0:
            interpretation = f"Calibration increased coverage by {change:.1f} percentage points. Detector v2 captures more instances of {domain.lower()} than v1, likely due to variant forms added in lexicon refinement."
        else:
            interpretation = f"Calibration decreased coverage by {abs(change):.1f} percentage points. This may indicate improved precision (fewer false positives) or missed variants that need further refinement."
        
        table_rows.append({
            'domain': domain,
            'v1_avg_coverage': f"{row['v1_avg_coverage']:.1f}%",
            'v2_avg_coverage': f"{row['v2_avg_coverage']:.1f}%",
            'change': f"{change:+.1f} pp",
            'percent_change': f"{row['percent_change']:+.1f}%",
            'trend_direction': trend_direction,
            'interpretation': interpretation
        })
    
    table_df = pd.DataFrame(table_rows)
    
    return table_df

def create_latex_table(table_df):
    """Create LaTeX version of table"""
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Largest Changes in Soft-Skill Detection After Calibration}",
        "\\label{tab:yc_calibration_changes}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Domain & V1 Avg & V2 Avg & Change & Interpretation \\\\",
        "\\midrule"
    ]
    
    for _, row in table_df.iterrows():
        domain = row['domain'].replace('&', '\\&')
        latex_lines.append(
            f"{domain} & {row['v1_avg_coverage']} & {row['v2_avg_coverage']} & {row['change']} & {row['interpretation'][:80]}... \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 8: Create Summary Table - Largest Changes")
    print("=" * 80)
    
    try:
        # Step 1: Load largest changes data
        print("\n1. Loading largest changes data...")
        print(f"   File: {LARGEST_CHANGES_FILE.name}")
        largest_changes_df = pd.read_csv(LARGEST_CHANGES_FILE)
        print(f"   [OK] Loaded {len(largest_changes_df):,} domains")
        
        # Step 2: Load trend comparison for trend direction
        print("\n2. Loading trend comparison data...")
        print(f"   File: {TREND_COMPARISON_FILE.name}")
        comparison_df = pd.read_csv(TREND_COMPARISON_FILE)
        print(f"   [OK] Loaded {len(comparison_df):,} data points")
        
        # Step 3: Create summary table
        table_df = create_summary_table(largest_changes_df, comparison_df)
        
        # Step 4: Save results
        print("\n4. Saving results...")
        table_df.to_csv(OUTPUT_TABLE, index=False)
        print(f"   [OK] Saved table to {OUTPUT_TABLE.name}")
        
        # Create LaTeX version
        latex_table = create_latex_table(table_df)
        with open(OUTPUT_TABLE_TEX, 'w') as f:
            f.write(latex_table)
        print(f"   [OK] Saved LaTeX table to {OUTPUT_TABLE_TEX.name}")
        
        # Step 5: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("SUMMARY TABLE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary Table Preview:")
        print(table_df[['domain', 'change', 'percent_change', 'interpretation']].head().to_string(index=False))
        print(f"\nOutput files:")
        print(f"  CSV Table: {OUTPUT_TABLE}")
        print(f"  LaTeX Table: {OUTPUT_TABLE_TEX}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Summary table creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
