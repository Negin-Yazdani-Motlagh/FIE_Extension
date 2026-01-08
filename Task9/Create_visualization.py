"""
Task 9: Create Visualization - YC 2024 vs LinkedIn 2024
Goal: Generate compact comparison figure (rank comparison or paired bar chart)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
COMPARISON_FILE = OUTPUT_DIR / "yc_vs_linkedin_2024_comparison.csv"

OUTPUT_FIGURE_PNG = OUTPUT_DIR / "yc_vs_linkedin_2024_comparison.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "yc_vs_linkedin_2024_comparison.pdf"

def create_comparison_figure(comparison_df):
    """Create compact comparison figure"""
    print("\n3. Creating comparison figure...")
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by YC coverage for consistent ordering
    comparison_df = comparison_df.sort_values('yc_2024_coverage_percent', ascending=True)
    
    # Create positions for bars
    y_pos = range(len(comparison_df))
    bar_width = 0.35
    
    # Create bars
    bars1 = ax.barh([y - bar_width/2 for y in y_pos], 
                    comparison_df['yc_2024_coverage_percent'],
                    bar_width, 
                    label='YC 2024', 
                    color='#1f77b4',
                    alpha=0.8)
    
    bars2 = ax.barh([y + bar_width/2 for y in y_pos], 
                    comparison_df['linkedin_2024_coverage_percent'],
                    bar_width, 
                    label='LinkedIn 2024', 
                    color='#ff7f0e',
                    alpha=0.8)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['domain'], fontsize=9)
    ax.set_xlabel('Coverage %', fontsize=11, fontweight='bold')
    ax.set_title('Soft-Skill Domain Coverage: YC 2024 vs LinkedIn 2024\n(Bridge-Year Generalizability)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        width1 = bar1.get_width()
        width2 = bar2.get_width()
        if width1 > 0:
            ax.text(width1 + 0.5, bar1.get_y() + bar1.get_height()/2, 
                   f'{width1:.1f}%', va='center', fontsize=8)
        if width2 > 0:
            ax.text(width2 + 0.5, bar2.get_y() + bar2.get_height()/2, 
                   f'{width2:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    
    return fig

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 9: Create Visualization - YC vs LinkedIn 2024")
    print("=" * 80)
    
    try:
        # Step 1: Load comparison data
        print("\n1. Loading comparison data...")
        print(f"   File: {COMPARISON_FILE.name}")
        if not COMPARISON_FILE.exists():
            print(f"   [ERROR] File not found: {COMPARISON_FILE}")
            print("   [INFO] Please run task9_compare_yc_linkedin_2024.py first")
            return
        
        comparison_df = pd.read_csv(COMPARISON_FILE)
        print(f"   [OK] Loaded comparison data for {len(comparison_df)} domains")
        
        # Step 2: Create figure
        fig = create_comparison_figure(comparison_df)
        
        # Step 3: Save figure
        print("\n2. Saving figure...")
        fig.savefig(OUTPUT_FIGURE_PNG, dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved PNG to {OUTPUT_FIGURE_PNG.name}")
        
        fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches='tight')
        print(f"   [OK] Saved PDF to {OUTPUT_FIGURE_PDF.name}")
        
        plt.close()
        
        # Step 4: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Domains compared: {len(comparison_df)}")
        print(f"\nOutput files:")
        print(f"  PNG: {OUTPUT_FIGURE_PNG}")
        print(f"  PDF: {OUTPUT_FIGURE_PDF}")
        print(f"\nTotal time: {duration:.1f} seconds")
        
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
