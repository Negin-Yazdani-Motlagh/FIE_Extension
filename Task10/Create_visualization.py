"""
Task 10: Create Visualization - Skill Bundles
Goal: Generate visualization of top skill bundles
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
BUNDLES_FILE = OUTPUT_DIR / "skill_bundles.csv"
CURRICULUM_FILE = OUTPUT_DIR / "curriculum_alignment_matrix.csv"

OUTPUT_BUNDLES_FIGURE = OUTPUT_DIR / "skill_bundles_figure.png"
OUTPUT_BUNDLES_PDF = OUTPUT_DIR / "skill_bundles_figure.pdf"

def create_bundles_figure(bundles_df):
    """Create visualization of top skill bundles"""
    print("\n3. Creating bundles figure...")
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 8)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Select top 10 bundles
    top_bundles = bundles_df.head(10).copy()
    
    # Create bundle labels
    bundle_labels = [
        f"{row['soft_domain'][:30]}...\n+ {row['technical_skill'][:30]}..."
        for _, row in top_bundles.iterrows()
    ]
    
    # Create horizontal bar chart
    y_pos = range(len(top_bundles))
    bars = ax.barh(y_pos, top_bundles['cooccurrence_count'], 
                    color='steelblue', alpha=0.8)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {label}" for i, label in enumerate(bundle_labels)], 
                        fontsize=9)
    ax.set_xlabel('Co-Occurrence Count', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Soft-Skill + Technical-Skill Bundles\n(LinkedIn 2024)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, top_bundles['cooccurrence_count'])):
        ax.text(count + 100, bar.get_y() + bar.get_height()/2, 
               f'{int(count):,}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    return fig

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 10: Create Visualization - Skill Bundles")
    print("=" * 80)
    
    try:
        # Step 1: Load bundles data
        print("\n1. Loading bundles data...")
        print(f"   File: {BUNDLES_FILE.name}")
        if not BUNDLES_FILE.exists():
            print(f"   [ERROR] File not found: {BUNDLES_FILE}")
            print("   [INFO] Please run task10_extract_bundles.py first")
            return
        
        bundles_df = pd.read_csv(BUNDLES_FILE)
        print(f"   [OK] Loaded {len(bundles_df)} bundles")
        
        # Step 2: Create figure
        fig = create_bundles_figure(bundles_df)
        
        # Step 3: Save figure
        print("\n2. Saving figure...")
        fig.savefig(OUTPUT_BUNDLES_FIGURE, dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved PNG to {OUTPUT_BUNDLES_FIGURE.name}")
        
        fig.savefig(OUTPUT_BUNDLES_PDF, bbox_inches='tight')
        print(f"   [OK] Saved PDF to {OUTPUT_BUNDLES_PDF.name}")
        
        plt.close()
        
        # Step 4: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Bundles visualized: {min(10, len(bundles_df))}")
        print(f"\nOutput files:")
        print(f"  PNG: {OUTPUT_BUNDLES_FIGURE}")
        print(f"  PDF: {OUTPUT_BUNDLES_PDF}")
        print(f"\nTotal time: {duration:.1f} seconds")
        
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
