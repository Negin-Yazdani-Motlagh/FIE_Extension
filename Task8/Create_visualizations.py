"""
Task 8: Create Visualizations - YC Trends Pre/Post Calibration
Goal: Generate main figure showing trend curves for focal domains
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
TREND_COMPARISON_FILE = OUTPUT_DIR / "yc_trend_comparison.csv"

OUTPUT_FIGURE_PNG = OUTPUT_DIR / "yc_trend_comparison_figure.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "yc_trend_comparison_figure.pdf"

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

# Focal domains to plot (select 3-5 most interesting)
# Will be determined from largest changes, but can be overridden
FOCAL_DOMAINS = None  # Will be set based on data

def select_focal_domains(comparison_df, n=5):
    """Select focal domains based on largest absolute changes"""
    # Calculate average absolute change per domain
    domain_changes = comparison_df.groupby('domain')['change'].apply(lambda x: abs(x.mean())).sort_values(ascending=False)
    
    # Select top N
    focal = domain_changes.head(n).index.tolist()
    
    return focal

def create_trend_figure(comparison_df, focal_domains):
    """Create main trend comparison figure"""
    print("\n3. Creating trend comparison figure...")
    
    # Set style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    else:
        plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create subplots: one per focal domain
    n_domains = len(focal_domains)
    n_cols = 2
    n_rows = (n_domains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_domains == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('YC Soft-Skill Trends: Pre vs Post Calibration (2012-2024)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, domain in enumerate(focal_domains):
        ax = axes[idx]
        
        domain_df = comparison_df[comparison_df['domain'] == domain].sort_values('year')
        
        # Plot v1 and v2 trends
        ax.plot(domain_df['year'], domain_df['v1_coverage_percent'], 
                marker='o', label='Detector v1 (Pre-calibration)', 
                linewidth=2, markersize=6, color='#1f77b4')
        ax.plot(domain_df['year'], domain_df['v2_coverage_percent'], 
                marker='s', label='Detector v2 (Post-calibration)', 
                linewidth=2, markersize=6, color='#ff7f0e')
        
        # Formatting
        ax.set_title(domain, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Coverage %', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
        # Set y-axis limits (same for both to show change clearly)
        y_min = min(domain_df['v1_coverage_percent'].min(), domain_df['v2_coverage_percent'].min())
        y_max = max(domain_df['v1_coverage_percent'].max(), domain_df['v2_coverage_percent'].max())
        y_range = y_max - y_min
        ax.set_ylim(max(0, y_min - y_range * 0.1), min(100, y_max + y_range * 0.1))
    
    # Hide extra subplots
    for idx in range(len(focal_domains), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 8: Create Visualizations - YC Trends")
    print("=" * 80)
    
    try:
        # Step 1: Load trend comparison data
        print("\n1. Loading trend comparison data...")
        print(f"   File: {TREND_COMPARISON_FILE.name}")
        comparison_df = pd.read_csv(TREND_COMPARISON_FILE)
        print(f"   [OK] Loaded {len(comparison_df):,} data points")
        
        # Step 2: Select focal domains
        print("\n2. Selecting focal domains...")
        if FOCAL_DOMAINS is None:
            focal_domains = select_focal_domains(comparison_df, n=5)
        else:
            focal_domains = FOCAL_DOMAINS
        
        print(f"   [OK] Selected {len(focal_domains)} focal domains:")
        for domain in focal_domains:
            print(f"     - {domain}")
        
        # Step 3: Create figure
        fig = create_trend_figure(comparison_df, focal_domains)
        
        # Step 4: Save figure
        print("\n4. Saving figure...")
        fig.savefig(OUTPUT_FIGURE_PNG, dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved PNG to {OUTPUT_FIGURE_PNG.name}")
        
        fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches='tight')
        print(f"   [OK] Saved PDF to {OUTPUT_FIGURE_PDF.name}")
        
        plt.close()
        
        # Step 5: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Focal domains: {len(focal_domains)}")
        print(f"  Years covered: {comparison_df['year'].min()} - {comparison_df['year'].max()}")
        print(f"\nOutput files:")
        print(f"  PNG: {OUTPUT_FIGURE_PNG}")
        print(f"  PDF: {OUTPUT_FIGURE_PDF}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
