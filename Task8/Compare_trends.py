"""
Task 8: Compare Trends - v1 vs v2
Goal: Calculate year-by-year trends and identify largest changes after calibration
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent

# File paths
V1_RESULTS_FILE = OUTPUT_DIR / "yc_detector_v1_results.parquet"
V2_RESULTS_FILE = OUTPUT_DIR / "yc_detector_v2_results.parquet"

OUTPUT_TREND_COMPARISON = OUTPUT_DIR / "yc_trend_comparison.csv"
OUTPUT_LARGEST_CHANGES = OUTPUT_DIR / "yc_largest_changes.csv"
OUTPUT_AI_TRENDS = OUTPUT_DIR / "yc_ai_vs_nonai_trends.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "task8_trend_comparison_summary.json"

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

def calculate_trends(df, version='v1'):
    """Calculate year-by-year trends for a detector version"""
    if 'year' not in df.columns:
        return {}
    
    trends = {}
    
    for year in sorted(df['year'].dropna().unique()):
        year_df = df[df['year'] == year]
        year_stats = {}
        
        for domain in DOMAIN_NAMES:
            present_col = f'text_domain_present_{domain}_{version}'
            if present_col in year_df.columns:
                n_present = year_df[present_col].sum()
                coverage = 100 * n_present / len(year_df) if len(year_df) > 0 else 0
                year_stats[domain] = {
                    'coverage_percent': round(coverage, 2),
                    'postings_with_domain': int(n_present),
                    'total_postings': len(year_df)
                }
        
        trends[int(year)] = year_stats
    
    return trends

def calculate_overall_changes(v1_trends, v2_trends):
    """Calculate overall changes across all years"""
    changes = {}
    
    for domain in DOMAIN_NAMES:
        v1_avg = 0
        v2_avg = 0
        years_counted = 0
        
        # Calculate average coverage across all years
        for year in sorted(set(list(v1_trends.keys()) + list(v2_trends.keys()))):
            if year in v1_trends and domain in v1_trends[year]:
                v1_avg += v1_trends[year][domain]['coverage_percent']
            if year in v2_trends and domain in v2_trends[year]:
                v2_avg += v2_trends[year][domain]['coverage_percent']
            years_counted += 1
        
        if years_counted > 0:
            v1_avg /= years_counted
            v2_avg /= years_counted
        
        change = v2_avg - v1_avg
        
        changes[domain] = {
            'v1_avg_coverage': round(v1_avg, 2),
            'v2_avg_coverage': round(v2_avg, 2),
            'change': round(change, 2),
            'change_percent': round((change / v1_avg * 100) if v1_avg > 0 else 0, 2)
        }
    
    return changes

def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("TASK 8: Compare Trends - v1 vs v2")
    print("=" * 80)
    
    try:
        # Step 1: Load v1 results
        print("\n1. Loading detector v1 results...")
        print(f"   File: {V1_RESULTS_FILE.name}")
        v1_df = pd.read_parquet(V1_RESULTS_FILE)
        print(f"   [OK] Loaded {len(v1_df):,} postings")
        
        # Step 2: Load v2 results
        print("\n2. Loading detector v2 results...")
        print(f"   File: {V2_RESULTS_FILE.name}")
        v2_df = pd.read_parquet(V2_RESULTS_FILE)
        print(f"   [OK] Loaded {len(v2_df):,} postings")
        
        # Step 3: Calculate trends
        print("\n3. Calculating trends...")
        v1_trends = calculate_trends(v1_df, version='v1')
        v2_trends = calculate_trends(v2_df, version='v2')
        
        print(f"   [OK] Calculated trends for {len(v1_trends)} years (v1) and {len(v2_trends)} years (v2)")
        
        # Step 4: Create year-by-year comparison
        print("\n4. Creating year-by-year comparison...")
        comparison_rows = []
        
        all_years = sorted(set(list(v1_trends.keys()) + list(v2_trends.keys())))
        
        for year in all_years:
            for domain in DOMAIN_NAMES:
                v1_coverage = v1_trends.get(year, {}).get(domain, {}).get('coverage_percent', 0)
                v2_coverage = v2_trends.get(year, {}).get(domain, {}).get('coverage_percent', 0)
                change = v2_coverage - v1_coverage
                
                comparison_rows.append({
                    'year': int(year),
                    'domain': domain,
                    'v1_coverage_percent': v1_coverage,
                    'v2_coverage_percent': v2_coverage,
                    'change': round(change, 2),
                    'change_percent': round((change / v1_coverage * 100) if v1_coverage > 0 else 0, 2)
                })
        
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Step 5: Calculate overall changes
        print("\n5. Calculating overall changes...")
        overall_changes = calculate_overall_changes(v1_trends, v2_trends)
        
        # Step 6: Identify largest changes
        print("\n6. Identifying largest changes...")
        largest_changes_rows = []
        
        for domain, change_data in overall_changes.items():
            largest_changes_rows.append({
                'domain': domain,
                'v1_avg_coverage': change_data['v1_avg_coverage'],
                'v2_avg_coverage': change_data['v2_avg_coverage'],
                'absolute_change': change_data['change'],
                'percent_change': change_data['change_percent'],
                'interpretation': f"Calibration {'increased' if change_data['change'] > 0 else 'decreased'} coverage by {abs(change_data['change']):.1f} percentage points"
            })
        
        largest_changes_df = pd.DataFrame(largest_changes_rows)
        largest_changes_df = largest_changes_df.sort_values('absolute_change', key=abs, ascending=False)
        
        # Step 7: AI vs non-AI analysis (if available)
        print("\n7. Analyzing AI vs non-AI trends...")
        ai_trends_rows = []
        ai_analysis_available = False
        
        if 'is_ai' in v1_df.columns and 'is_ai' in v2_df.columns:
            for ai_class in [True, False]:
                ai_label = 'AI' if ai_class else 'Non-AI'
                v1_ai_df = v1_df[v1_df['is_ai'] == ai_class]
                v2_ai_df = v2_df[v2_df['is_ai'] == ai_class]
                
                if len(v1_ai_df) > 0 and len(v2_ai_df) > 0:
                    v1_ai_trends = calculate_trends(v1_ai_df, version='v1')
                    v2_ai_trends = calculate_trends(v2_ai_df, version='v2')
                    
                    for year in sorted(set(list(v1_ai_trends.keys()) + list(v2_ai_trends.keys()))):
                        for domain in DOMAIN_NAMES:
                            v1_coverage = v1_ai_trends.get(year, {}).get(domain, {}).get('coverage_percent', 0)
                            v2_coverage = v2_ai_trends.get(year, {}).get(domain, {}).get('coverage_percent', 0)
                            change = v2_coverage - v1_coverage
                            
                            ai_trends_rows.append({
                                'year': int(year),
                                'ai_classification': ai_label,
                                'domain': domain,
                                'v1_coverage_percent': v1_coverage,
                                'v2_coverage_percent': v2_coverage,
                                'change': round(change, 2)
                            })
            
            if ai_trends_rows:
                ai_trends_df = pd.DataFrame(ai_trends_rows)
                ai_trends_df.to_csv(OUTPUT_AI_TRENDS, index=False)
                print(f"   [OK] Saved AI vs non-AI trends to {OUTPUT_AI_TRENDS.name}")
                ai_analysis_available = True
            else:
                print("   [INFO] No AI classification data available")
        else:
            print("   [INFO] No AI classification column found in data")
        
        # Step 8: Save results
        print("\n8. Saving results...")
        comparison_df.to_csv(OUTPUT_TREND_COMPARISON, index=False)
        print(f"   [OK] Saved trend comparison to {OUTPUT_TREND_COMPARISON.name}")
        
        largest_changes_df.to_csv(OUTPUT_LARGEST_CHANGES, index=False)
        print(f"   [OK] Saved largest changes to {OUTPUT_LARGEST_CHANGES.name}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_years': len(all_years),
            'years': [int(y) for y in all_years],
            'overall_changes': overall_changes,
            'largest_increases': largest_changes_df.nlargest(3, 'absolute_change')[['domain', 'absolute_change']].to_dict('records'),
            'largest_decreases': largest_changes_df.nsmallest(3, 'absolute_change')[['domain', 'absolute_change']].to_dict('records'),
            'ai_analysis_available': ai_analysis_available
        }
        
        with open(OUTPUT_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] Saved summary to {OUTPUT_SUMMARY.name}")
        
        # Step 9: Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TREND COMPARISON COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Years analyzed: {len(all_years)} ({min(all_years)}-{max(all_years)})")
        print(f"\nLargest Changes (Top 5):")
        for idx, row in largest_changes_df.head(5).iterrows():
            print(f"  {row['domain']}: {row['absolute_change']:+.2f} percentage points ({row['percent_change']:+.1f}%)")
        print(f"\nOutput files:")
        print(f"  Trend Comparison: {OUTPUT_TREND_COMPARISON}")
        print(f"  Largest Changes: {OUTPUT_LARGEST_CHANGES}")
        if ai_analysis_available:
            print(f"  AI vs Non-AI Trends: {OUTPUT_AI_TRENDS}")
        print(f"  Summary: {OUTPUT_SUMMARY}")
        print(f"\nTotal time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n[ERROR] Trend comparison failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
