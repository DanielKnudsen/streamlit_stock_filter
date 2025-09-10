"""
Before/After Quarterly Analysis
Compares ranking changes from day before quarterly report to day after
"""

import sqlite3
import pandas as pd
import logging
from datetime import date, timedelta
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def analyze_before_after_changes(db_path: str, days_back: int = 30) -> pd.DataFrame:
    """Analyze before/after ranking changes for quarterly reports"""
    
    with sqlite3.connect(db_path) as conn:
        # Get before/after pairs from recent captures
        cutoff_date = date.today() - timedelta(days=days_back)
        
        query = """
        WITH before_after_pairs AS (
            SELECT 
                b.ticker,
                b.dimension,
                b.value as before_value,
                b.date as before_date,
                b.quarter_diff as before_quarter_diff,
                a.value as after_value,
                a.date as after_date,
                a.quarter_diff as after_quarter_diff,
                (a.value - b.value) as change_value,
                CASE 
                    WHEN b.dimension LIKE '%Rank%' THEN (b.value - a.value)  -- For ranks, lower is better
                    ELSE (a.value - b.value)  -- For values, higher is better
                END as improvement_score
            FROM ranking_history b
            JOIN ranking_history a ON (
                b.ticker = a.ticker 
                AND b.dimension = a.dimension
                AND b.capture_type = 'before_quarterly'
                AND a.capture_type = 'after_quarterly'
                AND a.date = date(b.date, '+1 day')  -- After is day after before
            )
            WHERE b.date >= ?
        )
        SELECT 
            ticker,
            COUNT(DISTINCT dimension) as dimensions_compared,
            AVG(improvement_score) as avg_improvement,
            SUM(CASE WHEN improvement_score > 0 THEN 1 ELSE 0 END) as dimensions_improved,
            SUM(CASE WHEN improvement_score < 0 THEN 1 ELSE 0 END) as dimensions_worsened,
            MAX(before_date) as analysis_date,
            MAX(before_quarter_diff) as quarter_diff_before,
            MAX(after_quarter_diff) as quarter_diff_after
        FROM before_after_pairs
        GROUP BY ticker
        ORDER BY avg_improvement DESC
        """
        
        results = pd.read_sql(query, conn, params=(cutoff_date,))
        
        if results.empty:
            logging.info("No before/after pairs found in the specified period")
            return pd.DataFrame()
        
        # Calculate improvement percentage
        results['improvement_rate'] = (results['dimensions_improved'] / results['dimensions_compared'] * 100).round(1)
        
        # Add priority classification
        results['impact_classification'] = pd.cut(
            results['avg_improvement'], 
            bins=[-float('inf'), -1, 0, 1, float('inf')], 
            labels=['Significant_Decline', 'Slight_Decline', 'Slight_Improvement', 'Significant_Improvement']
        )
        
        return results

def get_detailed_changes(db_path: str, ticker: str, days_back: int = 30) -> pd.DataFrame:
    """Get detailed before/after changes for a specific ticker"""
    
    with sqlite3.connect(db_path) as conn:
        cutoff_date = date.today() - timedelta(days=days_back)
        
        query = """
        SELECT 
            b.dimension,
            b.value as before_value,
            a.value as after_value,
            (a.value - b.value) as raw_change,
            CASE 
                WHEN b.dimension LIKE '%Rank%' THEN (b.value - a.value)
                ELSE (a.value - b.value)
            END as improvement_score,
            b.date as before_date,
            a.date as after_date
        FROM ranking_history b
        JOIN ranking_history a ON (
            b.ticker = a.ticker 
            AND b.dimension = a.dimension
            AND b.capture_type = 'before_quarterly'
            AND a.capture_type = 'after_quarterly'
            AND a.date = date(b.date, '+1 day')
        )
        WHERE b.ticker = ? AND b.date >= ?
        ORDER BY improvement_score DESC
        """
        
        return pd.read_sql(query, conn, params=(ticker, cutoff_date))

def generate_before_after_report(db_path: str) -> str:
    """Generate a before/after analysis report"""
    
    setup_logging()
    
    # Get before/after analysis
    analysis = analyze_before_after_changes(db_path)
    
    if analysis.empty:
        return "No before/after quarterly data available yet. Run the system after QuarterDiff changes are detected."
    
    report = ["📊 Before/After Quarterly Impact Analysis", "=" * 50, ""]
    
    # Top improvers
    top_improvers = analysis.head(10)
    if not top_improvers.empty:
        report.append("🚀 Top 10 Stocks - Most Improved After Quarterly Report:")
        report.append("-" * 55)
        for _, stock in top_improvers.iterrows():
            report.append(f"• {stock['ticker']}: Avg Improvement {stock['avg_improvement']:.2f}")
            report.append(f"  📈 {stock['dimensions_improved']}/{stock['dimensions_compared']} dimensions improved ({stock['improvement_rate']:.1f}%)")
            report.append(f"  📅 Analysis Date: {stock['analysis_date']}")
            report.append(f"  📊 Quarter Diff: {stock['quarter_diff_before']:.1f} → {stock['quarter_diff_after']:.1f}")
            report.append("")
    
    # Bottom performers
    bottom_performers = analysis.tail(5)
    if not bottom_performers.empty:
        report.append("⚠️  Bottom 5 Stocks - Declined After Quarterly Report:")
        report.append("-" * 50)
        for _, stock in bottom_performers.iterrows():
            report.append(f"• {stock['ticker']}: Avg Change {stock['avg_improvement']:.2f}")
            report.append(f"  📉 {stock['dimensions_worsened']}/{stock['dimensions_compared']} dimensions worsened")
            report.append("")
    
    # Summary statistics
    report.extend([
        "📈 Summary Statistics:",
        f"Total stocks analyzed: {len(analysis)}",
        f"Average improvement score: {analysis['avg_improvement'].mean():.2f}",
        f"Stocks with net improvement: {(analysis['avg_improvement'] > 0).sum()}",
        f"Stocks with net decline: {(analysis['avg_improvement'] < 0).sum()}",
        f"Average improvement rate: {analysis['improvement_rate'].mean():.1f}%",
        ""
    ])
    
    return "\n".join(report)

def main():
    """Main function to demonstrate before/after analysis"""
    
    # Use local database path
    base_path = Path(__file__).parent
    db_path = base_path / 'data' / 'local' / 'ranking_history.db'
    
    if not db_path.exists():
        print("❌ Ranking history database not found")
        print(f"Expected: {db_path}")
        return
    
    # Generate before/after analysis report
    report = generate_before_after_report(str(db_path))
    print(report)
    
    # Save report to file
    report_path = base_path / 'before_after_analysis.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Before/After analysis saved to: {report_path}")

if __name__ == "__main__":
    main()
