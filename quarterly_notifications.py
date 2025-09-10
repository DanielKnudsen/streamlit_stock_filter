"""
Quarterly Notification Script
Identifies stocks with significant ranking changes during quarterly reports
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

def find_quarterly_ranking_changes(db_path: str, days_back: int = 30) -> pd.DataFrame:
    """Find stocks with significant ranking changes in quarterly captures"""
    
    with sqlite3.connect(db_path) as conn:
        # Get quarterly captures from the last N days
        cutoff_date = date.today() - timedelta(days=days_back)
        
        query = """
        SELECT 
            rh.ticker,
            rh.date as capture_date,
            rh.quarter_diff,
            qdt.total_captures,
            COUNT(DISTINCT rh.dimension) as dimensions_captured,
            AVG(CASE WHEN rh.dimension LIKE '%Rank%' THEN rh.value END) as avg_rank
        FROM ranking_history rh
        JOIN quarter_diff_tracking qdt ON rh.ticker = qdt.ticker
        WHERE rh.capture_type = 'quarterly_triggered'
          AND rh.date >= ?
        GROUP BY rh.ticker, rh.date, rh.quarter_diff, qdt.total_captures
        ORDER BY rh.date DESC, rh.ticker
        """
        
        results = pd.read_sql(query, conn, params=(cutoff_date,))
        
        if results.empty:
            logging.info("No quarterly captures found in the specified period")
            return pd.DataFrame()
        
        # Add analysis of ranking improvements
        results['avg_rank_score'] = 100 - results['avg_rank']  # Higher score = better rank
        results['notification_priority'] = pd.cut(
            results['avg_rank_score'], 
            bins=[0, 60, 80, 100], 
            labels=['Low', 'Medium', 'High']
        )
        
        return results

def generate_notification_report(db_path: str) -> str:
    """Generate a notification report for quarterly ranking changes"""
    
    setup_logging()
    
    # Find recent quarterly changes
    changes = find_quarterly_ranking_changes(db_path)
    
    if changes.empty:
        return "No quarterly ranking changes detected in the last 30 days."
    
    report = ["📊 Quarterly Stock Ranking Report", "=" * 40, ""]
    
    # Group by priority
    for priority in ['High', 'Medium', 'Low']:
        priority_stocks = changes[changes['notification_priority'] == priority]
        
        if not priority_stocks.empty:
            report.append(f"🔥 {priority} Priority Stocks ({len(priority_stocks)} stocks):")
            report.append("-" * 30)
            
            for _, stock in priority_stocks.iterrows():
                report.append(f"• {stock['ticker']}: Avg Rank Score {stock['avg_rank_score']:.1f}")
                report.append(f"  📅 Captured: {stock['capture_date']}")
                report.append(f"  📈 Quarter Diff: {stock['quarter_diff']}")
                report.append(f"  📊 Dimensions: {stock['dimensions_captured']}")
                report.append("")
    
    # Summary statistics
    report.extend([
        "📈 Summary:",
        f"Total stocks with quarterly updates: {len(changes)}",
        f"Average ranking score: {changes['avg_rank_score'].mean():.1f}",
        f"Date range: {changes['capture_date'].min()} to {changes['capture_date'].max()}",
        ""
    ])
    
    return "\n".join(report)

def main():
    """Main function to demonstrate notification system"""
    
    # Use local database path
    base_path = Path(__file__).parent
    db_path = base_path / 'data' / 'local' / 'ranking_history.db'
    
    if not db_path.exists():
        print("❌ Ranking history database not found")
        print(f"Expected: {db_path}")
        return
    
    # Generate notification report
    report = generate_notification_report(str(db_path))
    print(report)
    
    # Save report to file
    report_path = base_path / 'quarterly_ranking_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_path}")

if __name__ == "__main__":
    main()
