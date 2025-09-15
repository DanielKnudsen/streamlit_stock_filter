"""
Quarterly Notification Script
Identifies stocks with significant ranking changes during quarterly reports
Integrates with automated content generation pipeline
"""

import sqlite3
import pandas as pd
import logging
import subprocess
import os
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

def should_trigger_content_generation(changes: pd.DataFrame) -> bool:
    """Determine if quarterly changes warrant automated content generation"""
    
    if changes.empty:
        return False
    
    # Content generation thresholds
    min_high_priority = 2
    min_total_stocks = 5
    min_avg_score = 70.0
    
    high_priority_count = len(changes[changes['notification_priority'] == 'High'])
    total_stocks = len(changes)
    avg_score = changes['avg_rank_score'].mean()
    
    criteria = [
        high_priority_count >= min_high_priority,
        total_stocks >= min_total_stocks,
        avg_score >= min_avg_score
    ]
    
    meets_threshold = sum(criteria) >= 2  # At least 2 of 3 criteria
    
    logging.info("📊 Content generation criteria check:")
    logging.info(f"  - High priority stocks: {high_priority_count} (need {min_high_priority})")
    logging.info(f"  - Total stocks: {total_stocks} (need {min_total_stocks})")
    logging.info(f"  - Average score: {avg_score:.1f} (need {min_avg_score})")
    logging.info(f"  - Criteria met: {sum(criteria)}/3")
    
    return meets_threshold

def trigger_content_generation() -> bool:
    """Trigger automated content generation workflow"""
    try:
        import subprocess
        import os
        
        logging.info("🚀 Triggering automated content generation...")
        
        # Set environment for content generation
        env = os.environ.copy()
        env['ENVIRONMENT'] = os.getenv('ENVIRONMENT', 'local')
        
        # Run content generator
        result = subprocess.run(
            ['uv', 'run', 'python', 'content_generator.py'], 
            capture_output=True, 
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logging.info("✅ Content generation completed successfully")
            logging.info(f"Output: {result.stdout}")
            return True
        else:
            logging.error(f"❌ Content generation failed with code {result.returncode}")
            logging.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ Content generation timed out after 5 minutes")
        return False
    except Exception as e:
        logging.error(f"❌ Failed to trigger content generation: {e}")
        return False

def process_quarterly_notifications_with_content(db_path: str) -> dict:
    """Main workflow: analyze quarterly changes and trigger content if warranted"""
    
    setup_logging()
    
    # Generate notification report
    changes = find_quarterly_ranking_changes(db_path)
    
    # Generate text report for logging/debugging
    _ = generate_notification_report(db_path)  # Generate but don't store unused report
    
    results = {
        'report_generated': True,
        'total_stocks': len(changes),
        'content_triggered': False,
        'content_success': False
    }
    
    # Check if content generation should be triggered
    if should_trigger_content_generation(changes):
        logging.info("📝 Quarterly changes meet content generation criteria")
        
        content_success = trigger_content_generation()
        results['content_triggered'] = True
        results['content_success'] = content_success
        
        if content_success:
            logging.info("🎉 Automated content generation workflow completed")
        else:
            logging.warning("⚠️ Content generation triggered but failed")
    else:
        logging.info("📊 Quarterly changes don't meet content generation thresholds")
    
    return results

def main():
    """Main function - can be used for manual testing or automation"""
    
    # Use local database path
    base_path = Path(__file__).parent
    db_path = base_path / 'data' / 'local' / 'ranking_history.db'
    
    if not db_path.exists():
        print("❌ Ranking history database not found")
        print(f"Expected: {db_path}")
        return
    
    print("🔔 Quarterly Notifications & Content Generation")
    print("=" * 50)
    
    # Process notifications with potential content generation
    results = process_quarterly_notifications_with_content(str(db_path))
    
    # Display results
    print("\n📊 Results Summary:")
    print(f"  - Total stocks analyzed: {results['total_stocks']}")
    print(f"  - Content generation triggered: {results['content_triggered']}")
    
    if results['content_triggered']:
        if results['content_success']:
            print("  - Content generation: ✅ SUCCESS")
        else:
            print("  - Content generation: ❌ FAILED")
    
    # Also generate and save the standard notification report
    report = generate_notification_report(str(db_path))
    print(f"\n{report}")
    
    # Save report to file
    report_path = base_path / 'quarterly_ranking_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
