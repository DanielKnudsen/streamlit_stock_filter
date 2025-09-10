"""
Daily Capture Script for Ranking History - QuarterDiff Triggered Approach
Captures ranking data only when QuarterDiff changes (quarterly reports published)
"""

import os
import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from ranking_history_tracker import RankingHistoryTracker

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def detect_environment():
    """Detect if running locally or in GitHub Actions"""
    is_github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
    
    if is_github_actions:
        workspace = os.getenv('GITHUB_WORKSPACE', '/github/workspace')
        csv_path = os.path.join(workspace, 'data', 'local', 'stock_evaluation_results.csv')
        db_path = os.path.join(workspace, 'data', 'local', 'ranking_history.db')
    else:
        # Local development paths
        base_path = Path(__file__).parent
        csv_path = base_path / 'data' / 'local' / 'stock_evaluation_results.csv'
        db_path = base_path / 'data' / 'local' / 'ranking_history.db'
    
    return {
        'is_github_actions': is_github_actions,
        'csv_path': str(csv_path),
        'db_path': str(db_path),
        'workspace': workspace if is_github_actions else str(base_path)
    }

def main():
    setup_logging()
    
    # Detect environment
    env = detect_environment()
    logging.info(f"Environment: {'GitHub Actions' if env['is_github_actions'] else 'Local Development'}")
    logging.info(f"Workspace: {env['workspace']}")
    logging.info(f"CSV Path: {env['csv_path']}")
    logging.info(f"DB Path: {env['db_path']}")
    
    # Check if CSV file exists
    if not os.path.exists(env['csv_path']):
        logging.error(f"CSV file not found: {env['csv_path']}")
        # List available files for debugging
        data_dir = os.path.dirname(env['csv_path'])
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            logging.info(f"Files in {data_dir}: {files}")
        return 1
    
    try:
        # Initialize tracker
        tracker = RankingHistoryTracker(env['db_path'])
        
        # Capture rankings using quarterly-triggered approach
        logging.info("Starting quarterly-triggered ranking capture...")
        stats = tracker.capture_quarterly_triggered_rankings(env['csv_path'])
        
        # Log detailed results
        logging.info("Quarterly-triggered capture completed successfully!")
        logging.info(f"Records inserted: {stats['records_inserted']}")
        logging.info(f"Tickers with changes: {stats['tickers_with_changes']}")
        logging.info(f"QuarterDiff changes detected: {stats['quarter_diff_changes']}")
        logging.info(f"Before captures: {stats.get('before_captures', 0)}")
        logging.info(f"After captures: {stats.get('after_captures', 0)}")
        
        if stats['quarter_diff_changes'] == 0:
            logging.info("No quarterly changes detected - database remains unchanged")
        else:
            logging.info(f"Before/after quarterly snapshots captured for {stats['tickers_with_changes']} stocks")
        
        return 0
        
    except Exception as e:
        logging.error(f"Failed to capture rankings: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

import sys
import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from ranking_history_tracker import RankingHistoryTracker, auto_capture_daily

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging to both file and console
    log_file = Path("logs/daily_capture.log")
    log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Capture daily ranking data")
    parser.add_argument("--date", type=str, help="Capture date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--backfill", type=int, help="Backfill N days of historical data (simulated)")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logging.info("🚀 Starting daily ranking capture")
    
    try:
        # Determine capture date
        if args.date:
            capture_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            capture_date = date.today()
        
        logging.info(f"📅 Capture date: {capture_date}")
        
        # Initialize tracker
        tracker = RankingHistoryTracker()
        
        # Determine CSV path based on environment
        import os
        environment = os.getenv('ENVIRONMENT', 'local')
        if environment == 'remote':
            csv_path = "data/remote/stock_evaluation_results.csv"
        else:
            csv_path = "data/local/stock_evaluation_results.csv"
        
        logging.info(f"🌍 Environment: {environment}")
        logging.info(f"📁 CSV path: {csv_path}")
        
        # Check if CSV file exists
        if not Path(csv_path).exists():
            logging.error(f"❌ CSV file not found: {csv_path}")
            return 1
        
        # Handle backfill scenario
        if args.backfill:
            logging.info(f"📈 Backfilling {args.backfill} days of data (simulated)")
            
            for days_ago in range(args.backfill, -1, -1):
                backfill_date = capture_date - timedelta(days=days_ago)
                
                # Skip if we already have data for this date
                existing_dates = tracker.get_available_dates()
                if backfill_date.strftime("%Y-%m-%d") in existing_dates:
                    logging.info(f"⏭️  Skipping {backfill_date} - data already exists")
                    continue
                
                logging.info(f"📊 Capturing data for {backfill_date}")
                stats = tracker.capture_current_rankings(csv_path, backfill_date)
                logging.info(f"✅ Captured {stats['records_inserted']} records for {backfill_date}")
        
        else:
            # Single day capture
            # Check if we already have data for today
            existing_dates = tracker.get_available_dates()
            if capture_date.strftime("%Y-%m-%d") in existing_dates:
                logging.warning(f"⚠️  Data for {capture_date} already exists - will overwrite")
            
            # Capture current rankings
            stats = tracker.capture_current_rankings(csv_path, capture_date)
            
            logging.info(f"✅ Capture complete!")
            logging.info(f"📊 Records inserted: {stats['records_inserted']:,}")
            logging.info(f"📏 Dimensions captured: {stats['dimensions_captured']}")
            logging.info(f"🏢 Tickers processed: {stats['tickers_processed']}")
        
        # Show database status
        all_dates = tracker.get_available_dates()
        all_dimensions = tracker.get_available_dimensions()
        
        logging.info(f"📅 Total dates in database: {len(all_dates)}")
        logging.info(f"📏 Total dimensions tracked: {len(all_dimensions)}")
        
        if all_dates:
            logging.info(f"📈 Date range: {all_dates[-1]} to {all_dates[0]}")
        
        logging.info("🎉 Daily capture completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"❌ Daily capture failed: {e}")
        logging.exception("Full error details:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
