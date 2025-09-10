"""
📊 Ranking History Tracker
==========================

Normalized database system for tracking rank and TTM value changes over time.
Pivots stock_evaluation_results.csv into time-series format for trend analysis.

Schema:
    ranking_history(ticker, date, dimension, value)

Features:
- Daily automated capture
- Significant jump detection  
- Trend visualization
- Price correlation analysis
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import re

class RankingHistoryTracker:
    """Manages ranking and TTM value tracking database"""
    
    def __init__(self, db_path: str = "data/local/ranking_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main table with composite primary key including capture_type
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ranking_history (
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    dimension TEXT NOT NULL,
                    value REAL,
                    capture_type TEXT DEFAULT 'daily',  -- 'before_quarterly', 'after_quarterly', 'daily'
                    quarter_diff REAL,  -- Store the QuarterDiff value that triggered this capture
                    PRIMARY KEY (ticker, date, dimension, capture_type)
                )
            """)
            
            # Create QuarterDiff tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarter_diff_tracking (
                    ticker TEXT PRIMARY KEY,
                    last_quarter_diff REAL,
                    last_update_date DATE,
                    total_captures INTEGER DEFAULT 0
                )
            """)
            
            # Performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON ranking_history(ticker, date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dimension_date ON ranking_history(dimension, date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_only ON ranking_history(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_capture_type ON ranking_history(capture_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quarter_diff ON ranking_history(quarter_diff)")
            
            # Metadata table for tracking capture history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS capture_log (
                    capture_date DATE PRIMARY KEY,
                    records_inserted INTEGER,
                    dimensions_captured INTEGER,
                    tickers_processed INTEGER,
                    quarter_diff_changes INTEGER DEFAULT 0,
                    capture_mode TEXT DEFAULT 'daily',
                    status TEXT,
                    notes TEXT
                )
            """)
            
            conn.commit()
    
    def extract_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract columns that contain ranking data and related metrics"""
        # Skip basic identifying columns
        skip_columns = {'Unnamed: 0', 'Name', 'Sektor', 'Lista', 'QuarterDiff'}
        
        target_columns = []
        for col in df.columns:
            if col in skip_columns:
                continue
            # Include ranking columns and difference columns
            if ('Rank' in col or 'rank' in col or 
                '_diff' in col or 'Diff' in col or
                'Growth' in col or 'Value' in col or
                'Avg' in col or 'catRank' in col):
                target_columns.append(col)
        
        logging.info(f"Identified {len(target_columns)} target columns for ranking data")
        return target_columns

    def detect_quarter_diff_changes(self, df: pd.DataFrame) -> List[Dict]:
        """Detect stocks with QuarterDiff changes since last capture"""
        if 'QuarterDiff' not in df.columns:
            logging.warning("QuarterDiff column not found in data")
            return []
        
        # Get ticker column (first column)
        ticker_col = df.columns[0]
        
        changes = []
        with sqlite3.connect(self.db_path) as conn:
            for _, row in df.iterrows():
                ticker = row[ticker_col]
                current_quarter_diff = row['QuarterDiff']
                
                # Skip if QuarterDiff is NaN
                if pd.isna(current_quarter_diff):
                    continue
                
                # Get last known QuarterDiff for this ticker
                last_record = conn.execute("""
                    SELECT last_quarter_diff, last_update_date, total_captures
                    FROM quarter_diff_tracking 
                    WHERE ticker = ?
                """, (ticker,)).fetchone()
                
                if last_record is None:
                    # First time seeing this ticker - record initial state
                    changes.append({
                        'ticker': ticker,
                        'old_quarter_diff': None,
                        'new_quarter_diff': current_quarter_diff,
                        'change_type': 'initial'
                    })
                else:
                    old_quarter_diff, last_date, total_captures = last_record
                    
                    # Check if QuarterDiff has changed
                    if abs(current_quarter_diff - old_quarter_diff) > 0.01:  # Small tolerance for float comparison
                        changes.append({
                            'ticker': ticker,
                            'old_quarter_diff': old_quarter_diff,
                            'new_quarter_diff': current_quarter_diff,
                            'change_type': 'quarterly_update',
                            'last_date': last_date,
                            'total_captures': total_captures
                        })
        
        return changes

    def capture_quarterly_triggered_rankings(self, csv_path: str, capture_date: date = None) -> Dict[str, int]:
        """Capture rankings only when QuarterDiff changes (before/after snapshots)"""
        if capture_date is None:
            capture_date = date.today()
            
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} records from {csv_path}")
            
            # Detect QuarterDiff changes
            changes = self.detect_quarter_diff_changes(df)
            logging.info(f"Detected {len(changes)} QuarterDiff changes")
            
            if not changes:
                logging.info("No QuarterDiff changes detected - no capture needed")
                return {'records_inserted': 0, 'tickers_with_changes': 0, 'quarter_diff_changes': 0, 'before_captures': 0, 'after_captures': 0}
            
            # Get tickers that have changes
            changed_tickers = [change['ticker'] for change in changes]
            logging.info(f"Tickers with quarterly changes: {changed_tickers}")
            
            total_records_inserted = 0
            before_captures = 0
            after_captures = 0
            
            with sqlite3.connect(self.db_path) as conn:
                # For each ticker with changes, capture before/after snapshots
                for change in changes:
                    ticker = change['ticker']
                    
                    # 1. Capture "BEFORE" state if this is not an initial capture
                    if change['change_type'] != 'initial':
                        # Get the previous ranking data for this ticker (most recent capture)
                        previous_data = conn.execute("""
                            SELECT dimension, value 
                            FROM ranking_history 
                            WHERE ticker = ? 
                            AND date = (SELECT MAX(date) FROM ranking_history WHERE ticker = ?)
                        """, (ticker, ticker)).fetchall()
                        
                        if previous_data:
                            # Insert "before" snapshot with yesterday's date
                            before_date = capture_date - timedelta(days=1)
                            for dimension, value in previous_data:
                                conn.execute("""
                                    INSERT INTO ranking_history 
                                    (ticker, date, dimension, value, capture_type, quarter_diff)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (ticker, before_date, dimension, value, 'before_quarterly', change['old_quarter_diff']))
                                total_records_inserted += 1
                            before_captures += 1
                            logging.info(f"Captured BEFORE snapshot for {ticker} ({len(previous_data)} dimensions)")
                
                # 2. Capture "AFTER" state with current data
                # Filter dataframe to only include changed tickers
                ticker_col = df.columns[0]
                changed_df = df[df[ticker_col].isin(changed_tickers)].copy()
                
                # Transform to normalized format for changed tickers only
                normalized_df = self.pivot_data(changed_df, capture_date)
                
                # Add capture metadata for "after" captures
                normalized_df['capture_type'] = 'after_quarterly'
                
                # Add QuarterDiff values
                quarter_diff_map = {change['ticker']: change['new_quarter_diff'] for change in changes}
                normalized_df['quarter_diff'] = normalized_df['ticker'].map(quarter_diff_map)
                
                logging.info(f"Pivoted to {len(normalized_df)} normalized records for AFTER quarterly changes")
                
                # Insert "after" ranking data
                normalized_df.to_sql('ranking_history', conn, if_exists='append', index=False)
                total_records_inserted += len(normalized_df)
                after_captures = len(changed_tickers)
                
                # Update QuarterDiff tracking
                for change in changes:
                    conn.execute("""
                        INSERT OR REPLACE INTO quarter_diff_tracking 
                        (ticker, last_quarter_diff, last_update_date, total_captures)
                        VALUES (?, ?, ?, COALESCE(
                            (SELECT total_captures FROM quarter_diff_tracking WHERE ticker = ?) + 1, 1
                        ))
                    """, (change['ticker'], change['new_quarter_diff'], capture_date, change['ticker']))
                
                # Log capture metadata
                stats = {
                    'records_inserted': total_records_inserted,
                    'tickers_with_changes': len(changed_tickers),
                    'quarter_diff_changes': len(changes),
                    'before_captures': before_captures,
                    'after_captures': after_captures
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO capture_log 
                    (capture_date, records_inserted, dimensions_captured, tickers_processed, 
                     quarter_diff_changes, capture_mode, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (capture_date, stats['records_inserted'], 
                     normalized_df['dimension'].nunique() if len(normalized_df) > 0 else 0,
                     len(changed_tickers), stats['quarter_diff_changes'], 
                     'before_after_quarterly', 'success', 
                     f"Before: {before_captures}, After: {after_captures} tickers. Changes: {', '.join(changed_tickers[:5])}"))
                
                conn.commit()
                
            logging.info(f"Successfully captured before/after quarterly rankings for {capture_date}")
            logging.info(f"Before captures: {before_captures}, After captures: {after_captures}")
            return stats
            
        except Exception as e:
            logging.error(f"Failed to capture quarterly-triggered rankings: {e}")
            # Log failure
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO capture_log 
                    (capture_date, records_inserted, dimensions_captured, tickers_processed, 
                     quarter_diff_changes, capture_mode, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (capture_date, 0, 0, 0, 0, 'before_after_quarterly', 'failed', str(e)))
                conn.commit()
            raise
        """Extract columns containing 'Rank' or 'ttm' patterns"""
        target_columns = []
        
        for col in df.columns:
            # Match rank columns: *_ratioRank, *_catRank, *_clusterRank
            if re.search(r'.*Rank$', col, re.IGNORECASE):
                target_columns.append(col)
            # Match TTM columns: *_ttm_ratioValue, *_ttm_ratioRank  
            elif re.search(r'.*_ttm_.*', col, re.IGNORECASE):
                target_columns.append(col)
        
        return sorted(target_columns)
    
    def pivot_data(self, df: pd.DataFrame, capture_date: date = None) -> pd.DataFrame:
        """Transform wide CSV format to normalized long format"""
        if capture_date is None:
            capture_date = date.today()
            
        # Get ticker column (first column without data, usually index 0)
        ticker_col = df.columns[0]  # Assuming first column is ticker
        
        # Extract target columns
        target_columns = self.extract_target_columns(df)
        
        # Create base dataframe with ticker and target columns
        subset_df = df[[ticker_col] + target_columns].copy()
        
        # Melt the dataframe to long format
        melted_df = pd.melt(
            subset_df,
            id_vars=[ticker_col],
            value_vars=target_columns,
            var_name='dimension',
            value_name='value'
        )
        
        # Add capture date
        melted_df['date'] = capture_date
        
        # Rename ticker column
        melted_df = melted_df.rename(columns={ticker_col: 'ticker'})
        
        # Remove rows with null values
        melted_df = melted_df.dropna(subset=['value'])
        
        # Reorder columns to match schema
        melted_df = melted_df[['ticker', 'date', 'dimension', 'value']]
        
        return melted_df
    
    def capture_current_rankings(self, csv_path: str, capture_date: date = None) -> Dict[str, int]:
        """Capture current rankings from CSV and store in database"""
        if capture_date is None:
            capture_date = date.today()
            
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} records from {csv_path}")
            
            # Transform to normalized format
            normalized_df = self.pivot_data(df, capture_date)
            logging.info(f"Pivoted to {len(normalized_df)} normalized records")
            
            # Insert into database (replace if exists for same date)
            with sqlite3.connect(self.db_path) as conn:
                # Remove existing data for this date
                conn.execute("DELETE FROM ranking_history WHERE date = ?", (capture_date,))
                
                # Insert new data
                normalized_df.to_sql('ranking_history', conn, if_exists='append', index=False)
                
                # Log capture metadata
                stats = {
                    'records_inserted': len(normalized_df),
                    'dimensions_captured': normalized_df['dimension'].nunique(),
                    'tickers_processed': normalized_df['ticker'].nunique()
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO capture_log 
                    (capture_date, records_inserted, dimensions_captured, tickers_processed, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (capture_date, stats['records_inserted'], stats['dimensions_captured'], 
                     stats['tickers_processed'], 'success', f"Auto-captured from {csv_path}"))
                
                conn.commit()
                
            logging.info(f"Successfully captured {stats['records_inserted']} records for {capture_date}")
            return stats
            
        except Exception as e:
            logging.error(f"Failed to capture rankings: {e}")
            # Log failure
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO capture_log 
                    (capture_date, records_inserted, dimensions_captured, tickers_processed, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (capture_date, 0, 0, 0, 'failed', str(e)))
                conn.commit()
            raise
    
    def get_significant_jumps(self, days_back: int = 7, min_change: float = 10.0) -> pd.DataFrame:
        """Find stocks with significant rank changes in recent period"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                WITH latest_data AS (
                    SELECT ticker, dimension, value as current_value,
                           ROW_NUMBER() OVER (PARTITION BY ticker, dimension ORDER BY date DESC) as rn
                    FROM ranking_history 
                    WHERE date >= date('now', '-{days} days')
                ),
                previous_data AS (
                    SELECT ticker, dimension, value as previous_value,
                           ROW_NUMBER() OVER (PARTITION BY ticker, dimension ORDER BY date DESC) as rn
                    FROM ranking_history 
                    WHERE date < date('now', '-{days} days')
                      AND date >= date('now', '-{days_extended} days')
                )
                SELECT 
                    l.ticker,
                    l.dimension,
                    l.current_value,
                    p.previous_value,
                    (l.current_value - p.previous_value) as change,
                    ROUND(((l.current_value - p.previous_value) / p.previous_value * 100), 2) as pct_change
                FROM latest_data l
                JOIN previous_data p ON l.ticker = p.ticker AND l.dimension = p.dimension
                WHERE l.rn = 1 AND p.rn = 1
                  AND ABS(l.current_value - p.previous_value) >= {min_change}
                ORDER BY ABS(l.current_value - p.previous_value) DESC
            """.format(days=days_back, days_extended=days_back*2, min_change=min_change)
            
            return pd.read_sql_query(query, conn)
    
    def get_ranking_trend(self, ticker: str, dimension: str, days_back: int = 30) -> pd.DataFrame:
        """Get ranking trend for specific ticker and dimension"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT date, value
                FROM ranking_history
                WHERE ticker = ? AND dimension = ?
                  AND date >= date('now', '-{days} days')
                ORDER BY date
            """.format(days=days_back)
            
            return pd.read_sql_query(query, conn, params=(ticker, dimension))
    
    def get_dimension_stats(self, dimension: str, days_back: int = 30) -> pd.DataFrame:
        """Get statistics for a specific dimension across all tickers"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    ticker,
                    COUNT(*) as data_points,
                    MIN(value) as min_value,
                    MAX(value) as max_value,
                    AVG(value) as avg_value,
                    (MAX(value) - MIN(value)) as volatility
                FROM ranking_history
                WHERE dimension = ?
                  AND date >= date('now', '-{days} days')
                GROUP BY ticker
                HAVING COUNT(*) >= 3
                ORDER BY volatility DESC
            """.format(days=days_back)
            
            return pd.read_sql_query(query, conn, params=(dimension,))
    
    def get_available_dates(self) -> List[str]:
        """Get all available capture dates"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT DISTINCT date FROM ranking_history ORDER BY date DESC"
            result = conn.execute(query).fetchall()
            return [row[0] for row in result]
    
    def get_available_dimensions(self) -> List[str]:
        """Get all available dimensions"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT DISTINCT dimension FROM ranking_history ORDER BY dimension"
            result = conn.execute(query).fetchall()
            return [row[0] for row in result]
    
    def get_capture_log(self) -> pd.DataFrame:
        """Get capture history for monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM capture_log ORDER BY capture_date DESC", conn)


# 🚀 Utility Functions for Daily Automation
def auto_capture_daily():
    """Automated daily capture function - can be called by scheduler"""
    csv_path = "data/local/stock_evaluation_results.csv"
    tracker = RankingHistoryTracker()
    
    try:
        stats = tracker.capture_current_rankings(csv_path)
        print(f"✅ Daily capture successful: {stats}")
        return True
    except Exception as e:
        print(f"❌ Daily capture failed: {e}")
        return False


# 📊 Analytics Helper Functions
def detect_rank_movers(min_change: float = 15.0, days: int = 7) -> pd.DataFrame:
    """Quick function to find significant rank movements"""
    tracker = RankingHistoryTracker()
    return tracker.get_significant_jumps(days_back=days, min_change=min_change)


def get_top_volatile_stocks(dimension: str = "Latest_clusterRank", days: int = 30) -> pd.DataFrame:
    """Find most volatile stocks for specific ranking dimension"""
    tracker = RankingHistoryTracker()
    return tracker.get_dimension_stats(dimension, days_back=days)


if __name__ == "__main__":
    # Demo usage
    print("🚀 Ranking History Tracker Demo")
    print("=" * 40)
    
    # Initialize tracker
    tracker = RankingHistoryTracker()
    
    # Capture current data
    print("📊 Capturing current rankings...")
    stats = tracker.capture_current_rankings("data/local/stock_evaluation_results.csv")
    print(f"✅ Captured: {stats}")
    
    # Show available data
    print(f"\n📅 Available dates: {tracker.get_available_dates()[:5]}")
    print(f"📏 Available dimensions: {len(tracker.get_available_dimensions())} total")
    
    # Example queries
    print("\n🔍 Recent significant jumps:")
    jumps = tracker.get_significant_jumps(days_back=30, min_change=10.0)
    if not jumps.empty:
        print(jumps.head())
    else:
        print("No significant jumps found (need more historical data)")
