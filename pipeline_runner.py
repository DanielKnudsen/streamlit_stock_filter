#!/usr/bin/env python3
"""
Main Pipeline Runner - Redesigned Swedish Stock Analysis Pipeline
================================================================

This script orchestrates the complete redesigned pipeline using the new secure,
function-based approach with temporal analysis framework.

Pipeline Stages:
1. Ticker Discovery & Data Fetching
2. Financial Data Processing (FinancialDataProcessor)
3. Ratio Calculations (RatioCalculator)
4. Results Export and Storage

Usage:
    python pipeline_runner.py [--environment local|remote] [--skip-fetch]

Environment Variables:
    ENVIRONMENT: 'local' or 'remote' (default: 'local')
    FETCH_DATA: 'Yes' or 'No' (default: 'Yes')

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import traceback
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_runner.log')
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from data_fetcher import fetch_yfinance_data, read_tickers_from_csv
    from FinancialDataProcessor import FinancialDataProcessor
    from RatioCalculator import RatioCalculator
    from io_utils import save_csv, save_pickle
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Load environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
FETCH_DATA = os.getenv('FETCH_DATA', 'Yes').lower() == 'yes'
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')


def setup_environment() -> None:
    """Set up the environment and verify dependencies."""
    logger.info(f"Pipeline Runner starting - Environment: {ENVIRONMENT}")
    logger.info(f"Data path: {CSV_PATH}")
    logger.info(f"Fetch data: {FETCH_DATA}")
    
    # Create directories
    CSV_PATH.mkdir(parents=True, exist_ok=True)
    
    # Verify dependencies
    try:
        import pandas  # noqa: F401
        import yfinance  # noqa: F401
        import yaml  # noqa: F401
        import numpy  # noqa: F401
        logger.info("All dependencies verified successfully")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def load_tickers(test_mode: bool = False) -> list:
    """Load ticker list from CSV file."""
    try:
        tickers_file = CSV_PATH / "tickers_lists.csv"
        if not tickers_file.exists():
            logger.error(f"Tickers file not found: {tickers_file}")
            sys.exit(1)
        
        tickers = read_tickers_from_csv(str(tickers_file))
        
        # Apply test mode restriction if enabled
        if test_mode and len(tickers) > 5:
            tickers = tickers[:5]
            logger.info(f"TEST MODE: Using only first {len(tickers)} tickers")
        
        logger.info(f"Loaded {len(tickers)} tickers from {tickers_file}")
        return tickers
        
    except Exception as e:
        logger.error(f"Failed to load tickers: {e}")
        sys.exit(1)


def fetch_financial_data(tickers: list) -> Dict[str, Any]:
    """Fetch financial data for all tickers."""
    if not FETCH_DATA:
        logger.info("Skipping data fetch (FETCH_DATA=False)")
        # Try to load existing data
        try:
            import pickle
            data_file = CSV_PATH / "raw_financial_data.pkl"
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.error("No existing data found and FETCH_DATA=False")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            sys.exit(1)
    
    logger.info(f"Fetching financial data for {len(tickers)} tickers...")
    
    try:
        # Load configuration to get years parameter
        import yaml
        try:
            with open('ratios_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # Calculate years from quarters (4 quarters per year)
            quarters_to_collect = config.get('data_collection', {}).get('quarters_to_collect', 16)
            years = max(4, quarters_to_collect // 4)  # At least 4 years, based on quarters
        except Exception as e:
            logger.warning(f"Could not load config for years parameter: {e}, using default 4 years")
            years = 4
        
        logger.info(f"Fetching {years} years of data per ticker")
        
        # Fetch data for each ticker
        from tqdm import tqdm
        raw_data = {}
        failed_tickers = []
        
        # Progress bar (disabled in GitHub Actions)
        disable_progress = os.getenv('ENVIRONMENT', 'local') != 'local'
        
        for ticker in tqdm(tickers, desc="Fetching financial data", disable=disable_progress):
            try:
                # Fetch both annual and quarterly data
                annual_data = fetch_yfinance_data(ticker, years, "annual")
                quarterly_data = fetch_yfinance_data(ticker, years, "quarterly")
                
                # Combine data if both successful
                if annual_data and quarterly_data:
                    raw_data[ticker] = {
                        'annual': annual_data,
                        'quarterly': quarterly_data
                    }
                elif annual_data:
                    # Use only annual data if quarterly fails
                    raw_data[ticker] = {
                        'annual': annual_data,
                        'quarterly': None
                    }
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
                failed_tickers.append(ticker)
        
        # Log summary
        successful_tickers = len(raw_data)
        logger.info(f"Successfully fetched data for {successful_tickers}/{len(tickers)} tickers")
        if failed_tickers:
            logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
        
        # Save raw data
        raw_data_file = CSV_PATH / "raw_financial_data.pkl"
        save_pickle(raw_data, str(raw_data_file))
        logger.info(f"Raw financial data saved to {raw_data_file}")
        
        return raw_data
        
    except Exception as e:
        logger.error(f"Failed to fetch financial data: {e}")
        traceback.print_exc()
        sys.exit(1)


def process_financial_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process raw financial data using FinancialDataProcessor."""
    logger.info("Processing financial data...")
    
    try:
        # Initialize FinancialDataProcessor with ratios_config.yaml
        processor = FinancialDataProcessor('ratios_config.yaml')
        
        # Restructure raw data to match FinancialDataProcessor expectations
        # FinancialDataProcessor expects root-level financial statements, not nested under 'quarterly'
        restructured_data = {}
        for ticker, ticker_data in raw_data.items():
            if ticker_data and isinstance(ticker_data, dict):
                quarterly_data = ticker_data.get('quarterly', {})
                restructured_data[ticker] = {
                    'balance_sheet': quarterly_data.get('balance_sheet'),
                    'income_statement': quarterly_data.get('income_statement'),
                    'cash_flow': quarterly_data.get('cash_flow'),
                    'current_price': quarterly_data.get('current_price'),
                    'shares_outstanding': quarterly_data.get('shares_outstanding'),
                    'market_cap': quarterly_data.get('market_cap'),
                    'dividendRate': quarterly_data.get('dividendRate'),
                    'dividends': quarterly_data.get('dividends'),
                    'latest_report_date': quarterly_data.get('latest_report_date'),
                    'info': quarterly_data.get('info', {})
                }
        
        # Process the restructured data into standardized format
        processed_df = processor.process_raw_financial_data(restructured_data)
        logger.info(f"Successfully processed data for {len(processed_df)} tickers")
        
        # Save processed data
        processed_data_file = CSV_PATH / "processed_financial_data.pkl"
        save_pickle(processed_df, str(processed_data_file))
        logger.info(f"Processed data saved to {processed_data_file}")
        
        # Also save as CSV for inspection
        processed_csv_file = CSV_PATH / "processed_financial_data.csv"
        save_csv(processed_df, str(processed_csv_file))
        logger.info(f"Processed data CSV saved to {processed_csv_file}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Failed to process financial data: {e}")
        traceback.print_exc()
        return raw_data  # Fall back to raw data


def calculate_ratios(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate ratios using RatioCalculator with enhanced temporal analysis."""
    logger.info("Calculating financial ratios...")
    
    try:
        # Initialize RatioCalculator with ratios_config.yaml
        calculator = RatioCalculator('ratios_config.yaml')
        
        # PRIORITY: Use enhanced processed data with quarterly TTM time series if available
        processed_file = CSV_PATH / "processed_financial_data.pkl"
        
        if processed_file.exists():
            logger.info("Using enhanced processed data with quarterly TTM time series")
            
            # Load the enhanced processed data
            from io_utils import load_pickle
            enhanced_processed_data = load_pickle(str(processed_file))
            
            total_tickers = len(enhanced_processed_data)
            logger.info(f"Processing {total_tickers} tickers with enhanced temporal analysis")
            
            results = {}
            for i, (ticker, ticker_data) in enumerate(enhanced_processed_data.items()):
                if ticker_data is not None and not ticker_data.empty:
                    try:
                        # Use the new method that accepts pre-processed data with quarterly TTM time series
                        analysis = calculator.calculate_ratios_from_processed_data(
                            ticker=ticker,
                            processed_data=ticker_data
                        )
                        results[ticker] = analysis
                        
                        if (i + 1) % 50 == 0:  # Progress reporting
                            logger.info(f"Processed {i + 1}/{total_tickers} tickers")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {ticker}: {e}")
                        continue
            
            logger.info(f"Processed {len(results)} tickers using enhanced temporal analysis")
            
        elif isinstance(processed_data, pd.DataFrame):
            logger.info("Detected processed DataFrame - need to load original raw data for RatioCalculator")
            
            # RatioCalculator needs the original raw financial data, not processed DataFrame
            # Load the raw data that was saved during data fetching
            raw_data_file = CSV_PATH / "raw_financial_data_quarterly.pkl"
            if raw_data_file.exists():
                from io_utils import load_pickle
                raw_financial_data = load_pickle(str(raw_data_file))
                logger.info(f"Loaded raw financial data for {len(raw_financial_data)} tickers")
                
                # Convert structure to what RatioCalculator expects
                # Process each ticker individually using calculate_stock_ratios
                # This avoids the batch processing issue with data structure expectations
                results = {}
                total_tickers = len(raw_financial_data)
                
                for i, (ticker, ticker_data) in enumerate(raw_financial_data.items()):
                    if ticker_data and isinstance(ticker_data, dict):
                        try:
                            # Extract quarterly data and format for FinancialDataProcessor
                            # FinancialDataProcessor expects root-level financial statements
                            quarterly_data = ticker_data.get('quarterly', {})
                            formatted_ticker_data = {
                                'balance_sheet': quarterly_data.get('balance_sheet'),
                                'income_statement': quarterly_data.get('income_statement'),
                                'cash_flow': quarterly_data.get('cash_flow'),
                                'current_price': quarterly_data.get('current_price'),
                                'shares_outstanding': quarterly_data.get('shares_outstanding'),
                                'market_cap': quarterly_data.get('market_cap'),
                                'dividendRate': quarterly_data.get('dividendRate'),
                                'dividends': quarterly_data.get('dividends'),
                                'latest_report_date': quarterly_data.get('latest_report_date'),
                                'info': quarterly_data.get('info', {})
                            }
                            
                            # For calculate_stock_ratios, we need to pass the data in the format
                            # that process_raw_financial_data expects: {ticker: ticker_data}
                            single_ticker_data = {ticker: formatted_ticker_data}
                            
                            # Calculate ratios for this individual ticker
                            analysis = calculator.calculate_stock_ratios(
                                ticker=ticker,
                                raw_financial_data=single_ticker_data,
                                price_data=None
                            )
                            results[ticker] = analysis
                            
                            if (i + 1) % 50 == 0:  # Progress reporting
                                logger.info(f"Processed {i + 1}/{total_tickers} tickers")
                            
                        except Exception as e:
                            logger.error(f"Failed to process {ticker}: {e}")
                            continue
                
                logger.info(f"Processed {len(results)} tickers individually using calculate_stock_ratios")
            else:
                logger.error(f"Raw financial data file not found: {raw_data_file}")
                return {}
            
        else:
            logger.info("Using raw data format for ratio calculations")
            # Use batch calculation method
            results = calculator.calculate_batch_ratios(processed_data)
            logger.info(f"Calculated ratios for {len(results)} tickers")
        
        # Save calculated ratios
        ratios_file = CSV_PATH / "calculated_ratios.pkl"
        save_pickle(results, str(ratios_file))
        logger.info(f"Calculated ratios saved to {ratios_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to calculate ratios: {e}")
        traceback.print_exc()
        
        # Create fallback results
        results = {}
        if isinstance(processed_data, dict):
            for ticker, data in processed_data.items():
                if data is not None:
                    results[ticker] = {
                        'ticker': ticker,
                        'processed': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
        
        logger.warning(f"Created fallback results for {len(results)} tickers")
        return results
        
        logger.warning(f"Created fallback results for {len(results)} tickers")
        return results


def export_results(results: Dict[str, Any]) -> None:
    """Export results to CSV and other formats."""
    logger.info("Exporting results...")
    
    try:
        if results:
            # Check if results are from RatioCalculator (comprehensive format)
            # or fallback format (simple placeholders)
            first_result = next(iter(results.values()))
            
            if isinstance(first_result, dict) and 'processed' in first_result:
                # Fallback format - create minimal compatible CSV
                logger.warning("Exporting fallback results format")
                df = pd.DataFrame.from_dict(results, orient='index')
                
                # Add minimal required columns for compatibility
                if 'ticker' not in df.columns:
                    df['ticker'] = df.index
                    
            else:
                # Full RatioCalculator results - convert to DataFrame
                logger.info("Exporting comprehensive ratio calculation results")
                
                # Handle different potential formats from RatioCalculator
                if isinstance(results, pd.DataFrame):
                    df = results
                elif isinstance(results, dict):
                    # Convert dictionary of results to DataFrame
                    df = pd.DataFrame.from_dict(results, orient='index')
                else:
                    logger.error(f"Unexpected results format: {type(results)}")
                    return
            
            # Ensure ticker is the first column for compatibility
            if 'ticker' in df.columns and df.columns[0] != 'ticker':
                cols = df.columns.tolist()
                cols.insert(0, cols.pop(cols.index('ticker')))
                df = df[cols]
            
            # Export to CSV (compatible with existing workflow)
            results_file = CSV_PATH / "stock_evaluation_results.csv"
            save_csv(df, str(results_file))
            logger.info(f"Results exported to {results_file} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Also save as pickle for detailed data
            results_pickle = CSV_PATH / "stock_evaluation_results.pkl"
            save_pickle(results, str(results_pickle))
            logger.info(f"Detailed results saved to {results_pickle}")
            
            # Log column info for debugging
            logger.info(f"Exported columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
        else:
            logger.warning("No results to export")
            # Create empty file for workflow compatibility
            empty_df = pd.DataFrame({'ticker': [], 'message': []})
            results_file = CSV_PATH / "stock_evaluation_results.csv"
            save_csv(empty_df, str(results_file))
            logger.info(f"Empty results file created: {results_file}")
            
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        traceback.print_exc()


def main():
    """Main pipeline execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Redesigned Swedish Stock Analysis Pipeline')
    parser.add_argument('--environment', choices=['local', 'remote'], 
                       help='Override ENVIRONMENT variable')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip data fetching and use existing data')
    parser.add_argument('--config', default='ratios_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with only first 5 tickers')
    
    args = parser.parse_args()
    
    # Override environment variables if provided
    global ENVIRONMENT, FETCH_DATA, CSV_PATH
    if args.environment:
        ENVIRONMENT = args.environment
        CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')
    
    if args.skip_fetch:
        FETCH_DATA = False
    
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("REDESIGNED SWEDISH STOCK ANALYSIS PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Stage 1: Setup and validation
        setup_environment()
        
        # Stage 2: Load tickers
        tickers = load_tickers(test_mode=args.test_mode)
        
        # Stage 3: Fetch financial data
        raw_data = fetch_financial_data(tickers)
        
        # Stage 4: Process financial data
        processed_data = process_financial_data(raw_data)
        
        # Stage 5: Calculate ratios (passing processed DataFrame triggers raw data reload)
        results = calculate_ratios(processed_data)
        
        # Stage 6: Export results
        export_results(results)
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration}")
        logger.info(f"Tickers processed: {len(results) if results else 0}")
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Output directory: {CSV_PATH}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()