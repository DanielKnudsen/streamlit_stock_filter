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
from typing import Dict, Any, Optional
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
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
    from ratio_functions import roe, vinstmarginal, soliditet, skuldsattningsgrad
    from price_data_collector import PriceDataCollector
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
        # Try to load existing data - prefer quarterly data if available
        try:
            import pickle
            # Check for quarterly data first (more comprehensive)
            quarterly_data_file = CSV_PATH / "raw_financial_data_quarterly.pkl"
            regular_data_file = CSV_PATH / "raw_financial_data.pkl"
            
            if quarterly_data_file.exists():
                logger.info("Loading existing quarterly financial data")
                with open(quarterly_data_file, 'rb') as f:
                    return pickle.load(f)
            elif regular_data_file.exists():
                logger.info("Loading existing regular financial data")
                with open(regular_data_file, 'rb') as f:
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
                # Fetch only quarterly data (annual data not needed for TTM analysis)
                quarterly_data = fetch_yfinance_data(ticker)
                
                # Store quarterly data if successful
                if quarterly_data:
                    raw_data[ticker] = {
                        'quarterly': quarterly_data
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


def fetch_price_data(tickers: list, quarterly_dates: list = None) -> Dict[str, Any]:
    """Fetch comprehensive price data for all tickers using PriceDataCollector."""
    if not FETCH_DATA:
        logger.info("Skipping price data fetch (FETCH_DATA=False)")
        # Try to load existing price data
        try:
            import pickle
            price_data_file = CSV_PATH / "price_data_comprehensive.pkl"
            if price_data_file.exists():
                with open(price_data_file, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.warning("No existing price data found, but FETCH_DATA=False")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load existing price data: {e}")
            return {}
    
    logger.info(f"Fetching comprehensive price data for {len(tickers)} tickers...")
    
    try:
        # Initialize price data collector
        collector = PriceDataCollector(environment=ENVIRONMENT)
        
        # Collect comprehensive price data
        price_data = collector.collect_comprehensive_price_data(tickers, quarterly_dates)
        
        # Save the comprehensive price data
        collector.save_price_data(price_data, "price_data_comprehensive")
        
        logger.info(f"Price data collection completed for {len(price_data)} tickers")
        
        return price_data
        
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        traceback.print_exc()
        return {}


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


def calculate_ratios(processed_data: Dict[str, Any], price_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Calculate ratios using RatioCalculator with enhanced temporal analysis and comprehensive price data."""
    logger.info("Calculating financial ratios...")
    
    try:
        # Initialize RatioCalculator with ratios_config.yaml
        calculator = RatioCalculator('ratios_config.yaml')
        
        # Log price data availability
        if price_data:
            price_tickers_count = len([t for t, data in price_data.items() if data.get('data_quality', {}).get('has_data', False)])
            logger.info(f"Using comprehensive price data for {price_tickers_count}/{len(price_data)} tickers")
        else:
            logger.warning("No price data provided - using basic pricing from financial data")
        
        # PRIORITY: Use enhanced processed data with quarterly TTM time series if available
        processed_file = CSV_PATH / "processed_financial_data.pkl"
        
        if processed_file.exists():
            logger.info("Using enhanced processed data with quarterly TTM time series")
            
            # Load the enhanced processed data
            from io_utils import load_pickle
            enhanced_processed_data = load_pickle(str(processed_file))
            
            # The processed data is a DataFrame, convert it to the format the RatioCalculator expects
            if isinstance(enhanced_processed_data, pd.DataFrame):
                # Convert DataFrame to dictionary where each row is a ticker's data
                enhanced_processed_dict = {}
                for ticker in enhanced_processed_data.index:
                    # Convert the row to a DataFrame for the RatioCalculator
                    ticker_row_df = enhanced_processed_data.loc[[ticker]]
                    enhanced_processed_dict[ticker] = ticker_row_df
                enhanced_processed_data = enhanced_processed_dict
            
            total_tickers = len(enhanced_processed_data)
            logger.info(f"Processing {total_tickers} tickers with enhanced temporal analysis")
            
            results = {}
            for i, (ticker, ticker_data) in enumerate(enhanced_processed_data.items()):
                if ticker_data is not None and not ticker_data.empty:
                    try:
                        # Get price data for this ticker if available
                        ticker_price_data = None
                        if price_data and ticker in price_data:
                            ticker_price_data = price_data[ticker]
                        
                        # Use the new method that accepts pre-processed data with quarterly TTM time series
                        analysis = calculator.calculate_ratios_from_processed_data(
                            ticker=ticker,
                            processed_data=ticker_data,
                            price_data=ticker_price_data
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
                            ticker_price_data = None
                            if price_data and ticker in price_data:
                                ticker_price_data = convert_price_data_for_calculator(ticker, price_data[ticker])
                            
                            analysis = calculator.calculate_stock_ratios(
                                ticker=ticker,
                                raw_financial_data=single_ticker_data,
                                price_data=ticker_price_data
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


        return {}


def convert_price_data_for_calculator(ticker: str, comprehensive_price_data: Dict) -> Optional[pd.DataFrame]:
    """
    Convert comprehensive price data to DataFrame format expected by RatioCalculator.
    
    Args:
        ticker: Stock ticker symbol
        comprehensive_price_data: Price data from PriceDataCollector
        
    Returns:
        DataFrame with price data or None if no data available
    """
    if not comprehensive_price_data or not comprehensive_price_data.get('data_quality', {}).get('has_data', False):
        return None
    
    try:
        # Extract current market data and daily prices
        current_market = comprehensive_price_data.get('current_market', {})
        daily_prices = comprehensive_price_data.get('daily_prices', pd.DataFrame())
        
        # Create a simple DataFrame with the most recent price data
        # RatioCalculator primarily needs current price information
        price_df = pd.DataFrame({
            'current_price': [current_market.get('current_price', np.nan)],
            'market_cap': [current_market.get('market_cap', np.nan)],
            'shares_outstanding': [current_market.get('shares_outstanding', np.nan)],
            'enterprise_value': [current_market.get('enterprise_value', np.nan)],
            'book_value': [current_market.get('book_value', np.nan)],
            'beta': [current_market.get('beta', np.nan)],
            'trailing_pe': [current_market.get('trailing_pe', np.nan)],
            'dividend_yield': [current_market.get('dividend_yield', np.nan)]
        }, index=[ticker])
        
        # Add recent price statistics if daily prices are available
        if not daily_prices.empty:
            recent_days = min(30, len(daily_prices))  # Last 30 days or available data
            recent_prices = daily_prices.tail(recent_days)
            
            price_df['avg_volume_30d'] = recent_prices['Volume'].mean()
            price_df['price_volatility_30d'] = recent_prices['Close'].std()
            price_df['price_change_30d'] = ((recent_prices['Close'].iloc[-1] - recent_prices['Close'].iloc[0]) / recent_prices['Close'].iloc[0]) if len(recent_prices) > 1 else 0
        
        return price_df
        
    except Exception as e:
        logger.warning(f"Failed to convert price data for {ticker}: {e}")
        return None


def convert_to_rankings(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw ratio calculations to 0-100 percentile rankings.
    
    According to PIPELINE_REDESIGN.md, the final output should be 0-100 rankings,
    not raw composite scores.
    
    Args:
        results: Dictionary of StockAnalysis objects with raw ratio calculations
        
    Returns:
        Dictionary with converted 0-100 rankings for individuals, categories, and temporal perspectives
    """
    logger.info("Converting ratio calculations to 0-100 percentile rankings")
    
    if not results:
        logger.warning("No results to convert to rankings")
        return {}
    
    # Extract data for ranking conversion
    ticker_data = []
    for ticker, analysis in results.items():
        row_data = {'ticker': ticker}
        
        # Add overall score
        row_data['overall_score'] = analysis.overall_score
        
        # Add category scores
        for category, score in analysis.category_scores.items():
            row_data[f'category_{category}'] = score
            
        # Add individual ratio composite scores for temporal perspectives
        for ratio_name, ratio_result in analysis.ratio_results.items():
            if hasattr(ratio_result, 'composite_score') and ratio_result.composite_score is not None:
                row_data[f'ratio_{ratio_name}'] = ratio_result.composite_score
                
                # Add temporal perspective values if available
                if hasattr(ratio_result, 'current_value') and ratio_result.current_value is not None:
                    row_data[f'current_ttm_{ratio_name}'] = ratio_result.current_value
                if hasattr(ratio_result, 'trend_value') and ratio_result.trend_value is not None:
                    row_data[f'trend_ttm_{ratio_name}'] = ratio_result.trend_value
                if hasattr(ratio_result, 'stability_value') and ratio_result.stability_value is not None:
                    row_data[f'stability_ttm_{ratio_name}'] = ratio_result.stability_value
        
        ticker_data.append(row_data)
    
    # Convert to DataFrame for easier ranking calculation
    df = pd.DataFrame(ticker_data)
    df.set_index('ticker', inplace=True)
    
    # Convert all numeric columns to 0-100 percentile rankings
    ranking_df = df.copy()
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64', 'float32', 'int32']:
            # Convert to percentile rankings (0-100)
            # Use rank method with pct=True to get 0-1, then multiply by 100
            valid_data = df[column].dropna()
            if len(valid_data) > 1:
                ranking_df[column] = df[column].rank(pct=True) * 100
            else:
                ranking_df[column] = 50.0  # Default to middle ranking for single values
    
    # Convert back to the original results format but with rankings
    converted_results = {}
    for ticker in ranking_df.index:
        # Create a new StockAnalysis object with rankings
        row = ranking_df.loc[ticker]
        
        # Create converted ratio results
        ratio_results = {}
        for ratio_name, original_ratio_result in results[ticker].ratio_results.items():
            # Create a new ratio result with ranking values
            ratio_ranking = type(original_ratio_result)(
                ratio_name=ratio_name,
                current_value=row.get(f'current_ttm_{ratio_name}', original_ratio_result.current_value),
                trend_value=row.get(f'trend_ttm_{ratio_name}', original_ratio_result.trend_value),
                stability_value=row.get(f'stability_ttm_{ratio_name}', original_ratio_result.stability_value),
                composite_score=row.get(f'ratio_{ratio_name}', original_ratio_result.composite_score),
                data_quality=original_ratio_result.data_quality,
                calculation_notes=original_ratio_result.calculation_notes
            )
            ratio_results[ratio_name] = ratio_ranking
        
        # Create converted category scores (now as rankings)
        category_rankings = {}
        for category in results[ticker].category_scores.keys():
            category_rankings[category] = row.get(f'category_{category}', 50.0)
        
        # Create new StockAnalysis with rankings
        converted_analysis = type(results[ticker])(
            ticker=ticker,
            calculation_date=results[ticker].calculation_date,
            ratio_results=ratio_results,
            category_scores=category_rankings,
            overall_score=row.get('overall_score', results[ticker].overall_score),
            data_quality_score=results[ticker].data_quality_score,
            processing_notes=results[ticker].processing_notes
        )
        
        converted_results[ticker] = converted_analysis
    
    logger.info(f"Converted {len(converted_results)} tickers to 0-100 percentile rankings")
    
    # Log some ranking statistics
    overall_rankings = [analysis.overall_score for analysis in converted_results.values()]
    logger.info(f"Overall ranking range: {min(overall_rankings):.1f} - {max(overall_rankings):.1f}")
    logger.info(f"Overall ranking mean: {np.mean(overall_rankings):.1f}")
    
    return converted_results


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


def export_quarterly_ratios(processed_data: Any) -> None:
    """
    Extract and export quarterly ratio time series for Streamlit plotting.
    
    This function calculates ratios for each quarter and saves them in a format
    that's easy to use for time series plotting in the Streamlit app.
    
    Args:
        processed_data: Either DataFrame or dict with quarterly TTM time series
    """
    logger.info("Extracting quarterly ratios for Streamlit plotting...")
    
    quarterly_ratios = {}
    
    # Handle DataFrame format (enhanced processed data)
    if isinstance(processed_data, pd.DataFrame):
        for ticker in processed_data.index:
            ticker_data = processed_data.loc[ticker]
            quarterly_series = ticker_data.get('quarterly_ttm_series')
            
            if quarterly_series and isinstance(quarterly_series, dict):
                quarterly_ratios[ticker] = calculate_quarterly_ratios(quarterly_series)
    
    # Handle dict format (raw data)
    elif isinstance(processed_data, dict):
        for ticker, ticker_data in processed_data.items():
            if ticker_data and isinstance(ticker_data, pd.DataFrame):
                # Extract quarterly series from DataFrame
                quarterly_series = ticker_data.get('quarterly_ttm_series', {})
                if quarterly_series:
                    quarterly_ratios[ticker] = calculate_quarterly_ratios(quarterly_series)
    
    if quarterly_ratios:
        # Save as pickle for detailed access
        quarterly_ratios_file = CSV_PATH / "quarterly_ratios_timeseries.pkl"
        save_pickle(quarterly_ratios, str(quarterly_ratios_file))
        
        # Save as CSV for human inspection
        quarterly_ratios_csv = CSV_PATH / "quarterly_ratios_timeseries.csv"
        save_quarterly_ratios_to_csv(quarterly_ratios, str(quarterly_ratios_csv))
        
        logger.info(f"Quarterly ratios exported for {len(quarterly_ratios)} tickers")
        logger.info(f"Saved to: {quarterly_ratios_file} and {quarterly_ratios_csv}")
    else:
        logger.warning("No quarterly ratios to export")


def calculate_quarterly_ratios(quarterly_series: Dict) -> Dict:
    """Calculate ratios for each quarter in the time series"""
    
    ratios_by_quarter = {}
    
    # Get the length of time series (should be consistent across metrics)
    revenue_series = quarterly_series.get('Total Revenue TTM', [])
    num_quarters = len(revenue_series) if revenue_series else 0
    
    # Try to get quarter labels (like "Q1-25", "Q2-24") and dates if available
    quarter_labels = quarterly_series.get('quarter_labels', [])
    quarter_dates = quarterly_series.get('quarter_dates', [])
    
    for i in range(num_quarters):
        # Use quarter label with year if available, otherwise fallback to generic
        if i < len(quarter_labels) and quarter_labels[i]:
            quarter_key = quarter_labels[i]  # e.g., "Q1-25", "Q2-24"
        else:
            quarter_key = f"Q{i+1}"  # Fallback to Q1, Q2, Q3, etc.
        
        quarter_ratios = {}
        
        try:
            # Extract values for this quarter
            net_income = quarterly_series.get('Net Income TTM', [None] * num_quarters)[i]
            stockholders_equity = quarterly_series.get('Stockholders Equity', [None] * num_quarters)[i] if i == 0 else None
            total_revenue = quarterly_series.get('Total Revenue TTM', [None] * num_quarters)[i]
            total_assets = quarterly_series.get('Total Assets', [None] * num_quarters)[i] if i == 0 else None
            total_debt = quarterly_series.get('Total Debt', [None] * num_quarters)[i] if i == 0 else None
            
            # Calculate ROE (Return on Equity)
            if net_income is not None and stockholders_equity is not None:
                quarter_ratios['ROE'] = roe(net_income, stockholders_equity)
            
            # Calculate Vinstmarginal (Net Margin)
            if net_income is not None and total_revenue is not None and total_revenue != 0:
                quarter_ratios['Vinstmarginal'] = vinstmarginal(net_income, total_revenue)
            
            # Calculate Soliditet (Equity Ratio) - only for most recent quarter
            if i == 0 and stockholders_equity is not None and total_assets is not None:
                quarter_ratios['Soliditet'] = soliditet(stockholders_equity, total_assets)
            
            # Calculate Skuldsättningsgrad (Debt-to-Equity) - only for most recent quarter
            if i == 0 and total_debt is not None and stockholders_equity is not None:
                quarter_ratios['Skuldsattningsgrad'] = skuldsattningsgrad(total_debt, stockholders_equity)
            
            # Add raw financial data for reference
            quarter_ratios['Total_Revenue_TTM'] = total_revenue
            quarter_ratios['Net_Income_TTM'] = net_income
            quarter_ratios['Stockholders_Equity'] = stockholders_equity
            quarter_ratios['Total_Assets'] = total_assets
            quarter_ratios['Total_Debt'] = total_debt
            
            # Add quarter date information
            if i < len(quarter_dates) and quarter_dates[i]:
                quarter_ratios['Quarter_Date'] = quarter_dates[i]
            quarter_ratios['Quarter_Index'] = i  # Position in time series (0=most recent)
            quarter_ratios['Quarter_Label'] = quarter_key  # Store the formatted label
            
        except Exception as e:
            logger.warning(f"Error calculating ratios for {quarter_key}: {e}")
            quarter_ratios['error'] = str(e)
        
        ratios_by_quarter[quarter_key] = quarter_ratios
    
    return ratios_by_quarter


def save_quarterly_ratios_to_csv(quarterly_ratios: Dict, csv_path: str) -> None:
    """Save quarterly ratios to CSV format for easy inspection"""
    rows = []
    
    for ticker, quarters_data in quarterly_ratios.items():
        for quarter, ratios in quarters_data.items():
            row = {'Ticker': ticker, 'Quarter': quarter}
            
            # Add quarter date, index, and label for better identification
            if 'Quarter_Date' in ratios:
                row['Quarter_Date'] = ratios['Quarter_Date']
            if 'Quarter_Index' in ratios:
                row['Quarter_Index'] = ratios['Quarter_Index']
            if 'Quarter_Label' in ratios:
                row['Quarter_Label'] = ratios['Quarter_Label']
                
            # Add the ratio calculations and financial data
            for key, value in ratios.items():
                if key not in ['Quarter_Date', 'Quarter_Index', 'Quarter_Label']:  # Avoid duplicates
                    row[key] = value
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort by ticker and quarter index for better readability
        if 'Quarter_Index' in df.columns:
            df = df.sort_values(['Ticker', 'Quarter_Index'])
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Quarterly ratios CSV saved with {len(df)} rows")
        
        # Log some sample quarter information
        if 'Quarter_Label' in df.columns:
            sample_quarters = df[['Ticker', 'Quarter', 'Quarter_Label']].head(5)
            logger.info(f"Sample quarter labels:\n{sample_quarters.to_string(index=False)}")
    else:
        logger.warning("No quarterly ratios data to save")


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
        
        # Stage 4: Fetch comprehensive price data
        # For now, generate standard quarterly dates for price alignment
        # In future iterations, we can extract actual quarterly dates from processed data
        quarterly_dates = []
        current_year = 2021  # Start from 4 years ago
        end_year = 2025
        
        while current_year <= end_year:
            # Use last day of each quarter (properly calculated)
            for quarter_month in [3, 6, 9, 12]:  # March, June, September, December
                if quarter_month in [3, 12]:
                    day = 31
                elif quarter_month == 6:
                    day = 30
                else:  # September
                    day = 30
                
                quarter_end = pd.Timestamp(year=current_year, month=quarter_month, day=day)
                if quarter_end <= pd.Timestamp.now():
                    quarterly_dates.append(quarter_end)
            current_year += 1
        
        logger.info(f"Using {len(quarterly_dates)} standard quarterly dates for price alignment")
        
        price_data = fetch_price_data(tickers, quarterly_dates)
        
        # Stage 5: Process financial data
        processed_data = process_financial_data(raw_data)
        
        # Stage 6: Calculate ratios with comprehensive price data
        results = calculate_ratios(processed_data, price_data)
        
        # Stage 7: Convert to 0-100 rankings
        rankings = convert_to_rankings(results)
        
        # Stage 8: Export results
        export_results(rankings)
        
        # Stage 9: Export quarterly ratios for Streamlit plotting
        export_quarterly_ratios(processed_data)
        
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