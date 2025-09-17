#!/usr/bin/env python3
"""Debug script to examine the FinancialDataProcessor output structure."""

import pickle
from pathlib import Path
import pandas as pd
import logging
from FinancialDataProcessor import FinancialDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_processor():
    """Debug the FinancialDataProcessor output to see column structure."""
    
    # Load raw data
    data_path = Path('data/local/raw_financial_data_quarterly.pkl')
    with open(data_path, 'rb') as f:
        raw_financial_data = pickle.load(f)
    
    # Pick a ticker with good data
    ticker = 'AAK'
    if ticker not in raw_financial_data:
        ticker = list(raw_financial_data.keys())[8]  # Pick the 9th ticker (AAK)
    
    ticker_data = raw_financial_data[ticker]
    print(f"\n=== Debugging FinancialDataProcessor output for: {ticker} ===")
    
    # Apply the same transformation as pipeline_runner
    formatted_ticker_data = {
        'quarterly': {
            'balance_sheet': ticker_data.get('balance_sheet'),
            'income_statement': ticker_data.get('income_statement'),
            'cash_flow': ticker_data.get('cash_flow'),
            'info': {
                'sharesOutstanding': ticker_data.get('shares_outstanding'),
                'marketCap': ticker_data.get('market_cap'),
                **(ticker_data.get('info', {}))
            }
        },
        'current_price': ticker_data.get('current_price'),
        'dividendRate': ticker_data.get('dividendRate'),
        'dividends': ticker_data.get('dividends'),
        'latest_report_date': ticker_data.get('latest_report_date')
    }
    
    # Initialize FinancialDataProcessor and process the data
    processor = FinancialDataProcessor()
    single_ticker_data = {ticker: formatted_ticker_data}
    
    print(f"Processing data through FinancialDataProcessor...")
    processed_data = processor.process_raw_financial_data(single_ticker_data)
    
    print(f"\nProcessed data type: {type(processed_data)}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed data columns: {list(processed_data.columns)}")
    print(f"Processed data index: {processed_data.index}")
    
    # Check for date-related columns
    date_columns = [col for col in processed_data.columns if 'date' in col.lower()]
    print(f"Date-related columns: {date_columns}")
    
    # Check first few rows
    print(f"\nFirst few rows:")
    print(processed_data.head())
    
    # Check if there's a 'ticker' column (which would indicate row-per-ticker format)
    if 'ticker' in processed_data.columns:
        print(f"\nTickers in processed data: {processed_data['ticker'].unique()}")
    
    print(f"\nData types:")
    print(processed_data.dtypes)

if __name__ == "__main__":
    debug_data_processor()