#!/usr/bin/env python3
"""Debug script to examine the data flow in ratio calculations."""

import pickle
from pathlib import Path
import pandas as pd
import logging
from RatioCalculator import RatioCalculator

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_single_ticker():
    """Debug a single ticker's data transformation and ratio calculation."""
    
    # Load raw data
    data_path = Path('data/local/raw_financial_data_quarterly.pkl')
    with open(data_path, 'rb') as f:
        raw_financial_data = pickle.load(f)
    
    # Pick a ticker with good data (AAK seemed to have data quality 0.5625)
    ticker = 'AAK'
    if ticker not in raw_financial_data:
        ticker = list(raw_financial_data.keys())[8]  # Pick the 9th ticker (AAK)
    
    ticker_data = raw_financial_data[ticker]
    print(f"\n=== Debugging ticker: {ticker} ===")
    print(f"Original data keys: {list(ticker_data.keys())}")
    
    # Check the quarterly data types and shapes
    print(f"\nChecking quarterly data:")
    for key in ['balance_sheet', 'income_statement', 'cash_flow']:
        if key in ticker_data:
            data = ticker_data[key]
            print(f"{key}: type={type(data)}")
            if isinstance(data, pd.DataFrame):
                print(f"  shape: {data.shape}")
                print(f"  index (periods): {list(data.index)[:3]}...")  # First 3 periods
                print(f"  columns: {list(data.columns)[:5]}...")  # First 5 columns
            else:
                print(f"  value: {data}")
    
    # Check shares outstanding
    print(f"\nShares outstanding: {ticker_data.get('shares_outstanding')}")
    print(f"Market cap: {ticker_data.get('market_cap')}")
    
    # Now try the transformation that pipeline_runner does
    print(f"\n=== Applying pipeline_runner transformation ===")
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
    
    print(f"Formatted data keys: {list(formatted_ticker_data.keys())}")
    print(f"Quarterly section keys: {list(formatted_ticker_data['quarterly'].keys())}")
    print(f"Info section keys: {list(formatted_ticker_data['quarterly']['info'].keys())}")
    
    # Verify the DataFrames are still DataFrames
    quarterly = formatted_ticker_data['quarterly']
    for key in ['balance_sheet', 'income_statement', 'cash_flow']:
        data = quarterly[key]
        print(f"{key}: type={type(data)}, is_dataframe={isinstance(data, pd.DataFrame)}")
        if isinstance(data, pd.DataFrame) and not data.empty:
            print(f"  shape: {data.shape}, index length: {len(data.index)}")
    
    # Try the ratio calculator
    print(f"\n=== Testing RatioCalculator ===")
    try:
        calculator = RatioCalculator()
        single_ticker_data = {ticker: formatted_ticker_data}
        
        # This is what pipeline_runner calls
        analysis = calculator.calculate_stock_ratios(
            ticker=ticker,
            raw_financial_data=single_ticker_data,
            price_data=None
        )
        
        print(f"Analysis successful!")
        print(f"Overall score: {analysis.overall_score}")
        print(f"Data quality: {analysis.data_quality_score}")
        
        # Check a few ratio results
        for i, (ratio_name, ratio_result) in enumerate(analysis.ratio_results.items()):
            if i < 3:  # First 3 ratios
                print(f"{ratio_name}: current={ratio_result.current_value}, composite={ratio_result.composite_score}")
                if ratio_result.calculation_notes:
                    print(f"  notes: {ratio_result.calculation_notes[:2]}")  # First 2 notes
        
    except Exception as e:
        import traceback
        print(f"RatioCalculator failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_single_ticker()