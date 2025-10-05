import pandas as pd
from typing import Any, Dict, List, Optional
from data_io import load_csv, save_csv


def extract_ttm_values(csv_path: str, agr_dimensions: List[str], file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract ttm values for each ticker and metric in agr_dimensions from summarized quarterly data and save to CSV.

    Args:
        csv_path (str): Path to the summarized quarterly data CSV.
        agr_dimensions (List[str]): List of metrics to extract.
        file_path (str): Path to the output CSV file.

    Returns:
        Dict[str, Dict[str, Any]]: ttm values per ticker and metric.
    """
    df = load_csv(csv_path, index_col='Ticker')
    ttm_values = {}
    for ticker in df.index:
        ttm_values[ticker] = {}
        for dim in agr_dimensions:
            # Replace spaces with underscores to match column names if needed
            col_name = dim.replace(" ", "_")
            # Try both original and underscored column names
            if dim in df.columns:
                ttm_values[ticker][f"{col_name}_ttm"] = df.loc[ticker, dim]
            elif col_name in df.columns:
                ttm_values[ticker][f"{col_name}_ttm"] = df.loc[ticker, col_name]
            else:
                ttm_values[ticker][f"{col_name}_ttm"] = None
    # Save to CSV
    ttm_df = pd.DataFrame.from_dict(ttm_values, orient='index')
    save_csv(ttm_df, file_path, index=True)
    print(f"ttm values extracted and saved to {file_path}")
    return ttm_values

def combine_quarterly_summaries_for_ttm_trends(
    quarterly_summarized_0: Dict[str, Any], 
    quarterly_summarized_1: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine two quarterly summarized dictionaries to create TTM trend data.
    
    This function merges the most recent TTM data (quarters_back=0) with the 
    previous TTM data (quarters_back=1) to enable trend analysis across TTM periods.
    
    Args:
        quarterly_summarized_0 (Dict[str, Any]): Most recent 4 quarters summarized (quarters_back=0)
        quarterly_summarized_1 (Dict[str, Any]): Previous 4 quarters summarized (quarters_back=1)
    
    Returns:
        Dict[str, Any]: Combined data with multiple TTM periods per ticker
    """
    combined_data = {}
    
    # Get all unique tickers from both dictionaries
    all_tickers = set(quarterly_summarized_0.keys()) | set(quarterly_summarized_1.keys())
    
    for ticker in all_tickers:
        data_0 = quarterly_summarized_0.get(ticker)
        data_1 = quarterly_summarized_1.get(ticker)
        
        # Skip if both are None
        if data_0 is None and data_1 is None:
            continue
            
        # Initialize combined data structure
        combined_ticker_data = {
            'balance_sheet': pd.DataFrame(),
            'income_statement': pd.DataFrame(), 
            'cash_flow': pd.DataFrame(),
            'current_price': None,
            'shares_outstanding': None,
            'market_cap': None,
            'info': None
        }
        
        # Process each financial statement segment
        for segment in ['balance_sheet', 'income_statement', 'cash_flow']:
            dfs_to_combine = []
            
            # Add data from most recent TTM (quarters_back=0)
            if data_0 is not None and segment in data_0 and data_0[segment] is not None and not data_0[segment].empty:
                df_0 = data_0[segment].copy()
                # Rename index to indicate this is the most recent TTM
                if len(df_0.index) > 0:
                    #new_index = pd.Timestamp(df_0.index[0]).replace(day=15)  # Use mid-month for TTM
                    #df_0.index = [new_index]
                    dfs_to_combine.append(df_0)
            
            # Add data from previous TTM (quarters_back=1) 
            if data_1 is not None and segment in data_1 and data_1[segment] is not None and not data_1[segment].empty:
                df_1 = data_1[segment].copy()
                # Rename index to indicate this is the previous TTM (approximately 3 months earlier)
                if len(df_1.index) > 0:
                    #new_index = pd.Timestamp(df_1.index[0]).replace(day=15)# - pd.DateOffset(months=3)
                    #df_1.index = [new_index]
                    dfs_to_combine.append(df_1)
            
            # Combine the DataFrames
            if dfs_to_combine:
                combined_segment = pd.concat(dfs_to_combine, sort=False)
                combined_segment = combined_segment.sort_index()  # Sort by date
                combined_ticker_data[segment] = combined_segment
        
        # Set scalar values (use most recent where available)
        if data_0 is not None:
            combined_ticker_data.update({
                'current_price': data_0.get('current_price'),
                'shares_outstanding': data_0.get('shares_outstanding'), 
                'market_cap': data_0.get('market_cap'),
                'info': data_0.get('info')
            })
        elif data_1 is not None:
            combined_ticker_data.update({
                'current_price': data_1.get('current_price'),
                'shares_outstanding': data_1.get('shares_outstanding'),
                'market_cap': data_1.get('market_cap'),
                'info': data_1.get('info')
            })
        
        combined_data[ticker] = combined_ticker_data
        
        # Debug info
        bs_rows = len(combined_ticker_data['balance_sheet']) if not combined_ticker_data['balance_sheet'].empty else 0
        is_rows = len(combined_ticker_data['income_statement']) if not combined_ticker_data['income_statement'].empty else 0
        cf_rows = len(combined_ticker_data['cash_flow']) if not combined_ticker_data['cash_flow'].empty else 0
        print(f"Combined {ticker}: BS={bs_rows} rows, IS={is_rows} rows, CF={cf_rows} rows")
    
    print(f"Combined TTM data for {len(combined_data)} tickers")
    return combined_data

def filter_metrics_for_agr_dimensions(csv_path: str, agr_dimensions: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    # ...existing code...
    pass
