import pandas as pd
from typing import Any, Dict
from data_io import load_csv

def save_last_SMA_to_csv(read_from: str, save_to: str) -> None:
    """
    Save the latest SMA data to a separate CSV file and calculate CAGR for 'Close' per ticker.

    Args:
        read_from (str): Path to the input CSV file with price data.
        save_to (str): Path to the output CSV file.
    """
    try:
        df = load_csv(read_from, index_col='Date', parse_dates=True)
        if 'SMA_short' in df.columns and 'SMA_medium' in df.columns and 'SMA_long' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            # Get the latest row for each ticker
            last_rows = df.groupby('Ticker').tail(1)[['Ticker', 'pct_Close_vs_SMA_short', 'pct_SMA_short_vs_SMA_medium', 'pct_SMA_medium_vs_SMA_long','pct_ch_20_d']]
            # Calculate CAGR for each ticker
            cagr_list = []
            for ticker, group in df.groupby('Ticker'):
                group = group.sort_index()
                if len(group) > 1:
                    start_price = group['Close'].iloc[0]
                    end_price = group['Close'].iloc[-1]
                    num_years = (group.index[-1] - group.index[0]).days / 365.25
                    if start_price > 0 and num_years > 0:
                        cagr = (((end_price / start_price) ** (1 / num_years)) - 1)# * 100  # CAGR in percent
                        cagr_list.append({'Ticker': ticker, 'CAGR': cagr})
                    else:
                        cagr_list.append({'Ticker': ticker, 'CAGR': 0})  # If no valid CAGR can be calculated, set to 0
                else:
                    cagr_list.append({'Ticker': ticker, 'CAGR': 0})  # If no valid CAGR can be calculated, set to 0
            df_cagr = pd.DataFrame(cagr_list).set_index('Ticker')
            last_rows = last_rows.set_index('Ticker')
            last_rows['cagr_close'] = df_cagr['CAGR']
            last_rows.reset_index(inplace=True)
            last_rows.to_csv(save_to, index=False)
            print(f"Senaste SMA-data per ticker sparad i '{save_to}'")
        else:
            print("Fel: Saknar nödvändiga kolumner i prisdata.")
    except Exception as e:
        print(f"Fel vid sparande av senaste SMA-data: {e}")

def add_historical_prices_to_filtered_data(filtered_raw_data: Dict[str, Any], price_data_path: str, days_window: int = 5) -> Dict[str, Any]:
    """
    Add historical stock prices around report dates to filtered raw data.
    
    For each ticker and each report date found in the financial statements,
    calculates the average stock price from -days_window to +days_window days
    around the report date using the price_data.csv file.
    
    Args:
        filtered_raw_data (Dict[str, Any]): Filtered financial data per ticker.
        price_data_path (str): Path to the price_data.csv file.
        days_window (int, optional): Number of days before/after report date to average. Defaults to 5.
    
    Returns:
        Dict[str, Any]: Enhanced filtered data with historical prices.
    """
    # Load price data
    try:
        price_df = load_csv(price_data_path, parse_dates=['Date'])
        price_df = price_df.set_index('Date')
        print(f"Loaded price data with {len(price_df)} rows from {len(price_df['Ticker'].unique())} tickers")
    except Exception as e:
        print(f"Warning: Could not load price data from {price_data_path}: {e}")
        return filtered_raw_data
    
    enhanced_data = {}
    
    for ticker, data in filtered_raw_data.items():
        if data is None:
            enhanced_data[ticker] = None
            continue
            
        enhanced_data[ticker] = data.copy()
        
        # Get all unique report dates from financial statements
        report_dates = set()
        
        for segment in ['balance_sheet', 'income_statement', 'cash_flow']:
            if segment in data and hasattr(data[segment], 'index'):
                # Convert index to datetime if not already
                try:
                    dates = pd.to_datetime(data[segment].index, errors='coerce')
                    # Add 15 days to each date to match TTM convention
                    # dates = dates + pd.Timedelta(days=15)
                    valid_dates = dates.dropna()
                    report_dates.update(valid_dates)
                except Exception as e:
                    print(f"Warning: Could not extract dates from {ticker} {segment}: {e}")
        
        # Convert to sorted list
        report_dates = sorted(list(report_dates))
        
        if not report_dates:
            print(f"Warning: No valid report dates found for {ticker}")
            continue
            
        # Filter price data for this ticker
        ticker_prices = price_df[price_df['Ticker'] == ticker]
        
        if ticker_prices.empty:
            print(f"Warning: No price data found for ticker {ticker}")
            continue
            
        # Calculate average prices around each report date
        historical_prices = {}
        
        for report_date in report_dates:
            try:
                # Define the date range
                start_date = report_date - pd.Timedelta(days=days_window)
                end_date = report_date + pd.Timedelta(days=days_window)
                
                # Get prices within the window
                price_window = ticker_prices.loc[start_date:end_date]
                
                if not price_window.empty and 'Close' in price_window.columns:
                    avg_price = price_window['Close'].mean()
                    if not pd.isna(avg_price):
                        # Format date key (use the same format as in financial statements)
                        date_key = report_date.strftime('%Y-%m-%d')
                        historical_prices[date_key] = avg_price
                        # print(f"Added historical price for {ticker} on {date_key}: {avg_price:.2f}")
                    else:
                        print(f"Warning: No valid prices found for {ticker} around {report_date.date()}")
                else:
                    print(f"Warning: No price data available for {ticker} around {report_date.date()}")
                    
            except Exception as e:
                print(f"Warning: Error processing price for {ticker} on {report_date.date()}: {e}")
        
        # Add historical prices to the ticker data
        if historical_prices:
            enhanced_data[ticker]['historical_prices'] = historical_prices
            # print(f"Added {len(historical_prices)} historical prices for {ticker}")
        else:
            enhanced_data[ticker]['historical_prices'] = {}
            print(f"No historical prices could be calculated for {ticker}")
    
    return enhanced_data
