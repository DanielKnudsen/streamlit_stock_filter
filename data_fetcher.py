import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional

def get_price_data(
    SMA_short: int,
    SMA_medium: int,
    SMA_long: int,
    tickers: List[str],
    data_fetch_years: int,
    price_data_file_path: str
) -> None:
    """
    Fetch and save stock price data and moving averages for a list of tickers.

    Args:
        SMA_short (int): Window for short-term moving average.
        SMA_medium (int): Window for medium-term moving average.
        SMA_long (int): Window for long-term moving average.
        tickers (List[str]): List of ticker symbols.
        data_fetch_years (int): Number of years of price data to fetch.
        price_data_file_path (str): Path to the output CSV file.
    """
    import os
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
    import pandas as pd
    import yfinance as yf
    from tqdm import tqdm
    df_complete = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Fetching stock price data", disable=False if ENVIRONMENT == "local" else True):
        try:
            yf_ticker = f"{ticker}.ST"
            stock = yf.Ticker(yf_ticker)
            df_price_data = stock.history(period=f"{data_fetch_years}y")[['Close', 'Volume']]
            df_price_data['Ticker'] = ticker
            if df_price_data.index.tz is not None:
                df_price_data.index = df_price_data.index.tz_localize(None)
            if df_price_data.empty:
                print(f"Ingen data hämtad för {yf_ticker}")
            # Calculate moving averages
            df_price_data['SMA_short'] = df_price_data['Close'].rolling(window=SMA_short).mean()
            df_price_data['SMA_medium'] = df_price_data['Close'].rolling(window=SMA_medium).mean()
            df_price_data['SMA_long'] = df_price_data['Close'].rolling(window=SMA_long).mean()
            # Calculate percent differences
            df_price_data['pct_SMA_medium_vs_SMA_long'] = (((df_price_data['SMA_medium'] - df_price_data['SMA_long']) / df_price_data['SMA_long']) * 100).fillna(0)
            df_price_data['pct_SMA_short_vs_SMA_medium'] = (((df_price_data['SMA_short'] - df_price_data['SMA_medium']) / df_price_data['SMA_medium']) * 100).fillna(0)
            df_price_data['pct_Close_vs_SMA_short'] = (((df_price_data['Close'] - df_price_data['SMA_short']) / df_price_data['SMA_short']) * 100).fillna(0)
            df_complete = pd.concat([df_complete, df_price_data])
        except Exception as e:
            print(f"Fel vid hämtning av data för {yf_ticker}: {str(e)}")
    if df_complete.empty:
        print("Ingen prisdata hämtad för några tickers.")
    df_complete.to_csv(price_data_file_path, index=True)


def fetch_yfinance_data(ticker: str, years: int, period_type: str = "annual") -> Optional[Dict]:
    """
    Fetch financial data for a ticker from Yahoo Finance.

    Args:
        ticker (str): Ticker symbol.
        years (int): Number of years of data to fetch.
        period_type (str, optional): 'annual' or 'quarterly'. Defaults to 'annual'.

    Returns:
        Optional[Dict]: Dictionary of financial data, or None if fetch fails.
    """
    try:
        yf_ticker = f"{ticker}.ST"
        ticker_obj = yf.Ticker(yf_ticker)
        if period_type == "quarterly":
            bs = ticker_obj.quarterly_balance_sheet.transpose()
            is_ = ticker_obj.quarterly_income_stmt.transpose()
            cf = ticker_obj.quarterly_cash_flow.transpose()
            info = ticker_obj.info
        else:
            bs = ticker_obj.balance_sheet.transpose()
            is_ = ticker_obj.income_stmt.transpose()
            cf = ticker_obj.cash_flow.transpose()
            info = ticker_obj.info
            longBusinessSummary = info.get('longBusinessSummary', 'No summary available')
            dividendRate = info.get('dividendRate', None)
            lastDividendDate = info.get('lastDividendDate', None)
            dividends = ticker_obj.dividends

        shares_outstanding = info.get('sharesOutstanding', None)
        current_price = info.get('currentPrice', None)
        market_cap = info.get('marketCap', None)

        latest_report_date = None
        if hasattr(bs, 'index') and len(bs.index) > 0:
            try:
                latest_report_date = pd.to_datetime(bs.index[0])
            except Exception:
                latest_report_date = str(bs.index[0])

        if not all([bs is not None, is_ is not None, cf is not None, market_cap is not None, info is not None]):
            print(f"Warning: Incomplete data for {ticker}. Skipping.")
            return None

        bs = bs.head(years).copy().infer_objects(copy=False).fillna(0)
        is_ = is_.head(years).copy().infer_objects(copy=False).fillna(0)
        cf = cf.head(years).copy().infer_objects(copy=False).fillna(0)
        
        return {
            'balance_sheet': bs,
            'income_statement': is_,
            'cash_flow': cf,
            'current_price': current_price,
            'shares_outstanding': shares_outstanding,
            'info': info,
            'dividendRate': dividendRate if period_type == "annual" else None,
            'lastDividendDate': lastDividendDate if period_type == "annual" else None,
            'longBusinessSummary': longBusinessSummary if period_type == "annual" else None,
            'market_cap': market_cap,
            'dividends': dividends if period_type == "annual" else None,
            'latest_report_date': latest_report_date if latest_report_date else None,
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def read_tickers_from_csv(csv_file_path: str) -> List[str]:
    """
    Read tickers from a CSV file and return as a list of strings.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        List[str]: List of ticker symbols.
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_file_path)
        if 'ticker' in df.columns:
            return df['ticker'].tolist()
        elif 'symbol' in df.columns:
            return df['symbol'].tolist()
        elif 'Instrument' in df.columns:
            return df['Instrument'].tolist()
        else:
            print("Error: Could not find a 'ticker', 'symbol' or 'Instrument' column in the CSV file.")
            return []
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return []
