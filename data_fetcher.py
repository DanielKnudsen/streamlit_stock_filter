import yfinance as yf
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any

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

def get_raw_financial_data(tickers, years, quarters) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], set]:
    """
    Fetch raw financial data for a list of tickers from Yahoo Finance.
    Makes sure that that there are 'years' of annual data and 'quarters' of quarterly data available, otherwise skips the ticker.

    Args:
        tickers (List[str]): List of ticker symbols.
        years (int): Number of years of data to fetch.
        quarters (int): Number of quarters of data to fetch.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], set]: Tuple containing raw financial data dictionaries and a set of valid tickers.
    """
    raw_financial_data_annual = {}
    raw_financial_data_quarterly = {}
    raw_financial_info = {}
    raw_financial_data_dividends = {}

    for ticker in tqdm(tickers):
        try:
            yf_ticker = f"{ticker}.ST"
            ticker_obj = yf.Ticker(yf_ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        # Get additional info
        info = ticker_obj.info
        raw_financial_info[ticker] = info

        # Get dividend info
        dividends = ticker_obj.dividends
        dividendRate = info.get('dividendRate', None)
        lastDividendDate = info.get('lastDividendDate', None)
        raw_financial_data_dividends[ticker] = {
            'dividends': dividends,
            'dividendRate': dividendRate,
            'lastDividendDate': lastDividendDate
        }

        # Quarterly data, transposed for easier access
        bs_quarterly = ticker_obj.quarterly_balance_sheet.transpose()
        is_quarterly = ticker_obj.quarterly_income_stmt.transpose()
        cf_quarterly = ticker_obj.quarterly_cash_flow.transpose()

        # Ensure we have enough data, infer better data types and fill NaNs with 0
        bs_quarterly = bs_quarterly.head(quarters).copy().infer_objects(copy=False).fillna(0)
        is_quarterly = is_quarterly.head(quarters).copy().infer_objects(copy=False).fillna(0)
        cf_quarterly = cf_quarterly.head(quarters).copy().infer_objects(copy=False).fillna(0)

        # Get latest report date for quarterly data
        latest_report_date_quarterly = None
        if hasattr(bs_quarterly, 'index') and len(bs_quarterly.index) > 0:
            try:
                latest_report_date_quarterly = pd.to_datetime(bs_quarterly.index[0])
            except Exception as e:
                latest_report_date_quarterly = str(bs_quarterly.index[0])

        shares_outstanding = info.get('sharesOutstanding', None)
        market_cap = info.get('marketCap', None)

        # Store quarterly data in quarterly dictionary
        raw_financial_data_quarterly[ticker] = {
            'balance_sheet': bs_quarterly,
            'income_statement': is_quarterly,
            'cash_flow': cf_quarterly,
            'latest_report_date': latest_report_date_quarterly,
            'shares_outstanding': shares_outstanding,
            'market_cap': market_cap
        }

        # Annual data, transposed for easier access
        bs_annual = ticker_obj.balance_sheet.transpose()
        is_annual = ticker_obj.income_stmt.transpose()
        cf_annual = ticker_obj.cash_flow.transpose()

        # Ensure we have enough data, infer better data types and fill NaNs with 0
        bs_annual = bs_annual.head(years).copy().infer_objects(copy=False).fillna(0)
        is_annual = is_annual.head(years).copy().infer_objects(copy=False).fillna(0)
        cf_annual = cf_annual.head(years).copy().infer_objects(copy=False).fillna(0)
        
        # Get latest report date for annual data
        latest_report_date_annual = None
        if hasattr(bs_annual, 'index') and len(bs_annual.index) > 0:
            try:
                latest_report_date_annual = pd.to_datetime(bs_annual.index[0])
            except Exception as e:
                latest_report_date_annual = str(bs_annual.index[0])
        shares_outstanding = info.get('sharesOutstanding', None)
        market_cap = info.get('marketCap', None)

        # Store annual data in annual dictionary
        raw_financial_data_annual[ticker] = {
            'balance_sheet': bs_annual,
            'income_statement': is_annual,
            'cash_flow': cf_annual,
            'latest_report_date': latest_report_date_annual,
            'shares_outstanding': shares_outstanding,
            'market_cap': market_cap
        }

        # Only keep the ticker if we have enough data
        if len(bs_annual) < years or len(is_annual) < years or len(cf_annual) < years or len(bs_quarterly) < quarters or len(is_quarterly) < quarters or len(cf_quarterly) < quarters:
            print(f"Warning: Not enough annual data for {ticker}. Skipping.")
            del raw_financial_data_annual[ticker]
            del raw_financial_data_quarterly[ticker]
            del raw_financial_info[ticker]
            continue

    # return only the tickers with enough data in all three dictionaries
    valid_tickers = set(raw_financial_data_annual.keys()) & set(raw_financial_data_quarterly.keys()) & set(raw_financial_info.keys())
    raw_financial_data_annual = {k: v for k, v in raw_financial_data_annual.items() if k in valid_tickers}
    raw_financial_data_quarterly = {k: v for k, v in raw_financial_data_quarterly.items() if k in valid_tickers}
    raw_financial_info = {k: v for k, v in raw_financial_info.items() if k in valid_tickers}
    raw_financial_data_dividends = {k: v for k, v in raw_financial_data_dividends.items() if k in valid_tickers}

    return raw_financial_data_annual, raw_financial_data_quarterly, raw_financial_info, raw_financial_data_dividends, valid_tickers

def fetch_yfinance_data(ticker: str, years: int, period_type: str = "annual") -> Optional[Dict]:
    """
    Fetch financial data for a ticker from Yahoo Finance.
    TODO: extract all dicts at once instead of individual fields
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
