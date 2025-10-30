import os
import pandas as pd
import datetime
from typing import Any, Dict, Optional, List
from io_utils import load_csv, save_csv

def save_results_to_csv(results_df: pd.DataFrame, file_path: str) -> None:
    """
    Save the final DataFrame to a CSV file, ensuring the target directory exists.

    Args:
        results_df (pd.DataFrame): The DataFrame to save.
        file_path (str): Path to the output CSV file.
    """
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    save_csv(results_df, file_path, index=True)

def save_raw_data_to_csv(raw_data: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save raw financial data to a CSV file in long format.

    Args:
        raw_data (Dict[str, Any]): Raw financial data per ticker.
        csv_file_path (str): Path to the output CSV file.
    """
    segments = ['balance_sheet','income_statement','cash_flow']
    df = pd.DataFrame()
    for ticker,data in raw_data.items():
        for segment in segments:
            df_segment_data = data[segment].T
            df_temp = (df_segment_data
                              .reset_index()
                              .rename(columns={'index': 'Metric'}))
            df_long_format = pd.melt(df_temp,id_vars=['Metric'], 
                         var_name='Date', 
                         value_name='Value')
            df_long_format['Ticker'] = ticker
            df_long_format['Segment'] = segment
            df_long_format['Date'] = pd.to_datetime(df_long_format['Date'], errors='coerce')
            df_long_format['Date'] = df_long_format['Date'].dt.date
            df = pd.concat([df, df_long_format], ignore_index=True)

    save_csv(df, csv_file_path, index=False)



def save_longBusinessSummary_to_csv(raw_data: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save long business summaries to a CSV file.

    Args:
        raw_data (Dict[str, Any]): Raw financial data per ticker.
        csv_file_path (str): Path to the output CSV file.
    """
    df = pd.DataFrame()
    for ticker, data in raw_data.items():
        if 'longBusinessSummary' in data:
            summary = data['longBusinessSummary']
            df_temp = pd.DataFrame({'Ticker': [ticker], 'LongBusinessSummary': [summary]})
            df = pd.concat([df, df_temp], ignore_index=True)
        else:
            print(f"Warning: No longBusinessSummary for {ticker}. Skipping.")
    
    save_csv(df, csv_file_path, index=False)

def save_market_cap_to_csv(raw_data: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save market cap information for each ticker to a CSV file.

    Args:
        raw_data (Dict[str, Any]): Financial data for each ticker, including market cap info.
        csv_file_path (str): Path to the output CSV file.
    """
    rows = []
    for ticker, data in raw_data.items():
        market_cap = None

        if 'market_cap' in raw_data[ticker]:
            market_cap = raw_data[ticker]['market_cap'] if raw_data[ticker]['market_cap'] is not None else None
        else:
            market_cap = None

        rows.append({'Ticker': ticker, 'market_cap': market_cap})

    df = pd.DataFrame(rows)
    save_csv(df, csv_file_path, index=False)

def save_info_to_csv(raw_data: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save long business summaries to a CSV file.

    Args:
        raw_data (Dict[str, Any]): Raw financial data per ticker.
        csv_file_path (str): Path to the output CSV file.
    """
    df = pd.DataFrame()
    for ticker, data in raw_data.items():
        if 'longBusinessSummary' in data:
            summary = data['longBusinessSummary']
            df_temp = pd.DataFrame({'Ticker': [ticker], 'LongBusinessSummary': [summary]})
            df = pd.concat([df, df_temp], ignore_index=True)
        else:
            print(f"Warning: No longBusinessSummary for {ticker}. Skipping.")
    
    save_csv(df, csv_file_path, index=False)

def save_latest_report_dates_to_csv(raw_data: Dict[str, Any], csv_file_path: str, period_type: str = "Y") -> None:
    """
    Save the latest report dates to a CSV file.

    Args:
        raw_data (Dict[str, Any]): Raw financial data per ticker.
        csv_file_path (str): Path to the output CSV file.
        period_type (str, optional): Period type, 'Y' for yearly or 'Q' for quarterly. Defaults to 'Y'.
    """
    df = pd.DataFrame()
    for ticker, data in raw_data.items():
        if 'latest_report_date' in data:
            report_date = data['latest_report_date']
            df_temp = pd.DataFrame({'Ticker': [ticker], f'LatestReportDate_{period_type}': [report_date]})
            df = pd.concat([df, df_temp], ignore_index=True)
        else:
            print(f"Warning: No latest_report_date for {ticker}. Skipping.")

    save_csv(df, csv_file_path, index=False)

def save_dividends_to_csv(raw_data: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save dividend information for each ticker to a CSV file.

    Args:
        raw_data (Dict[str, Any]): Financial data for each ticker, including dividend info.
        csv_file_path (str): Path to the output CSV file.
    """
    rows = []
    for ticker, data in raw_data.items():
        dividend_last_year = None
        last_dividend_year = None

        if 'dividendRate' in raw_data[ticker]:
            dividend_last_year = raw_data[ticker]['dividendRate'] if raw_data[ticker]['dividendRate'] is not None else None
        else:
            dividend_last_year = None
        if 'lastDividendDate' in raw_data[ticker]:
            last_dividend_year = int(datetime.datetime.fromtimestamp(raw_data[ticker]['lastDividendDate']).strftime('%Y')) if raw_data[ticker]['lastDividendDate'] is not None else None
        else:
            last_dividend_year = None
        if 'dividends' in data and len(data['dividends']) > 0:
            dividends = data['dividends']
            # dividends can be a pandas Series, DataFrame, or dict
            if isinstance(dividends, pd.DataFrame):
                # Already a DataFrame with Date and Value columns
                div_df = dividends.copy()
            elif isinstance(dividends, (pd.Series, dict)):
                # Convert Series or dict to DataFrame
                div_df = pd.DataFrame(list(dividends.items()), columns=['Date', 'Value'])
            else:
                print(f"Warning: Unexpected dividends type for {ticker}: {type(dividends)}")
                continue
                
            div_df['Date'] = pd.to_datetime(div_df['Date'], errors='coerce')
            div_df['Year'] = div_df['Date'].dt.year
            # Aggregate by year (sum all dividends for the same year)
            yearly = div_df.groupby('Year')['Value'].sum().reset_index()
            for _, row in yearly.iterrows():
                if last_dividend_year is not None and int(row['Year']) == last_dividend_year and dividend_last_year is not None:
                    rows.append({'Ticker': ticker, 'Year': int(row['Year']), 'Value': dividend_last_year})
                else:
                    rows.append({'Ticker': ticker, 'Year': int(row['Year']), 'Value': row['Value']})
        else:
            print(f"Warning: No dividends for {ticker}. Skipping.")

    df = pd.DataFrame(rows)
    save_csv(df, csv_file_path, index=False)

def save_calculated_ratios_to_csv(calculated_ratios: Dict[str, Any], csv_file_path: str, period_type: str = "annual") -> None:
    """
    Save calculated ratios to a CSV file.

    Args:
        calculated_ratios (Dict[str, Any]): Calculated ratios per ticker.
        csv_file_path (str): Path to the output CSV file.
        period_type (str, optional): 'annual' or 'quarterly'. Defaults to 'annual'.
    """
    df = pd.DataFrame()
    for ticker, data in calculated_ratios.items():
        """if period_type == "quarterly":
            # Only keep metrics ending with '_latest_ratioValue' and rename to '_ttm'
            filtered_data = {k.replace('_latest_ratioValue', '_ttm_ratioValue'): v for k, v in data.items() if k.endswith('_latest_ratioValue')}
        else:"""
        filtered_data = data
        df_metrics = pd.DataFrame(filtered_data, index=[ticker]).T
        df_metrics.columns = ['Values']
        df_temp = df_metrics.reset_index().rename(columns={'index': 'Metric'})
        df_temp['Ticker'] = ticker
        # Reorder columns to put 'Ticker' first
        cols = df_temp.columns.tolist()
        if 'Ticker' in cols:
            cols.insert(0, cols.pop(cols.index('Ticker')))
            df_temp = df_temp[cols]
        df = pd.concat([df, df_temp], ignore_index=False)
    save_csv(df, csv_file_path, index=False)

def save_dict_of_dicts_to_csv(
    data_dict: Dict[str, Dict[str, Any]],
    csv_file_path: str,
    index_col_name: str = "Key",
    index: bool = True
) -> None:
    """
    Save a dict-of-dicts to CSV.

    Args:
        data_dict (Dict[str, Dict[str, Any]]): Data to save.
        csv_file_path (str): Path to the output CSV file.
        index_col_name (str, optional): Name for the first column (e.g. 'Rank', 'Category'). Defaults to 'Key'.
        index (bool, optional): Whether to write row names (index). Defaults to True.
    """
    df = pd.DataFrame()
    for ticker, data in data_dict.items():
        df_metrics = pd.DataFrame(data, index=[ticker]).T
        df_metrics.columns = ['Values']
        df_temp = df_metrics.reset_index().rename(columns={'index': index_col_name})
        df_temp['Ticker'] = ticker
        df = pd.concat([df, df_temp], ignore_index=False)
    save_csv(df, csv_file_path, index=index)

def reduce_price_data(read_from: str, save_to: str, columns: List[str]):
    df = pd.read_csv(read_from, usecols=columns)
    df = df.round(2)
    df.to_csv(save_to, index=False)
