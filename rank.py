import pandas as pd
import numpy as np
import datetime
import os
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
from io_utils import load_yaml, load_csv, save_csv, load_pickle, save_pickle
from data_fetcher import read_tickers_from_csv, get_price_data, get_raw_financial_data
from ratios import calculate_all_ratios
from ranking import create_ratios_to_ranks

# Ladda .env-filen endast om den finns
if Path('.env').exists():
    load_dotenv()
    
# Bestäm miljön (default till 'local')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
FETCH_DATA = os.getenv('FETCH_DATA', 'Yes')

# Välj CSV-path
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')

# --- Funktionsdefinitioner ---

def load_config(config_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from a YAML file.

    Args:
        config_file_path (str): Path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary or None if not found.
    """
    try:
        return load_yaml(config_file_path)
    except FileNotFoundError:
        print(f"Error: The file {config_file_path} was not found.")
        return None

def combine_all_results(valid_tickers: List[str],
    calculated_ratios: Dict[str, Dict[str, Any]],
    calculated_ratios_ttm_trends: Dict[str, Dict[str, Any]],
    complete_ranks: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Combine all results into a single DataFrame.

    Args:
        calculated_ratios (Dict[str, Dict[str, Any]]): Calculated ratios per ticker.
        calculated_ratios_ttm_trends (Dict[str, Dict[str, Any]]): TTM trends per ticker.
        complete_ranks (Dict[str, Dict[str, Any]]): Category scores per ticker.

    Returns:
        pd.DataFrame: Combined results DataFrame.
    """
    df_calculated = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    df_calculated_ttm_trends = pd.DataFrame.from_dict(calculated_ratios_ttm_trends, orient='index')
    df_complete_ranks = pd.DataFrame.from_dict(complete_ranks, orient='index')
    """df_scores = pd.DataFrame.from_dict(category_scores, orient='index')
    df_cluster_ranks = pd.DataFrame.from_dict(cluster_ranks, orient='index')"""
    df_agr = load_csv(CSV_PATH / "agr_results.csv", index_col=0)
    df_agr_dividends = load_csv(CSV_PATH / "agr_dividend_results.csv", index_col=0)
    tickers_file = CSV_PATH / config.get("input_ticker_file")
    df_tickers = load_csv(tickers_file, index_col='Instrument')
    df_tickers = df_tickers.rename(columns={'Instrument': 'Ticker'})
    df_latest_report_dates = load_csv(CSV_PATH / "latest_report_dates.csv", index_col='Ticker')
    df_latest_report_dates_quarterly = load_csv(CSV_PATH / "latest_report_dates_quarterly.csv", index_col='Ticker')
    #df_ttm_values = load_csv(CSV_PATH / "ttm_values.csv", index_col='Ticker')
    df_last_SMA = load_csv(CSV_PATH / "last_SMA.csv", index_col='Ticker')
    df_long_business_summary = load_csv(CSV_PATH / "longBusinessSummary.csv", index_col='Ticker')
    #df_calculated_quarterly_long = load_csv(CSV_PATH / "calculated_ratios_quarterly.csv")
    #df_calculated_quarterly = df_calculated_quarterly_long.pivot(index='Ticker', columns='Metric', values='Values')
    #df_calculated_quarterly.index = df_calculated_quarterly.index.astype(str)
    df_market_cap = load_csv(CSV_PATH / "market_cap.csv", index_col='Ticker')
    final_df = pd.concat([
        df_tickers, df_calculated, df_calculated_ttm_trends, df_complete_ranks, df_last_SMA,
        df_agr, df_agr_dividends, df_latest_report_dates, df_latest_report_dates_quarterly, df_long_business_summary, df_market_cap
    ], axis=1)

    # only keep rows for valid tickers
    return final_df[final_df.index.isin(valid_tickers)]

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
        if period_type == "quarterly":
            # Only keep metrics ending with '_latest_ratioValue' and rename to '_ttm'
            filtered_data = {k.replace('_latest_ratioValue', '_ttm_ratioValue'): v for k, v in data.items() if k.endswith('_latest_ratioValue')}
        else:
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

def filter_metrics_for_agr_dimensions(csv_path: str, agr_dimensions: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Filter and pivot metrics for AGR dimensions from a CSV file.

    Args:
        csv_path (str): Path to the input CSV file.
        agr_dimensions (List[str]): List of metrics to filter.
        output_path (Optional[str], optional): Path to save the filtered output. Defaults to None.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = load_csv(csv_path)
    # Normalize agr_dimensions to match possible column values (spaces/underscores)
    agr_set = set(agr_dimensions) | set([dim.replace(' ', '_') for dim in agr_dimensions])
    filtered = df[df['Metric'].isin(agr_set)]
    
    # Pivot: Ticker as index, Metric as columns, Value as values
    filtered_pivot = filtered.pivot(index='Ticker', columns='Metric', values='Value')
    # fill spaces in column names with "_" and add "<_ttm>"
    filtered_pivot.columns = filtered_pivot.columns.str.replace(' ', '_') + "_ttm"
    if output_path:
        save_csv(filtered_pivot, output_path, index=True)
    return filtered

def summarize_quarterly_data_to_yearly(raw_financial_data_quarterly: Dict[str, Any], quarters_back: int) -> Dict[str, Any]:
    """
    Summarize quarterly data (4 quarters) to yearly by summing or latest depending on segment.
    Can be any 4 quarters in a row, e.g. last 4, (pos 0,1,2,3) or starting n quarters back and then taking e.g pos 1,2,3,4 

    Args:
        raw_financial_data_quarterly (Dict[str, Any]): Quarterly financial data per ticker.
        quarters_back (int): Number of quarters back for the last data point. E.g 0 for including the most recent one and the three before,
        1 for omitting the most recent quarter and then getting the 4 quarters before the most recent one.

    Returns:
        Dict[str, Any]: Summarized yearly data per ticker.
    """
    sum_metrics = [
        'Net Income', 'EBIT', 'Pretax Income', 'Tax Provision', 'Interest Expense',
        'Gross Profit', 'Total Revenue', 'Operating Income', 'Basic EPS', 'EBITDA',
        'Operating Cash Flow', 'Free Cash Flow'
    ]
    latest_metrics = [
        'Stockholders Equity', 'Total Assets', 'Total Debt', 'Cash And Cash Equivalents',
        'sharesOutstanding', 'currentPrice', 'marketCap'
    ]
    summarized_data = {}
    for ticker, data in raw_financial_data_quarterly.items():
        if data is None:
            summarized_data[ticker] = None
            continue
        summarized = {
            'balance_sheet': pd.DataFrame(),
            'income_statement': pd.DataFrame(),
            'cash_flow': pd.DataFrame(),
            'current_price': data.get('current_price'),
            'shares_outstanding': data.get('shares_outstanding'),
            'info': data.get('info'),
            'market_cap': data.get('market_cap'),
        }
        for segment in ['balance_sheet', 'income_statement', 'cash_flow']:
            df = data.get(segment)
            if df is None or df.empty:
                summarized[segment] = pd.DataFrame()
                continue
            df = df.sort_index(ascending=False)
            df = df.iloc[quarters_back:quarters_back + 4]
            
            # Check if we have enough data after filtering
            if df.empty:
                summarized[segment] = pd.DataFrame()
                continue
                
            agg_dict = {}
            for col in df.columns:
                if col in sum_metrics:
                    agg_dict[col] = df[col].sum()
                elif col in latest_metrics:
                    agg_dict[col] = df[col].iloc[0]
                else:
                    agg_dict[col] = df[col].iloc[0]
            latest_idx = df.index[0] if len(df.index) > 0 else None
            summarized[segment] = pd.DataFrame([agg_dict], index=[latest_idx])
        summarized_data[ticker] = summarized
    return summarized_data


def calculate_agr_for_ticker(csv_path: str, tickers: List[str], dimensions: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate average growth rate (AGR) for each ticker and metric.

    Args:
        csv_path (str): Path to the input CSV file.
        tickers (List[str]): List of tickers.
        dimensions (List[str]): List of metrics to calculate AGR for.

    Returns:
        Dict[str, Dict[str, Any]]: AGR results per ticker and metric.
    """
    df = load_csv(csv_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
    agr_results = {}
    grouped = df.groupby(['Ticker', 'Metric'])
    for ticker in tickers:
        ticker_agr = {}
        for dim in dimensions:
            try:
                group = grouped.get_group((ticker, dim)).sort_values('Date')
                values = pd.to_numeric(group['Value'].values, errors='coerce')
                years = group['Date'].values
            except KeyError:
                values = np.array([])
                years = np.array([])
            growth_rates = []
            for i in range(1, len(values)):
                prev = values[i - 1]
                curr = values[i]
                if pd.isna(prev) or pd.isna(curr) or prev == 0:
                    continue
                growth = (curr - prev) / abs(prev)
                growth_rates.append(growth)
            if growth_rates:
                agr = np.mean(growth_rates)
            else:
                agr = np.nan
            dim_key = dim.replace(" ", "_") + "_AvgGrowth"
            ticker_agr[dim_key] = agr
            dim_data_key = dim.replace(" ", "_")
            for year, value in zip(years, values):
                year_str = str(year)
                ticker_agr[f"{dim_data_key}_year_{year_str}"] = value
        agr_results[ticker] = ticker_agr
    return agr_results


def calculate_agr_dividend_for_ticker(csv_path: str, tickers: List[str], n_years: int = 4) -> Dict[str, Dict[str, Any]]:
    """
    Calculate average dividend growth rate for each ticker.

    Args:
        csv_path (str): Path to the dividends CSV file.
        tickers (List[str]): List of tickers.
        n_years (int, optional): Number of years to consider. Defaults to 4.

    Returns:
        Dict[str, Dict[str, Any]]: Dividend AGR results per ticker.
    """
    df = load_csv(csv_path, parse_dates=['Year'])
    df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
    agr_results = {}
    for ticker in tickers:
        group = df[df['Ticker'] == ticker]
        yearly = group.groupby('Year')['Value'].sum().sort_index()
        current_year = pd.Timestamp.today().year
        min_year = current_year - n_years
        yearly = yearly[yearly.index >= min_year]
        years = yearly.index.values
        values = yearly.values
        growth_rates = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            if pd.isna(prev) or pd.isna(curr) or prev == 0:
                continue
            growth = (curr - prev) / abs(prev)
            growth_rates.append(growth)
        if growth_rates:
            agr = np.mean(growth_rates)
        else:
            agr = np.nan
        ticker_dict = {'Dividend_AvgGrowth': agr}
        for year, value in zip(years, values):
            ticker_dict[f"Dividend_year_{year}"] = value
        agr_results[ticker] = ticker_dict
    return agr_results


def save_agr_results_to_csv(agr_results: Dict[str, Dict[str, Any]], csv_file_path: str) -> None:
    """
    Save AGR results to a CSV file and add percentile ranks.

    Args:
        agr_results (Dict[str, Dict[str, Any]]): AGR results per ticker.
        csv_file_path (str): Path to the output CSV file.
    """
    df = pd.DataFrame.from_dict(agr_results, orient='index')
    for col in df.columns:
        if col.endswith('_AvgGrowth'):
            ranks = df[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            rank_col = col.replace('_AvgGrowth', '_AvgGrowth_Rank')
            df[rank_col] = ranks
    save_csv(df, csv_file_path, index=True)
    print(f"AGR-resultat sparade till {csv_file_path}")


def post_processing(final_df: pd.DataFrame, rank_decimals: int, ratio_definitions: Dict[str, Any]) -> pd.DataFrame:
    """
    Perform post-processing on the final results DataFrame, including ranking and difference calculations.

    Args:
        final_df (pd.DataFrame): Combined results DataFrame.
        rank_decimals (int): Number of decimals for rank columns.
        ratio_definitions (Dict[str, Any]): Ratio definitions for ranking.

    Returns:
        pd.DataFrame: Post-processed DataFrame.
    """
    def group_by_sector(column: str) -> pd.Series:
        """ 
        Group by sector and calculate mean for the specified column
        Calculate the difference between each value and the sector mean
        Return two columns: one with the sector mean and one with the difference
        """
        sector_mean = final_df.groupby('Sektor')[column].transform('mean')
        return sector_mean, final_df[column] - sector_mean

    def get_quarter(dt):
        if pd.isnull(dt):
            return None
        return (dt.month - 1) // 3 + 1
    def quarter_diff(row):
        if pd.isnull(row['LatestReportDate_Q']) or pd.isnull(row['LatestReportDate_Y']):
            return None
        y1, q1 = row['LatestReportDate_Y'].year, get_quarter(row['LatestReportDate_Y'])
        y2, q2 = row['LatestReportDate_Q'].year, get_quarter(row['LatestReportDate_Q'])
        return (y2 - y1) * 4 + (q2 - q1)
    final_df['LatestReportDate_Q'] = pd.to_datetime(final_df['LatestReportDate_Q'], errors='coerce')
    final_df['LatestReportDate_Y'] = pd.to_datetime(final_df['LatestReportDate_Y'], errors='coerce')
    final_df['QuarterDiff'] = final_df.apply(quarter_diff, axis=1)
    all_ratios = []
    for category, ratios in config['kategorier'].items():
        all_ratios.extend(ratios)
    # Calculate difference between most recent ttm value and the ttm value one quarter back TODO
    """for ratio in all_ratios:
        final_df[f'{ratio}_ttm_diff'] = (final_df[f'{ratio}_ttm_ratioValue'] - final_df[f'{ratio}_latest_ratioValue'])"""
    for agr_temp in config['agr_dimensions']:
        agr = agr_temp.replace(" ", "_")
        latest_full_year_value = final_df.apply(
            lambda row: row.get(f"{agr}_year_{row['LatestReportDate_Y'].year}") if pd.notnull(row['LatestReportDate_Y']) and f"{agr}_year_{row['LatestReportDate_Y'].year}" in final_df.columns else np.nan,
            axis=1
        )
        final_df[f'{agr}_ttm_diff'] = final_df.get(f'{agr}_ttm', pd.Series(np.nan, index=final_df.index)) - latest_full_year_value
    all_ratios = []
    for category, ratios in config['kategorier'].items():
        all_ratios.extend(f"{ratio}_ttm_diff" for ratio in ratios)
    """for col in all_ratios:
        ratio_name = col.replace('_ttm_diff', '')
        is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
        ranked = final_df[col].rank(pct=True, ascending=is_better) * 100
        ranked = ranked.fillna(50)
        final_df[f'{ratio_name}_ttm_ratioRank'] = final_df.index.map(ranked)"""
        
    final_df.index = final_df.index.astype(str)
    final_df['Name'] = final_df['Name'].astype(str)
    final_df = final_df.assign(
        pct_ch_20_d_mean=group_by_sector('pct_ch_20_d')[0],
        pct_ch_20_d_diff=group_by_sector('pct_ch_20_d')[1],
        TTM_clusterRank_sector_mean=group_by_sector('TTM_clusterRank')[0],
        TTM_clusterRank_sector_diff=group_by_sector('TTM_clusterRank')[1],
        Latest_clusterRank_mean=group_by_sector('Latest_clusterRank')[0],
        Latest_clusterRank_diff=group_by_sector('Latest_clusterRank')[1]
    )
    final_df['TTM_diff_vs_pct_ch_20_d_diff'] = final_df['TTM_clusterRank_sector_diff'] - final_df['pct_ch_20_d_diff']
    final_df['Latest_diff_vs_pct_ch_20_d_diff'] = final_df['Latest_clusterRank_diff'] - final_df['pct_ch_20_d_diff']

    for col in final_df.columns:
        if "Rank" in col:
            final_df[col] = final_df[col].round(rank_decimals)
    return final_df

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

def trim_unused_columns(final_results: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Remove columns from final_results that are not used in app.py.

    Args:
        final_results (pd.DataFrame): The full results DataFrame.
        config (dict): The configuration dictionary (for dynamic columns).

    Returns:
        pd.DataFrame: DataFrame with only used columns.
    """
    # Always keep these columns if present
    keep_cols = {
        'Lista', 'Sektor', 'Trend_clusterRank', 'Latest_clusterRank', 
        'marketCap', 'cagr_close', 'Name', 'pct_Close_vs_SMA_short', 'pct_SMA_short_vs_SMA_medium', 'pct_SMA_medium_vs_SMA_long',
        'QuarterDiff', 'LatestReportDate_Q', 'LatestReportDate_Y'
    }
    # Add columns ending with these patterns
    patterns = [
        '_latest_ratioValue', '_trend_ratioValue',
        '_latest_ratioRank', '_trend_ratioRank',
        '_AvgGrowth_Rank', '_AvgGrowth', 'catRank', 
        '_ttm_diff', '_ttm_ratioRank'
    ]
    # Add dynamically from config
    if 'category_ratios' in config:
        for cat, ratios in config['category_ratios'].items():
            keep_cols.add(cat)
            keep_cols.update(ratios)
    if 'kategorier' in config:
        for cat, ratios in config['kategorier'].items():
            keep_cols.add(cat)
            keep_cols.update(ratios)
    # Add all columns matching patterns
    for col in final_results.columns:
        if any(col.endswith(pat) for pat in patterns) or any(pat in col for pat in patterns):
            keep_cols.add(col)
    # Only keep columns that exist in the DataFrame
    keep_cols = [col for col in keep_cols if col in final_results.columns]
    final_result_trimmed = final_results[keep_cols].copy()
    return final_result_trimmed

def reduce_price_data(read_from: str, save_to: str, columns: List[str]):
    # round price data to only two decimals and keep only necessary columns
    
    df = pd.read_csv(read_from, usecols=columns)
    df = df.round(2)    
    df.to_csv(save_to, index=False)

# --- Main Execution ---

if __name__ == "__main__":
    # Load configuration from YAML
    config = load_config("rank-config.yaml")
    if config:
        TICKERS_FILE_NAME = config["input_ticker_file"]
        if not TICKERS_FILE_NAME:
            print("No tickers file name found. Please check your CSV file.")
        else:
            # Step 0: Read tickers from CSV file
            print(f"Reading tickers from {CSV_PATH / TICKERS_FILE_NAME}...")
            tickers = read_tickers_from_csv(CSV_PATH / TICKERS_FILE_NAME)

            # Step 1: Fetch financial data for each ticker
            if not tickers:
                print("No tickers found in the file. Exiting.")
                exit(1)

            if FETCH_DATA == "Yes":
                raw_financial_data, raw_financial_data_quarterly, raw_financial_info, raw_financial_data_dividends, valid_tickers = get_raw_financial_data(tickers, config["data_fetch_years"], config["data_fetch_quarterly"])
                if ENVIRONMENT == "local":
                    save_pickle(raw_financial_data, CSV_PATH / "raw_financial_data.pkl")
                    save_pickle(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.pkl")
                    save_pickle(raw_financial_info, CSV_PATH / "raw_financial_info.pkl")
                    save_pickle(raw_financial_data_dividends, CSV_PATH / "raw_financial_data_dividends.pkl")
            else:
                print("Loading raw financial data from pickles...")
                try:
                    raw_financial_data = load_pickle(CSV_PATH / "raw_financial_data.pkl")
                    raw_financial_data_quarterly = load_pickle(CSV_PATH / "raw_financial_data_quarterly.pkl")
                    raw_financial_info = load_pickle(CSV_PATH / "raw_financial_info.pkl")
                    raw_financial_data_dividends = load_pickle(CSV_PATH / "raw_financial_data_dividends.pkl")
                    valid_tickers = list(raw_financial_data.keys())
                except FileNotFoundError:
                    print("No raw financial data found. Please fetch data first.")
                    exit(1)

            print(f"Fetched data for {len(valid_tickers)} valid tickers out of {len(tickers)} total tickers.")

            # Save raw financial data and business summaries
            save_raw_data_to_csv(raw_financial_data, CSV_PATH / "raw_financial_data.csv")
            save_raw_data_to_csv(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.csv")
            # save_info_to_csv(raw_financial_info, CSV_PATH / "raw_financial_info.csv")
            save_longBusinessSummary_to_csv(raw_financial_info, CSV_PATH / "longBusinessSummary.csv")
            save_dividends_to_csv(raw_financial_data_dividends, CSV_PATH / "dividends.csv")
            save_latest_report_dates_to_csv(raw_financial_data, CSV_PATH / "latest_report_dates.csv", period_type="Y")
            save_latest_report_dates_to_csv(raw_financial_data_quarterly, CSV_PATH / "latest_report_dates_quarterly.csv", period_type="Q")
            save_market_cap_to_csv(raw_financial_data, CSV_PATH / "market_cap.csv")
            # Step 2: Fetch and process stock price data
            
            if FETCH_DATA == "Yes":
                print("Fetching stock price data...")
                get_price_data(config["SMA_short"],config["SMA_medium"], config["SMA_long"],config['SMA_sector'],
                           valid_tickers,config["price_data_years"],CSV_PATH / config["price_data_file_raw"])
            
                save_last_SMA_to_csv(
                    read_from=CSV_PATH / config["price_data_file_raw"],
                    save_to=CSV_PATH / "last_SMA.csv"
                )

                # reduce price data to only necessary columns
                reduce_price_data(
                    read_from=CSV_PATH / config["price_data_file_raw"],
                    save_to=CSV_PATH / config["price_data_file"],
                    columns=['Date','Close','Volume','Ticker']
                )
            
                # Clean up: Remove the raw price data file as it's no longer needed

                raw_file_path = CSV_PATH / config["price_data_file_raw"]
                if raw_file_path.exists():
                    os.remove(raw_file_path)
                    print(f"Cleaned up raw price data file: {raw_file_path}")

            # Step 3: Calculate ratios and rankings
            # Define which keys are needed for ratio calculations
            ratio_keys = ['balance_sheet', 'income_statement', 'cash_flow', 'shares_outstanding', 'market_cap']
            
            # Filter raw_financial_data to only include needed keys
            filtered_raw_data = {
                ticker: {key: data[key] for key in ratio_keys if key in data}
                for ticker, data in raw_financial_data.items()
            }
            
            # Add historical prices around report dates
            print("Adding historical prices to filtered data...")
            filtered_raw_data_with_prices = add_historical_prices_to_filtered_data(
                filtered_raw_data, 
                CSV_PATH / config["price_data_file"]
            )
            
            # Calculate annual ratios
            calculated_ratios = calculate_all_ratios(filtered_raw_data_with_prices, config["ratio_definitions"])
            save_calculated_ratios_to_csv(calculated_ratios, CSV_PATH / "calculated_ratios.csv", period_type="annual")

            # Summarize quarterly data to yearly for most recent 4 quarters (0 quarters back)
            quarters_back=0
            raw_financial_data_quarterly_summarized_0 = summarize_quarterly_data_to_yearly(raw_financial_data_quarterly,quarters_back)
            
            # Summarize quarterly data to yearly for 4 quarters back (1 quarter back)
            quarters_back=1
            raw_financial_data_quarterly_summarized_1 = summarize_quarterly_data_to_yearly(raw_financial_data_quarterly,quarters_back)
            
            # Combine the two TTM summaries for trend analysis
            print("Combining TTM summaries for trend analysis...")
            combined_ttm_data = combine_quarterly_summaries_for_ttm_trends(
                raw_financial_data_quarterly_summarized_0,
                raw_financial_data_quarterly_summarized_1
            )
            
            # Filter combined TTM data
            filtered_combined_ttm = {
                ticker: {key: data[key] for key in ratio_keys if key in data}
                for ticker, data in combined_ttm_data.items()
                if data is not None
            }
            
            # Add historical prices to combined TTM data
            filtered_combined_ttm_with_prices = add_historical_prices_to_filtered_data(
                filtered_combined_ttm, 
                CSV_PATH / config["price_data_file"]
            )
            
            # Calculate TTM trend ratios
            calculated_ratios_ttm_trends = calculate_all_ratios(
                filtered_combined_ttm_with_prices, 
                config["ratio_definitions"],
                period_type="quarterly"
            )
            save_calculated_ratios_to_csv(
                calculated_ratios_ttm_trends, 
                CSV_PATH / "calculated_ratios_ttm_trends.csv", 
                period_type="quarterly"
            )

            # Create ratios to ranks 
            complete_ranks = create_ratios_to_ranks(
                calculated_ratios,
                calculated_ratios_ttm_trends,
                config["ratio_definitions"],
                config["category_ratios"]
            )
            save_dict_of_dicts_to_csv(complete_ranks, CSV_PATH / "complete_ranks.csv")

            # Step 5: Calculate AGR results
            agr_results = calculate_agr_for_ticker(CSV_PATH / "raw_financial_data.csv", tickers, config['agr_dimensions'])
            save_agr_results_to_csv(agr_results, CSV_PATH / "agr_results.csv")

            agr_dividend = calculate_agr_dividend_for_ticker(CSV_PATH / "dividends.csv", tickers, config.get('data_fetch_years', 4))
            save_agr_results_to_csv(agr_dividend, CSV_PATH / "agr_dividend_results.csv")

            # Step 6: Extract ttm values for agr dimensions TODO: check if needed
            """filter_metrics_for_agr_dimensions(CSV_PATH / "raw_financial_data_quarterly_summarized.csv", 
                               config['agr_dimensions'],
                               CSV_PATH / "ttm_values.csv")"""

            # Step 7: Combine all results and save final output
            combined_results = combine_all_results(valid_tickers,
                calculated_ratios,
                calculated_ratios_ttm_trends,
                complete_ranks
            )

            final_results = post_processing(combined_results, config["rank_decimals"], config["ratio_definitions"])
            #final_results_trimmed = trim_unused_columns(final_results, config)  # Trim unused columns
            save_results_to_csv(final_results, CSV_PATH / config["results_file"])

            print(f"Stock evaluation completed and saved to {CSV_PATH / config['results_file']}")
    else:
        print("Could not load configuration. Exiting.")
