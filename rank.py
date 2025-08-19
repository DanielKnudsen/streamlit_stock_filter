import pandas as pd
import numpy as np
import datetime
import os
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from io_utils import load_yaml, load_csv, save_csv, load_pickle, save_pickle
from data_fetcher import fetch_yfinance_data, read_tickers_from_csv, get_price_data
from ratios import calculate_all_ratios
from ranking import rank_all_ratios, aggregate_category_ranks, aggregate_cluster_ranks

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

def combine_all_results(
    calculated_ratios: Dict[str, Dict[str, Any]],
    ranked_ratios: Dict[str, Dict[str, Any]],
    category_scores: Dict[str, Dict[str, Any]],
    cluster_ranks: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Combine all results into a single DataFrame.

    Args:
        calculated_ratios (Dict[str, Dict[str, Any]]): Calculated ratios per ticker.
        ranked_ratios (Dict[str, Dict[str, Any]]): Ranked ratios per ticker.
        category_scores (Dict[str, Dict[str, Any]]): Category scores per ticker.
        cluster_ranks (Dict[str, Dict[str, Any]]): Cluster ranks per ticker.

    Returns:
        pd.DataFrame: Combined results DataFrame.
    """
    df_calculated = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    df_ranked = pd.DataFrame.from_dict(ranked_ratios, orient='index')
    df_scores = pd.DataFrame.from_dict(category_scores, orient='index')
    df_cluster_ranks = pd.DataFrame.from_dict(cluster_ranks, orient='index')
    df_agr = load_csv(CSV_PATH / "agr_results.csv", index_col=0)
    df_agr_dividends = load_csv(CSV_PATH / "agr_dividend_results.csv", index_col=0)
    tickers_file = CSV_PATH / config.get("input_ticker_file")
    df_tickers = load_csv(tickers_file, index_col='Instrument')
    df_tickers = df_tickers.rename(columns={'Instrument': 'Ticker'})
    df_latest_report_dates = load_csv(CSV_PATH / "latest_report_dates.csv", index_col='Ticker')
    df_latest_report_dates_quarterly = load_csv(CSV_PATH / "latest_report_dates_quarterly.csv", index_col='Ticker')
    df_ttm_values = load_csv(CSV_PATH / "ttm_values.csv", index_col='Ticker')
    df_last_SMA = load_csv(CSV_PATH / "last_SMA.csv", index_col='Ticker')
    df_long_business_summary = load_csv(CSV_PATH / "longBusinessSummary.csv", index_col='Ticker')
    df_calculated_quarterly_long = load_csv(CSV_PATH / "calculated_ratios_quarterly.csv")
    df_calculated_quarterly = df_calculated_quarterly_long.pivot(index='Ticker', columns='Metric', values='Values')
    df_calculated_quarterly.index = df_calculated_quarterly.index.astype(str)
    final_df = pd.concat([
        df_tickers, df_calculated, df_calculated_quarterly, df_ranked, df_scores, df_last_SMA, df_cluster_ranks,
        df_agr, df_agr_dividends, df_latest_report_dates, df_latest_report_dates_quarterly, df_ttm_values, df_long_business_summary
    ], axis=1)
    return final_df

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
        if 'info' in raw_data[ticker]:
            if 'dividendRate' in raw_data[ticker]['info']:
                dividend_last_year = raw_data[ticker]['info']['dividendRate'] if raw_data[ticker]['info']['dividendRate'] is not None else None
            else:
                dividend_last_year = None
            if 'lastDividendDate' in raw_data[ticker]['info']:
                last_dividend_year = int(datetime.datetime.fromtimestamp(raw_data[ticker]['info']['lastDividendDate']).strftime('%Y'))
            else:
                last_dividend_year = None
        if 'dividends' in data:
            dividends = data['dividends']
            # dividends is a pandas Series: index=date, value=dividend
            if hasattr(dividends, 'items'):
                # Build a DataFrame for aggregation
                div_df = pd.DataFrame(list(dividends.items()), columns=['Date', 'Value'])
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
            # Only keep metrics ending with '_latest_ratioValue' and rename to '_TTM'
            filtered_data = {k.replace('_latest_ratioValue', '_TTM'): v for k, v in data.items() if k.endswith('_latest_ratioValue')}
        else:
            filtered_data = data
        df_metrics = pd.DataFrame(filtered_data, index=[ticker]).T
        df_metrics.columns = ['Values']
        df_temp = df_metrics.reset_index().rename(columns={'index': 'Metric'})
        df_temp['Ticker'] = ticker
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
            last_rows = df.groupby('Ticker').tail(1)[['Ticker', 'pct_Close_vs_SMA_short', 'pct_SMA_short_vs_SMA_medium', 'pct_SMA_medium_vs_SMA_long']]
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
    Extract TTM values for each ticker and metric in agr_dimensions from summarized quarterly data and save to CSV.

    Args:
        csv_path (str): Path to the summarized quarterly data CSV.
        agr_dimensions (List[str]): List of metrics to extract.
        file_path (str): Path to the output CSV file.

    Returns:
        Dict[str, Dict[str, Any]]: TTM values per ticker and metric.
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
                ttm_values[ticker][f"{col_name}_TTM"] = df.loc[ticker, dim]
            elif col_name in df.columns:
                ttm_values[ticker][f"{col_name}_TTM"] = df.loc[ticker, col_name]
            else:
                ttm_values[ticker][f"{col_name}_TTM"] = None
    # Save to CSV
    ttm_df = pd.DataFrame.from_dict(ttm_values, orient='index')
    save_csv(ttm_df, file_path, index=True)
    print(f"TTM values extracted and saved to {file_path}")
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
    # fill spaces in column names with "_" and add "<_TTM>"
    filtered_pivot.columns = filtered_pivot.columns.str.replace(' ', '_') + "_TTM"
    if output_path:
        save_csv(filtered_pivot, output_path, index=True)
    return filtered

def summarize_quarterly_data_to_yearly(raw_financial_data_quarterly: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize quarterly data to yearly by summing or averaging depending on segment.

    Args:
        raw_financial_data_quarterly (Dict[str, Any]): Quarterly financial data per ticker.

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
            df = df.head(4)
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
    for ratio in all_ratios:
        final_df[f'{ratio}_TTM_diff'] = (final_df[f'{ratio}_TTM'] - final_df[f'{ratio}_latest_ratioValue'])
    for agr_temp in config['agr_dimensions']:
        agr = agr_temp.replace(" ", "_")
        latest_full_year_value = final_df.apply(
            lambda row: row.get(f"{agr}_year_{row['LatestReportDate_Y'].year}") if pd.notnull(row['LatestReportDate_Y']) and f"{agr}_year_{row['LatestReportDate_Y'].year}" in final_df.columns else np.nan,
            axis=1
        )
        final_df[f'{agr}_TTM_diff'] = final_df.get(f'{agr}_TTM', pd.Series(np.nan, index=final_df.index)) - latest_full_year_value
    all_ratios = []
    for category, ratios in config['kategorier'].items():
        all_ratios.extend(f"{ratio}_TTM_diff" for ratio in ratios)
    for col in all_ratios:
        ratio_name = col.replace('_TTM_diff', '')
        is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
        ranked = final_df[col].rank(pct=True, ascending=is_better) * 100
        ranked = ranked.fillna(50)
        final_df[f'{ratio_name}_ttm_ratioRank'] = final_df.index.map(ranked)
        
    final_df.index = final_df.index.astype(str)
    final_df['Name'] = final_df['Name'].astype(str)
    for col in final_df.columns:
        if "Rank" in col:
            final_df[col] = final_df[col].round(rank_decimals)
    return final_df

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
                raw_financial_data = {}
                raw_financial_data_quarterly = {}
                print("Fetching financial data...")
                for ticker in tqdm(tickers, desc="Fetching financial data", disable=True if ENVIRONMENT == "remote" else False):
                    raw_financial_data[ticker] = fetch_yfinance_data(ticker, config["data_fetch_years"], period_type="annual")
                    raw_financial_data_quarterly[ticker] = fetch_yfinance_data(ticker, config["data_fetch_quarterly"], period_type="quarterly")
                    if ENVIRONMENT == "local":
                        save_pickle(raw_financial_data, CSV_PATH / "raw_financial_data.pkl")
                        save_pickle(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.pkl")
            else:
                print("Loading raw financial data from pickles...")
                try:
                    raw_financial_data = load_pickle(CSV_PATH / "raw_financial_data.pkl")
                    raw_financial_data_quarterly = load_pickle(CSV_PATH / "raw_financial_data_quarterly.pkl")
                except FileNotFoundError:
                    print("No raw financial data found. Please fetch data first.")
                    exit(1)

            # Remove tickers with no data
            raw_financial_data = {ticker: data for ticker, data in raw_financial_data.items() if data is not None}
            raw_financial_data_quarterly = {ticker: data for ticker, data in raw_financial_data_quarterly.items() if data is not None}

            # Save raw financial data and business summaries
            save_raw_data_to_csv(raw_financial_data, CSV_PATH / "raw_financial_data.csv")
            save_raw_data_to_csv(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.csv")
            save_longBusinessSummary_to_csv(raw_financial_data, CSV_PATH / "longBusinessSummary.csv")
            save_dividends_to_csv(raw_financial_data, CSV_PATH / "dividends.csv")
            save_latest_report_dates_to_csv(raw_financial_data, CSV_PATH / "latest_report_dates.csv", period_type="Y")
            save_latest_report_dates_to_csv(raw_financial_data_quarterly, CSV_PATH / "latest_report_dates_quarterly.csv", period_type="Q")

            # Step 2: Fetch and process stock price data
            
            if FETCH_DATA == "Yes":
                print("Fetching stock price data...")
                get_price_data(config["SMA_short"],config["SMA_medium"], config["SMA_long"],
                           raw_financial_data.keys(),config["price_data_years"],CSV_PATH / config["price_data_file"])
            
            save_last_SMA_to_csv(
                read_from=CSV_PATH / config["price_data_file"],
                save_to=CSV_PATH / "last_SMA.csv"
)

            # Step 3: Calculate ratios and rankings
            raw_financial_data_quarterly_summarized=summarize_quarterly_data_to_yearly(raw_financial_data_quarterly)
            save_raw_data_to_csv(raw_financial_data_quarterly_summarized, CSV_PATH / "raw_financial_data_quarterly_summarized.csv")

            calculated_ratios_quarterly = calculate_all_ratios(raw_financial_data_quarterly_summarized, config["ratio_definitions"])
            save_calculated_ratios_to_csv(calculated_ratios_quarterly, CSV_PATH / "calculated_ratios_quarterly.csv", period_type="quarterly")

            calculated_ratios = calculate_all_ratios(raw_financial_data, config["ratio_definitions"])
            save_calculated_ratios_to_csv(calculated_ratios, CSV_PATH / "calculated_ratios.csv", period_type="annual")

            ranked_ratios = rank_all_ratios(calculated_ratios, config["ratio_definitions"])
            save_dict_of_dicts_to_csv(ranked_ratios, CSV_PATH / "ranked_ratios.csv", index_col_name="Rank", index=False)

            # Step 4: Aggregate category and cluster ranks
            category_ranks = aggregate_category_ranks(ranked_ratios, config["category_ratios"])
            save_dict_of_dicts_to_csv(category_ranks, CSV_PATH / "category_ranks.csv", index_col_name="Category", index=True)

            cluster_ranks = aggregate_cluster_ranks(category_ranks)
            save_dict_of_dicts_to_csv(cluster_ranks, CSV_PATH / "cluster_ranks.csv", index_col_name="Category", index=True)

            # Step 5: Calculate AGR results
            agr_results = calculate_agr_for_ticker(CSV_PATH / "raw_financial_data.csv", tickers, config['agr_dimensions'])
            save_agr_results_to_csv(agr_results, CSV_PATH / "agr_results.csv")

            agr_dividend = calculate_agr_dividend_for_ticker(CSV_PATH / "dividends.csv", tickers, config.get('data_fetch_years', 4))
            save_agr_results_to_csv(agr_dividend, CSV_PATH / "agr_dividend_results.csv")

            # Step 6: Extract TTM values for agr dimensions
            filter_metrics_for_agr_dimensions(CSV_PATH / "raw_financial_data_quarterly_summarized.csv", 
                               config['agr_dimensions'],
                               CSV_PATH / "ttm_values.csv")

            # Step 7: Combine all results and save final output
            combined_results = combine_all_results(
                calculated_ratios,
                ranked_ratios,
                category_ranks,
                cluster_ranks
            )

            final_results=post_processing(combined_results,config["rank_decimals"],config["ratio_definitions"])
            save_results_to_csv(final_results, CSV_PATH / config["results_file"])

            print(f"Stock evaluation completed and saved to {CSV_PATH / config['results_file']}")
    else:
        print("Could not load configuration. Exiting.")
