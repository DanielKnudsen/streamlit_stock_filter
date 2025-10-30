import numpy as np
import pandas as pd
from typing import Any, Dict, List
from data_io import load_csv, save_csv

def calculate_agr_for_ticker(csv_path: str, tickers: List[str], dimensions: List[str], period_type: str = "year") -> Dict[str, Dict[str, Any]]:
    """
    Calculate average growth rate (AGR) for each ticker and metric.

    Args:
        csv_path (str): Path to the input CSV file.
        tickers (List[str]): List of tickers.
        dimensions (List[str]): List of metrics to calculate AGR for.
        period_type (str, optional): Type of period ("year" or "quarterly"). Defaults to "year".

    Returns:
        Dict[str, Dict[str, Any]]: AGR results per ticker and metric.
    """
    df = load_csv(csv_path, parse_dates=['Date'])
    if period_type == 'quarterly':
        # period = f"{pd.to_datetime(idx).year}Q{pd.to_datetime(idx).quarter}"
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('Q').astype(str)
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
        # period = pd.to_datetime(idx).year

    #df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
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
            dim_key = dim.replace(" ", "_") + f"_{period_type}_AvgGrowth"
            ticker_agr[dim_key] = agr
            dim_data_key = dim.replace(" ", "_")
            for year, value in zip(years, values):
                year_str = str(year)
                ticker_agr[f"{dim_data_key}_{period_type}_{year_str}"] = value
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

def save_agr_results_melted_to_csv(agr_results: Dict[str, Dict[str, Any]], csv_file_path: str) -> None:
    """
    Save melted DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        csv_file_path (str): Path to the output CSV file.
    """
    df = pd.DataFrame.from_dict(agr_results, orient='index')
    for col in df.columns:
        if col.endswith('_AvgGrowth'):
            ranks = df[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            rank_col = col.replace('_AvgGrowth', '_AvgGrowth_Rank')
            df[rank_col] = ranks
    df = df.reset_index().melt(id_vars='index', var_name='Growth_Metric', value_name='Value')
    df = df.rename(columns={'index': 'Ticker'})
    df.sort_values(by=['Ticker', 'Growth_Metric'], inplace=True)
    # drop rows with NaN values in 'Value' column
    df = df.dropna(subset=['Value'])
    save_csv(df, csv_file_path, index=False)
    print(f"Melted data sparade till {csv_file_path}")