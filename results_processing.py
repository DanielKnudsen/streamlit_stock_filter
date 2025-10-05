import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from data_io import load_csv, save_csv
from config_utils import CSV_PATH, config

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
