import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import plotly.express as px


def get_ratio_values_by_period(ticker, ratio, df):
    """
    Get ratio values for a stock across annual and TTM periods.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAK')
        ratio: Ratio/metric name (e.g., 'ROE', 'Total_Revenue')
        df: DataFrame with stock evaluation results (index is ticker)
    
    Returns:
        DataFrame with:
        - One row: the ticker
        - Columns: Last 4 annual values + Last 2 TTM values (6 columns total)
        - Values: Ratio values for each period
    """
    # Find all year columns for this ratio
    year_cols = [col for col in df.columns if col.startswith(ratio + '_year_')]
    year_cols = [col for col in year_cols if not pd.isna(df.loc[ticker, col])]
    year_cols_sorted = sorted(year_cols, key=lambda x: int(x.split('_')[-1]), reverse=False)
    year_cols_last4 = year_cols_sorted[-4:]
    
    # Find all quarter/TTM columns for this ratio
    # Looking for columns with pattern like: {ratio}_quarter_2... or {ratio}_ttm_...
    quarter_cols = [col for col in df.columns if '_quarter_' in col and ratio in col and '2' in col]
    quarter_cols = [col for col in quarter_cols if not pd.isna(df.loc[ticker, col])]
    
    # Sort quarters chronologically
    def quarter_sort_key(col):
        last_part = col.split('_')[-1]  # e.g., "2025Q1"
        if 'Q' in last_part:
            try:
                year_part = last_part.split('Q')[0]
                quarter_part = last_part.split('Q')[1]
                return int(year_part), int(quarter_part)
            except (ValueError, IndexError):
                return (0, 0)
        return (0, 0)
    
    quarter_cols_sorted = sorted(quarter_cols, key=quarter_sort_key, reverse=False)
    quarter_cols_last2 = quarter_cols_sorted[-2:]
    
    # Build the data dictionary
    data = {}
    
    # Add the 4 annual values
    for col in year_cols_last4:
        year = col.split('_')[-1]
        data[f"Year {year}"] = df.loc[ticker, col]
    
    # Add the 2 TTM values
    for col in quarter_cols_last2:
        quarter = col.split('_')[-1]  # e.g., "2025Q1"
        data[f"TTM {quarter}"] = df.loc[ticker, col]
    
    # Convert to DataFrame with one row
    result_df = pd.DataFrame([data])
    result_df.index = [ratio]
    
    return result_df



def get_ratio_ranks_by_period(ticker, ratio, df, mappings):
    """
    Get ratio ranks for a stock across all periods.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAK')
        ratio: Ratio name (e.g., 'ROE', 'Total_Revenue')
        df: DataFrame with stock evaluation results (index is ticker)
        mappings: ConfigMappings instance
    
    Returns:
        DataFrame with:
        - One row: the ratio name
        - Columns: period_types (long_trend, ttm_momentum, ttm_current) with display names
        - Values: Ratio ranks for each period
    """
    # Create a dictionary to store the data
    data = {}
    
    # Iterate over periods
    for period in mappings.period_types:
        # Construct the column name for this ratio and period
        # Assuming pattern like: {ratio}_{period}_ratioRank
        col_name = f"{ratio}_{period}_ratioRank"
        
        # Get the value from the dataframe
        try:
            value = df.loc[ticker, col_name]
            data[period] = value
        except KeyError:
            data[period] = None
    
    # Convert to DataFrame with one row
    result_df = pd.DataFrame([data])
    result_df.index = [ratio]
    
    # Rename columns using display names from config
    col_names = {period: mappings.get_cluster_col_name(period) for period in mappings.period_types}
    result_df.rename(columns=col_names, inplace=True)
    
    return result_df

def get_category_ranks_by_period(ticker, df, mappings):
    """
    Get category ranks for a stock across all periods and categories.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAK')
        df: DataFrame with stock evaluation results (index is ticker)
        mappings: ConfigMappings instance
    
    Returns:
        DataFrame with:
        - Rows: category_base (Kvalitet, Hälsa, Lönsamhet, etc.) + Cluster rank row at the end
        - Columns: period_types (long_trend, ttm_momentum, ttm_current)
        - Values: Category ranks and cluster ranks from the evaluation results
    """
    # Create a dictionary to store the data
    data = {}
    
    # Iterate over periods (columns)
    for period in mappings.period_types:
        period_data = {}
        
        # Iterate over categories (rows)
        for category in mappings.category_bases:
            # Get the column name for this category and period
            col_name = mappings.get_category_rank_column(category, period)
            
            # Get the value from the dataframe
            try:
                value = df.loc[ticker, col_name]
                period_data[category] = value
            except KeyError:
                period_data[category] = None
        
        # Add cluster rank for this period
        cluster_col = mappings.cluster_mappings[period]
        try:
            cluster_value = df.loc[ticker, cluster_col]
            period_data['Agg. Rank'] = cluster_value
        except KeyError:
            period_data['Agg. Rank'] = None
        
        data[period] = period_data
    
    # Convert to DataFrame with categories as rows and periods as columns
    result_df = pd.DataFrame(data)

    # rename columns using mappings.period_descriptions
    col_names = {period: mappings.get_cluster_col_name(period) for period in mappings.period_types}
    result_df.rename(columns=col_names, inplace=True)

    return result_df

def visualize_dataframe_with_progress(color_progress, df_ranking, hide_index=False):
    """
    Display a styled DataFrame in Streamlit with progress bars for ranking columns.

    Parameters:
        color_progress (function): Function to apply background colors based on cell values.
        df_ranking (pd.DataFrame): DataFrame containing ranking values to display.

    Returns:
        None. Renders the styled DataFrame in the Streamlit app.
    """
    st.dataframe(
                    df_ranking
                    .style.map(color_progress),
                    hide_index=hide_index,
                    width="stretch",
                    column_config={
                        col: st.column_config.ProgressColumn(
                            col,
                            help="Rankingvärde (0-100)",
                            min_value=0,
                            max_value=100,
                            format="%.1f",
                            width="small"
                        )
                        for col in df_ranking.columns
                    }
                )