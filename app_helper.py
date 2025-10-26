import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st

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

def plot_ratio_values(df,mappings):
    """
    Create a bar plot from ratio values dataframe.

    Args:
        df: pandas.DataFrame with 1 row and 6 columns (result from get_ratio_values_by_period).
            First 4 columns: annual values (royalblue).
            Last 2 columns: TTM values (gold).
        mappings: ConfigMappings instance (object providing ratio metadata and period mappings).

    Returns:
        plotly.graph_objects.Figure: Bar plot visualizing ratio values and trends.

    Error Handling:
        - If `df` contains missing or non-numeric values, bars will display 'nan' and trend lines may be omitted.
        - If `mappings.is_higher_better(ratio_name)` returns None, defaults to True.
        - Function assumes correct DataFrame shape; malformed input may result in plotting errors.
    """
    # Get the values and column names
    values = df.iloc[0].values.astype(float)
    columns = df.columns.tolist()
    ratio_name = df.index[0]

    # get higher_is_better info from mappings (default to True if None)
    hib = mappings.is_higher_better(ratio_name)
    higher_is_better = hib if hib is not None else True

    # Create colors: first 4 blue, last 2 gold
    colors = ['royalblue'] * 4 + ['gold'] * 2
    #colors = ['#D3D3D3'] * 4 + ['#4A90E2'] * 2
    trend_color_improving = '#70AD47'
    trend_color_deterioating = '#C5504E'
    trend_line_width = 6
    font_size = 18

    # Create patterns: 5th bar has '/' pattern
    patterns = [''] * 4 + ['\\'] + ['']
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=columns,
            y=[f"{v:.3f}" for v in values],#values,
            marker_color=colors,
            marker_pattern_shape=patterns,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            textfont=dict(size=font_size),
            showlegend=False,
            name=ratio_name
        )
    ])
    
    # Add linear regression line for first 4 bars
    x_indices_1 = np.array([0, 1, 2, 3])
    y_values_1 = values[:4]
    coeffs_1 = np.polyfit(x_indices_1, y_values_1, 1)
    y_fit_1 = np.polyval(coeffs_1, x_indices_1)

    annual_diff =  values[3] - values[0]
    if (annual_diff > 0 and higher_is_better) or (annual_diff < 0 and not higher_is_better):
        annual_trend_color = trend_color_improving
    else:
        annual_trend_color = trend_color_deterioating
    
    fig.add_trace(go.Scatter(
        x=columns[:4],
        y=[f"{v:.3f}" for v in y_fit_1],#y_fit_1,
        mode='lines',
        name='Trend (4-year)',
        line=dict(color=annual_trend_color, width=trend_line_width, dash='dash'),
        showlegend=False
    ))

    # Add linear regression line for last 2 bars
    ttm_diff = values[5] - values[4]

    if (ttm_diff > 0 and higher_is_better) or (ttm_diff < 0 and not higher_is_better):
        ttm_trend_color = trend_color_improving
    else:
        ttm_trend_color = trend_color_deterioating

    x_indices_2 = np.array([0, 1])
    y_values_2 = values[4:6]
    coeffs_2 = np.polyfit(x_indices_2, y_values_2, 1)
    y_fit_2 = np.polyval(coeffs_2, x_indices_2)
    
    fig.add_trace(go.Scatter(
        x=columns[4:6],
        y=y_fit_2,
        mode='lines',
        name='Trend (TTM)',
        line=dict(color=ttm_trend_color, width=trend_line_width, dash='dash'),
        showlegend=False
    ))

    # Add annotation with the difference
    fig.add_annotation(
        x=columns[5],
        y=values[5],
        text=f"{ttm_diff:+.2g}",
        font=dict(size=font_size, color=ttm_trend_color),
        yshift=20,
        xshift=5,
        showarrow=False
    )
    
    # Add 'Data missing' annotations for each NaN value
    for i, v in enumerate(values):
        if pd.isna(v):
            fig.add_annotation(
                x=columns[i],
                y=0,
                text="Data missing",
                showarrow=False,
                font=dict(color="#b30000", size=13),
                bgcolor="#ffe5e5",
                bordercolor="#ffcccc",
                borderwidth=1,
                yshift=20
            )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        showlegend=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )
    
    return fig

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
                            format="%.1f"
                        )
                        for col in df_ranking.columns
                    }
                )