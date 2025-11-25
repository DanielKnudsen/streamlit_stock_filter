import streamlit as st
import numpy as np
import pandas as pd
import pwlf
import plotly.express as px
import plotly.graph_objects as go # Import Plotly

def generate_price_chart(config, CSV_PATH, add_moving_averages, get_display_name, selected_stock_ticker, selected_stock_dict):
    if selected_stock_ticker:
                # Add slider for PWLF
        label = "Antal linjesegment för trendlinje"
        linjesegments =[1, 2, 3, 4, 5]
        num_segments = st.segmented_control(label, linjesegments, selection_mode='single', default=1, key="pwlf_slider")
        label = "Historik (år bakåt i tiden)"
        linjesegments =[4,1, 0.5, 0.25] # years
        historik_segments = st.segmented_control(label, linjesegments, selection_mode='single', default=4, key="historik_segments")
        price_file_path = CSV_PATH / config["price_data_file"]
        if price_file_path.exists():
            df_price_all = pd.read_csv(price_file_path)
            df_price_all['Date'] = pd.to_datetime(df_price_all['Date']) # Convert 'Date' to datetime object before filtering
            df_price = df_price_all[df_price_all['Ticker'] == selected_stock_ticker].copy()
            df_price = add_moving_averages(df_price)
                    # Cap data to historik_segments years back
            if not df_price.empty and historik_segments:
                max_date = df_price['Date'].max()
                        # Use days for fractional years to avoid DateOffset ambiguity
                try:
                    years_float = float(historik_segments)
                    if years_float.is_integer():
                        min_date = max_date - pd.DateOffset(years=int(years_float))
                    else:
                                # Approximate 1 year as 365.25 days
                        days = int(years_float * 365.25)
                        min_date = max_date - pd.DateOffset(days=days)
                except Exception as e:
                    min_date = max_date - pd.DateOffset(years=1)  # fallback to 1 year
                df_price = df_price[df_price['Date'] >= min_date]
                    #df_price = add_moving_averages(df_price)
                    # PWLF calculation
            x_hat = None
            y_hat = None
            std_devs = None
            if len(df_price) > num_segments:
                x = np.arange(len(df_price['Date']))
                y = df_price['Close'].values

                my_pwlf = pwlf.PiecewiseLinFit(x, y)
                        # fit the data for a given number of line segments
                res = my_pwlf.fit(num_segments)
                        # predict for the determined breaks
                x_hat = np.linspace(x.min(), x.max(), 100)
                y_hat = my_pwlf.predict(x_hat)

                        # Calculate standard deviation of residuals (difference between actual and fitted)
                y_fitted = my_pwlf.predict(x)
                residuals = y - y_fitted
                std_devs = [np.std(residuals) * i for i in [1, 2, 3]]

                    # Create Plotly figure
            fig = go.Figure()

                    # Add Close price
            if 'Close' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['Close'],
                            mode='lines', name='Stängningskurs',
                            line=dict(color='blue', width=2)))

                    # Add PWLF trendline to the plot
            if x_hat is not None and y_hat is not None:
                        # Create a new date range for the predicted values
                date_range = pd.to_datetime(np.linspace(df_price['Date'].min().value, df_price['Date'].max().value, len(x_hat)))
                fig.add_trace(go.Scatter(x=date_range, y=y_hat,
                                    mode='lines', name='Trendlinje',
                                    line=dict(color='orange', width=3, dash='dash')))
                        # Add dotted lines for +- 1, 2, 3 standard deviations from trendline
                if std_devs is not None:
                    for i, std in enumerate(std_devs, 1):
                        fig.add_trace(go.Scatter(
                                    x=date_range, y=y_hat + std,
                                    mode='lines',
                                    name=f'+{i}σ',
                                    line=dict(color='gray', width=1, dash='dot'),
                                    showlegend=True
                                ))
                        fig.add_trace(go.Scatter(
                                    x=date_range, y=y_hat - std,
                                    mode='lines',
                                    name=f'-{i}σ',
                                    line=dict(color='gray', width=1, dash='dot'),
                                    showlegend=True
                                ))

                    # Add SMA_short
            if 'SMA_short' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_short'],
                            mode='lines', name=get_display_name('SMA_short'),
                            line=dict(color='red', width=1, dash='dot')))

                    # Add SMA_medium
            if 'SMA_medium' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_medium'],
                            mode='lines', name=get_display_name('SMA_medium'),
                            line=dict(color='green', width=1, dash='dash')))

                    # Add SMA_long
            if 'SMA_long' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_long'],
                            mode='lines', name=get_display_name('SMA_long'),
                            line=dict(color='purple', width=1, dash='longdash')))

                    # Add Volume as a secondary y-axis
            if 'Volume' in df_price.columns:
                fig.add_trace(go.Bar(x=df_price['Date'], y=df_price['Volume'],
                            name='Volym', marker_color='gray', opacity=0.3, yaxis='y2'))

                    # Update layout for the chart
            fig.update_layout(
                        title=f"Pris & Volym för {selected_stock_dict['Name']} ({selected_stock_ticker})",
                        xaxis_title="Datum",
                        yaxis_title="Pris",
                        hovermode=False,  # Disable all hover interactions
                        legend_title="Legend",
                        height=500,
                        yaxis2=dict(title="Volym", overlaying="y", side="right", showgrid=False),
                        legend=dict(
                            x=0.01,
                            y=0.99,
                            xanchor="left",
                            yanchor="top",
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="rgba(0,0,0,0.1)",
                            borderwidth=1
                        )
                    )

                    # Remove hover for all traces
            for trace in fig.data:
                trace.update(hoverinfo="skip", hovertemplate=None)

            st.plotly_chart(fig, config={"displayModeBar": False, "responsive": True})
        else:
            st.warning(f"Prisdatafil saknas: {price_file_path}. Kontrollera att filen finns i mappen '{CSV_PATH}/'.")

def plot_ratio_values(df,mappings):
    """
    Create a bar plot from ratio values dataframe.

    Args:
        df: pandas.DataFrame with 1 row and variable columns (result from get_ratio_values_by_period).
            Expected structure:
            - First 4 columns: annual values (Year YYYY format) (royalblue).
            - Last 2 columns: TTM values (TTM YYYYQX format) (gold).
            - If fewer than 4 annual columns exist, missing years are filled with NaN.
            - If fewer than 2 TTM columns exist, missing TTM periods are filled with NaN.
        mappings: ConfigMappings instance (object providing ratio metadata and period mappings).

    Returns:
        plotly.graph_objects.Figure: Bar plot visualizing ratio values and trends.

    Error Handling:
        - If `df` contains missing or non-numeric values, bars will display 'nan' and trend lines may be omitted.
        - If `mappings.is_higher_better(ratio_name)` returns None, defaults to True.
        - Automatically fills missing year columns with NaN to ensure consistent structure.
    """
    # Get the values and column names
    columns = df.columns.tolist()
    ratio_name = df.index[0]
    
    # Separate year and TTM columns
    year_cols = [col for col in columns if col.startswith('Year ')]
    ttm_cols = [col for col in columns if col.startswith('TTM ')]
    
    # Extract years from year columns and sort them
    year_nums = []
    for col in year_cols:
        try:
            year = int(col.replace('Year ', '').strip())
            year_nums.append((year, col))
        except ValueError:
            pass
    
    year_nums.sort(key=lambda x: x[0])
    years_only = [y[0] for y in year_nums]
    
    # Fill missing years between min and max with NaN
    if years_only:
        min_year = min(years_only)
        max_year = max(years_only)
        all_years = list(range(min_year, max_year + 1))
        
        # Create mapping for existing years
        existing_year_cols = {y[0]: y[1] for y in year_nums}
        
        # Build the complete year column list with NaN placeholders
        complete_year_cols = []
        for year in all_years:
            if year in existing_year_cols:
                complete_year_cols.append(existing_year_cols[year])
            else:
                complete_year_cols.append(f'Year {year}')
        
        # Take only the last 4 years
        complete_year_cols = complete_year_cols[-4:]
    else:
        complete_year_cols = []
    
    # Ensure we have exactly 4 year columns (pad with NaN if needed)
    while len(complete_year_cols) < 4:
        complete_year_cols.insert(0, f'Year {min_year - (4 - len(complete_year_cols))}')
    
    # Ensure we have exactly 2 TTM columns (pad with NaN if needed)
    while len(ttm_cols) < 2:
        ttm_cols.append(f'TTM {len(ttm_cols) + 1}')
    
    ttm_cols = ttm_cols[-2:]  # Take only the last 2
    
    # Reconstruct the dataframe with the complete column set
    ordered_columns = complete_year_cols + ttm_cols
    
    # Build the values array
    values = []
    for col in ordered_columns:
        if col in df.columns:
            val = df.iloc[0][col]
            values.append(float(val) if not pd.isna(val) else np.nan)
        else:
            values.append(np.nan)  # Fill missing columns with NaN
    
    values = np.array(values)
    columns = ordered_columns

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
    
    # Create bar plot with proper handling of NaN values
    bar_text = []
    for v in values:
        if pd.isna(v):
            bar_text.append('N/A')
        else:
            bar_text.append(f"{v:.2f}")
    
    fig = go.Figure(data=[
        go.Bar(
            x=columns,
            y=values,  # Use actual values (Plotly handles NaN)
            marker_color=colors,
            marker_pattern_shape=patterns,
            text=bar_text,
            textposition='auto',
            textfont=dict(size=font_size),
            showlegend=False,
            name=ratio_name
        )
    ])
    
    # Add linear regression line for first 4 bars (skip NaN values)
    x_indices_1 = np.array([0, 1, 2, 3])
    y_values_1 = values[:4]
    
    # Only calculate trend if we have at least 2 non-NaN values
    valid_indices_1 = [i for i, v in enumerate(y_values_1) if not pd.isna(v)]
    if len(valid_indices_1) >= 2:
        x_valid_1 = x_indices_1[valid_indices_1]
        y_valid_1 = y_values_1[valid_indices_1]
        coeffs_1 = np.polyfit(x_valid_1, y_valid_1, 1)
        y_fit_1 = np.polyval(coeffs_1, x_indices_1)
        
        # Determine trend color based on first and last non-NaN values
        annual_diff = y_values_1[valid_indices_1[-1]] - y_values_1[valid_indices_1[0]]
        if (annual_diff > 0 and higher_is_better) or (annual_diff < 0 and not higher_is_better):
            annual_trend_color = trend_color_improving
        else:
            annual_trend_color = trend_color_deterioating
        
        fig.add_trace(go.Scatter(
            x=columns[:4],
            y=y_fit_1,
            mode='lines',
            name='Trend (4-year)',
            line=dict(color=annual_trend_color, width=trend_line_width, dash='dash'),
            opacity=0.6,
            showlegend=False
        ))

    # Add linear regression line for last 2 bars (skip NaN values)
    x_indices_2 = np.array([0, 1])
    y_values_2 = values[4:6]
    
    # Only calculate trend if we have at least 2 non-NaN values
    valid_indices_2 = [i for i, v in enumerate(y_values_2) if not pd.isna(v)]
    if len(valid_indices_2) >= 2:
        x_valid_2 = x_indices_2[valid_indices_2]
        y_valid_2 = y_values_2[valid_indices_2]
        coeffs_2 = np.polyfit(x_valid_2, y_valid_2, 1)
        y_fit_2 = np.polyval(coeffs_2, x_indices_2)
        
        # Determine trend color
        ttm_diff = y_values_2[valid_indices_2[-1]] - y_values_2[valid_indices_2[0]]
        if (ttm_diff > 0 and higher_is_better) or (ttm_diff < 0 and not higher_is_better):
            ttm_trend_color = trend_color_improving
        else:
            ttm_trend_color = trend_color_deterioating

        fig.add_trace(go.Scatter(
            x=columns[4:6],
            y=y_fit_2,
            mode='lines',
            name='Trend (TTM)',
            line=dict(color=ttm_trend_color, width=trend_line_width, dash='dash'),
            opacity=0.6,
            showlegend=False
        ))
        
        # Add annotation with the difference (only if both values are not NaN)
        if not pd.isna(y_values_2[1]) and not pd.isna(y_values_2[0]):
            fig.add_annotation(
                x=columns[5],
                y=y_values_2[1],
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

def generate_scatter_plot(df_scatter_to_use):
    fig = px.scatter(df_scatter_to_use, 
        x='ttm_momentum_clusterRank', 
        y='pct_ch_3_m', 
        #color=df_scatter_to_use['is_sektor'].map({True: 'Sector Average', False: 'Individual Stock'}),
        color=df_scatter_to_use['Lista'],   
        hover_data=['Sektor'],
        hover_name=df_scatter_to_use.index)

    # Add vertical and horizontal lines for sector averages
    sector_data = df_scatter_to_use[df_scatter_to_use['is_sektor']]
    for idx, row in sector_data.iterrows():
        x_val = row['ttm_momentum_clusterRank']
        y_val = row['pct_ch_3_m']

        # Vertical line
        fig.add_vline(x=x_val, line_dash="dash", line_color="red", opacity=0.5)
        # Horizontal line
        fig.add_hline(y=y_val, line_dash="dash", line_color="red", opacity=0.5)

        fig.update_layout(title='TTM Momentum Cluster Rank vs 3-Month Percentage Change',
                        xaxis_title='TTM Momentum Cluster Rank',
                        yaxis_title='3-Month Percentage Change')
    return fig

def create_trend_momentum_plot(get_display_name, scatter_df):
    fig_scatter = px.scatter(
                        scatter_df,
                        x='long_trend_clusterRank',
                        y='ttm_momentum_clusterRank',
                        size='ttm_current_clusterRank',
                        color='Lista',
                        hover_name=scatter_df.index,
                        title='Trend vs Momentum Ranking (Size = Current Rank)',
                        labels={
                            'long_trend_clusterRank': get_display_name('long_trend_clusterRank'),
                            'ttm_momentum_clusterRank': get_display_name('ttm_momentum_clusterRank'),
                            'ttm_current_clusterRank': get_display_name('ttm_current_clusterRank')
                        },
                        size_max=60
                    )
    fig_scatter.update_layout(
                        height=500,
                        margin=dict(l=10, r=10, t=50, b=10),
                        hovermode='closest'
                    )
    return st.plotly_chart(fig_scatter, config={"responsive": True}, key="trend_momentum_scatter")