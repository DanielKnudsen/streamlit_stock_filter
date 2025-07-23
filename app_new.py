import streamlit as st
import pandas as pd
import os
import yaml
import plotly.graph_objects as go # Import Plotly
import plotly.express as px # Import Plotly Express for bubble plot
import numpy as np # For handling numerical operations
from collections import OrderedDict

# Define directories for CSV files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = "csv-data" # Directory for the main CSV (e.g., "stock_evaluation_results.csv")
PRICE_DATA_DIR = "data" # Directory for individual stock price CSV files (e.g., "data/Company A.csv")
CSV_FILE_NAME = "stock_evaluation_results.csv"
full_csv_path = os.path.join(CSV_DATA_DIR, CSV_FILE_NAME)

# --- Streamlit Application ---
st.set_page_config(layout="centered", page_title="Stock Evaluation", page_icon="📈")

st.title("📈 Stock Evaluation Results")
st.markdown("Here is an overview of stock information, filtered to include only columns with 'rank_Score'.")

try:
    # Read the CSV file. index_col=0 will set the first column (Stock) as the index (Ticker).
    df_new_ranks = pd.read_csv(full_csv_path, index_col=0)
    df_long_business_summary = pd.read_csv(os.path.join(CSV_DATA_DIR, "longBusinessSummary.csv"), index_col=0)

    # --- Add hover_summary column for bubble plot ---
    def get_truncated_summary(ticker):
        try:
            summary = str(df_long_business_summary.loc[ticker].values[0])
            return summary[:130] + ("..." if len(summary) > 130 else "")
        except Exception:
            return "No summary available."
    df_new_ranks['hover_summary'] = df_new_ranks.index.map(get_truncated_summary)
    
    # Filter columns that contain the string "catRank" for the main table
    # These columns will be displayed in the main table
    rank_score_columns = [col for col in df_new_ranks.columns if "catRank" in col]
    latest_columns = [col for col in rank_score_columns if "latest" in col.lower()]
    trend_columns = [col for col in rank_score_columns if "trend" in col.lower()]
    rank_score_columns = rank_score_columns + ['Latest_clusterRank', 'Trend_clusterRank']  # Include total scores
    # Initialize a DataFrame that will be filtered by sliders
    # This DataFrame contains all columns from the original file, with Ticker as index
    df_filtered_by_sliders = df_new_ranks.copy()

    # --- Load ranking categories from rank-config.yaml early so category_ratios is available for filter UI ---
    rank_config_path = os.path.join(BASE_DIR, "rank-config.yaml")
    if os.path.exists(rank_config_path):
        with open(rank_config_path, "r") as f:
            config = yaml.safe_load(f)
        category_ratios = config.get("category_ratios", {})
        categories = list(category_ratios.keys())
    else:
        category_ratios = {}
        categories = []

    # --- Unified Filtering Section ---
    st.subheader("Filter Stocks with Sliders")
    st.markdown("Use the sliders below to filter stocks based on total, 'latest', and 'trend' values from 'rank_Score' columns.")
    st.write(df_new_ranks.columns)  # Display the columns for debugging --- IGNORE ---
    # --- Filter by SMA differences ---
    # pct_Close_vs_SMA_short,pct_SMA_short_vs_SMA_medium,pct_SMA_medium_vs_SMA_long
    st.markdown("##### Filter by SMA Differences")
    col_diff_long_medium, col_diff_short_medium, col_diff_price_short = st.columns(3)
    with col_diff_long_medium:
        # Filter by Long-Medium SMA difference
        min_diff_long_medium = float(df_new_ranks['pct_SMA_short_vs_SMA_medium'].min())
        max_diff_long_medium = float(df_new_ranks['pct_SMA_short_vs_SMA_medium'].max())
        diff_long_medium_range = st.slider(
            'Long-Medium SMA Difference',
            min_value=min_diff_long_medium,
            max_value=max_diff_long_medium,
            value=(min_diff_long_medium, max_diff_long_medium),
            step=0.01
        )
    with col_diff_short_medium:
        # Filter by Short-Medium SMA difference
        min_diff_short_medium = float(df_new_ranks['pct_SMA_short_vs_SMA_medium'].min())
        max_diff_short_medium = float(df_new_ranks['pct_SMA_short_vs_SMA_medium'].max())
        diff_short_medium_range = st.slider(
            'Short-Medium SMA Difference',
            min_value=min_diff_short_medium,
            max_value=max_diff_short_medium,
            value=(min_diff_short_medium, max_diff_short_medium),
            step=0.01
        )
    with col_diff_price_short:
        # Filter by Price-Short SMA difference
        min_diff_price_short = float(df_new_ranks['pct_Close_vs_SMA_short'].min())
        max_diff_price_short = float(df_new_ranks['pct_Close_vs_SMA_short'].max())
        diff_price_short_range = st.slider(
            'Price-Short SMA Difference',
            min_value=min_diff_price_short,
            max_value=max_diff_price_short,
            value=(min_diff_price_short, max_diff_price_short),
            step=0.01
        )

    df_filtered_by_sliders = df_filtered_by_sliders[(df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] >= diff_long_medium_range[0]) & (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] <= diff_long_medium_range[1]) &
                                                      (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] >= diff_short_medium_range[0]) & (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] <= diff_short_medium_range[1]) &
                                                      (df_filtered_by_sliders['pct_Close_vs_SMA_short'] >= diff_price_short_range[0]) & (df_filtered_by_sliders['pct_Close_vs_SMA_short'] <= diff_price_short_range[1])]
    # --- Total Score Sliders (on top, now in two columns) ---
    st.markdown('##### Filter by Total Scores')
    col_total_trend, col_total_latest = st.columns(2)
    with col_total_trend:
        min_trend = float(df_new_ranks['Trend_clusterRank'].min())
        max_trend = float(df_new_ranks['Trend_clusterRank'].max())
        trend_range = st.slider(
            'Total Trend Score',
            min_value=min_trend,
            max_value=max_trend,
            value=(min_trend, max_trend),
            step=0.1
        )
    with col_total_latest:
        min_latest = float(df_new_ranks['Latest_clusterRank'].min())
        max_latest = float(df_new_ranks['Latest_clusterRank'].max())
        latest_range = st.slider(
            'Total Latest Score',
            min_value=min_latest,
            max_value=max_latest,
            value=(min_latest, max_latest),
            step=0.1
        )
    
    # Filter df_new_ranks for the rest of the app
    df_filtered_by_sliders = df_filtered_by_sliders[(df_filtered_by_sliders['Trend_clusterRank'] >= trend_range[0]) & (df_filtered_by_sliders['Trend_clusterRank'] <= trend_range[1]) &
                                (df_filtered_by_sliders['Latest_clusterRank'] >= latest_range[0]) & (df_filtered_by_sliders['Latest_clusterRank'] <= latest_range[1])]

    # --- Latest/Trend Sliders (below total sliders) ---
    with st.expander('Filter by Category Ranks', expanded=False):
        col_filter_left, col_filter_right = st.columns(2)
        with col_filter_left:
            st.markdown("###### Filter for category Trend ranks")
            if trend_columns:
                for col in trend_columns:
                    min_val = df_filtered_by_sliders[col].min()
                    max_val = df_filtered_by_sliders[col].max()
                    slider_min = float(min_val)
                    slider_max = float(max_val)
                    if slider_min == slider_max:
                        slider_max += 0.001
                    current_min, current_max = st.slider(
                        f"{col.replace('_trend_catRank', ' trend Rank')}",
                        min_value=slider_min,
                        max_value=slider_max,
                        value=(slider_min, slider_max),
                        key=f"slider_trend_{col}"
                    )
                    df_filtered_by_sliders = df_filtered_by_sliders[
                        (df_filtered_by_sliders[col] >= current_min) &
                        (df_filtered_by_sliders[col] <= current_max)
                    ]
                    category_name = col.replace("catRank", "ratioRank")
                    # Dynamically add tabs for each trend category using ratio names
                    ratio_name = [r for r in category_ratios[category_name]]
                    ratio_name_display = [r.replace("_trend_ratioRank", "") for r in ratio_name] 
                    tab_labels = ['Info'] + ratio_name_display
                    tabs = st.tabs(tab_labels)
                    tabs[0].write(f"Detailed filtering for *ratios* in {category_name.replace('_trend_ratioRank', '')}:")
                    # Add a slider for each ratio tab (from index 1 and upwards) trend_slope
                    for i, r in enumerate(ratio_name):
                        with tabs[i+1]:
                            #st.write(f"r: {r}")  # Debugging line to show current ratio
                            if r in df_filtered_by_sliders.columns:
                                min_val = float(df_filtered_by_sliders[r].min())
                                max_val = float(df_filtered_by_sliders[r].max())
                                if min_val == max_val:
                                    max_val += 0.001
                                slider_min, slider_max = st.slider(
                                    f"Filter {r.replace('_trend_ratioRank', ' trend Rank')} ",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"slider_tab_trend_{category_name}_{r}"
                                )
                                df_filtered_by_sliders = df_filtered_by_sliders[
                                    (df_filtered_by_sliders[r] >= slider_min) &
                                    (df_filtered_by_sliders[r] <= slider_max)
                                ]
                            else:
                                st.info(f"Column {r} not found in data.")
                            r_data = f"{r.replace('_trend_ratioRank', '_ratio_trendSlope')}"
                            if r_data in df_filtered_by_sliders.columns:
                                min_val = float(df_filtered_by_sliders[r_data].min())
                                max_val = float(df_filtered_by_sliders[r_data].max())
                                if min_val == max_val:
                                    max_val += 0.001
                                slider_min, slider_max = st.slider(
                                    f"Filter {r_data.replace('_ratio_trendSlope', ' trend Slope')}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"slider_tab_latest_{r_data}"
                                )
                                df_filtered_by_sliders = df_filtered_by_sliders[
                                    (df_filtered_by_sliders[r_data] >= slider_min) &
                                    (df_filtered_by_sliders[r_data] <= slider_max)
                                ]
                            else:
                                st.info(f"Column {r_data} not found in data.")
                    st.markdown("---")

            else:
                st.info("No 'trend' columns found among 'rank_Score' columns for filtering.")
        with col_filter_right:
            st.markdown("###### Filter for category Latest ranks")
            if latest_columns:
                for col in latest_columns:
                    min_val = df_filtered_by_sliders[col].min()
                    max_val = df_filtered_by_sliders[col].max()
                    slider_min = float(min_val)
                    slider_max = float(max_val)
                    if slider_min == slider_max:
                        slider_max += 0.001
                    current_min, current_max = st.slider(
                        f"{col.replace('_latest_catRank', ' latest Rank')}",
                        min_value=slider_min,
                        max_value=slider_max,
                        value=(slider_min, slider_max),
                        key=f"slider_latest_{col}"
                    )
                    df_filtered_by_sliders = df_filtered_by_sliders[
                        (df_filtered_by_sliders[col] >= current_min) &
                        (df_filtered_by_sliders[col] <= current_max)
                    ]
                    category_name = col.replace("catRank", "ratioRank")
                    # Dynamically add tabs for each latest category using ratio names
                    ratio_name = [r for r in category_ratios[category_name]]
                    ratio_name_display = [r.replace("_latest_ratioRank", "") for r in ratio_name] 
                    tab_labels = ['Info'] + ratio_name_display
                    tabs = st.tabs(tab_labels)
                    tabs[0].write(f"Detailed filtering for *ratios* in {category_name.replace('_latest_ratioRank', '')}:")
                    # Add a slider for each ratio tab (from index 1 and upwards)
                    for i, r in enumerate(ratio_name):
                        with tabs[i+1]:
                            if r in df_filtered_by_sliders.columns:
                                min_val = float(df_filtered_by_sliders[r].min())
                                max_val = float(df_filtered_by_sliders[r].max())
                                if min_val == max_val:
                                    max_val += 0.001
                                slider_min, slider_max = st.slider(
                                    f"Filter {r.replace('_latest_ratioRank', ' latest Rank')} ",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"slider_tab_latest_{category_name}_{r}"
                                )
                                df_filtered_by_sliders = df_filtered_by_sliders[
                                    (df_filtered_by_sliders[r] >= slider_min) &
                                    (df_filtered_by_sliders[r] <= slider_max)
                                ]
                            else:
                                st.info(f"Column {r} not found in data.")
                            r_data = f"{r.replace('_latest_ratioRank', '_ratio_latest')}"
                            if r_data in df_filtered_by_sliders.columns:
                                min_val = float(df_filtered_by_sliders[r_data].min())
                                max_val = float(df_filtered_by_sliders[r_data].max())
                                if min_val == max_val:
                                    max_val += 0.001
                                slider_min, slider_max = st.slider(
                                    f"Filter {r_data.replace('_ratio_latest', ' latest Value')}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"slider_tab_latest_{r_data}"
                                )
                                df_filtered_by_sliders = df_filtered_by_sliders[
                                    (df_filtered_by_sliders[r_data] >= slider_min) &
                                    (df_filtered_by_sliders[r_data] <= slider_max)
                                ]
                            else:
                                st.info(f"Column {r_data} not found in data.")
                    st.markdown("---")
            else:
                st.info("No 'latest' columns found among 'rank_Score' columns for filtering.")
    # --- Category Score Sliders: One expander per category (no nesting) ---
    for cat in categories:
        with st.expander(f"Filter by Category: {cat}", expanded=False):
            pass

    # --- Bubble Plot: Total_Trend_Score vs Total_Latest_Score (filtered) ---
    col_ticker, col_hover = st.columns(2)
    with col_ticker:
        show_tickers = st.toggle('Show tickers in bubble plot', value=True)
    with col_hover:
        show_hover = st.toggle('Show full info in hover in bubble plot', value=True)

    # --- Lista toggles for bubble plot ---
    lista_values = []
    if 'Lista' in df_filtered_by_sliders.columns:
        lista_values = df_filtered_by_sliders['Lista'].dropna().unique().tolist()
        lista_values = lista_values[:5]  # Limit to 5 unique values
        col_lista = st.columns(len(lista_values)) if lista_values else []
        lista_selected = []
        for idx, lista in enumerate(lista_values):
            with col_lista[idx]:
                show_lista = st.toggle(f"Show {lista}", value=True, key=f"toggle_lista_{lista}")
                if show_lista:
                    lista_selected.append(lista)
        # Filter df_filtered_by_sliders by selected Lista values
        if lista_selected:
            df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders['Lista'].isin(lista_selected)]
        else:
            df_filtered_by_sliders = df_filtered_by_sliders.iloc[0:0]  # Show nothing if none selected
    
    # Format marketCap for hover (MSEK, rounded, with space as thousands separator)
    if 'marketCap' in df_filtered_by_sliders.columns:
        df_filtered_by_sliders['marketCap_MSEK'] = (df_filtered_by_sliders['marketCap'] / 1_000_000).round().astype(int).map(lambda x: f"{x:,}".replace(",", " ") + " MSEK")
    if len(df_filtered_by_sliders) > 0:
        # Assign fixed colors to Lista values using all possible values from the full dataset
        lista_color_map = {
            'Large Cap': '#1f77b4',
            'Mid Cap': '#ff7f0e',
            'Small Cap': '#2ca02c',
            'First North': '#d62728',
            'Other': '#9467bd'
        }
        if 'Lista' in df_new_ranks.columns:
            # Get all unique Lista values from the full dataset for stable color mapping
            all_lista = df_new_ranks['Lista'].dropna().unique().tolist()
            plotly_colors = px.colors.qualitative.Plotly
            for i, lista in enumerate(all_lista):
                if lista not in lista_color_map:
                    lista_color_map[lista] = plotly_colors[i % len(plotly_colors)]
            # Use the full color map, but only show legend for filtered values
            color_discrete_map = {k: v for k, v in lista_color_map.items()}
        else:
            color_discrete_map = None
        # Define size_raw and size for bubble plot
        if 'marketCap' in df_filtered_by_sliders.columns:
            size_raw = df_filtered_by_sliders['marketCap'].copy()
        else:
            size = [20] * len(df_filtered_by_sliders)
        bubble_fig = px.scatter(
            df_filtered_by_sliders,
            x='Trend_clusterRank',
            y='Latest_clusterRank',
            color='Lista' if 'Lista' in df_filtered_by_sliders.columns else None,
            color_discrete_map=color_discrete_map,
            hover_name=df_filtered_by_sliders.index if show_tickers else None,
            text=df_filtered_by_sliders.index if show_tickers else None,
            size=size_raw if 'marketCap' in df_filtered_by_sliders.columns else size,
            hover_data={
                "hover_summary": True,
                "marketCap_MSEK": True
            } if show_hover else {},
            labels={
            'Trend_clusterRank': 'Total Trend Score',
            'Latest_clusterRank': 'Total Latest Score',
            'Lista': '',
            'hover_summary': 'Summary',
            'marketCap_MSEK': 'Market Cap'
            },
            title='Total Trend Score vs Total Latest Score',
            width=900,
            height=600
        )
        bubble_fig.update_layout(
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
            )
        )
        bubble_fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        if show_tickers:
            bubble_fig.update_traces(textposition='top center')
        st.plotly_chart(bubble_fig, use_container_width=True)
    else:
        st.info('No stocks in the selected score range.')

    # Create a DataFrame for display in the main table.
    # This DataFrame is now based on the slider-filtered data
    # and contains only 'rank_Score' columns, keeping the Ticker as index.
    df_display = df_filtered_by_sliders[rank_score_columns].copy() # Keep index

    # Rename the rank_Score columns for display
    # Create a dictionary for renaming
    rename_mapping = {col: col.replace("_rank_Score", "") for col in rank_score_columns}
    df_display.rename(columns=rename_mapping, inplace=True)

    # Update rank_score_columns to reflect the new names for shortlist display
    display_rank_score_columns = [col.replace("_rank_Score", "") for col in rank_score_columns]


    # Add a "Välj" column for plotting the graph
    # Initialize all checkboxes to False
    df_display['Välj'] = False

    # Add a "Shortlist" column to save stocks
    # Initialize all checkboxes to False
    df_display['Shortlist'] = False

    # Get the number of stocks after filtering by sliders
    num_filtered_stocks = len(df_display)
    st.subheader(f"Filtered Stock Information ({num_filtered_stocks} aktier)")
    st.info("Check the box under 'Välj' to display stock data. Check the box under 'Shortlist' to add the stock to your shortlist.")

    # Use st.data_editor to display the table with interactive checkboxes
    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        hide_index=False, # Set to False to always show the index (Ticker)
        column_config={
            # No need to configure "Ticker" as it's now the index
            "Välj": st.column_config.CheckboxColumn(
                "Välj", # Header for the checkbox column to plot
                help="Select a stock to display its price development",
                default=False,
                width="small"
            ),
            "Shortlist": st.column_config.CheckboxColumn(
                "Shortlist", # Header for the checkbox column for shortlist
                help="Add the stock to your personal shortlist",
                default=False,
                width="small"
            )
        },
        key="stock_selection_editor" # Unique key to manage state
    )

    # Logic to handle checkbox selection for plotting
    selected_rows_plot = edited_df[edited_df['Välj']]

    # Ensure only one stock can be selected at a time for plotting.
    if len(selected_rows_plot) > 1:
        st.warning("Only one stock can be selected at a time for price development. Displaying graph for the first selected stock.")
        selected_stock_ticker = selected_rows_plot.index[0] # Access Ticker from index
    elif len(selected_rows_plot) == 1:
        selected_stock_ticker = selected_rows_plot.index[0] # Access Ticker from index
    else:
        selected_stock_ticker = None # No stock selected for plotting

    # Logic to handle Shortlist
    shortlisted_stocks = edited_df[edited_df['Shortlist']]

    st.markdown("---")
    st.subheader("Your Shortlist")

    if not shortlisted_stocks.empty:
        # Display only Ticker (index) and the renamed rank_Score columns for shortlist
        st.dataframe(
            shortlisted_stocks[display_rank_score_columns], # Ticker is already the index
            hide_index=False, # Show the index (Ticker) for the shortlist as well
            use_container_width=True
        )
    else:
        st.info("Your shortlist is empty. Check the box under 'Shortlist' to add stocks.")

    st.markdown("---")
    st.subheader("Price Development")

    if selected_stock_ticker:
        price_file_path = os.path.join(CSV_DATA_DIR, "price_data.csv")
        if os.path.exists(price_file_path):
            df_price_all = pd.read_csv(price_file_path)
            df_price = df_price_all[df_price_all['Ticker'] == selected_stock_ticker].copy()
            df_price['Date'] = pd.to_datetime(df_price['Date']) # Convert 'Date' to datetime object

            # Create Plotly figure
            fig = go.Figure()

            # Add Close price
            if 'Close' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['Close'],
                         mode='lines', name='Close Price',
                         line=dict(color='blue', width=2)))

            # Add SMA_short
            if 'SMA_short' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_short'],
                         mode='lines', name='SMA Short',
                         line=dict(color='red', width=1, dash='dot')))

            # Add SMA_medium
            if 'SMA_medium' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_medium'],
                         mode='lines', name='SMA Medium',
                         line=dict(color='green', width=1, dash='dash')))

            # Add SMA_long
            if 'SMA_long' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_long'],
                         mode='lines', name='SMA Long',
                         line=dict(color='purple', width=1, dash='longdash')))

            # Add Volume as a secondary y-axis
            if 'Volume' in df_price.columns:
                fig.add_trace(go.Bar(x=df_price['Date'], y=df_price['Volume'],
                         name='Volume', marker_color='gray', opacity=0.3, yaxis='y2'))

            # Update layout for the chart
            fig.update_layout(
            title=f"Price & Volume for {selected_stock_ticker}",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            legend_title="Legend",
            height=500,
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Price data file not found: {price_file_path}. Check that the file exists in the '{CSV_DATA_DIR}/' folder.")

    else:
        st.info("Check a box under 'Välj' in the table above to display price development.")

    # Bar plot for all pct_ columns for selected_stock_ticker
    pct_cols = [col for col in df_new_ranks.columns if col.startswith('pct_')]
    if pct_cols and selected_stock_ticker:
        pct_values = df_new_ranks.loc[selected_stock_ticker, pct_cols].astype(float)
        fig_pct = go.Figure(go.Bar(
            x=pct_cols,
            y=pct_values,
            marker_color='royalblue',
            text=[f"{v:.2f}" for v in pct_values],
            textposition='auto',
        ))
        fig_pct.update_layout(
            title=f"Percentage Metrics for {selected_stock_ticker}",
            xaxis_title="Metric",
            yaxis_title="Percentage",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pct, use_container_width=True, key=f"pct_bar_{selected_stock_ticker}")
    
    if selected_stock_ticker is not None:
        longBusinessSummary = df_long_business_summary.loc[selected_stock_ticker]
        st.subheader(f"{selected_stock_ticker} - Business Summary")
        st.write(longBusinessSummary.values[0] if not longBusinessSummary.empty else "No long business summary available for this stock.")

    # get the ranking for each category of the selected stock
    st.subheader("Ranking for Selected Stock")
    if selected_stock_ticker and not df_filtered_by_sliders.empty and categories:
        # Prepare columns for display
        col_left, col_right = st.columns(2)
        # Find all latest and trend columns for the selected stock
        latest_rankings = {}
        trend_rankings = {}
        # Find all columns containing "_catRank"
        all_rank_score_columns = [col for col in df_new_ranks.columns if "_catRank" in col]
        # For each category, try to find the correct "latest" and "trend" columns
        for cat in all_rank_score_columns:
            # Find the latest column for this category
            latest_col = next((col for col in all_rank_score_columns if cat in col and "latest" in col.lower()), None)
            trend_col = next((col for col in all_rank_score_columns if cat in col and "trend" in col.lower()), None)
            if latest_col and latest_col in df_new_ranks.columns:
                latest_rankings[cat] = df_new_ranks.loc[selected_stock_ticker, latest_col]
            if trend_col and trend_col in df_new_ranks.columns:
                trend_rankings[cat] = df_new_ranks.loc[selected_stock_ticker, trend_col]
            if latest_col in df_new_ranks.columns:
                latest_rankings[cat] = df_new_ranks.loc[selected_stock_ticker, latest_col]
            if trend_col in df_new_ranks.columns:
                trend_rankings[cat] = df_new_ranks.loc[selected_stock_ticker, trend_col]
        with col_left:
            st.markdown(f"##### Latest Rankings for {selected_stock_ticker}")
            if latest_rankings:
                # Radar plot for latest rankings
                radar_categories = list(latest_rankings.keys())
                radar_values = [float(latest_rankings[cat]) for cat in radar_categories]
                # Close the radar loop
                radar_categories += [radar_categories[0]]
                radar_values += [radar_values[0]]
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=[f"<b style='font-size:1.2em'>{cat.replace('_latest_catRank', '').replace('_latest_catRank', '').replace('_', ' ').title()}</b>" for cat in radar_categories],
                    fill='toself',
                    name='Latest Rankings',
                    line=dict(color='blue'),
                    mode='lines+markers+text',
                    text=[f"<b style='font-size:1.3em'>{v:.1f}</b>" for v in radar_values],
                    textposition='middle center',
                    textfont=dict(size=18, color='black'),
                    hoverinfo='all',
                    hovertemplate='%{theta}: %{r:.1f}'
                ))
                

                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            showline=False,
                            showticklabels=True,
                            ticks='',
                            layer='below traces',
                            gridcolor='rgba(0,0,0,0.08)'
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=14)
                        )
                    ),
                    showlegend=False,
                    title="Latest Rankings Radar"
                )
                st.plotly_chart(radar_fig, use_container_width=True, key="radar_latest")
            else:
                st.info("No 'latest' rankings found for this stock.")

        with col_right:
            st.markdown(f"##### Trend Rankings for {selected_stock_ticker}")
            if trend_rankings:
                # Radar plot for trend rankings
                radar_categories = list(trend_rankings.keys())
                radar_values = [float(trend_rankings[cat]) for cat in radar_categories]
                # Close the radar loop
                radar_categories += [radar_categories[0]]
                radar_values += [radar_values[0]]
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=[f"<b style='font-size:1.2em'>{cat.replace('_trend_catRank', '').replace('_trend_catRank', '').replace('_', ' ').title()}</b>" for cat in radar_categories],
                    fill='toself',
                    name='Trend Rankings',
                    line=dict(color='#888888'),
                    mode='lines+markers+text',
                    text=[f"<b style='font-size:1.3em'>{v:.1f}</b>" for v in radar_values],
                    textposition='middle center',
                    textfont=dict(size=18, color='black'),
                    hoverinfo='all',
                    hovertemplate='%{theta}: %{r:.1f}'
                ))
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            showline=False,
                            showticklabels=True,
                            ticks='',
                            layer='below traces',
                            gridcolor='rgba(0,0,0,0.08)'
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=14)
                        )
                    ),
                    showlegend=False,
                    title="Trend Rankings Radar"
                )
                st.plotly_chart(radar_fig, use_container_width=True, key="radar_trend")
            else:
                st.info("No 'trend' rankings found for this stock.")

        # --- BEGIN: Show ratio bar charts for each _trend_rank category, all in one row per category ---
        st.markdown('---')
        st.subheader('Trend Ratio Breakdown (Last 4 Years)')
        # Load help texts from config if available
        ratio_help_texts = config.get('ratio_help_texts', {}) if 'config' in locals() or 'config' in globals() else {}
        #st.write("category_ratios:", category_ratios.items())  # Debugging line to show category_ratios
        for cat, cat_dict in category_ratios.items():
            if cat.endswith('trend_ratioRank'):
                display_cat = cat.replace('_trend_ratioRank', '')
                # Use a visually distinct box for each category, with extra margin for spacing
                with st.container():
                    st.markdown(f"<div style='background-color:#f5f7fa; border-radius:10px; padding:18px 10px 10px 10px; margin-top:38px; margin-bottom:38px; border:1px solid #e0e0e0;'><span style='font-size:1.2em; font-weight:bold'>{display_cat}</span></div>", unsafe_allow_html=True)
                    ratios = [ratio for ratio in cat_dict]
                    cols = st.columns(len(ratios)) if ratios else []
                    for idx, ratio in enumerate(ratios):
                        base_ratio = ratio.replace('_trend_ratioRank', '')
                        year_cols = [col for col in df_new_ranks.columns if col.startswith(base_ratio + '_year_')]
                        # Filter out columns where the value for the selected stock is NaN
                        year_cols = [col for col in year_cols if not pd.isna(df_new_ranks.loc[selected_stock_ticker, col])]
                        year_cols_sorted = sorted(year_cols, key=lambda x: int(x.split('_')[-1]), reverse=False)
                        year_cols_last4 = year_cols_sorted[-4:]
                        latest_rank_col = f"{base_ratio}_latest_ratioRank"
                        trend_rank_col = f"{base_ratio}_trend_ratioRank"
                        with cols[idx]:
                            if year_cols_last4:
                                values = df_new_ranks.loc[selected_stock_ticker, year_cols_last4].values.astype(float)
                                years = [int(col.split('_')[-1]) for col in year_cols_last4]
                                # Linear regression for trend line
                                if len(years) > 1:
                                    coeffs = np.polyfit(years, values, 1)
                                    trend_vals = np.polyval(coeffs, years)
                                else:
                                    trend_vals = values
                                fig = go.Figure()
                                colors = ['lightblue'] * (len(years) - 1) + ['royalblue']
                                fig.add_trace(go.Bar(x=years, y=values, marker_color=colors, name=base_ratio, showlegend=False))
                                fig.add_trace(go.Scatter(
                                    x=years, 
                                    y=trend_vals, 
                                    mode='lines', 
                                    name='Trend',
                                    line=dict(color='#888888', dash='dot', width=6),  # Medium-dark gray
                                    showlegend=False
                                ))
                                fig.update_layout(title=f"{base_ratio}", 
                                                  height=250, 
                                                  margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                                st.plotly_chart(fig, use_container_width=True, key=f"{cat}_{base_ratio}_bar")
                                latest_rank = df_new_ranks.loc[selected_stock_ticker, latest_rank_col] if latest_rank_col in df_new_ranks.columns else 'N/A'
                                trend_rank = df_new_ranks.loc[selected_stock_ticker, trend_rank_col] if trend_rank_col in df_new_ranks.columns else 'N/A'
                                # Bullet plots for the two ranks in two columns: trend (left), latest (right)

                                bullet_col_trend, bullet_col_latest = st.columns(2)
                                with bullet_col_trend:
                                    if trend_rank != 'N/A':
                                        st.metric(label='Trend Rank',value=f"{trend_rank:.1f}", delta=None, delta_color="normal", border=True)
                                        fig_trend = go.Figure(go.Indicator(
                                            mode="gauge",
                                            value=float(trend_rank),
                                            domain={'x': [0, 1 if len(ratios)==2 else 0.7], 'y': [0, 1]},
                                            gauge={
                                                'shape': "bullet",
                                                'axis': {'range': [0, 100]},
                                                'bar': {'color': "#888888"},
                                                'steps': [
                                                    {'range': [0, 20], 'color': '#ffcccc'},        # Light Red
                                                    {'range': [20, 40], 'color': '#ffe5cc'},       # Light Orange
                                                    {'range': [40, 60], 'color': '#ffffcc'},       # Light Yellow
                                                    {'range': [60, 80], 'color': '#e6ffe6'},       # Very Light Green
                                                    {'range': [80, 100], 'color': '#ccffcc'}       # Light Green
                                                ]
                                            }
                                        ))
                                        fig_trend.update_layout(height=70, margin=dict(l=5, r=5, t=10, b=10))
                                        st.plotly_chart(fig_trend, use_container_width=True, key=f"{cat}_{base_ratio}_trend_bullet_{selected_stock_ticker}")
                                        
                                with bullet_col_latest:
                                    if latest_rank != 'N/A':
                                        st.metric(label='Latest Rank',value=f"{latest_rank:.1f}", delta=None, delta_color="normal", border=True)
                                        fig_latest = go.Figure(go.Indicator(
                                            mode="gauge",  # Remove "number" to hide the value
                                            value=float(latest_rank),
                                            domain={'x': [0, 1 if len(ratios)==2 else 0.7], 'y': [0, 1]},
                                            gauge={
                                                'shape': "bullet",
                                                'axis': {'range': [0, 100]},
                                                'bar': {'color': "royalblue"},
                                                'steps': [
                                                    {'range': [0, 20], 'color': '#ffcccc'},        # Light Red
                                                    {'range': [20, 40], 'color': '#ffe5cc'},       # Light Orange
                                                    {'range': [40, 60], 'color': '#ffffcc'},       # Light Yellow
                                                    {'range': [60, 80], 'color': '#e6ffe6'},       # Very Light Green
                                                    {'range': [80, 100], 'color': '#ccffcc'}       # Light Green
                                                ]
                                            }
                                        ))
                                        fig_latest.update_layout(height=70, margin=dict(l=5, r=5, t=10, b=10))
                                        st.plotly_chart(fig_latest, use_container_width=True, key=f"{cat}_{base_ratio}_latest_bullet_{selected_stock_ticker}")
                                # Show help text for this ratio if available
                                # Show help text if available
                                help_key = f"{base_ratio}_latest_rank"# if base_ratio in ratio_help_texts else f"{ratio}" if 'ratio' in locals() and ratio in ratio_help_texts else None
                                if help_key and ratio_help_texts.get(help_key):
                                    with st.expander(f"Förklaring av {base_ratio}", expanded=False):
                                        st.write(ratio_help_texts[help_key])
                            else:
                                st.info(f"No year data found for {base_ratio}.")
        # --- END: Show ratio bar charts for each _trend_rank category ---

except FileNotFoundError:
    st.error(f"Error: Main file '{CSV_FILE_NAME}' not found in directory '{CSV_DATA_DIR}'. Check the path.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.subheader("About this application")
st.info("To run this app locally: Save the code as a .py file (e.g., `app.py`) and run `streamlit run app.py` in your terminal.")
st.caption("Make sure your CSV files are in the specified folders (`csv-data` for the main file and `data` for the price files).")

