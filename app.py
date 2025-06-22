from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analyzer import StockAnalyzer, IndicatorConfig
import glob
import os
import re
from scipy.stats import median_abs_deviation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@dataclass
class Filter:
    indicator: str
    min_value: float
    max_value: float

def convert_to_pandas_offset(period: str) -> str:
    """
    Convert yfinance-style period (e.g., '6mo', '7d', '12y') to Pandas offset (e.g., '6M', '7D', '12Y').
    Handles 'ytd' and 'max' as special cases.
    """
    period = period.lower()
    if period in ("ytd", "max"):
        return period.upper()
    match = re.match(r"(\d+)([a-z]+)", period)
    if match:
        num, unit = match.groups()
        unit_map = {
            "d": "D",
            "w": "W",
            "mo": "M",
            "y": "Y"
        }
        # Find the longest matching unit
        for k in sorted(unit_map, key=len, reverse=True):
            if unit.startswith(k):
                return f"{num}{unit_map[k]}"
    # Fallback: return as is
    return period.upper()


@st.cache_data
def load_all_data(tickers):
    all_data = {}
    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                all_data[ticker] = df
            except Exception as e:
                st.error(f"Error loading data for {ticker}: {str(e)}")
    return all_data

def plot_stock(ticker: str, data: pd.DataFrame, config: List[IndicatorConfig], period: str):
    # Determine which panels are needed
    has_middle = any(getattr(ind, "panel", "price") == "middle" for ind in config)
    has_lower = any(getattr(ind, "panel", "price") == "lower" for ind in config)

    # Decide number of rows
    if has_middle and has_lower:
        rows = 3
        row_heights = [0.5, 0.25, 0.25]
        subplot_titles = ("Price Panel", "Middle Panel", "Lower Panel")
    elif has_middle:
        rows = 2
        row_heights = [0.7, 0.3]
        subplot_titles = ("Price Panel", "Middle Panel")
    elif has_lower:
        rows = 2
        row_heights = [0.7, 0.3]
        subplot_titles = ("Price Panel", "Lower Panel")
    else:
        rows = 1
        row_heights = [1.0]
        subplot_titles = ("Price Panel",)

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )

    # Price panel (main)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Close Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Add indicators to the correct panel
    for indicator in config:
        if indicator.name in data.columns:
            panel = getattr(indicator, "panel", "price")
            trace = go.Scatter(
                x=data.index,
                y=data[indicator.name],
                name=indicator.name
            )
            if panel == "price":
                fig.add_trace(trace, row=1, col=1)
            elif panel == "middle" and has_middle:
                fig.add_trace(trace, row=2 if not has_lower else 2, col=1)
            elif panel == "lower" and has_lower:
                fig.add_trace(trace, row=3 if has_middle and has_lower else 2, col=1)

    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if has_middle:
        fig.update_yaxes(title_text="Middle Indicators", row=2 if not has_lower else 2, col=1)
    if has_lower:
        fig.update_yaxes(title_text="Lower Indicators", row=3 if has_middle and has_lower else 2, col=1)

    fig.update_layout(
        height=900 if has_middle and has_lower else 700,
        legend=dict(orientation="h"),
        xaxis=dict(title="Date"),
        title=f"{ticker} Stock Chart"  # <-- Add this line
    )

    return fig

# Define explanations for each fundamental
FUNDAMENTAL_EXPLANATIONS = {
    "earningsGrowth": "Year-over-year earnings growth rate.",
    "revenueGrowth": "Year-over-year revenue growth rate.",
    "profitMargins": "Net profit as a percentage of revenue.",
    "returnOnAssets": "Net income divided by total assets.",
    "priceToBook": "Share price divided by book value per share.",
    "forwardPE": "Forward price-to-earnings ratio."
}

def remove_outliers(series, n_mad=5):
    median = np.median(series)
    mad = median_abs_deviation(series, nan_policy='omit')
    if mad == 0:
        return series  # Avoid division by zero
    mask = np.abs(series - median) < n_mad * mad
    return series[mask]

def main():
    st.set_page_config(page_title="Stock Analyzer", layout="wide")
    st.title("Stock Technical Analysis Dashboard")
    
    try:
        analyzer = StockAnalyzer("config.yaml")
    except Exception as e:
        st.error(f"Error initializing StockAnalyzer: {str(e)}")
        return
    
    if not analyzer.tickers:
        st.error("No tickers found in tickers.csv.")
        return

    # Load all ticker data into memory ONCE
    all_data = load_all_data(analyzer.tickers)
    
    # Remove ranking calculations
    try:
        analyzer.fetch_data()
        analyzer.calculate_indicators()
        analyzer.fetch_fundamentals() 
        analyzer.save_data()
        st.success("Data fetched and processed successfully.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return
    
    st.sidebar.header("TA Filters")
    filters = []
    for indicator in analyzer.config.indicators:
        if getattr(indicator, "filter", False):
            # Build a list of filters except the current one
            other_filters = [f for f in filters if f.indicator != indicator.name]
            # Find tickers that match all other filters
            tickers_for_boxplot = []
            for ticker in analyzer.tickers:
                data = all_data.get(ticker)
                if data is None:
                    continue
                include = True
                for f in other_filters:
                    if f.indicator in data.columns:
                        series = data[f.indicator].dropna()
                        value = series.iloc[-1] if not series.empty else None
                        if value is not None and not (f.min_value <= value <= f.max_value):
                            include = False
                            break
                    else:
                        include = False
                        break
                if include:
                    tickers_for_boxplot.append(ticker)

            # Now build the values for the boxplot using these tickers
            values = []
            for ticker in tickers_for_boxplot:
                data = all_data.get(ticker)
                if data is not None and indicator.name in data.columns:
                    series = data[indicator.name].dropna()
                    if not series.empty:
                        v = series.iloc[-1]
                        if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v):
                            values.append(v)
            # Remove outliers
            if values:
                values = np.array(values)
                values = remove_outliers(values, n_mad=getattr(analyzer.config, "n_mad", 5))
                values = values.tolist()
            else:
                values = []

            if values:
                min_val = float(min(values))
                max_val = float(max(values))
            else:
                min_val = -100.0
                max_val = 100.0

            slider_min, slider_max = st.sidebar.slider(
                f"{indicator.name} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
                key=f"slider_{indicator.name}",
                help=getattr(indicator, "description", "")
            )

            # --- Add boxplot below the slider ---
            if len(values) > 1:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    x=values,
                    boxpoints='outliers',
                    orientation='h',
                    marker_color='lightblue',
                    name="",
                    hoverinfo="skip"
                ))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=80,
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title=None,
                )
                fig.update_traces(hoverinfo="skip", selector=dict(type="box"))
                st.sidebar.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

            # Add a horizontal rule between indicators
            st.sidebar.markdown("---")

            # Add the filter for this indicator (after the slider)
            filters.append(Filter(indicator=indicator.name, min_value=slider_min, max_value=slider_max))
    
    st.sidebar.header("Fundamental Filters")
    # Fundamental filters
    for field in getattr(analyzer.config, "fundamentals", []):
        # Build a list of filters except the current one
        other_filters = [f for f in filters if f.indicator != field]
        # Find tickers that match all other filters
        tickers_for_boxplot = []
        for ticker in analyzer.tickers:
            data = all_data.get(ticker)
            if data is None:
                continue
            include = True
            for f in other_filters:
                if f.indicator in data.columns:
                    series = data[f.indicator].dropna()
                    value = series.iloc[-1] if not series.empty else None
                    if value is not None and not (f.min_value <= value <= f.max_value):
                        include = False
                        break
                else:
                    include = False
                    break
            if include:
                tickers_for_boxplot.append(ticker)

        # Now build the values for the boxplot using these tickers
        values = []
        for ticker in tickers_for_boxplot:
            data = all_data.get(ticker)
            if data is not None and field in data.columns:
                series = data[field].dropna()
                if not series.empty:
                    v = series.iloc[-1]
                    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v):
                        values.append(v)
        # Remove outliers
        if values:
            values = np.array(values)
            values = remove_outliers(values, n_mad=getattr(analyzer.config, "n_mad", 5))
            values = values.tolist()
        else:
            values = []

        if values:
            min_val = float(min(values))
            max_val = float(max(values))
        else:
            min_val = -100.0
            max_val = 100.0

        slider_min, slider_max = st.sidebar.slider(
            f"{field} range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
            key=f"slider_{field}",
            help=FUNDAMENTAL_EXPLANATIONS.get(field, "")
        )

        # --- Add boxplot below the slider ---
        if len(values) > 1:
            fig = go.Figure()
            fig.add_trace(go.Box(
                x=values,
                boxpoints='outliers',
                orientation='h',
                marker_color='lightblue',
                name="",
                hoverinfo="skip"
            ))
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=80,
                showlegend=False,
                xaxis_title=None,
                yaxis_title=None,
            )
            fig.update_traces(hoverinfo="skip", selector=dict(type="box"))
            st.sidebar.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

        # Add a horizontal rule between indicators
        st.sidebar.markdown("---")

        # Add the filter for this field (after the slider)
        filters.append(Filter(indicator=field, min_value=slider_min, max_value=slider_max))
    
    # Apply filters (no ranking, just filter on indicator values)
    filtered_tickers = []
    for ticker in analyzer.tickers:
        include = True
        data = all_data.get(ticker)
        if data is None:
            continue
        for f in filters:
            # Use the last value of the indicator for filtering
            if f.indicator in data.columns:
                series = data[f.indicator].dropna()
                value = series.iloc[-1] if not series.empty else None
                # Only filter if value is present
                if value is not None and not (f.min_value <= value <= f.max_value):
                    include = False
                    break
                # If value is None, skip filtering for this field
            else:
                include = False
                break
        if include:
            filtered_tickers.append(ticker)
    
    if not filtered_tickers:
        st.warning("No tickers match the filter criteria. Adjust the filters or try again.")
        return

    # Prepare table data (no ranking columns)
    table_data = []
    for ticker in filtered_tickers:
        row = {"Ticker": ticker}
        data = all_data.get(ticker)
        if data is not None:
            # Add indicator values
            for ind in analyzer.config.indicators:
                if ind.name in data.columns:
                    row[ind.name] = data[ind.name].dropna().iloc[-1] if not data[ind.name].dropna().empty else None
            # Add fundamental values
            for field in getattr(analyzer.config, "fundamentals", []):
                if field in data.columns:
                    row[field] = data[field].dropna().iloc[-1] if not data[field].dropna().empty else None
        table_data.append(row)
    df_table = pd.DataFrame(table_data)

    st.subheader(f"Filtered Stocks ({len(filtered_tickers)})")

    # Add a Select column for checkboxes
    df_table["Select"] = False
    df_table = df_table.set_index("Ticker")

    # Use session state to persist selection
    if "selection_df" not in st.session_state or not st.session_state.selection_df.index.equals(df_table.index):
        st.session_state.selection_df = df_table.copy()

    edited_df = st.data_editor(
        st.session_state.selection_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select")
        },
        key="filtered_stocks_editor"
    )

    st.session_state.selection_df = edited_df

    selected_tickers = edited_df[edited_df["Select"]].index.tolist()

    if not selected_tickers:
        st.info("Select at least one stock to visualize.")
        return

    for selected_ticker in selected_tickers:
        data = all_data.get(selected_ticker)
        if data is not None:
            try:
                pandas_offset = convert_to_pandas_offset(analyzer.config.display_period)
                data = data.last(pandas_offset)
                fig = plot_stock(selected_ticker, data, analyzer.config.indicators, analyzer.config.display_period)
                st.plotly_chart(fig, use_container_width=True)

                # Display extra company info
                info = analyzer.fundamentals_data.get(selected_ticker, {})
                for field in getattr(analyzer.config, "extra_fundamental_fields", []):
                    value = info.get(field, "N/A")
                    st.markdown(f"**{field}:** {value}")

                for field in getattr(analyzer.config, "fundamentals", []):
                    # Gather all values for this fundamental (across all tickers)
                    values = []
                    for ticker in filtered_tickers:
                        data = all_data.get(ticker)
                        if data is not None and field in data.columns:
                            series = data[field].dropna()
                            if not series.empty:
                                v = series.iloc[-1]
                                if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v):
                                    values.append(v)
                        # Remove outliers
                        if values:
                            values = np.array(values)
                            values = remove_outliers(values, n_mad=getattr(analyzer.config, "n_mad", 5))
                            values = values.tolist()
                            
                    # Value for the selected stock
                    data_selected = all_data.get(selected_ticker)
                    selected_value = None
                    if data_selected is not None and field in data_selected.columns:
                        series_selected = data_selected[field].dropna()
                        if not series_selected.empty:
                            selected_value = series_selected.iloc[-1]

                    if len(values) > 1 and selected_value is not None:
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            x=values,
                            name=field,
                            boxpoints='outliers',
                            orientation='h',
                            marker_color='lightblue'
                        ))
                        # Highlight the selected stock's value
                        fig.add_trace(go.Scatter(
                            x=[selected_value],
                            y=[field],
                            mode='markers',
                            marker=dict(color='red', size=14, symbol='diamond'),
                            name=f"{selected_ticker}"
                        ))
                        fig.update_layout(
                            title=f"Distribution of {field} (red = {selected_ticker})",
                            xaxis_title=field,
                            showlegend=False,
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif selected_value is not None:
                        st.info(f"Not enough data to show a box plot for {field}.")

            except Exception as e:
                st.error(f"Error plotting data for {selected_ticker}: {str(e)}")


if __name__ == "__main__":
    main()