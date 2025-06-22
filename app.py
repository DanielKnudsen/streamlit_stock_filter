from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analyzer import StockAnalyzer, IndicatorConfig
import glob
import os
import re

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
def load_latest_data(ticker: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        st.error(f"No data found for ticker {ticker}. Run stock_analyzer.py to generate data.")
        return None
    try:
        # Parse Date as datetime and set as index
        df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
        # Ensure timezone-naive DatetimeIndex
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # Verify index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

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
    
    # Remove ranking calculations
    try:
        analyzer.fetch_data()
        analyzer.calculate_indicators()
        analyzer.save_data()
        st.success("Data fetched and processed successfully.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return
    
    st.sidebar.header("Filters")
    filters = []
    for indicator in analyzer.config.indicators:
        if getattr(indicator, "filter", False):  # Only show if filter: true
            # Gather all last values for this indicator across tickers
            values = []
            for ticker in analyzer.tickers:
                data = load_latest_data(ticker)
                if data is not None and indicator.name in data.columns:
                    series = data[indicator.name].dropna()
                    if not series.empty:
                        values.append(series.iloc[-1])
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
                key=f"slider_{indicator.name}"
            )
            filters.append(Filter(indicator=indicator.name, min_value=slider_min, max_value=slider_max))
            
    # Apply filters (no ranking, just filter on indicator values)
    filtered_tickers = []
    for ticker in analyzer.tickers:
        include = True
        data = load_latest_data(ticker)
        if data is None:
            continue
        for f in filters:
            # Use the last value of the indicator for filtering
            if f.indicator in data.columns:
                value = data[f.indicator].dropna().iloc[-1] if not data[f.indicator].dropna().empty else None
                if value is None or not (f.min_value <= value <= f.max_value):
                    include = False
                    break
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
        data = load_latest_data(ticker)
        if data is not None:
            for ind in analyzer.config.indicators:
                if ind.name in data.columns:
                    row[ind.name] = data[ind.name].dropna().iloc[-1] if not data[ind.name].dropna().empty else None
        table_data.append(row)
    df_table = pd.DataFrame(table_data)

    st.subheader("Filtered Stocks")

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
        data = load_latest_data(selected_ticker)
        if data is not None:
            try:
                pandas_offset = convert_to_pandas_offset(analyzer.config.display_period)
                data = data.last(pandas_offset)
                fig = plot_stock(selected_ticker, data, analyzer.config.indicators, analyzer.config.display_period)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting data for {selected_ticker}: {str(e)}")

if __name__ == "__main__":
    main()