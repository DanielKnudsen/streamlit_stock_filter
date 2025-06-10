from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from stock_analyzer import StockAnalyzer, IndicatorConfig
import glob
import os

@dataclass
class Filter:
    indicator: str
    min_value: float
    max_value: float

def convert_to_pandas_offset(period: str) -> str:
    """Convert yfinance-style period (e.g., '6mo') to Pandas offset (e.g., '6M')."""
    period_map = {
        "1d": "1D",
        "5d": "5D",
        "1mo": "1M",
        "3mo": "3M",
        "6mo": "6M",
        "1y": "1Y",
        "2y": "2Y",
        "5y": "5Y",
        "10y": "10Y",
        "ytd": "YTD",
        "max": "max"
    }
    return period_map.get(period, period)

@st.cache_data
def load_latest_data(ticker: str) -> pd.DataFrame:
    files = glob.glob(f"data/{ticker}_*.csv")
    if not files:
        st.error(f"No data found for ticker {ticker}. Run stock_analyzer.py to generate data.")
        return None
    latest_file = max(files, key=os.path.getctime)
    try:
        # Parse Date as datetime and set as index
        df = pd.read_csv(latest_file, index_col='Date', parse_dates=['Date'])
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Close Price',
        line=dict(color='blue')
    ))
    
    for indicator in config:
        if indicator.name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[indicator.name],
                name=indicator.name,
                yaxis='y2' if indicator.type in ['rsi', 'macd'] else 'y'
            ))
    
    fig.update_layout(
        title=f"{ticker} Stock Data",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Indicator', overlaying='y', side='right'),
        showlegend=True
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
    
    # Ensure analyzer has loaded tickers
    if not analyzer.tickers:
        st.error("No tickers found in tickers.csv.")
        return
    
    # Run data fetch and calculations if not already done
    if not analyzer.ranking:
        try:
            analyzer.fetch_data()
            analyzer.calculate_indicators()
            analyzer.calculate_ranking()
            analyzer.save_data()
            st.success("Data fetched and processed successfully.")
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return
    
    # Filters
    st.sidebar.header("Filters")
    filters = []
    for indicator in analyzer.config.indicators:
        if indicator.rank:
            min_val = st.sidebar.number_input(
                f"Min {indicator.name}", value=-100.0, step=0.1, key=f"min_{indicator.name}"
            )
            max_val = st.sidebar.number_input(
                f"Max {indicator.name}", value=100.0, step=0.1, key=f"max_{indicator.name}"
            )
            filters.append(Filter(indicator=indicator.name, min_value=min_val, max_value=max_val))
            
    # Apply filters
    filtered_tickers = []
    for ticker in analyzer.tickers:
        include = True
        for f in filters:
            rank = analyzer.ranking.get(f.indicator, {}).get(ticker, 0)
            if not (f.min_value <= rank <= f.max_value):
                include = False
                break
        if include:
            filtered_tickers.append(ticker)
    
    if not filtered_tickers:
        st.warning("No tickers match the filter criteria. Adjust the filters or try again.")
        return
    
    # Ticker selection
    selected_ticker = st.selectbox("Select Ticker", filtered_tickers)
    
    # Plot
    if selected_ticker:
        data = load_latest_data(selected_ticker)
        if data is not None:
            try:
                # Convert display_period to Pandas offset
                pandas_offset = convert_to_pandas_offset(analyzer.config.display_period)
                data = data.last(pandas_offset)
                fig = plot_stock(selected_ticker, data, analyzer.config.indicators, 
                                analyzer.config.display_period)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display rankings
                st.subheader("Rankings")
                rank_data = {ind.name: analyzer.ranking.get(ind.name, {}).get(selected_ticker, 0)
                            for ind in analyzer.config.indicators if ind.rank}
                st.table(pd.DataFrame([rank_data], index=[selected_ticker]))
            except Exception as e:
                st.error(f"Error plotting data for {selected_ticker}: {str(e)}")

if __name__ == "__main__":
    main()