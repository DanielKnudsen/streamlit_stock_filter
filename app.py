from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from stock_analyzer import StockAnalyzer, IndicatorConfig
import glob
import os
from datetime import datetime, timedelta

@dataclass
class Filter:
    indicator: str
    min_value: float
    max_value: float

def parse_period(period: str) -> timedelta:
    """Convert a period string (e.g., '6mo', '1y') to a timedelta."""
    period = period.lower()
    if period.endswith('mo'):
        months = int(period[:-2])
        return timedelta(days=months * 30)  # Approximate months to days
    elif period.endswith('y'):
        years = int(period[:-1])
        return timedelta(days=years * 365)
    else:
        raise ValueError(f"Invalid period format: {period}")

@st.cache_data
def load_latest_data(ticker: str) -> pd.DataFrame:
    files = glob.glob(f"data/{ticker}_*.csv")
    if not files:
        st.error(f"No data found for ticker {ticker}. Run stock_analyzer.py to generate data.")
        return None
    latest_file = max(files, key=os.path.getctime)
    try:
        df = pd.read_csv(latest_file, index_col='Date', parse_dates=True)
        # Ensure index is DatetimeIndex and handle timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC')
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
            filters.append(Filter(indicator.name, min_val, max_val))
    
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
        st.warning("No tickers match the filter criteria. Adjust filters and try again.")
        return
    
    # Ticker selection
    selected_ticker = st.selectbox("Select Ticker", filtered_tickers)
    
    # Plot
    if selected_ticker:
        data = load_latest_data(selected_ticker)
        if data is not None:
            try:
                # Filter data to the display period
                period_delta = parse_period(analyzer.config.display_period)
                cutoff_date = data.index.max() - period_delta
                data = data[data.index >= cutoff_date]
                
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