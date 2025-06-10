from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from stock_analyzer import StockAnalyzer, IndicatorConfig

@dataclass
class Filter:
    indicator: str
    min_value: float
    max_value: float

def load_latest_data(ticker: str) -> pd.DataFrame:
    import glob
    import os
    files = glob.glob(f"data/{ticker}_*.csv")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return pd.read_csv(latest_file, index_col='Date', parse_dates=True)

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
    
    analyzer = StockAnalyzer("config.yaml")
    tickers = analyzer.tickers
    
    # Filters
    st.sidebar.header("Filters")
    filters = []
    for indicator in analyzer.config.indicators:
        if indicator.rank:
            min_val = st.sidebar.number_input(
                f"Min {indicator.name}", value=-100.0, step=0.1
            )
            max_val = st.sidebar.number_input(
                f"Max {indicator.name}", value=100.0, step=0.1
            )
            filters.append(Filter(indicator.name, min_val, max_val))
    
    # Apply filters
    filtered_tickers = []
    for ticker in tickers:
        include = True
        for f in filters:
            rank = analyzer.ranking.get(f.indicator, {}).get(ticker, 0)
            if not (f.min_value <= rank <= f.max_value):
                include = False
                break
        if include:
            filtered_tickers.append(ticker)
    
    # Ticker selection
    selected_ticker = st.selectbox("Select Ticker", filtered_tickers)
    
    # Plot
    if selected_ticker:
        data = load_latest_data(selected_ticker)
        if data is not None:
            data = data.last(analyzer.config.display_period)
            fig = plot_stock(selected_ticker, data, analyzer.config.indicators, 
                           analyzer.config.display_period)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display rankings
            st.subheader("Rankings")
            rank_data = {ind.name: analyzer.ranking.get(ind.name, {}).get(selected_ticker, 0)
                        for ind in analyzer.config.indicators if ind.rank}
            st.table(pd.DataFrame([rank_data], index=[selected_ticker]))

if __name__ == "__main__":
    main()