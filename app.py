from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analyzer import StockAnalyzer, IndicatorConfig
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
    """Konverterar yfinance-period (t.ex. '6mo') till Pandas offset (t.ex. '6M')."""
    period = period.lower()
    if period in ("ytd", "max"):
        return period.upper()
    match = re.match(r"(\d+)([a-z]+)", period)
    if match:
        num, unit = match.groups()
        unit_map = {"d": "D", "w": "W", "mo": "M", "y": "Y"}
        for k in sorted(unit_map, key=len, reverse=True):
            if unit.startswith(k):
                return f"{num}{unit_map[k]}"
    return period.upper()

@st.cache_data
def load_summary_data(tickers):
    """Laddar den senaste raden för varje ticker för filtrering och boxplot."""
    summary = {}
    fundamentals_file = os.path.join(DATA_DIR, "fundamentals.csv")
    fundamentals_df = None
    if os.path.exists(fundamentals_file):
        try:
            fundamentals_df = pd.read_csv(fundamentals_file, index_col='Instrument')
        except Exception as e:
            st.error(f"Fel vid laddning av fundamentals.csv: {str(e)}")
    
    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
                if not df.empty:
                    last_row = df.iloc[-1].copy()
                    # Lägg till fundamental data från fundamentals_df om den finns
                    if fundamentals_df is not None and ticker in fundamentals_df.index:
                        for col in fundamentals_df.columns:
                            last_row[col] = fundamentals_df.loc[ticker, col]
                    # Säkerställ att 'sector' finns, annars sätt "Unknown"
                    if 'sector' not in last_row or pd.isna(last_row['sector']):
                        last_row['sector'] = "Unknown"
                    summary[ticker] = last_row
            except Exception as e:
                st.error(f"Fel vid laddning av sammanfattning för {ticker}: {str(e)}")
    summary_df = pd.DataFrame(summary).T
    return summary_df

@st.cache_data
def load_full_data(ticker):
    """Laddar fullständig data för en specifik ticker vid behov."""
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            st.error(f"Fel vid laddning av full data för {ticker}: {str(e)}")
            return None
    return None

def plot_stock(ticker: str, data: pd.DataFrame, config: List[IndicatorConfig], period: str):
    """Skapar ett diagram för en given ticker med angivna indikatorer."""
    has_middle = any(getattr(ind, "panel", "price") == "middle" for ind in config)
    has_lower = any(getattr(ind, "panel", "price") == "lower" for ind in config)
    if has_middle and has_lower:
        rows, row_heights = 3, [0.5, 0.25, 0.25]
        subplot_titles = ("Prispanel", "Mellanpanel", "Nedre panel")
    elif has_middle:
        rows, row_heights = 2, [0.7, 0.3]
        subplot_titles = ("Prispanel", "Mellanpanel")
    elif has_lower:
        rows, row_heights = 2, [0.7, 0.3]
        subplot_titles = ("Prispanel", "Nedre panel")
    else:
        rows, row_heights = 1, [1.0]
        subplot_titles = ("Prispanel",)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.05, subplot_titles=subplot_titles)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stängningspris', line=dict(color='blue')), row=1, col=1)
    for indicator in config:
        if indicator.name in data.columns:
            panel = getattr(indicator, "panel", "price")
            trace = go.Scatter(x=data.index, y=data[indicator.name], name=indicator.name)
            if panel == "price":
                fig.add_trace(trace, row=1, col=1)
            elif panel == "middle" and has_middle:
                fig.add_trace(trace, row=2 if not has_lower else 2, col=1)
            elif panel == "lower" and has_lower:
                fig.add_trace(trace, row=3 if has_middle else 2, col=1)
    fig.update_yaxes(title_text="Pris", row=1, col=1)
    if has_middle:
        fig.update_yaxes(title_text="Mellanindikatorer", row=2 if not has_lower else 2, col=1)
    if has_lower:
        fig.update_yaxes(title_text="Nedre indikatorer", row=3 if has_middle else 2, col=1)
    fig.update_layout(height=900 if has_middle and has_lower else 700, legend=dict(orientation="h"), xaxis=dict(title="Datum"), title=f"{ticker} Aktiediagram")
    return fig

FUNDAMENTAL_EXPLANATIONS = {
    "earningsGrowth": "Årlig vinsttillväxt.",
    "revenueGrowth": "Årlig intäktstillväxt.",
    "profitMargins": "Nettovinst som procent av intäkter.",
    "returnOnAssets": "Nettoinkomst dividerat med totala tillgångar.",
    "priceToBook": "Aktiepris dividerat med bokfört värde per aktie.",
    "forwardPE": "Framtida pris/vinst-förhållande."
}

def remove_outliers(series, n_mad=5):
    """Tar bort extremvärden baserat på median absolut avvikelse."""
    median = np.median(series)
    mad = median_abs_deviation(series, nan_policy='omit')
    if mad == 0:
        return series
    mask = np.abs(series - median) < n_mad * mad
    return series[mask]

def main():
    st.set_page_config(page_title="Stock Analyzer", layout="wide")
    st.title("Dashboard för Aktieanalys")
    
    try:
        analyzer = StockAnalyzer("config.yaml")
    except Exception as e:
        st.error(f"Fel vid initiering av StockAnalyzer: {str(e)}")
        return
    
    if not analyzer.tickers:
        st.error("Inga tickers hittades i tickers.csv.")
        return
    
    summary_df = load_summary_data(analyzer.tickers)
    if summary_df.empty:
        st.error("Ingen data tillgänglig. Säkerställ att data är hämtad och sparad.")
        return
    all_markets = sorted(analyzer.tickers_df['Lista'].dropna().unique())
    selected_markets = st.sidebar.multiselect(
        "Market",
        options=all_markets,
        default=all_markets,
    )

    all_sectors = sorted(set(summary_df['sector'].dropna().unique())) if 'sector' in summary_df.columns else ["Unknown"]
    selected_sectors = st.sidebar.multiselect("Affärssektor", options=all_sectors, default=all_sectors)
    
    st.sidebar.header("Tekniska Filter")
    filters = []
    
    def get_filtered_tickers(exclude_filter=None):
        market_mask = summary_df.index.isin(
            analyzer.tickers_df[analyzer.tickers_df['Lista'].isin(selected_markets)]['Instrument']
        )
        sector_mask = summary_df['sector'].isin(selected_sectors) if 'sector' in summary_df.columns else pd.Series(True, index=summary_df.index)
        filter_masks = {
            f.indicator: summary_df[f.indicator].astype(float).between(f.min_value, f.max_value)
            if f.indicator in summary_df.columns else pd.Series(False, index=summary_df.index)
            for f in filters if f != exclude_filter
        }
        all_masks = [market_mask, sector_mask] + list(filter_masks.values())
        final_mask = np.logical_and.reduce(all_masks) if all_masks else pd.Series(True, index=summary_df.index)
        return summary_df.index[final_mask].tolist()
    
    for indicator in analyzer.config.indicators:
        if getattr(indicator, "filter", False):
            filtered_tickers = get_filtered_tickers()
            values = summary_df.loc[filtered_tickers, indicator.name].dropna().astype(float)
            
            if not values.empty and np.isfinite(values.min()) and np.isfinite(values.max()):
                min_val = float(values.min())
                max_val = float(values.max())
                if min_val == max_val:
                    min_val -= 1.0
                    max_val += 1.0
            else:
                min_val = 0.0
                max_val = 1.0

            slider_min, slider_max = st.sidebar.slider(
                f"{indicator.name} intervall", min_value=min_val, max_value=max_val,
                value=(min_val, max_val), step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
                key=f"slider_{indicator.name}", help=getattr(indicator, "description", "")
            )
            
            if not values.empty:
                fig = go.Figure(go.Violin(x=values, points=False, orientation='h', marker_color='lightblue', name=""))
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=180, showlegend=False)
                st.sidebar.plotly_chart(fig, use_container_width=True)
            st.sidebar.markdown("---")
            filters.append(Filter(indicator=indicator.name, min_value=slider_min, max_value=slider_max))
    
    st.sidebar.header("Fundamentala Filter")
    for field in getattr(analyzer.config, "fundamentals", []):
        filtered_tickers = get_filtered_tickers()
        values = summary_df.loc[filtered_tickers, field].dropna().astype(float)
        
        if not values.empty and np.isfinite(values.min()) and np.isfinite(values.max()):
            min_val = float(values.min())
            max_val = float(values.max())
            if min_val == max_val:
                min_val -= 1.0
                max_val += 1.0
        else:
            min_val = 0.0
            max_val = 1.0

        slider_min, slider_max = st.sidebar.slider(
            f"{field} intervall", min_value=min_val, max_value=max_val,
            value=(min_val, max_val), step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
            key=f"slider_{field}", help=FUNDAMENTAL_EXPLANATIONS.get(field, "")
        )
        
        if not values.empty:
            fig = go.Figure(go.Violin(x=values, points=False, orientation='h', marker_color='lightblue', name=""))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=180, showlegend=False)
            st.sidebar.plotly_chart(fig, use_container_width=True)
        st.sidebar.markdown("---")
        filters.append(Filter(indicator=field, min_value=slider_min, max_value=slider_max))
    
    filtered_tickers = get_filtered_tickers()
    if not filtered_tickers:
        st.warning("Inga tickers matchar filterkriterierna. Justera filtren och försök igen.")
        return
    st.subheader(f"Filtrerade aktier ({len(filtered_tickers)})")
    st.markdown(f"**Antal filtrerade aktier:** {len(filtered_tickers)}")
    st.markdown(f"**Valda sektorer:** {', '.join(selected_sectors)}")
    st.markdown("---")

    table_data = summary_df.loc[filtered_tickers].copy()
    table_data["Välj"] = False

    market_info = analyzer.tickers_df.set_index('Instrument')['Lista']
    table_data['Market'] = table_data.index.map(market_info)

    table_data.drop(
        columns=['Open','High','Low','Dividends','Volume','Stock Splits','longBusinessSummary','Capital Gains'],
        inplace=True, errors='ignore'
    )
    edited_df = st.data_editor(
        table_data, use_container_width=True, hide_index=False,
        column_config={"Välj": st.column_config.CheckboxColumn("Välj")}
    )
    
    selected_tickers = edited_df[edited_df["Välj"]].index.tolist()
    if not selected_tickers:
        st.info("Välj minst en aktie att visualisera.")
        return
    
    for selected_ticker in selected_tickers:
        data = load_full_data(selected_ticker)
        if data is not None:
            try:
                pandas_offset = convert_to_pandas_offset(analyzer.config.display_period)
                data = data.last(pandas_offset)
                fig = plot_stock(selected_ticker, data, analyzer.config.indicators, analyzer.config.display_period)
                st.plotly_chart(fig, use_container_width=True)
                
                info = summary_df.loc[selected_ticker]
                for field in getattr(analyzer.config, "extra_fundamental_fields", []):
                    st.markdown(f"**{field}:** {info.get(field, 'N/A')}")
                
                for field in getattr(analyzer.config, "fundamentals", []):
                    values = summary_df.loc[filtered_tickers, field].dropna().astype(float)
                    selected_value = info.get(field)
                    if not values.empty and selected_value is not None:
                        fig = go.Figure()
                        fig.add_trace(go.Violin(
                            x=values,
                            y=[field]*len(values),
                            orientation='h',
                            marker_color='lightblue',
                            name="Distribution"
                        ))
                        fig.add_trace(go.Scatter(
                            x=[selected_value],
                            y=[field],
                            mode='markers',
                            marker=dict(color='red', size=14, symbol='diamond'),
                            name=selected_ticker
                        ))
                        fig.update_layout(
                            title=f"Distribution av {field} (röd = {selected_ticker})",
                            height=300,
                            showlegend=False
                        )
                        fig.data = (fig.data[0], fig.data[1])
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Fel vid plotting av {selected_ticker}: {str(e)}")

if __name__ == "__main__":
    main()