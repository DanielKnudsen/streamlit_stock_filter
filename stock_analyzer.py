from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from finta import TA
import yaml
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@dataclass
class IndicatorConfig:
    name: str
    type: str
    period: Optional[int] = None
    short_sma: Optional[str] = None
    long_sma: Optional[str] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    filter: bool = False
    panel: str = "price"
    cluster: Optional[str] = None
    order: Optional[str] = None

@dataclass
class DataConfig:
    history_period: str
    display_period: str
    tickers_file: str
    indicators: List[IndicatorConfig]
    n_mad: int = 5
    fundamentals: List[str] = None
    extra_fundamental_fields: List[str] = None

class StockAnalyzer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.tickers = self._load_tickers()
        self.data = {}
        self.indicators_data = {}
        self.fundamentals_data = {}

    def _load_config(self, config_path: str) -> DataConfig:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        indicators = [IndicatorConfig(**ind) for ind in config_data['indicators']]
        fundamentals = config_data["fundamentals"]
        fundamental_names = [f["name"] for f in fundamentals]
        return DataConfig(
            history_period=config_data['data']['history_period'],
            display_period=config_data['data']['display_period'],
            n_mad=config_data['data'].get('n_mad', 5),
            tickers_file=config_data['tickers_file'],
            indicators=indicators,
            fundamentals=fundamental_names,
            extra_fundamental_fields=config_data.get('extra_fundamental_fields', [])
        )

    def _load_tickers(self) -> List[str]:
        df = pd.read_csv(self.config.tickers_file)
        self.tickers_df = df  # Save the full DataFrame for later use
        return df['Instrument'].tolist()

    def fetch_data(self):
        print(f"Hämtar data för {len(self.tickers)} tickers. Exempel: {self.tickers[:5]}")
        for ticker in self.tickers:
            try:
                yf_ticker = f"{ticker}.ST"
                stock = yf.Ticker(yf_ticker)
                self.data[ticker] = stock.history(period=self.config.history_period)
                if self.data[ticker].index.tz is not None:
                    self.data[ticker].index = self.data[ticker].index.tz_localize(None)
                if self.data[ticker].empty:
                    print(f"Ingen data hämtad för {yf_ticker}")
            except Exception as e:
                print(f"Fel vid hämtning av data för {yf_ticker}: {str(e)}")

    def calculate_indicators(self):
        print(f"Beräknar indikatorer för tickers {len(self.tickers)} tickers. Exempel: {self.tickers[:5]}")
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"Hoppar över indikatorberäkning för {ticker} p.g.a. saknad data")
                continue
            df = self.data[ticker].copy()
            self.indicators_data[ticker] = {}
            for indicator in self.config.indicators:
                try:
                    if indicator.type == "sma":
                        self.indicators_data[ticker][indicator.name] = TA.SMA(df, period=indicator.period)
                    elif indicator.type == "sma_diff":
                        short = self.indicators_data[ticker][indicator.short_sma]
                        long = self.indicators_data[ticker][indicator.long_sma]
                        self.indicators_data[ticker][indicator.name] = ((short - long) / long * 100)
                    elif indicator.type == "rsi":
                        self.indicators_data[ticker][indicator.name] = TA.RSI(df, period=indicator.period)
                    elif indicator.type == "macd":
                        macd = TA.MACD(df, period_fast=indicator.fast_period, period_slow=indicator.slow_period, period_signal=indicator.signal_period)
                        self.indicators_data[ticker][indicator.name] = macd['MACD']
                except Exception as e:
                    print(f"Fel vid beräkning av {indicator.name} för {ticker}: {str(e)}")

    def fetch_fundamentals(self):
        print(f"Hämtar fundamentala data för tickers: {len(self.tickers)} tickers. Exempel: {self.tickers[:5]}")
        self.fundamentals_data = {}
        for ticker in self.tickers:
            try:
                yf_ticker = yf.Ticker(f"{ticker}.ST")
                info = yf_ticker.info
                self.fundamentals_data[ticker] = {}
                for field in getattr(self.config, "fundamentals", []):
                    self.fundamentals_data[ticker][field] = info.get(field, None)
                for field in getattr(self.config, "extra_fundamental_fields", []):
                    self.fundamentals_data[ticker][field] = info.get(field, "Unknown" if field == "sector" else None)
            except Exception as e:
                print(f"Fel vid hämtning av fundamentala data för {ticker}: {str(e)}")

    def save_fundamentals(self):
        """Sparar fundamental data i en separat CSV-fil med en rad per ticker."""
        print("Sparar fundamental data till fundamentals.csv")
        if not self.fundamentals_data:
            print("Ingen fundamental data att spara")
            return
        try:
            fundamentals_df = pd.DataFrame(self.fundamentals_data).T
            fundamentals_df.index.name = 'Instrument'
            os.makedirs(DATA_DIR, exist_ok=True)
            fundamentals_df.to_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
        except Exception as e:
            print(f"Fel vid sparande av fundamental data: {str(e)}")

    def save_data(self):
        print(f"Sparar data för tickers:{len(self.tickers)} tickers. Exempel: {self.tickers[:5]}")
        os.makedirs(DATA_DIR, exist_ok=True)
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"Hoppar över sparande för {ticker} p.g.a. saknad data")
                continue
            df = self.data[ticker].copy()
            for indicator in self.config.indicators:
                if ticker in self.indicators_data and indicator.name in self.indicators_data[ticker]:
                    df[indicator.name] = self.indicators_data[ticker][indicator.name]
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), index=True, index_label='Date')
            except Exception as e:
                print(f"Fel vid sparande av data för {ticker}: {str(e)}")

    def calculate_rank(self):
        """
        Calculates 0-100 ranks for all indicators and fundamentals in config.yaml where 'cluster' is not null.
        If order is 'High', higher values get higher rank. If 'Low', lower values get higher rank.
        Results are stored in ranks.csv (one row per ticker, one column per rank).
        """
        # Prepare rank targets from config
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
        fundamentals = [f for f in config_data["fundamentals"] if f.get("cluster") is not None]
        indicators = [i for i in config_data["indicators"] if i.get("cluster") is not None]

        # Collect values for all tickers
        rank_data = {}
        for ticker in self.tickers:
            rank_data[ticker] = {}
            # Fundamentals
            for f in fundamentals:
                val = self.fundamentals_data.get(ticker, {}).get(f["name"], np.nan)
                rank_data[ticker][f["name"]] = val
            # Indicators (use last available value)
            for i in indicators:
                val = np.nan
                if ticker in self.indicators_data and i["name"] in self.indicators_data[ticker]:
                    series = self.indicators_data[ticker][i["name"]]
                    if hasattr(series, "iloc"):
                        val = series.iloc[-1]
                    else:
                        val = series
                rank_data[ticker][i["name"]] = val

        df = pd.DataFrame(rank_data).T

        # Convert all columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate ranks for each column
        rank_cols = []
        for f in fundamentals:
            col = f["name"]
            order = f.get("order", "High")
            if df[col].nunique(dropna=True) > 1:
                if order == "Low":
                    ranks = df[col].rank(method='dense',ascending=False, na_option="keep", pct=True) * 100
                else:
                    ranks = df[col].rank(method='dense',ascending=True, na_option="keep", pct=True) * 100
                df[f"{col}_rank"] = ranks.round(0)
            else:
                df[f"{col}_rank"] = np.nan
            rank_cols.append(f"{col}_rank")
        for i in indicators:
            col = i["name"]
            order = i.get("order", "High")
            if df[col].nunique(dropna=True) > 1:
                if order == "Low":
                    ranks = df[col].rank(method='dense',ascending=False, na_option="keep", pct=True) * 100
                else:
                    ranks = df[col].rank(method='dense',ascending=True, na_option="keep", pct=True) * 100
                df[f"{col}_rank"] = ranks.round(0)
            else:
                df[f"{col}_rank"] = np.nan
            rank_cols.append(f"{col}_rank")

        # Save only rank columns
        df_ranks = df[rank_cols]
        df_ranks.index.name = "Instrument"
        df_ranks.to_csv(os.path.join(DATA_DIR, "ranks.csv"))

    def calculate_cluster_rank(self):
        """
        Summarizes the ranks per cluster and calculates a cluster rank (0-100, 100=best) for each ticker.
        Also calculates an overall rank as the average of cluster ranks, scaled 0-100.
        Results are saved in cluster_ranks.csv (one row per ticker, one column per cluster rank plus overall rank).
        """
        # Load config to get clusters
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
        fundamentals = [f for f in config_data["fundamentals"] if f.get("cluster") is not None]
        indicators = [i for i in config_data["indicators"] if i.get("cluster") is not None]

        # Map: cluster -> [field names]
        cluster_fields = {}
        for f in fundamentals:
            cluster = f["cluster"]
            if cluster:
                cluster_fields.setdefault(cluster, []).append(f"{f['name']}_rank")
        for i in indicators:
            cluster = i["cluster"]
            if cluster:
                cluster_fields.setdefault(cluster, []).append(f"{i['name']}_rank")

        # Load ranks.csv
        ranks_path = os.path.join(DATA_DIR, "ranks.csv")
        if not os.path.exists(ranks_path):
            print("ranks.csv not found, run calculate_rank() first.")
            return
        ranks_df = pd.read_csv(ranks_path, index_col="Instrument")

        # Calculate cluster means
        cluster_rank_data = {}
        for cluster, fields in cluster_fields.items():
            # Only use fields that exist in ranks_df
            valid_fields = [f for f in fields if f in ranks_df.columns]
            if not valid_fields:
                continue
            cluster_rank_data[cluster] = ranks_df[valid_fields].mean(axis=1, skipna=True)

        cluster_ranks_df = pd.DataFrame(cluster_rank_data)

        # Rank within each cluster (0-100, 100=best)
        for cluster in cluster_ranks_df.columns:
            col = cluster
            if cluster_ranks_df[col].nunique(dropna=True) > 1:
                ranks = cluster_ranks_df[col].rank(ascending=True, na_option="keep", pct=True) * 100
                cluster_ranks_df[f"{col}_cluster_rank"] = ranks.round(0)
            else:
                cluster_ranks_df[f"{col}_cluster_rank"] = np.nan

        # Calculate overall rank as the average of cluster ranks
        cluster_rank_cols = [c for c in cluster_ranks_df.columns if c.endswith("_cluster_rank")]
        if cluster_rank_cols:
            cluster_ranks_df['overall_rank'] = cluster_ranks_df[cluster_rank_cols].mean(axis=1, skipna=True)
            # Scale overall rank to 0-100 (100=best) if there are valid values
            if cluster_ranks_df['overall_rank'].nunique(dropna=True) > 1:
                cluster_ranks_df['overall_rank'] = (
                    cluster_ranks_df['overall_rank'].rank(ascending=True, na_option="keep", pct=True) * 100
                ).round(0)
            else:
                cluster_ranks_df['overall_rank'] = np.nan
        else:
            cluster_ranks_df['overall_rank'] = np.nan

        # Save only cluster_rank and overall_rank columns
        output_cols = [c for c in cluster_ranks_df.columns if c.endswith("_cluster_rank") or c == "overall_rank"]
        out_df = cluster_ranks_df[output_cols]
        out_df.index.name = "Instrument"
        out_df.to_csv(os.path.join(DATA_DIR, "cluster_ranks.csv"))

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    analyzer = StockAnalyzer("config.yaml")
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    analyzer.fetch_fundamentals()
    analyzer.save_data()
    analyzer.save_fundamentals()
    analyzer.calculate_rank()
    analyzer.calculate_cluster_rank()