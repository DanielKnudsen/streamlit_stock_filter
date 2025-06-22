from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from finta import TA
import yaml
from datetime import datetime
import os

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
    panel: str = "price"  # <-- Add this line

@dataclass
class DataConfig:
    history_period: str
    display_period: str
    tickers_file: str
    indicators: List[IndicatorConfig]

class StockAnalyzer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.tickers = self._load_tickers()
        self.data = {}
        self.indicators_data = {}
        self.ranking = {}

    def _load_config(self, config_path: str) -> DataConfig:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        indicators = [
            IndicatorConfig(**ind) for ind in config_data['indicators']
        ]
        return DataConfig(
            history_period=config_data['data']['history_period'],
            display_period=config_data['data']['display_period'],
            tickers_file=config_data['tickers_file'],
            indicators=indicators
        )

    def _load_tickers(self) -> List[str]:
        df = pd.read_csv(self.config.tickers_file)
        return df['ticker'].tolist()

    def fetch_data(self):
        for ticker in self.tickers:
            try:
                # Dynamically append .ST for Stockholm Stock Exchange
                yf_ticker = f"{ticker}.ST"
                stock = yf.Ticker(yf_ticker)
                self.data[ticker] = stock.history(period=self.config.history_period)
                # Ensure timezone-naive index
                if self.data[ticker].index.tz is not None:
                    self.data[ticker].index = self.data[ticker].index.tz_localize(None)
                if self.data[ticker].empty:
                    print(f"No data retrieved for {yf_ticker}")
            except Exception as e:
                print(f"Error fetching data for {yf_ticker}: {str(e)}")

    def calculate_indicators(self):
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"Skipping indicator calculation for {ticker} due to missing data")
                continue
            df = self.data[ticker].copy()
            self.indicators_data[ticker] = {}
            
            for indicator in self.config.indicators:
                try:
                    if indicator.type == "sma":
                        self.indicators_data[ticker][indicator.name] = TA.SMA(
                            df, period=indicator.period
                        )
                    elif indicator.type == "sma_diff":
                        short = self.indicators_data[ticker][indicator.short_sma]
                        long = self.indicators_data[ticker][indicator.long_sma]
                        self.indicators_data[ticker][indicator.name] = (
                            (short - long) / long * 100
                        )
                    elif indicator.type == "rsi":
                        self.indicators_data[ticker][indicator.name] = TA.RSI(
                            df, period=indicator.period
                        )
                    elif indicator.type == "macd":
                        macd = TA.MACD(
                            df,
                            period_fast=indicator.fast_period,
                            period_slow=indicator.slow_period,
                            period_signal=indicator.signal_period
                        )
                        # Finta returns a DataFrame with 'MACD' and 'SIGNAL'
                        self.indicators_data[ticker][indicator.name] = macd['MACD']
                except Exception as e:
                    print(f"Error calculating {indicator.name} for {ticker}: {str(e)}")

    def save_data(self):
        os.makedirs("data", exist_ok=True)  # <-- Ensure the data directory exists
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"Skipping save for {ticker} due to missing data")
                continue
            df = self.data[ticker].copy()
            for indicator in self.config.indicators:
                if ticker in self.indicators_data and indicator.name in self.indicators_data[ticker]:
                    df[indicator.name] = self.indicators_data[ticker][indicator.name]
            try:
                # Save with timezone-naive index
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_csv(f"data/{ticker}.csv", index=True, index_label='Date')
            except Exception as e:
                print(f"Error saving data for {ticker}: {str(e)}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    analyzer = StockAnalyzer("config.yaml")
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    analyzer.save_data()