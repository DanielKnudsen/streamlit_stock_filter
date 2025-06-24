from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from finta import TA
import yaml
import os

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
        return DataConfig(
            history_period=config_data['data']['history_period'],
            display_period=config_data['data']['display_period'],
            n_mad=config_data['data'].get('n_mad', 5),
            tickers_file=config_data['tickers_file'],
            indicators=indicators,
            fundamentals=config_data.get('fundamentals', []),
            extra_fundamental_fields=config_data.get('extra_fundamental_fields', [])
        )

    def _load_tickers(self) -> List[str]:
        df = pd.read_csv(self.config.tickers_file)
        self.tickers_df = df  # Save the full DataFrame for later use
        return df['Instrument'].tolist()

    def fetch_data(self):
        print("Hämtar data för tickers:", self.tickers)
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
        print("Beräknar indikatorer för tickers:", self.tickers)
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
        print("Hämtar fundamentala data för tickers:", self.tickers)
        self.fundamentals_data = {}
        for ticker in self.tickers:
            try:
                yf_ticker = yf.Ticker(f"{ticker}.ST")
                info = yf_ticker.info
                self.fundamentals_data[ticker] = {}
                for field in getattr(self.config, "fundamentals", []):
                    self.fundamentals_data[ticker][field] = info.get(field, None)
                for field in getattr(self.config, "extra_fundamental_fields", []):
                    # Använd "Unknown" för 'sector' om det saknas
                    self.fundamentals_data[ticker][field] = info.get(field, "Unknown" if field == "sector" else None)
            except Exception as e:
                print(f"Fel vid hämtning av fundamentala data för {ticker}: {str(e)}")

    def save_data(self):
        print("Sparar data för tickers:", self.tickers)
        os.makedirs(DATA_DIR, exist_ok=True)
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"Hoppar över sparande för {ticker} p.g.a. saknad data")
                continue
            df = self.data[ticker].copy()
            for indicator in self.config.indicators:
                if ticker in self.indicators_data and indicator.name in self.indicators_data[ticker]:
                    df[indicator.name] = self.indicators_data[ticker][indicator.name]
            if hasattr(self, "fundamentals_data"):
                for field in getattr(self.config, "fundamentals", []):
                    value = self.fundamentals_data.get(ticker, {}).get(field, None)
                    df[field] = value
                for field in getattr(self.config, "extra_fundamental_fields", []):
                    value = self.fundamentals_data.get(ticker, {}).get(field, "Unknown" if field == "sector" else None)
                    df[field] = value
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), index=True, index_label='Date')
            except Exception as e:
                print(f"Fel vid sparande av data för {ticker}: {str(e)}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    analyzer = StockAnalyzer("config.yaml")
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    analyzer.fetch_fundamentals()
    analyzer.save_data()