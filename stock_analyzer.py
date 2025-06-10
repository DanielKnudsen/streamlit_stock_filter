from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from finta import TA
import yaml
from datetime import datetime

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
    rank: bool = False

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

    def calculate_ranking(self):
        for indicator in self.config.indicators:
            if not indicator.rank:
                continue
            values = []
            for ticker in self.tickers:
                if ticker in self.indicators_data and indicator.name in self.indicators_data[ticker]:
                    last_value = self.indicators_data[ticker][indicator.name].iloc[-1]
                    if pd.notna(last_value):
                        values.append((ticker, last_value))
            
            if not values:
                print(f"No valid data for ranking {indicator.name}")
                continue
            
            # Sort and normalize to 0-100
            values.sort(key=lambda x: x[1])
            min_val = values[0][1]
            max_val = values[-1][1]
            range_val = max_val - min_val if max_val != min_val else 1
            
            self.ranking[indicator.name] = {
                ticker: ((val - min_val) / range_val) * 100
                for ticker, val in values
            }

    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d")
        for ticker in self.tickers:
            if ticker not in self.data or self.data[ticker].empty:
                print(f"Skipping save for {ticker} due to missing data")
                continue
            df = self.data[ticker].copy()
            for indicator in self.config.indicators:
                if ticker in self.indicators_data and indicator.name in self.indicators_data[ticker]:
                    df[indicator.name] = self.indicators_data[ticker][indicator.name]
            try:
                # Explicitly save the index (Date) to ensure it's preserved
                df.to_csv(f"data/{ticker}_{timestamp}.csv", index=True, index_label='Date')
            except Exception as e:
                print(f"Error saving data for {ticker}: {str(e)}")

if __name__ == "__main__":
    analyzer = StockAnalyzer("config.yaml")
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    analyzer.calculate_ranking()
    analyzer.save_data()