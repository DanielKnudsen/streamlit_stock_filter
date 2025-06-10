from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
import talib
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
            stock = yf.Ticker(ticker)
            self.data[ticker] = stock.history(period=self.config.history_period)

    def calculate_indicators(self):
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            self.indicators_data[ticker] = {}
            
            for indicator in self.config.indicators:
                if indicator.type == "sma":
                    self.indicators_data[ticker][indicator.name] = talib.SMA(
                        df['Close'], timeperiod=indicator.period
                    )
                elif indicator.type == "sma_diff":
                    short = self.indicators_data[ticker][indicator.short_sma]
                    long = self.indicators_data[ticker][indicator.long_sma]
                    self.indicators_data[ticker][indicator.name] = (
                        (short - long) / long * 100
                    )
                elif indicator.type == "rsi":
                    self.indicators_data[ticker][indicator.name] = talib.RSI(
                        df['Close'], timeperiod=indicator.period
                    )
                elif indicator.type == "macd":
                    macd, signal, _ = talib.MACD(
                        df5050['Close'],
                        fastperiod=indicator.fast_period,
                        slowperiod=indicator.slow_period,
                        signalperiod=indicator.signal_period
                    )
                    self.indicators_data[ticker][indicator.name] = macd - signal

    def calculate_ranking(self):
        for indicator in self.config.indicators:
            if not indicator.rank:
                continue
            values = []
            for ticker in self.tickers:
                last_value = self.indicators_data[ticker][indicator.name].iloc[-1]
                if pd.notna(last_value):
                    values.append((ticker, last_value))
            
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
            df = self.data[ticker].copy()
            for indicator in self.config.indicators:
                df[indicator.name] = self.indicators_data[ticker][indicator.name]
            df.to_csv(f"data/{ticker}_{timestamp}.csv")

if __name__ == "__main__":
    analyzer = StockAnalyzer("config.yaml")
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    analyzer.calculate_ranking()
    analyzer.save_data()