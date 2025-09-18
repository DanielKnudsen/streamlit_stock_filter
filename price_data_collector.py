"""
Comprehensive Price Data Collector for Swedish Stock Analysis

This module implements the price data collection strategy outlined in PIPELINE_REDESIGN.md,
providing both historical quarterly-aligned prices and current market data for valuation
calculations.

Key Features:
- Daily price history (4+ years) for technical analysis
- Quarterly-aligned historical prices for temporal alignment with financial data
- Current market data for real-time valuation assessment
- Business day fallback logic for weekends/holidays
- Data quality assessment and error handling
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceDataCollector:
    """
    Comprehensive price data collector for Swedish stocks
    
    Implements the strategy outlined in PIPELINE_REDESIGN.md for collecting:
    1. Historical daily price data (4+ years)
    2. Quarterly-aligned historical prices
    3. Current market data for valuation
    """
    
    def __init__(self, environment: str = None):
        """Initialize the price data collector"""
        self.environment = environment or os.getenv('ENVIRONMENT', 'local')
        self.data_path = Path('data') / ('local' if self.environment == 'local' else 'remote')
        self.logs_path = Path('logs')
        
        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Setup error logging
        self.error_log = self.logs_path / 'price_data_errors.log'
        
    def collect_comprehensive_price_data(self, tickers: List[str], quarterly_dates: List[pd.Timestamp] = None) -> Dict:
        """
        Collect both historical quarterly-aligned prices and current prices for valuation calculations
        
        Strategy:
        1. Historical Alignment: Get prices on or right after the end of quarter date
        2. Current Pricing: Get most recent price for current valuation assessment
        3. Daily Price Series: Maintain complete price history for technical analysis
        
        Args:
            tickers: List of Swedish stock symbols (without .ST suffix)
            quarterly_dates: List of quarter-end dates from financial data collection
            
        Returns:
            Comprehensive price dataset with multiple temporal perspectives
        """
        logger.info(f"Collecting comprehensive price data for {len(tickers)} tickers")
        
        # If no quarterly dates provided, generate standard quarterly dates for last 4 years
        if quarterly_dates is None:
            quarterly_dates = self._generate_standard_quarterly_dates()
        
        price_data = {}
        failed_tickers = []
        
        # Progress bar (disabled in GitHub Actions)
        disable_progress = self.environment != 'local'
        
        for ticker in tqdm(tickers, desc="Collecting comprehensive price data", disable=disable_progress):
            try:
                yf_symbol = f"{ticker}.ST"
                stock = yf.Ticker(yf_symbol)
                
                # 1. Get 4+ years of daily price history (covers 16+ quarters)
                daily_prices = self._get_daily_price_history(stock, years=4)
                
                # 2. Extract quarterly-aligned historical prices
                quarterly_aligned_prices = self._get_quarterly_aligned_prices(daily_prices, quarterly_dates)
                
                # 3. Get current market data
                current_market_data = self._get_current_market_data(stock)
                
                # 4. Calculate data quality metrics
                data_quality = self._calculate_price_data_quality(daily_prices, quarterly_aligned_prices)
                
                price_data[ticker] = {
                    'daily_prices': daily_prices,
                    'quarterly_aligned': quarterly_aligned_prices,
                    'current_market': current_market_data,
                    'data_quality': data_quality,
                    'collection_timestamp': pd.Timestamp.now()
                }
                
            except Exception as e:
                error_msg = f"Error collecting price data for {ticker}: {e}"
                logger.warning(error_msg)
                self._log_error(error_msg)
                failed_tickers.append(ticker)
                price_data[ticker] = self._create_empty_price_record()
        
        # Log collection summary
        successful_count = len([t for t in price_data.keys() if price_data[t]['data_quality']['has_data']])
        logger.info(f"Price data collection completed: {successful_count}/{len(tickers)} successful")
        
        if failed_tickers:
            logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
        
        return price_data
    
    def _get_daily_price_history(self, stock: yf.Ticker, years: int = 4) -> pd.DataFrame:
        """
        Get complete daily price history for the specified number of years
        
        Args:
            stock: yfinance Ticker object
            years: Number of years of history to collect
            
        Returns:
            DataFrame with daily OHLCV data
        """
        try:
            # Calculate start date
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=years)
            
            # Fetch daily price data
            hist = stock.history(start=start_date, end=end_date, interval='1d')
            
            if hist.empty:
                logger.warning(f"No daily price data found for {stock.ticker}")
                return pd.DataFrame()
            
            # Clean and validate data
            hist = hist.dropna(subset=['Close'])  # Remove rows without closing prices
            
            # Ensure we have reasonable data coverage (at least 50% of expected trading days)
            expected_trading_days = years * 252  # Approximate trading days per year
            actual_days = len(hist)
            
            if actual_days < expected_trading_days * 0.5:
                logger.warning(f"Low data coverage for {stock.ticker}: {actual_days}/{expected_trading_days} days")
            
            return hist
            
        except Exception as e:
            logger.error(f"Failed to get daily price history for {stock.ticker}: {e}")
            return pd.DataFrame()
    
    def _get_quarterly_aligned_prices(self, daily_prices: pd.DataFrame, quarterly_dates: List[pd.Timestamp]) -> Dict:
        """
        Extract historical stock prices aligned with quarterly report publication dates
        
        Timing Strategy:
        - Use price on or right after the end of quarter date
        - This represents market pricing at the time of quarterly data
        - Ensures temporal alignment between price and financial data
        
        Fallback Strategy:
        - If exact date unavailable (weekends/holidays), use nearest business day
        - Priority: Next business day > Previous business day > Skip quarter
        """
        if daily_prices.empty:
            return {}
        
        aligned_prices = {}
        
        for quarter_end in quarterly_dates:
            try:
                # Calculate target pricing date (on or right after quarter end date)
                target_date = self._get_next_business_day(quarter_end)
                
                # Find actual price using fallback strategy
                actual_price = self._find_nearest_business_day_price(daily_prices, target_date)
                
                if actual_price is not None:
                    quarter_key = f"{quarter_end.year}-Q{((quarter_end.month - 1) // 3) + 1}"
                    aligned_prices[quarter_key] = {
                        'quarter_end_date': quarter_end,
                        'pricing_date': actual_price['date'],
                        'close_price': actual_price['close'],
                        'volume': actual_price['volume'],
                        'days_after_quarter_end': (actual_price['date'] - quarter_end).days,
                        'open_price': actual_price.get('open'),
                        'high_price': actual_price.get('high'),
                        'low_price': actual_price.get('low')
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to align price for quarter {quarter_end}: {e}")
                continue
        
        return aligned_prices
    
    def _get_current_market_data(self, stock: yf.Ticker) -> Dict:
        """Get current real-time market data for immediate valuation calculations"""
        try:
            info = stock.info
            
            # Get current price from multiple sources for reliability
            current_price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose')
            )
            
            return {
                'current_price': current_price,
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'enterprise_value': info.get('enterpriseValue'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'average_volume': info.get('averageVolume'),
                'last_updated': pd.Timestamp.now(),
                'currency': info.get('currency', 'SEK'),
                'exchange': info.get('exchange'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
        except Exception as e:
            logger.warning(f"Failed to get current market data for {stock.ticker}: {e}")
            return self._create_empty_current_market_data()
    
    def _get_next_business_day(self, start_date: pd.Timestamp) -> pd.Timestamp:
        """Get the next business day from the given date"""
        current_date = start_date
        
        # If it's already a business day, return it
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            return current_date
        
        # Otherwise, find the next business day
        while current_date.weekday() >= 5:  # Weekend
            current_date += pd.Timedelta(days=1)
            
        return current_date
    
    def _find_nearest_business_day_price(self, daily_prices: pd.DataFrame, target_date: pd.Timestamp) -> Optional[Dict]:
        """
        Find stock price for nearest available business day to target date
        
        Priority order:
        1. Exact target date
        2. Next 1-7 business days
        3. Previous 1-7 business days
        4. Return None if no data found
        """
        if daily_prices.empty:
            return None
        
        # Convert target_date to date for comparison
        target_date_only = target_date.date()
        
        # Try exact date first
        matching_rows = daily_prices[daily_prices.index.date == target_date_only]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            return {
                'date': target_date,
                'close': row['Close'],
                'volume': row['Volume'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low']
            }
        
        # Try next few business days
        for offset in range(1, 8):
            check_date = (target_date + pd.Timedelta(days=offset)).date()
            matching_rows = daily_prices[daily_prices.index.date == check_date]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                return {
                    'date': target_date + pd.Timedelta(days=offset),
                    'close': row['Close'],
                    'volume': row['Volume'],
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low']
                }
        
        # Try previous few business days
        for offset in range(1, 8):
            check_date = (target_date - pd.Timedelta(days=offset)).date()
            matching_rows = daily_prices[daily_prices.index.date == check_date]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                return {
                    'date': target_date - pd.Timedelta(days=offset),
                    'close': row['Close'],
                    'volume': row['Volume'],
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low']
                }
        
        logger.warning(f"No price data found near target date {target_date_only}")
        return None
    
    def _calculate_price_data_quality(self, daily_prices: pd.DataFrame, quarterly_aligned: Dict) -> Dict:
        """Calculate data quality metrics for the collected price data"""
        if daily_prices.empty:
            return {
                'has_data': False,
                'daily_coverage': 0.0,
                'quarterly_coverage': 0.0,
                'data_gaps': [],
                'quality_score': 0.0
            }
        
        # Calculate daily data coverage (last 4 years)
        expected_trading_days = 4 * 252  # Approximate
        actual_trading_days = len(daily_prices)
        daily_coverage = min(1.0, actual_trading_days / expected_trading_days)
        
        # Calculate quarterly alignment coverage
        expected_quarters = 16  # 4 years * 4 quarters
        actual_quarters = len(quarterly_aligned)
        quarterly_coverage = actual_quarters / expected_quarters if expected_quarters > 0 else 0.0
        
        # Identify data gaps (more than 10 consecutive missing trading days)
        data_gaps = self._identify_data_gaps(daily_prices)
        
        # Calculate overall quality score
        quality_score = (daily_coverage * 0.6 + quarterly_coverage * 0.4) * (1.0 - min(0.5, len(data_gaps) * 0.1))
        
        return {
            'has_data': True,
            'daily_coverage': daily_coverage,
            'quarterly_coverage': quarterly_coverage,
            'data_gaps': data_gaps,
            'quality_score': quality_score,
            'total_daily_records': actual_trading_days,
            'total_quarterly_records': actual_quarters
        }
    
    def _identify_data_gaps(self, daily_prices: pd.DataFrame, max_gap_days: int = 10) -> List[Dict]:
        """Identify significant gaps in daily price data"""
        if daily_prices.empty or len(daily_prices) < 2:
            return []
        
        gaps = []
        dates = daily_prices.index.date
        
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i-1]).days
            if gap_days > max_gap_days:
                gaps.append({
                    'start_date': dates[i-1],
                    'end_date': dates[i],
                    'gap_days': gap_days
                })
        
        return gaps
    
    def _generate_standard_quarterly_dates(self) -> List[pd.Timestamp]:
        """Generate standard quarterly end dates for the last 4 years"""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=4)
        
        quarterly_dates = []
        current_year = start_date.year
        
        while current_year <= end_date.year:
            for quarter_month in [3, 6, 9, 12]:  # March, June, September, December
                quarter_end = pd.Timestamp(year=current_year, month=quarter_month, day=31)
                if quarter_end <= end_date:
                    quarterly_dates.append(quarter_end)
            current_year += 1
        
        return sorted(quarterly_dates)
    
    def _create_empty_price_record(self) -> Dict:
        """Create an empty price record for failed data collection"""
        return {
            'daily_prices': pd.DataFrame(),
            'quarterly_aligned': {},
            'current_market': self._create_empty_current_market_data(),
            'data_quality': {
                'has_data': False,
                'daily_coverage': 0.0,
                'quarterly_coverage': 0.0,
                'data_gaps': [],
                'quality_score': 0.0
            },
            'collection_timestamp': pd.Timestamp.now()
        }
    
    def _create_empty_current_market_data(self) -> Dict:
        """Create empty current market data structure"""
        return {
            'current_price': np.nan,
            'market_cap': np.nan,
            'shares_outstanding': np.nan,
            'enterprise_value': np.nan,
            'book_value': np.nan,
            'price_to_book': np.nan,
            'trailing_pe': np.nan,
            'forward_pe': np.nan,
            'dividend_yield': np.nan,
            'beta': np.nan,
            'fifty_two_week_low': np.nan,
            'fifty_two_week_high': np.nan,
            'average_volume': np.nan,
            'last_updated': pd.Timestamp.now(),
            'currency': 'SEK',
            'exchange': None,
            'sector': None,
            'industry': None
        }
    
    def _log_error(self, error_msg: str):
        """Log error to dedicated error log file"""
        try:
            with open(self.error_log, 'a', encoding='utf-8') as f:
                timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {error_msg}\n")
        except Exception as e:
            logger.error(f"Failed to write to error log: {e}")
    
    def save_price_data(self, price_data: Dict, filename: str = "price_data_comprehensive"):
        """Save comprehensive price data to both pickle and summary CSV"""
        try:
            import pickle
            
            # Save full data as pickle
            pickle_path = self.data_path / f"{filename}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(price_data, f)
            logger.info(f"Comprehensive price data saved to {pickle_path}")
            
            # Create summary CSV
            summary_data = []
            for ticker, data in price_data.items():
                summary_data.append({
                    'ticker': ticker,
                    'has_data': data['data_quality']['has_data'],
                    'current_price': data['current_market']['current_price'],
                    'market_cap': data['current_market']['market_cap'],
                    'daily_records': data['data_quality'].get('total_daily_records', 0),
                    'quarterly_records': data['data_quality'].get('total_quarterly_records', 0),
                    'data_quality_score': data['data_quality']['quality_score'],
                    'collection_timestamp': data['collection_timestamp']
                })
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = self.data_path / f"{filename}_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"Price data summary saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save price data: {e}")


# Convenience functions for direct use
def collect_comprehensive_price_data(tickers: List[str], quarterly_dates: List[pd.Timestamp] = None) -> Dict:
    """
    Convenience function to collect comprehensive price data
    
    Args:
        tickers: List of Swedish stock symbols (without .ST suffix)
        quarterly_dates: Optional list of quarter-end dates for alignment
        
    Returns:
        Comprehensive price dataset
    """
    collector = PriceDataCollector()
    return collector.collect_comprehensive_price_data(tickers, quarterly_dates)


def save_price_data(price_data: Dict, filename: str = "price_data_comprehensive"):
    """
    Convenience function to save price data
    
    Args:
        price_data: Price data dictionary from collect_comprehensive_price_data
        filename: Base filename for saving (without extension)
    """
    collector = PriceDataCollector()
    collector.save_price_data(price_data, filename)


if __name__ == "__main__":
    # Example usage for testing
    test_tickers = ['ASSA-B', 'VOLV-B', 'SHB-A']
    
    collector = PriceDataCollector()
    price_data = collector.collect_comprehensive_price_data(test_tickers)
    collector.save_price_data(price_data, "test_price_data")
    
    print(f"Collected price data for {len(price_data)} tickers")
    for ticker, data in price_data.items():
        quality = data['data_quality']
        print(f"{ticker}: Quality Score: {quality['quality_score']:.3f}, Daily Records: {quality.get('total_daily_records', 0)}")