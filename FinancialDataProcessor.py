"""
Swedish Stock Analysis - Financial Data Processor
=================================================

FinancialDataProcessor standardizes raw yfinance data into clean DataFrame format
for ratio calculations. Handles field mapping, data validation, TTM calculations,
and missing data handling for the Swedish stock analysis pipeline.

Key responsibilities:
1. Transform raw yfinance API data into standardized format
2. Calculate TTM (Trailing Twelve Months) values from quarterly data
3. Handle missing data and data quality validation
4. Apply Swedish market-specific processing rules
5. Prepare data for ratio calculation functions

This class bridges the gap between raw financial data and the ratio calculation engine.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os
from dataclasses import dataclass
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics for tracking data quality throughout processing."""
    total_tickers: int = 0
    processed_tickers: int = 0
    failed_tickers: int = 0
    field_completeness: Dict[str, float] = None
    ttm_calculation_success_rate: float = 0.0
    quarters_coverage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.field_completeness is None:
            self.field_completeness = {}
        if self.quarters_coverage is None:
            self.quarters_coverage = {}


class FinancialDataProcessor:
    """
    Processes raw yfinance data into standardized format for ratio calculations.
    
    Handles the complete data transformation pipeline from raw API data to
    analysis-ready DataFrames with proper TTM calculations and quality controls.
    """
    
    # YFINANCE FIELD MAPPING - Comprehensive mapping from raw API to standardized names
    YFINANCE_FIELD_MAPPING = {
        # Income Statement Fields (TTM versions)
        'Total Revenue TTM': ['totalRevenue', 'Total Revenue', 'totalRev', 'revenue'],
        'Net Income TTM': ['netIncome', 'Net Income', 'netIncomeCommonStockholders'],
        'Gross Profit TTM': ['grossProfit', 'Gross Profit', 'totalGrossProfit'],
        'Operating Income TTM': ['operatingIncome', 'Operating Income', 'operatingRevenue'],
        'EBIT TTM': ['ebit', 'EBIT', 'operatingIncome'],  # Often same as operating income
        'EBITDA TTM': ['ebitda', 'EBITDA', 'normalizedEBITDA'],
        'Basic EPS TTM': ['basicEPS', 'Basic EPS', 'trailingEps', 'basicEpsFromOps'],
        'Tax Provision TTM': ['taxProvision', 'Tax Provision', 'incomeTaxExpense'],
        'Pretax Income TTM': ['pretaxIncome', 'Pretax Income', 'incomeBeforeTax'],
        
        # Balance Sheet Fields (Most Recent Quarter)
        'Total Assets': ['totalAssets', 'Total Assets', 'totalAssetsGrowth'],
        'Stockholders Equity': ['stockholdersEquity', 'Stockholders Equity', 'totalStockholderEquity', 'shareholdersEquity'],
        'Total Debt': ['totalDebt', 'Total Debt', 'totalDebtEquity', 'longTermDebt'],
        'Cash And Cash Equivalents': ['cash', 'Cash And Cash Equivalents', 'cashAndCashEquivalents', 'totalCash'],
        'Shares Outstanding': ['sharesOutstanding', 'Shares Outstanding', 'impliedSharesOutstanding', 'basicSharesOutstanding'],
        
        # Cash Flow Statement Fields (TTM versions)
        'Operating Cash Flow TTM': ['operatingCashFlow', 'Operating Cash Flow', 'totalCashFromOperatingActivities', 'cashFlowFromOperations'],
        'Free Cash Flow TTM': ['freeCashFlow', 'Free Cash Flow', 'freeCashflow'],
        
        # Market Data Fields
        'Market Cap': ['marketCap', 'Market Cap', 'enterpriseValue'],
        'Enterprise Value': ['enterpriseValue', 'Enterprise Value', 'ev'],
    }
    
    # Required fields for basic analysis (must be present for valid processing)
    REQUIRED_FIELDS = [
        'Total Revenue TTM',
        'Net Income TTM', 
        'Stockholders Equity',
        'Total Assets',
        'Shares Outstanding'
    ]
    
    # Fields that should be positive (validation rules)
    POSITIVE_FIELDS = [
        'Total Revenue TTM',
        'Total Assets',
        'Stockholders Equity',
        'Shares Outstanding',
        'Cash And Cash Equivalents'
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the FinancialDataProcessor.
        
        Args:
            config_path: Path to ratios_config.yaml file
        """
        self.config = self._load_config(config_path)
        self.data_quality_metrics = DataQualityMetrics()
        self.environment = os.getenv('ENVIRONMENT', 'local')
        
        # Configure data quality thresholds from config
        self.min_field_completeness = self.config.get('data_quality', {}).get('min_field_completeness', 0.70)
        self.min_quarters_for_analysis = self.config.get('data_collection', {}).get('minimum_quarters', 4)
        self.quarters_to_collect = self.config.get('data_collection', {}).get('quarters_to_collect', 16)
        
        logger.info(f"FinancialDataProcessor initialized for {self.environment} environment")
        logger.info(f"Quality thresholds: {self.min_field_completeness:.0%} field completeness, {self.min_quarters_for_analysis} min quarters")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from ratios_config.yaml."""
        if config_path is None:
            config_path = 'ratios_config.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def process_raw_financial_data(self, raw_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Process raw yfinance data for multiple tickers into standardized DataFrame.
        
        Args:
            raw_data: Dictionary with ticker as key and raw yfinance data as value
            
        Returns:
            pd.DataFrame: Processed data with standardized columns and TTM calculations
        """
        logger.info(f"Processing raw financial data for {len(raw_data)} tickers")
        
        processed_rows = []
        self.data_quality_metrics.total_tickers = len(raw_data)
        
        for ticker, ticker_data in raw_data.items():
            try:
                processed_ticker_data = self._process_single_ticker(ticker, ticker_data)
                if processed_ticker_data is not None:
                    processed_rows.append(processed_ticker_data)
                    self.data_quality_metrics.processed_tickers += 1
                else:
                    self.data_quality_metrics.failed_tickers += 1
                    logger.warning(f"Failed to process ticker {ticker}")
                    
            except Exception as e:
                self.data_quality_metrics.failed_tickers += 1
                logger.error(f"Error processing ticker {ticker}: {e}")
        
        if not processed_rows:
            logger.error("No tickers were successfully processed")
            return pd.DataFrame()
        
        # Combine all ticker data into single DataFrame
        result_df = pd.DataFrame(processed_rows)
        result_df.set_index('ticker', inplace=True)
        
        # Calculate data quality metrics
        self._calculate_quality_metrics(result_df)
        
        logger.info(f"Successfully processed {len(result_df)} tickers out of {len(raw_data)}")
        return result_df
    
    def _process_single_ticker(self, ticker: str, ticker_data: Dict) -> Optional[Dict]:
        """
        Process raw data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            ticker_data: Raw yfinance data for the ticker
            
        Returns:
            Dict: Processed ticker data or None if processing failed
        """
        try:
            # Extract quarterly financial data
            quarterly_data = self._extract_quarterly_data(ticker_data)
            if len(quarterly_data) < self.min_quarters_for_analysis:
                logger.warning(f"{ticker}: Insufficient quarters ({len(quarterly_data)}) for analysis")
                return None
            
            # Calculate TTM values
            ttm_data = self._calculate_ttm_values(quarterly_data)
            if not ttm_data:
                logger.warning(f"{ticker}: Failed to calculate TTM values")
                return None
            
            # Calculate quarterly TTM time series for temporal analysis
            ttm_time_series = self._calculate_quarterly_ttm_time_series(quarterly_data)
            
            # Extract balance sheet data (most recent)
            balance_sheet_data = self._extract_balance_sheet_data(ticker_data)
            
            # Extract market data
            market_data = self._extract_market_data(ticker_data)
            
            # Combine all data
            combined_data = {
                'ticker': ticker,
                **ttm_data,
                **balance_sheet_data,
                **market_data,
                'quarterly_ttm_series': ttm_time_series  # Add time series data
            }
            
            # Validate data quality
            if not self._validate_ticker_data_quality(ticker, combined_data):
                return None
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            return None
    
    def _extract_quarterly_data(self, ticker_data: Dict) -> List[Dict]:
        """
        Extract quarterly financial data from raw yfinance data.
        
        Args:
            ticker_data: Raw yfinance data for a ticker (direct from raw_financial_data_quarterly.pkl)
            
        Returns:
            List[Dict]: List of quarterly data dictionaries
        """
        quarterly_data = []
        
        # Extract DataFrames directly from ticker_data (no nested 'quarterly' structure)
        balance_sheet_df = ticker_data.get('balance_sheet')
        income_statement_df = ticker_data.get('income_statement') 
        cash_flow_df = ticker_data.get('cash_flow')
        
        if balance_sheet_df is None or income_statement_df is None or cash_flow_df is None:
            logger.warning("Missing required quarterly financial statements")
            return []
            
        # Convert DataFrames to list of quarterly dictionaries
        # DataFrames have periods as index, financial items as columns
        try:
            for period in income_statement_df.index:
                quarter_data = {'period': str(period)}
                
                # Add balance sheet data
                if period in balance_sheet_df.index:
                    quarter_data.update({f"BS_{col}": val for col, val in balance_sheet_df.loc[period].items()})
                
                # Add income statement data  
                if period in income_statement_df.index:
                    quarter_data.update({f"IS_{col}": val for col, val in income_statement_df.loc[period].items()})
                
                # Add cash flow data
                if period in cash_flow_df.index:
                    quarter_data.update({f"CF_{col}": val for col, val in cash_flow_df.loc[period].items()})
                
                quarterly_data.append(quarter_data)
        
        except Exception as e:
            logger.error(f"Error converting quarterly DataFrames to list format: {e}")
            return []
        
        # Ensure we have the required number of quarters
        quarterly_data = quarterly_data[:self.quarters_to_collect]  # Take most recent quarters
        
        # Sort by period (most recent first)
        try:
            quarterly_data.sort(key=lambda x: pd.to_datetime(x.get('period', '1900-01-01')), reverse=True)
        except Exception:
            logger.warning("Could not sort quarterly data by period")
        
        return quarterly_data
    
    def _calculate_ttm_values(self, quarterly_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate Trailing Twelve Months (TTM) values from quarterly data.
        
        Args:
            quarterly_data: List of quarterly data dictionaries
            
        Returns:
            Dict: TTM calculated values
        """
        if len(quarterly_data) < 4:
            logger.warning("Insufficient quarters for TTM calculation")
            return {}
        
        ttm_results = {}
        
        # Mapping from TTM field names to yfinance field names
        field_mapping = {
            'Total Revenue TTM': 'IS_Total Revenue',
            'Net Income TTM': 'IS_Net Income', 
            'Gross Profit TTM': 'IS_Gross Profit',
            'Operating Income TTM': 'IS_Operating Income',
            'EBIT TTM': 'IS_EBIT',
            'EBITDA TTM': 'IS_EBITDA',
            'Tax Provision TTM': 'IS_Tax Provision',
            'Pretax Income TTM': 'IS_Pretax Income',
            'Operating Cash Flow TTM': 'CF_Operating Cash Flow',
            'Free Cash Flow TTM': 'CF_Free Cash Flow'
        }
        
        for ttm_field, source_field in field_mapping.items():
            try:
                # Get values from last 4 quarters
                quarterly_values = []
                for quarter in quarterly_data[:4]:  # Most recent 4 quarters
                    value = self._extract_field_value(quarter, source_field)
                    if value is not None and not pd.isna(value):
                        quarterly_values.append(float(value))
                
                # Calculate TTM as sum of 4 quarters
                if len(quarterly_values) >= 3:  # Allow some missing data
                    ttm_value = sum(quarterly_values)
                    ttm_results[ttm_field] = ttm_value
                else:
                    ttm_results[ttm_field] = np.nan
                    logger.debug(f"Insufficient data for {ttm_field} TTM calculation")
                    
            except Exception as e:
                logger.warning(f"Error calculating TTM for {ttm_field}: {e}")
                ttm_results[ttm_field] = np.nan
        
        # Calculate Basic EPS TTM separately (handle per-share calculations)
        ttm_results['Basic EPS TTM'] = self._calculate_eps_ttm(quarterly_data)
        
        return ttm_results
    
    def _calculate_quarterly_ttm_time_series(self, quarterly_data: List[Dict]) -> Dict[str, List[float]]:
        """
        Calculate quarterly TTM time series for temporal analysis.
        For each available period, calculate TTM using the 4 quarters ending at that period.
        
        Args:
            quarterly_data: List of quarterly data dictionaries (sorted most recent first)
            
        Returns:
            Dict: TTM time series data with field names mapped to lists of quarterly TTM values
        """
        if len(quarterly_data) < 4:
            logger.warning("Insufficient quarters for TTM time series calculation")
            return {}
        
        ttm_time_series = {}
        
        # Mapping from TTM field names to yfinance field names
        field_mapping = {
            'Total Revenue TTM': 'IS_Total Revenue',
            'Net Income TTM': 'IS_Net Income', 
            'Gross Profit TTM': 'IS_Gross Profit',
            'Operating Income TTM': 'IS_Operating Income',
            'EBIT TTM': 'IS_EBIT',
            'EBITDA TTM': 'IS_EBITDA',
            'Tax Provision TTM': 'IS_Tax Provision',
            'Pretax Income TTM': 'IS_Pretax Income',
            'Operating Cash Flow TTM': 'CF_Operating Cash Flow',
            'Free Cash Flow TTM': 'CF_Free Cash Flow'
        }
        
        # Calculate maximum number of TTM periods we can create
        max_ttm_periods = len(quarterly_data) - 3  # Need 4 quarters for each TTM
        
        for ttm_field, source_field in field_mapping.items():
            ttm_values = []
            
            # For each possible TTM period (starting from most recent)
            for period_start in range(max_ttm_periods):
                try:
                    # Get 4 quarters starting from this period
                    quarterly_values = []
                    for i in range(4):
                        quarter_idx = period_start + i
                        if quarter_idx < len(quarterly_data):
                            value = self._extract_field_value(quarterly_data[quarter_idx], source_field)
                            if value is not None and not pd.isna(value):
                                quarterly_values.append(float(value))
                    
                    # Calculate TTM as sum of 4 quarters
                    if len(quarterly_values) >= 3:  # Allow some missing data
                        ttm_value = sum(quarterly_values)
                        ttm_values.append(ttm_value)
                    else:
                        ttm_values.append(np.nan)
                        
                except Exception as e:
                    logger.warning(f"Error calculating TTM for {ttm_field} at period {period_start}: {e}")
                    ttm_values.append(np.nan)
            
            ttm_time_series[ttm_field] = ttm_values
        
        # Calculate Basic EPS TTM time series separately
        ttm_time_series['Basic EPS TTM'] = self._calculate_eps_ttm_time_series(quarterly_data)
        
        return ttm_time_series

    def _calculate_eps_ttm_time_series(self, quarterly_data: List[Dict]) -> List[float]:
        """
        Calculate EPS TTM time series with proper handling of share count changes.
        
        Args:
            quarterly_data: List of quarterly data dictionaries (sorted most recent first)
            
        Returns:
            List[float]: EPS TTM values for each available period
        """
        if len(quarterly_data) < 4:
            return []
        
        eps_ttm_values = []
        max_ttm_periods = len(quarterly_data) - 3
        
        for period_start in range(max_ttm_periods):
            try:
                # Method 1: Try to get TTM EPS directly from most recent quarter of this period
                eps_ttm = self._extract_field_value(quarterly_data[period_start], 'Basic EPS TTM')
                if eps_ttm is not None and not pd.isna(eps_ttm):
                    eps_ttm_values.append(float(eps_ttm))
                    continue
                
                # Method 2: Calculate from Net Income TTM and current shares for this period
                net_income_ttm = 0.0
                valid_quarters = 0
                
                for i in range(4):
                    quarter_idx = period_start + i
                    if quarter_idx < len(quarterly_data):
                        net_income = self._extract_field_value(quarterly_data[quarter_idx], 'IS_Net Income')
                        if net_income is not None and not pd.isna(net_income):
                            net_income_ttm += float(net_income)
                            valid_quarters += 1
                
                if valid_quarters >= 3:
                    # Use shares outstanding from the most recent quarter of this period
                    shares = self._extract_field_value(quarterly_data[period_start], 'BS_Ordinary Shares Number')
                    if shares is not None and float(shares) > 0:
                        eps_ttm_values.append(net_income_ttm / float(shares))
                    else:
                        eps_ttm_values.append(np.nan)
                else:
                    eps_ttm_values.append(np.nan)
                    
            except Exception as e:
                logger.warning(f"Error calculating EPS TTM for period {period_start}: {e}")
                eps_ttm_values.append(np.nan)
        
        return eps_ttm_values
    
    def _calculate_eps_ttm(self, quarterly_data: List[Dict]) -> float:
        """
        Calculate EPS TTM with proper handling of share count changes.
        
        Args:
            quarterly_data: List of quarterly data dictionaries
            
        Returns:
            float: EPS TTM value or np.nan
        """
        try:
            # Method 1: Try to get TTM EPS directly
            for quarter in quarterly_data[:1]:  # Most recent quarter
                eps_ttm = self._extract_field_value(quarter, 'Basic EPS TTM')
                if eps_ttm is not None and not pd.isna(eps_ttm):
                    return float(eps_ttm)
            
            # Method 2: Calculate from Net Income TTM and current shares
            net_income_ttm = 0.0
            valid_quarters = 0
            
            for quarter in quarterly_data[:4]:
                net_income = self._extract_field_value(quarter, 'Net Income TTM')
                if net_income is not None and not pd.isna(net_income):
                    net_income_ttm += float(net_income)
                    valid_quarters += 1
            
            if valid_quarters >= 3:
                # Use most recent shares outstanding
                shares = self._extract_field_value(quarterly_data[0], 'Shares Outstanding')
                if shares is not None and float(shares) > 0:
                    return net_income_ttm / float(shares)
            
            return np.nan
            
        except Exception as e:
            logger.warning(f"Error calculating EPS TTM: {e}")
            return np.nan
    
    def _extract_balance_sheet_data(self, ticker_data: Dict) -> Dict[str, float]:
        """
        Extract balance sheet data (most recent values).
        
        Args:
            ticker_data: Raw yfinance data for a ticker (direct from raw_financial_data_quarterly.pkl)
            
        Returns:
            Dict: Balance sheet data
        """
        balance_sheet_results = {}
        
        # Mapping from expected field names to yfinance field names
        field_mapping = {
            'Total Assets': 'BS_Total Assets',
            'Stockholders Equity': 'BS_Stockholders Equity',
            'Total Debt': 'BS_Total Debt',
            'Cash And Cash Equivalents': 'BS_Cash And Cash Equivalents',
            'Shares Outstanding': 'shares_outstanding'  # From info section
        }
        
        # Get balance sheet DataFrame directly from ticker_data
        balance_sheet_df = ticker_data.get('balance_sheet')
        if balance_sheet_df is None or balance_sheet_df.empty:
            logger.warning("No balance sheet data found")
            return {field: np.nan for field in field_mapping.keys()}
        
        # Extract quarterly data (already processed in _extract_quarterly_data)
        quarterly_data = self._extract_quarterly_data(ticker_data)
        if not quarterly_data:
            logger.warning("No quarterly data available for balance sheet extraction")
            return {field: np.nan for field in field_mapping.keys()}
        
        # Get most recent quarter for balance sheet items
        most_recent_quarter = quarterly_data[0] if quarterly_data else {}
        
        for field, source_field in field_mapping.items():
            if source_field == 'shares_outstanding':
                # Shares outstanding comes from info section (directly from ticker_data)
                info_data = ticker_data.get('info', {})
                field_value = info_data.get('sharesOutstanding', np.nan)
            else:
                # Balance sheet items come from quarterly data
                field_value = self._extract_field_value(most_recent_quarter, source_field)
            
            balance_sheet_results[field] = field_value if field_value is not None else np.nan
        
        return balance_sheet_results
    
    def _extract_market_data(self, ticker_data: Dict) -> Dict[str, float]:
        """
        Extract market-related data.
        
        Args:
            ticker_data: Raw yfinance data for a ticker (direct from raw_financial_data_quarterly.pkl)
            
        Returns:
            Dict: Market data
        """
        market_results = {}
        
        # Get info data directly from ticker_data
        info_data = ticker_data.get('info', {})
        
        # Market data field mapping
        field_mapping = {
            'Market Cap': 'marketCap',
            'Enterprise Value': 'enterpriseValue'
        }
        
        for field, source_field in field_mapping.items():
            field_value = info_data.get(source_field, np.nan)
            market_results[field] = field_value if field_value is not None else np.nan
        
        return market_results
    
    def _extract_field_value(self, data_dict: Dict, field_name: str) -> Optional[float]:
        """
        Extract a field value using the field mapping.
        
        Args:
            data_dict: Dictionary containing raw data
            field_name: Standardized field name to extract
            
        Returns:
            Optional[float]: Field value or None if not found
        """
        if not isinstance(data_dict, dict):
            return None
        
        # Get possible field names for this standardized field
        possible_names = self.YFINANCE_FIELD_MAPPING.get(field_name, [field_name])
        
        for name in possible_names:
            if name in data_dict:
                try:
                    value = data_dict[name]
                    if value is not None and str(value).lower() not in ['none', 'nan', '']:
                        return float(value)
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _validate_ticker_data_quality(self, ticker: str, data: Dict) -> bool:
        """
        Validate data quality for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data: Processed ticker data
            
        Returns:
            bool: True if data quality is acceptable
        """
        # Check for required fields
        missing_required = []
        for field in self.REQUIRED_FIELDS:
            if field not in data or pd.isna(data[field]):
                missing_required.append(field)
        
        if missing_required:
            logger.warning(f"{ticker}: Missing required fields: {missing_required}")
            return False
        
        # Check field completeness
        total_fields = len(self.YFINANCE_FIELD_MAPPING)
        valid_fields = sum(1 for field in self.YFINANCE_FIELD_MAPPING.keys() 
                          if field in data and not pd.isna(data[field]))
        completeness = valid_fields / total_fields
        
        if completeness < self.min_field_completeness:
            logger.warning(f"{ticker}: Low field completeness ({completeness:.1%})")
            return False
        
        # Validate positive fields
        for field in self.POSITIVE_FIELDS:
            if field in data and not pd.isna(data[field]):
                if float(data[field]) <= 0:
                    logger.warning(f"{ticker}: Invalid negative/zero value for {field}: {data[field]}")
                    return False
        
        return True
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> None:
        """
        Calculate overall data quality metrics.
        
        Args:
            df: Processed DataFrame
        """
        if df.empty:
            return
        
        # Field completeness by field
        for field in self.YFINANCE_FIELD_MAPPING.keys():
            if field in df.columns:
                non_null_count = df[field].count()
                total_count = len(df)
                self.data_quality_metrics.field_completeness[field] = non_null_count / total_count
        
        # TTM calculation success rate
        ttm_fields = [f for f in self.YFINANCE_FIELD_MAPPING.keys() if 'TTM' in f]
        ttm_success_count = 0
        ttm_total_count = 0
        
        for field in ttm_fields:
            if field in df.columns:
                ttm_success_count += df[field].count()
                ttm_total_count += len(df)
        
        if ttm_total_count > 0:
            self.data_quality_metrics.ttm_calculation_success_rate = ttm_success_count / ttm_total_count
    
    def get_data_quality_report(self) -> Dict:
        """
        Get comprehensive data quality report.
        
        Returns:
            Dict: Data quality metrics and recommendations
        """
        report = {
            'processing_summary': {
                'total_tickers': self.data_quality_metrics.total_tickers,
                'processed_tickers': self.data_quality_metrics.processed_tickers,
                'failed_tickers': self.data_quality_metrics.failed_tickers,
                'success_rate': (self.data_quality_metrics.processed_tickers / 
                               self.data_quality_metrics.total_tickers 
                               if self.data_quality_metrics.total_tickers > 0 else 0)
            },
            'field_completeness': self.data_quality_metrics.field_completeness,
            'ttm_calculation_success_rate': self.data_quality_metrics.ttm_calculation_success_rate,
            'quality_thresholds': {
                'min_field_completeness': self.min_field_completeness,
                'min_quarters_for_analysis': self.min_quarters_for_analysis
            }
        }
        
        # Add recommendations
        recommendations = []
        success_rate = report['processing_summary']['success_rate']
        
        if success_rate < 0.8:
            recommendations.append("Low ticker processing success rate - check data quality")
        
        if self.data_quality_metrics.ttm_calculation_success_rate < 0.9:
            recommendations.append("Low TTM calculation success rate - verify quarterly data availability")
        
        report['recommendations'] = recommendations
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Save processed data to file.
        
        Args:
            df: Processed DataFrame
            output_path: Optional custom output path
            
        Returns:
            str: Path where data was saved
        """
        if output_path is None:
            # Use environment-aware path
            data_dir = Path('data') / ('local' if self.environment == 'local' else 'remote')
            data_dir.mkdir(parents=True, exist_ok=True)
            output_path = data_dir / 'processed_financial_data.csv'
        
        try:
            df.to_csv(output_path)
            logger.info(f"Processed data saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise


if __name__ == "__main__":
    # Basic testing when module is run directly
    print("Financial Data Processor - Basic Tests")
    print("=" * 50)
    
    # Create test processor
    processor = FinancialDataProcessor()
    
    # Test field mapping
    test_data = {
        'totalRevenue': 10000000,
        'netIncome': 1000000,
        'totalAssets': 20000000,
        'stockholdersEquity': 5000000,
        'sharesOutstanding': 1000000
    }
    
    # Test field extraction
    revenue = processor._extract_field_value(test_data, 'Total Revenue TTM')
    net_income = processor._extract_field_value(test_data, 'Net Income TTM')
    
    print(f"✓ Field extraction test: Revenue={revenue}, Net Income={net_income}")
    
    # Test data quality validation
    test_ticker_data = {
        'ticker': 'TEST',
        'Total Revenue TTM': 10000000,
        'Net Income TTM': 1000000,
        'Total Assets': 20000000,
        'Stockholders Equity': 5000000,
        'Shares Outstanding': 1000000
    }
    
    is_valid = processor._validate_ticker_data_quality('TEST', test_ticker_data)
    print(f"✓ Data validation test: {is_valid}")
    
    print(f"\nProcessor initialized for {processor.environment} environment")
    print(f"Quality thresholds: {processor.min_field_completeness:.0%} completeness, {processor.min_quarters_for_analysis} quarters")
    print("Module ready for integration with pipeline.")