#!/usr/bin/env python3
"""
RatioCalculator.py - Comprehensive ratio calculation engine for Swedish stock analysis

This module orchestrates the complete ratio calculation pipeline, integrating:
- FinancialDataProcessor for data standardization
- ratio_functions for secure calculations
- ratios_config for configuration-driven behavior
- Three temporal perspectives framework (Current TTM, Trend TTM, Stability TTM)

Swedish Market Focus:
- All 13 Swedish financial ratios (ROE, ROIC, Soliditet, etc.)
- Temporal analysis with TTM-to-TTM comparison
- Dual pricing strategy for valuation ratios
- Comprehensive error handling and data quality validation

Author: AI Assistant
Date: September 2025
Environment: Development/Production with local/remote data separation
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass

# Local imports
from FinancialDataProcessor import FinancialDataProcessor
import ratio_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TemporalPerspective:
    """Data class for temporal perspective configuration"""
    name: str
    weight: float
    description: str
    quarters_back: int


@dataclass
class RatioResult:
    """Data class for individual ratio calculation result"""
    ratio_name: str
    current_value: Optional[float]
    trend_value: Optional[float]
    stability_value: Optional[float]
    composite_score: Optional[float]
    data_quality: float
    calculation_notes: List[str]


@dataclass
class StockAnalysis:
    """Data class for complete stock analysis result"""
    ticker: str
    calculation_date: datetime
    ratio_results: Dict[str, RatioResult]
    category_scores: Dict[str, float]
    overall_score: Optional[float]
    data_quality_score: float
    processing_notes: List[str]


class RatioCalculator:
    """
    Main ratio calculation engine that orchestrates the complete analysis pipeline.
    
    Responsibilities:
    1. Load and validate configuration from ratios_config.yaml
    2. Initialize FinancialDataProcessor for data standardization
    3. Execute three temporal perspectives framework
    4. Calculate all Swedish ratios using secure ratio_functions
    5. Aggregate results with proper weighting
    6. Provide comprehensive error handling and validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RatioCalculator with configuration and data processor.
        
        Args:
            config_path: Optional path to ratios_config.yaml, defaults to project root
        """
        self.config_path = config_path or "ratios_config.yaml"
        self.config = self._load_configuration()
        self.data_processor = FinancialDataProcessor()
        self.temporal_perspectives = self._initialize_temporal_perspectives()
        self.ratio_functions_registry = self._validate_ratio_functions()
        
        logger.info(f"RatioCalculator initialized with {len(self.config['ratio_definitions'])} ratios")
        logger.info(f"Temporal perspectives: {[tp.name for tp in self.temporal_perspectives]}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load and validate ratios configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['ratio_definitions', 'category_weights', 'temporal_perspectives', 'data_quality']
            missing_sections = [section for section in required_sections if section not in config]
            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {missing_sections}")
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_temporal_perspectives(self) -> List[TemporalPerspective]:
        """Initialize temporal perspective objects from configuration"""
        perspectives = []
        
        for perspective_name, perspective_config in self.config['temporal_perspectives'].items():
            # UPDATED DATA ARCHITECTURE: FinancialDataProcessor outputs quarterly TTM time series
            # Use quarters_back=0 for current, 1-3 for trend/stability based on available data
            if perspective_name == 'current_ttm':
                quarters_back = 0  # Use current available TTM data
            elif perspective_name == 'trend_ttm':
                quarters_back = 1  # Use time series for trend analysis
            elif perspective_name == 'stability_ttm':
                quarters_back = 1  # Use time series for stability analysis
            else:
                quarters_back = 1  # Default for any other perspectives
                
            perspective = TemporalPerspective(
                name=perspective_name,
                weight=perspective_config['default_weight'],
                description=perspective_config['description'],
                quarters_back=quarters_back
            )
            perspectives.append(perspective)
        
        # Validate weights sum to 1.0
        total_weight = sum(tp.weight for tp in perspectives)
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Temporal perspective weights sum to {total_weight}, not 1.0")
        
        return perspectives
    
    def _validate_ratio_functions(self) -> Dict[str, callable]:
        """Validate that all configured ratios have corresponding functions"""
        registry = {}
        missing_functions = []
        
        for ratio_name, ratio_config in self.config['ratio_definitions'].items():
            function_name = ratio_config['function']
            
            if hasattr(ratio_functions, function_name):
                registry[ratio_name] = getattr(ratio_functions, function_name)
            else:
                missing_functions.append(function_name)
        
        if missing_functions:
            raise ValueError(f"Missing ratio functions: {missing_functions}")
        
        logger.info(f"Validated {len(registry)} ratio functions")
        return registry
    
    def calculate_stock_ratios(self, ticker: str, raw_financial_data: Dict[str, Dict], 
                             price_data: Optional[pd.DataFrame] = None) -> StockAnalysis:
        """
        Calculate all ratios for a single stock using three temporal perspectives.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL.ST')
            raw_financial_data: Raw financial data from yfinance in dictionary format
            price_data: Optional price data for valuation ratios
            
        Returns:
            StockAnalysis object with complete ratio analysis
        """
        logger.info(f"Starting ratio calculation for {ticker}")
        
        try:
            # Process raw data through FinancialDataProcessor
            processed_data = self.data_processor.process_raw_financial_data(raw_financial_data)
            
            return self.calculate_ratios_from_processed_data(ticker, processed_data)
            
        except Exception as e:
            logger.error(f"Failed to calculate ratios for {ticker}: {e}")
            raise

    def calculate_ratios_from_processed_data(self, ticker: str, processed_data: pd.DataFrame, 
                                           price_data: Optional[Dict] = None) -> StockAnalysis:
        """
        Calculate all ratios from pre-processed financial data.
        
        Args:
            ticker: Stock ticker symbol
            processed_data: Pre-processed financial data from FinancialDataProcessor
            price_data: Optional price data dictionary with current_price and historical prices
            
        Returns:
            StockAnalysis object with complete ratio analysis
        """
        logger.info(f"Starting ratio calculation for {ticker} from processed data")
        
        try:
            # Integrate price data if provided
            if price_data:
                processed_data = processed_data.copy()
                
                # Add current price to the processed data for valuation ratios
                if 'current_market' in price_data and 'current_price' in price_data['current_market']:
                    current_price = price_data['current_market']['current_price']
                    processed_data['current_price'] = current_price
                    processed_data['Current Price'] = current_price  # For backward compatibility
                    logger.debug(f"Added current price {current_price} for {ticker}")
                
                # Add quarterly aligned prices for historical ratio calculations
                if 'quarterly_aligned' in price_data and price_data['quarterly_aligned']:
                    # Get the most recent quarterly price for historical ratios
                    quarterly_prices = price_data['quarterly_aligned']
                    if quarterly_prices:
                        # Sort by quarter to get most recent
                        sorted_quarters = sorted(quarterly_prices.keys(), reverse=True)
                        if sorted_quarters:
                            most_recent_quarter = sorted_quarters[0]
                            quarterly_price = quarterly_prices[most_recent_quarter]['close_price']
                            processed_data['quarterly_aligned_price'] = quarterly_price
                            logger.debug(f"Added quarterly aligned price {quarterly_price} from {most_recent_quarter} for {ticker}")
                            
                            # Also store the full quarterly price series for temporal analysis
                            quarterly_price_series = []
                            for quarter in sorted_quarters:
                                quarterly_price_series.append(quarterly_prices[quarter]['close_price'])
                            processed_data['quarterly_price_series'] = [quarterly_price_series]  # Wrap in list for consistency
            
            # Validate data quality
            data_quality_score = self._assess_data_quality(processed_data)
            if data_quality_score < self.config['data_quality']['min_field_completeness']:
                logger.warning(f"Low data quality for {ticker}: {data_quality_score:.1%}")
            
            # Calculate ratios for each temporal perspective
            ratio_results = {}
            processing_notes = []
            
            for ratio_name, ratio_config in self.config['ratio_definitions'].items():
                try:
                    ratio_result = self._calculate_ratio_temporal(
                        ratio_name, ratio_config, processed_data
                    )
                    ratio_results[ratio_name] = ratio_result
                    
                except Exception as e:
                    logger.error(f"Failed to calculate {ratio_name} for {ticker}: {e}")
                    processing_notes.append(f"Failed to calculate {ratio_name}: {str(e)}")
                    
                    # Create empty result for failed calculation
                    ratio_results[ratio_name] = RatioResult(
                        ratio_name=ratio_name,
                        current_value=None,
                        trend_value=None,
                        stability_value=None,
                        composite_score=None,
                        data_quality=0.0,
                        calculation_notes=[f"Calculation failed: {str(e)}"]
                    )
            
            # Calculate category scores
            category_scores = self._calculate_category_scores(ratio_results)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(category_scores)
            
            # Create analysis result
            analysis = StockAnalysis(
                ticker=ticker,
                calculation_date=datetime.now(),
                ratio_results=ratio_results,
                category_scores=category_scores,
                overall_score=overall_score,
                data_quality_score=data_quality_score,
                processing_notes=processing_notes
            )
            
            overall_score_str = f"{overall_score:.3f}" if overall_score is not None else "N/A"
            logger.info(f"Completed ratio calculation for {ticker} - Overall score: {overall_score_str}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to calculate ratios for {ticker}: {e}")
            raise
    
    def _calculate_ratio_temporal(self, ratio_name: str, ratio_config: Dict[str, Any], 
                                processed_data: pd.DataFrame) -> RatioResult:
        """
        Calculate a single ratio across all temporal perspectives.
        
        Args:
            ratio_name: Name of the ratio to calculate
            ratio_config: Configuration for this ratio
            processed_data: Processed financial data
            
        Returns:
            RatioResult with temporal perspective values
        """
        ratio_function = self.ratio_functions_registry[ratio_name]
        calculation_notes = []
        
        # Calculate for each temporal perspective
        current_value = None
        trend_value = None
        stability_value = None
        
        # Check if we have quarterly TTM time series data for advanced temporal analysis
        first_row = processed_data.iloc[0] if not processed_data.empty else None
        has_time_series = (first_row is not None and 
                          'quarterly_ttm_series' in first_row and 
                          isinstance(first_row['quarterly_ttm_series'], dict))
        
        for perspective in self.temporal_perspectives:
            try:
                # Extract data for this temporal perspective
                perspective_data = self._extract_perspective_data(
                    processed_data, perspective.quarters_back
                )
                
                if perspective_data is None or perspective_data.empty:
                    if perspective.quarters_back == 0:
                        calculation_notes.append(f"No current TTM data available for {perspective.name}")
                    else:
                        calculation_notes.append(f"Temporal analysis not available for {perspective.name} (requires quarterly TTM time series)")
                    continue
                
                # Handle different perspective calculations
                if perspective.name == 'current_ttm':
                    # Current perspective: use standard ratio calculation
                    required_fields = ratio_config.get('required_fields', [])
                    price_fields = ratio_config.get('price_fields', [])
                    parameters = self._extract_ratio_parameters(perspective_data, required_fields, price_fields)
                    
                    if parameters:
                        current_value = ratio_function(**parameters)
                    else:
                        calculation_notes.append(f"Missing required fields for {perspective.name}")
                
                elif perspective.name == 'trend_ttm' and has_time_series:
                    # Trend perspective: calculate from time series if available
                    time_series_data = first_row['quarterly_ttm_series']
                    trend_value = self._calculate_trend_from_time_series(
                        time_series_data, ratio_name, ratio_config, processed_data
                    )
                    if trend_value is None:
                        calculation_notes.append("Insufficient time series data for trend calculation")
                
                elif perspective.name == 'stability_ttm' and has_time_series:
                    # Stability perspective: calculate from time series if available
                    time_series_data = first_row['quarterly_ttm_series']
                    stability_value = self._calculate_stability_from_time_series(
                        time_series_data, ratio_name, ratio_config, processed_data
                    )
                    if stability_value is None:
                        calculation_notes.append("Insufficient time series data for stability calculation")
                
                else:
                    # Fallback: try standard calculation for historical perspectives
                    required_fields = ratio_config.get('required_fields', [])
                    price_fields = ratio_config.get('price_fields', [])
                    parameters = self._extract_ratio_parameters(perspective_data, required_fields, price_fields)
                    
                    if parameters:
                        value = ratio_function(**parameters)
                        if perspective.name == 'trend_ttm':
                            trend_value = value
                        elif perspective.name == 'stability_ttm':
                            stability_value = value
                    else:
                        calculation_notes.append(f"Missing required fields for {perspective.name}")
                
            except Exception as e:
                calculation_notes.append(f"Error in {perspective.name}: {str(e)}")
        
        # Calculate composite score using temporal perspective weights
        composite_score = self._calculate_composite_score(
            current_value, trend_value, stability_value
        )
        
        # Assess data quality for this ratio
        data_quality = self._assess_ratio_data_quality(
            processed_data, ratio_config.get('required_fields', [])
        )
        
        return RatioResult(
            ratio_name=ratio_name,
            current_value=current_value,
            trend_value=trend_value,
            stability_value=stability_value,
            composite_score=composite_score,
            data_quality=data_quality,
            calculation_notes=calculation_notes
        )
    
    def _extract_perspective_data(self, processed_data: pd.DataFrame, 
                                quarters_back: int) -> Optional[pd.DataFrame]:
        """
        Extract data for a specific temporal perspective.
        
        Args:
            processed_data: Full processed data
            quarters_back: Number of quarters to go back from most recent (0=current, >0=historical)
            
        Returns:
            DataFrame with data for this perspective, or None if not available
        """
        if processed_data.empty:
            return None
        
        # Check if quarterly TTM time series data is available
        first_row = processed_data.iloc[0]
        has_time_series = 'quarterly_ttm_series' in first_row and \
                         isinstance(first_row['quarterly_ttm_series'], dict)
        
        if quarters_back == 0:
            # Current perspective: use available single-row TTM data
            return processed_data.copy()
        
        elif has_time_series and quarters_back > 0:
            # Historical perspectives: use quarterly TTM time series if available
            time_series_data = first_row['quarterly_ttm_series']
            
            # For trend/stability analysis, we don't need specific historical quarters
            # We just need the full time series to be available for temporal calculations
            # The trend/stability logic will use the entire time series
            if any(isinstance(values, list) and len(values) >= self.config['data_quality']['min_quarters_for_trend']
                   for values in time_series_data.values()):
                
                # Return the original data with time series - temporal calculations will use the full series
                return processed_data.copy()
            else:
                # Not enough historical data available
                return None
        else:
            # Historical perspectives without time series data: not available
            return None
    
    def _extract_ratio_parameters(self, processed_data: pd.DataFrame, 
                                required_fields: List[str], price_fields: List[str] = None) -> Dict[str, float]:
        """
        Extract required parameters for ratio calculation from processed data.
        
        Args:
            processed_data: Processed financial data
            required_fields: List of field names required for the ratio
            price_fields: List of price field names required for the ratio
            
        Returns:
            Dictionary of parameter names and values for ratio function
        """
        if processed_data.empty:
            return {}
        
        # Initialize price_fields if None
        if price_fields is None:
            price_fields = []
        
        # Combine all required fields
        all_required_fields = required_fields + price_fields
        
        if not all_required_fields:
            return {}
        
        logger.debug(f"Extracting parameters for fields: {all_required_fields}")
        
        # For temporal calculations, use the most recent data point
        latest_data = processed_data.iloc[0] if len(processed_data) > 0 else processed_data
        
        # Create parameter mapping based on field names
        parameters = {}
        field_mapping = {
            # Direct field mappings - updated to match actual processed data column names
            'Net Income TTM': 'net_income_ttm',
            'Stockholders Equity': 'stockholders_equity', 
            'EBIT TTM': 'ebit_ttm',
            'Tax Provision TTM': 'tax_provision_ttm',
            'Pretax Income TTM': 'pretax_income_ttm',
            'Total Debt': 'total_debt',
            'Gross Profit TTM': 'gross_profit_ttm',
            'Total Revenue TTM': 'total_revenue_ttm',
            'Operating Income TTM': 'operating_income_ttm',
            'Total Assets': 'total_assets',
            'Operating Cash Flow TTM': 'operating_cash_flow_ttm',
            'Free Cash Flow TTM': 'free_cash_flow_ttm',
            'Current Price': 'price',  # Updated for ratio functions
            'Historical Price': 'price',  # Updated for ratio functions
            'current_price': 'price',  # For price_fields from config
            'quarterly_aligned_price': 'price',  # For historical price fields
            'Market Cap': 'market_cap',
            'Enterprise Value': 'enterprise_value',
            'EBITDA TTM': 'ebitda_ttm',
            'Cash And Cash Equivalents': 'cash_and_equivalents',  # Fixed parameter name
            'Shares Outstanding': 'shares_outstanding',
        }
        
        for field in all_required_fields:
            if field in latest_data and pd.notna(latest_data[field]):
                param_name = field_mapping.get(field, field.lower().replace(' ', '_'))
                parameters[param_name] = float(latest_data[field])
            else:
                logger.debug(f"Missing field '{field}' in data columns: {list(latest_data.index)}")
        
        logger.debug(f"Extracted parameters: {parameters}")
        return parameters
    
    def _calculate_composite_score(self, current_value: Optional[float], 
                                 trend_value: Optional[float], 
                                 stability_value: Optional[float]) -> Optional[float]:
        """
        Calculate composite score using temporal perspective weights.
        
        Args:
            current_value: Current TTM value
            trend_value: Trend TTM value
            stability_value: Stability TTM value
            
        Returns:
            Weighted composite score or None if insufficient data
        """
        values = [current_value, trend_value, stability_value]
        weights = [tp.weight for tp in self.temporal_perspectives]
        
        # Filter out None values and corresponding weights
        valid_pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
        
        if not valid_pairs:
            return None
        
        if len(valid_pairs) < len(values):
            # Renormalize weights for available values
            valid_values, valid_weights = zip(*valid_pairs)
            weight_sum = sum(valid_weights)
            normalized_weights = [w / weight_sum for w in valid_weights]
            
            return sum(v * w for v, w in zip(valid_values, normalized_weights))
        
        return sum(v * w for v, w in valid_pairs)
    
    def _calculate_category_scores(self, ratio_results: Dict[str, RatioResult]) -> Dict[str, float]:
        """
        Calculate weighted scores for each ratio category.
        
        Args:
            ratio_results: Dictionary of ratio calculation results
            
        Returns:
            Dictionary of category scores
        """
        category_scores = {}
        
        for category_name, category_config in self.config['category_weights'].items():
            category_ratios = category_config['ratios']
            
            # Calculate weighted average for this category
            total_score = 0.0
            total_weight = 0.0
            
            for ratio_name in category_ratios:
                if ratio_name in ratio_results:
                    ratio_result = ratio_results[ratio_name]
                    ratio_weight = self.config['ratio_definitions'][ratio_name].get('weight', 1.0)
                    
                    if ratio_result.composite_score is not None:
                        total_score += ratio_result.composite_score * ratio_weight
                        total_weight += ratio_weight
            
            # Calculate category score
            if total_weight > 0:
                category_scores[category_name] = total_score / total_weight
            else:
                category_scores[category_name] = 0.0
        
        return category_scores
    
    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> Optional[float]:
        """
        Calculate overall score from category scores using category weights.
        
        Args:
            category_scores: Dictionary of category scores
            
        Returns:
            Overall weighted score or None if insufficient data
        """
        total_score = 0.0
        total_weight = 0.0
        
        for category_name, category_score in category_scores.items():
            if category_score > 0:  # Only include categories with valid scores
                category_weight = self.config['category_weights'][category_name]['default_weight']
                total_score += category_score * category_weight
                total_weight += category_weight
        
        if total_weight > 0:
            return total_score / total_weight
        
        return None
    
    def _assess_data_quality(self, processed_data) -> float:
        """
        Assess overall data quality for a stock's processed data.
        
        Args:
            processed_data: Processed financial data (DataFrame or Series)
            
        Returns:
            Data quality score between 0.0 and 1.0
        """
        # Handle both DataFrame and Series input
        if hasattr(processed_data, 'empty') and processed_data.empty:
            return 0.0
        elif hasattr(processed_data, 'isna') and len(processed_data) == 0:
            return 0.0
        
        # Calculate field completeness
        if isinstance(processed_data, pd.DataFrame):
            total_fields = len(processed_data.columns) - 1  # Exclude date column
            total_values = len(processed_data) * total_fields
            non_null_values = processed_data.select_dtypes(include=[np.number]).count().sum()
        else:  # Series
            # For Series, exclude non-numeric columns like quarterly_ttm_series
            numeric_data = processed_data.select_dtypes(include=[np.number]) if hasattr(processed_data, 'select_dtypes') else processed_data
            if isinstance(numeric_data, pd.Series):
                total_fields = len(numeric_data)
                total_values = total_fields
                non_null_values = numeric_data.count()
            else:
                # Fallback calculation
                total_fields = len([v for k, v in processed_data.items() if isinstance(v, (int, float, np.number)) and k != 'quarterly_ttm_series'])
                total_values = total_fields
                non_null_values = len([v for k, v in processed_data.items() if pd.notna(v) and isinstance(v, (int, float, np.number)) and k != 'quarterly_ttm_series'])
        
        field_completeness = non_null_values / total_values if total_values > 0 else 0.0
        
        # Check temporal coverage
        min_quarters = self.config['data_quality']['min_quarters_for_trend']
        if isinstance(processed_data, pd.DataFrame):
            temporal_coverage = min(len(processed_data) / min_quarters, 1.0)
        else:
            # For Series, check if quarterly_ttm_series exists and has enough data
            if 'quarterly_ttm_series' in processed_data and isinstance(processed_data['quarterly_ttm_series'], dict):
                ttm_series = processed_data['quarterly_ttm_series']
                # Check if any metric has enough temporal data
                max_temporal_length = max(len(series) for series in ttm_series.values()) if ttm_series else 0
                temporal_coverage = min(max_temporal_length / min_quarters, 1.0)
            else:
                temporal_coverage = 0.2  # Default minimal coverage for single-period data
        
        # Combine metrics (equal weighting)
        overall_quality = (field_completeness + temporal_coverage) / 2.0
        
        return overall_quality
    
    def _assess_ratio_data_quality(self, processed_data: pd.DataFrame, 
                                 required_fields: List[str]) -> float:
        """
        Assess data quality for a specific ratio's required fields.
        
        Args:
            processed_data: Processed financial data
            required_fields: List of field names required for this ratio
            
        Returns:
            Data quality score for this ratio between 0.0 and 1.0
        """
        if not required_fields or processed_data.empty:
            return 0.0
        
        # Check field availability and completeness
        available_fields = [field for field in required_fields if field in processed_data.columns]
        field_availability = len(available_fields) / len(required_fields)
        
        if not available_fields:
            return 0.0
        
        # Check data completeness for available fields
        field_data = processed_data[available_fields]
        total_values = len(field_data) * len(available_fields)
        non_null_values = field_data.count().sum()
        
        field_completeness = non_null_values / total_values if total_values > 0 else 0.0
        
        # Combine metrics
        ratio_quality = (field_availability + field_completeness) / 2.0
        
        return ratio_quality
    
    def calculate_batch_ratios(self, stock_data: Dict[str, Dict[str, Any]], 
                             progress_callback: Optional[callable] = None) -> Dict[str, StockAnalysis]:
        """
        Calculate ratios for multiple stocks in batch.
        
        Args:
            stock_data: Dictionary with structure {ticker: {'financial': raw_dict, 'price': df}}
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            Dictionary of StockAnalysis results by ticker
        """
        results = {}
        total_stocks = len(stock_data)
        
        logger.info(f"Starting batch calculation for {total_stocks} stocks")
        
        for i, (ticker, data) in enumerate(stock_data.items()):
            try:
                raw_financial_data = data.get('financial')
                price_data = data.get('price')
                
                if raw_financial_data is None:
                    logger.warning(f"No financial data for {ticker}")
                    continue
                
                # Calculate ratios for this stock
                analysis = self.calculate_stock_ratios(ticker, raw_financial_data, price_data)
                results[ticker] = analysis
                
                # Report progress
                if progress_callback:
                    progress = (i + 1) / total_stocks
                    progress_callback(ticker, progress, analysis.overall_score)
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                continue
        
        logger.info(f"Completed batch calculation for {len(results)}/{total_stocks} stocks")
        return results
    
    def export_results(self, results: Dict[str, StockAnalysis], 
                      output_path: str, format: str = 'csv') -> None:
        """
        Export calculation results to file.
        
        Args:
            results: Dictionary of StockAnalysis results
            output_path: Path for output file
            format: Export format ('csv', 'excel', 'json')
        """
        if format == 'csv':
            self._export_to_csv(results, output_path)
        elif format == 'excel':
            self._export_to_excel(results, output_path)
        elif format == 'json':
            self._export_to_json(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported results to {output_path}")
    
    def _export_to_csv(self, results: Dict[str, StockAnalysis], output_path: str) -> None:
        """Export results to CSV format"""
        rows = []
        
        for ticker, analysis in results.items():
            row = {
                'ticker': ticker,
                'calculation_date': analysis.calculation_date.isoformat(),
                'overall_score': analysis.overall_score,
                'data_quality_score': analysis.data_quality_score
            }
            
            # Add category scores
            for category, score in analysis.category_scores.items():
                row[f'category_{category}'] = score
            
            # Add ratio scores (composite only for CSV simplicity)
            for ratio_name, ratio_result in analysis.ratio_results.items():
                row[f'ratio_{ratio_name}'] = ratio_result.composite_score
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    def _export_to_excel(self, results: Dict[str, StockAnalysis], output_path: str) -> None:
        """Export results to Excel format with multiple sheets"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_rows = []
            for ticker, analysis in results.items():
                summary_rows.append({
                    'ticker': ticker,
                    'overall_score': analysis.overall_score,
                    'data_quality_score': analysis.data_quality_score,
                    **{f'category_{k}': v for k, v in analysis.category_scores.items()}
                })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed ratios sheet
            ratio_rows = []
            for ticker, analysis in results.items():
                for ratio_name, ratio_result in analysis.ratio_results.items():
                    ratio_rows.append({
                        'ticker': ticker,
                        'ratio': ratio_name,
                        'current_ttm': ratio_result.current_value,
                        'trend_ttm': ratio_result.trend_value,
                        'stability_ttm': ratio_result.stability_value,
                        'composite_score': ratio_result.composite_score,
                        'data_quality': ratio_result.data_quality
                    })
            
            ratios_df = pd.DataFrame(ratio_rows)
            ratios_df.to_excel(writer, sheet_name='Detailed_Ratios', index=False)
    
    def _export_to_json(self, results: Dict[str, StockAnalysis], output_path: str) -> None:
        """Export results to JSON format"""
        json_data = {}
        
        for ticker, analysis in results.items():
            json_data[ticker] = {
                'calculation_date': analysis.calculation_date.isoformat(),
                'overall_score': analysis.overall_score,
                'data_quality_score': analysis.data_quality_score,
                'category_scores': analysis.category_scores,
                'ratio_results': {
                    ratio_name: {
                        'current_value': result.current_value,
                        'trend_value': result.trend_value,
                        'stability_value': result.stability_value,
                        'composite_score': result.composite_score,
                        'data_quality': result.data_quality,
                        'calculation_notes': result.calculation_notes
                    }
                    for ratio_name, result in analysis.ratio_results.items()
                },
                'processing_notes': analysis.processing_notes
            }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _calculate_trend_from_time_series(self, time_series_data: Dict[str, List[float]], 
                                         ratio_name: str, ratio_config: Dict[str, Any],
                                         current_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate trend value from quarterly TTM time series.
        Trend represents the average percentage change per quarter.
        
        Args:
            time_series_data: Dictionary with TTM field names and their quarterly values
            ratio_name: Name of the ratio being calculated
            ratio_config: Configuration for the ratio
            current_data: Current row data for non-time-series fields (balance sheet items)
            
        Returns:
            Trend value (average % change per quarter) or None if insufficient data
        """
        try:
            # Get required fields for this ratio
            required_fields = ratio_config.get('required_fields', [])
            ratio_function = self.ratio_functions_registry.get(ratio_name)
            
            if not ratio_function or not required_fields:
                return None
            
            # Calculate the ratio for each period in the time series
            ratio_values = []
            
            # Determine how many periods we can calculate (only for time series fields)
            time_series_fields = [field for field in required_fields 
                                 if field in time_series_data and isinstance(time_series_data[field], list)]
            
            if not time_series_fields:
                return None
            
            min_periods = min(len(time_series_data[field]) for field in time_series_fields)
            
            if min_periods < 2:  # Need at least 2 periods for trend
                return None
            
            # Calculate ratio for each available period
            for period_idx in range(min(min_periods, 8)):  # Limit to 8 quarters (2 years)
                try:
                    # Extract parameters for this period
                    period_params = {}
                    field_mapping = {
                        'Net Income TTM': 'net_income_ttm',
                        'Stockholders Equity': 'stockholders_equity',
                        'EBIT TTM': 'ebit_ttm',
                        'Total Revenue TTM': 'total_revenue_ttm',
                        'Total Assets': 'total_assets',
                        'Basic EPS TTM': 'basic_eps_ttm',
                        'Free Cash Flow TTM': 'free_cash_flow_ttm',
                        'Operating Cash Flow TTM': 'operating_cash_flow_ttm',
                        'Gross Profit TTM': 'gross_profit_ttm',
                        'Operating Income TTM': 'operating_income_ttm'
                    }
                    
                    for field in required_fields:
                        param_name = field_mapping.get(field, field.lower().replace(' ', '_'))
                        
                        if field in time_series_data and isinstance(time_series_data[field], list):
                            # Use time series data for this field
                            if len(time_series_data[field]) > period_idx:
                                period_params[param_name] = time_series_data[field][period_idx]
                        else:
                            # Use current data for non-time-series fields (balance sheet items)
                            if not current_data.empty and field in current_data.iloc[0]:
                                period_params[param_name] = current_data.iloc[0][field]
                    
                    # Calculate ratio if we have all required parameters
                    if len(period_params) == len(required_fields):
                        ratio_value = ratio_function(**period_params)
                        if ratio_value is not None and not np.isnan(ratio_value):
                            ratio_values.append(ratio_value)
                
                except Exception as e:
                    logger.debug(f"Error calculating ratio for period {period_idx}: {e}")
                    continue
            
            # Calculate trend as average percentage change per quarter
            if len(ratio_values) >= 2:
                percentage_changes = []
                for i in range(1, len(ratio_values)):
                    if ratio_values[i] != 0:  # Avoid division by zero
                        pct_change = (ratio_values[i-1] - ratio_values[i]) / abs(ratio_values[i]) * 100
                        if not np.isnan(pct_change) and abs(pct_change) < 1000:  # Filter outliers
                            percentage_changes.append(pct_change)
                
                if percentage_changes:
                    return np.mean(percentage_changes)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating trend for {ratio_name}: {e}")
            return None
    
    def _calculate_stability_from_time_series(self, time_series_data: Dict[str, List[float]], 
                                            ratio_name: str, ratio_config: Dict[str, Any],
                                            current_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate stability value from quarterly TTM time series.
        Stability represents consistency (inverse of coefficient of variation).
        
        Args:
            time_series_data: Dictionary with TTM field names and their quarterly values
            ratio_name: Name of the ratio being calculated
            ratio_config: Configuration for the ratio
            current_data: Current row data for non-time-series fields (balance sheet items)
            
        Returns:
            Stability value (1 / coefficient_of_variation) or None if insufficient data
        """
        try:
            # Get required fields for this ratio
            required_fields = ratio_config.get('required_fields', [])
            ratio_function = self.ratio_functions_registry.get(ratio_name)
            
            if not ratio_function or not required_fields:
                return None
            
            # Calculate the ratio for each period in the time series
            ratio_values = []
            
            # Determine how many periods we can calculate (only for time series fields)
            time_series_fields = [field for field in required_fields 
                                 if field in time_series_data and isinstance(time_series_data[field], list)]
            
            if not time_series_fields:
                return None
            
            min_periods = min(len(time_series_data[field]) for field in time_series_fields)
            
            if min_periods < 3:  # Need at least 3 periods for stability measure
                return None
            
            # Calculate ratio for each available period (same logic as trend)
            for period_idx in range(min(min_periods, 8)):  # Limit to 8 quarters (2 years)
                try:
                    period_params = {}
                    field_mapping = {
                        'Net Income TTM': 'net_income_ttm',
                        'Stockholders Equity': 'stockholders_equity', 
                        'EBIT TTM': 'ebit_ttm',
                        'Total Revenue TTM': 'total_revenue_ttm',
                        'Total Assets': 'total_assets',
                        'Basic EPS TTM': 'basic_eps_ttm',
                        'Free Cash Flow TTM': 'free_cash_flow_ttm',
                        'Operating Cash Flow TTM': 'operating_cash_flow_ttm',
                        'Gross Profit TTM': 'gross_profit_ttm',
                        'Operating Income TTM': 'operating_income_ttm'
                    }
                    
                    for field in required_fields:
                        param_name = field_mapping.get(field, field.lower().replace(' ', '_'))
                        
                        if field in time_series_data and isinstance(time_series_data[field], list):
                            # Use time series data for this field
                            if len(time_series_data[field]) > period_idx:
                                period_params[param_name] = time_series_data[field][period_idx]
                        else:
                            # Use current data for non-time-series fields (balance sheet items)
                            if not current_data.empty and field in current_data.iloc[0]:
                                period_params[param_name] = current_data.iloc[0][field]
                    
                    # Calculate ratio if we have all required parameters
                    if len(period_params) == len(required_fields):
                        ratio_value = ratio_function(**period_params)
                        if ratio_value is not None and not np.isnan(ratio_value):
                            ratio_values.append(ratio_value)
                
                except Exception as e:
                    logger.debug(f"Error calculating ratio for period {period_idx}: {e}")
                    continue
            
            # Calculate stability as inverse coefficient of variation
            if len(ratio_values) >= 3:
                mean_value = np.mean(ratio_values)
                if mean_value != 0:
                    std_value = np.std(ratio_values)
                    coefficient_of_variation = std_value / abs(mean_value)
                    
                    # Return stability as inverse of CV (higher = more stable)
                    # Cap the result to prevent extreme values
                    if coefficient_of_variation > 0:
                        stability = min(1.0 / coefficient_of_variation, 10.0)
                        return stability
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating stability for {ratio_name}: {e}")
            return None


def main():
    """
    Test the RatioCalculator with sample data.
    """
    try:
        # Initialize calculator
        calculator = RatioCalculator()
        
        # Create sample processed data directly (bypassing FinancialDataProcessor for testing)
        sample_processed_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=16, freq='QS'),
            'Total Revenue TTM': [10000000] * 16,
            'Net Income TTM': [1000000] * 16,
            'Total Assets': [50000000] * 16,
            'Stockholders Equity': [20000000] * 16,
            'Total Debt': [15000000] * 16,
            'Operating Cash Flow TTM': [1200000] * 16,
            'Free Cash Flow TTM': [800000] * 16,
            'Gross Profit TTM': [4000000] * 16,
            'Operating Income TTM': [1500000] * 16,
            'EBIT TTM': [1500000] * 16,
            'Tax Provision TTM': [200000] * 16,
            'Pretax Income TTM': [1200000] * 16,
        })
        
        print("Testing RatioCalculator with sample processed data...")
        
        # Test individual ratio calculations using parameter extraction
        print("\nTesting individual ratio calculations:")
        
        # Test ROE calculation
        try:
            roe_config = calculator.config['ratio_definitions']['ROE']
            roe_params = calculator._extract_ratio_parameters(
                sample_processed_data, roe_config['required_fields'], roe_config.get('price_fields', [])
            )
            print(f"ROE parameters: {roe_params}")
            if roe_params:
                roe_value = ratio_functions.roe(**roe_params)
                print(f"ROE: {roe_value:.3f}")
            else:
                print("ROE: Missing required parameters")
        except Exception as e:
            print(f"ROE calculation failed: {e}")
        
        # Test ROIC calculation  
        try:
            roic_config = calculator.config['ratio_definitions']['ROIC']
            roic_params = calculator._extract_ratio_parameters(
                sample_processed_data, roic_config['required_fields'], roic_config.get('price_fields', [])
            )
            print(f"ROIC parameters: {roic_params}")
            if roic_params:
                roic_value = ratio_functions.roic(**roic_params)
                print(f"ROIC: {roic_value:.3f}")
            else:
                print("ROIC: Missing required parameters")
        except Exception as e:
            print(f"ROIC calculation failed: {e}")
        
        # Test temporal perspective calculation
        print("\nTesting temporal perspective calculations:")
        
        for perspective in calculator.temporal_perspectives:
            print(f"  {perspective.name}: weight={perspective.weight}, quarters_back={perspective.quarters_back}")
            
            perspective_data = calculator._extract_perspective_data(
                sample_processed_data, perspective.quarters_back
            )
            print(f"    Data points: {len(perspective_data)}")
        
        # Test category configuration
        print("\nCategory configuration:")
        for category_name, category_config in calculator.config['category_weights'].items():
            print(f"  {category_name}: weight={category_config['default_weight']}, ratios={category_config['ratios']}")
        
        print("\nRatioCalculator test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()