"""
Swedish Stock Analysis - Ratio Functions Module
===============================================

Secure, testable ratio calculation functions for Swedish stock analysis platform.
Replaces eval() usage with explicit function calls for improved security and performance.

This module implements all 13 Swedish financial ratios across 5 categories:
- Kvalitet (Quality): ROE, ROIC
- Lönsamhet (Profitability): Bruttomarginal, Rörelsemarginal, Vinstmarginal
- Finansiell Hälsa (Financial Health): Soliditet, Skuldsättningsgrad
- Kassaflöde (Cash Flow): Rörelseflödesmarginal, Kassaflöde_till_Skuld
- Värdering (Valuation): PE_tal, PB_tal, EV_EBITDA

All functions handle missing data gracefully and return numpy.nan for invalid calculations.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

# Type aliases for clarity
FinancialValue = Union[float, int, np.number]
OptionalFinancialValue = Optional[FinancialValue]

# Constants for error handling
DIVISION_BY_ZERO_THRESHOLD = 1e-10
MIN_VALID_VALUE = 1e-6


def _safe_divide(numerator: OptionalFinancialValue, 
                denominator: OptionalFinancialValue, 
                allow_negative_denominator: bool = False) -> float:
    """
    Safely perform division with comprehensive error handling.
    
    Args:
        numerator: The dividend value
        denominator: The divisor value  
        allow_negative_denominator: Whether to allow negative denominators
        
    Returns:
        float: Division result or np.nan if invalid
    """
    # Check for None or NaN values
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    
    # Convert to float for consistent handling
    try:
        num_val = float(numerator)
        den_val = float(denominator)
    except (ValueError, TypeError):
        return np.nan
    
    # Check for invalid denominator
    if abs(den_val) < DIVISION_BY_ZERO_THRESHOLD:
        return np.nan
    
    # Check for negative denominator if not allowed
    if not allow_negative_denominator and den_val < 0:
        return np.nan
    
    # Perform division
    try:
        result = num_val / den_val
        # Check for infinite or invalid results
        if not np.isfinite(result):
            return np.nan
        return result
    except (ZeroDivisionError, OverflowError):
        return np.nan


def _validate_positive(value: OptionalFinancialValue, 
                      allow_zero: bool = False) -> bool:
    """
    Validate that a financial value is positive (and optionally non-zero).
    
    Args:
        value: The value to validate
        allow_zero: Whether zero is considered valid
        
    Returns:
        bool: True if valid, False otherwise
    """
    if pd.isna(value):
        return False
    
    try:
        val = float(value)
        if allow_zero:
            return val >= 0
        else:
            return val > MIN_VALID_VALUE
    except (ValueError, TypeError):
        return False


# =============================================================================
# KVALITET (QUALITY) - Management Efficiency & Capital Returns
# =============================================================================

def roe(net_income_ttm: OptionalFinancialValue, 
        stockholders_equity: OptionalFinancialValue) -> float:
    """
    Calculate Return on Equity (Avkastning på eget kapital).
    
    ROE measures how efficiently a company generates profit from shareholders' equity.
    Higher values indicate better management efficiency.
    
    Formula: Net Income TTM / Stockholders Equity
    
    Args:
        net_income_ttm: Net income for trailing twelve months
        stockholders_equity: Total shareholders' equity
        
    Returns:
        float: ROE ratio or np.nan if calculation invalid
        
    Example:
        >>> roe(1000000, 5000000)  # 20% ROE
        0.2
        >>> roe(None, 5000000)     # Missing data
        nan
    """
    # Validate equity is positive (negative equity makes ROE meaningless)
    if not _validate_positive(stockholders_equity):
        return np.nan
    
    return _safe_divide(net_income_ttm, stockholders_equity, allow_negative_denominator=False)


def roic(ebit_ttm: OptionalFinancialValue,
         tax_provision_ttm: OptionalFinancialValue,
         pretax_income_ttm: OptionalFinancialValue,
         total_debt: OptionalFinancialValue,
         stockholders_equity: OptionalFinancialValue) -> float:
    """
    Calculate Return on Invested Capital (Avkastning på investerat kapital).
    
    ROIC measures how efficiently a company uses both debt and equity to generate returns.
    Higher values indicate better capital allocation efficiency.
    
    Formula: (EBIT * (1 - Tax Rate)) / (Total Debt + Stockholders Equity)
    Where Tax Rate = Tax Provision / Pretax Income
    
    Args:
        ebit_ttm: Earnings before interest and taxes (TTM)
        tax_provision_ttm: Tax provision (TTM)  
        pretax_income_ttm: Pretax income (TTM)
        total_debt: Total debt
        stockholders_equity: Total shareholders' equity
        
    Returns:
        float: ROIC ratio or np.nan if calculation invalid
        
    Example:
        >>> roic(2000000, 400000, 1500000, 3000000, 5000000)
        0.175  # 17.5% ROIC
    """
    # Calculate tax rate with safety checks
    if pd.isna(tax_provision_ttm) or pd.isna(pretax_income_ttm):
        tax_rate = 0.0  # Assume no taxes if data missing
    else:
        try:
            if abs(float(pretax_income_ttm)) < DIVISION_BY_ZERO_THRESHOLD:
                tax_rate = 0.0
            else:
                tax_rate = float(tax_provision_ttm) / float(pretax_income_ttm)
                # Cap tax rate at reasonable bounds
                tax_rate = max(0.0, min(1.0, tax_rate))
        except (ValueError, TypeError):
            tax_rate = 0.0
    
    # Calculate after-tax EBIT
    if pd.isna(ebit_ttm):
        return np.nan
    
    try:
        after_tax_ebit = float(ebit_ttm) * (1 - tax_rate)
    except (ValueError, TypeError):
        return np.nan
    
    # Calculate invested capital (debt + equity)
    if pd.isna(total_debt):
        total_debt = 0.0  # Assume no debt if missing
    if not _validate_positive(stockholders_equity):
        return np.nan
    
    try:
        invested_capital = float(total_debt) + float(stockholders_equity)
    except (ValueError, TypeError):
        return np.nan
    
    return _safe_divide(after_tax_ebit, invested_capital)


# =============================================================================
# LÖNSAMHET (PROFITABILITY) - Operational Efficiency
# =============================================================================

def bruttomarginal(gross_profit_ttm: OptionalFinancialValue,
                   total_revenue_ttm: OptionalFinancialValue) -> float:
    """
    Calculate Gross Margin (Bruttomarginal).
    
    Gross margin measures production efficiency by showing what percentage
    of revenue remains after direct production costs.
    
    Formula: Gross Profit TTM / Total Revenue TTM
    
    Args:
        gross_profit_ttm: Gross profit for trailing twelve months
        total_revenue_ttm: Total revenue for trailing twelve months
        
    Returns:
        float: Gross margin ratio or np.nan if calculation invalid
        
    Example:
        >>> bruttomarginal(3000000, 10000000)
        0.3  # 30% gross margin
    """
    if not _validate_positive(total_revenue_ttm):
        return np.nan
    
    return _safe_divide(gross_profit_ttm, total_revenue_ttm)


def rorelsemarginal(operating_income_ttm: OptionalFinancialValue,
                    total_revenue_ttm: OptionalFinancialValue) -> float:
    """
    Calculate Operating Margin (Rörelsemarginal).
    
    Operating margin measures operational efficiency by showing what percentage
    of revenue remains after all operating expenses.
    
    Formula: Operating Income TTM / Total Revenue TTM
    
    Args:
        operating_income_ttm: Operating income for trailing twelve months
        total_revenue_ttm: Total revenue for trailing twelve months
        
    Returns:
        float: Operating margin ratio or np.nan if calculation invalid
        
    Example:
        >>> rorelsemarginal(1500000, 10000000)
        0.15  # 15% operating margin
    """
    if not _validate_positive(total_revenue_ttm):
        return np.nan
    
    return _safe_divide(operating_income_ttm, total_revenue_ttm)


def vinstmarginal(net_income_ttm: OptionalFinancialValue,
                  total_revenue_ttm: OptionalFinancialValue) -> float:
    """
    Calculate Net Profit Margin (Vinstmarginal).
    
    Net profit margin measures overall profitability by showing what percentage
    of revenue becomes profit after all expenses, taxes, and interest.
    
    Formula: Net Income TTM / Total Revenue TTM
    
    Args:
        net_income_ttm: Net income for trailing twelve months
        total_revenue_ttm: Total revenue for trailing twelve months
        
    Returns:
        float: Net profit margin ratio or np.nan if calculation invalid
        
    Example:
        >>> vinstmarginal(1000000, 10000000)
        0.1  # 10% net profit margin
    """
    if not _validate_positive(total_revenue_ttm):
        return np.nan
    
    return _safe_divide(net_income_ttm, total_revenue_ttm)


# =============================================================================
# FINANSIELL HÄLSA (FINANCIAL HEALTH) - Stability & Debt Management
# =============================================================================

def soliditet(stockholders_equity: OptionalFinancialValue,
              total_assets: OptionalFinancialValue) -> float:
    """
    Calculate Equity Ratio (Soliditet).
    
    Equity ratio measures financial stability by showing what percentage
    of assets are financed by shareholders' equity rather than debt.
    
    Formula: Stockholders Equity / Total Assets
    
    Args:
        stockholders_equity: Total shareholders' equity
        total_assets: Total assets
        
    Returns:
        float: Equity ratio or np.nan if calculation invalid
        
    Example:
        >>> soliditet(5000000, 10000000)
        0.5  # 50% equity ratio
    """
    if not _validate_positive(total_assets):
        return np.nan
    
    return _safe_divide(stockholders_equity, total_assets)


def skuldsattningsgrad(total_debt: OptionalFinancialValue,
                       stockholders_equity: OptionalFinancialValue) -> float:
    """
    Calculate Debt-to-Equity Ratio (Skuldsättningsgrad).
    
    Debt-to-equity ratio measures financial leverage by comparing
    total debt to shareholders' equity. Lower values indicate less financial risk.
    
    Formula: Total Debt / Stockholders Equity
    
    Args:
        total_debt: Total debt
        stockholders_equity: Total shareholders' equity
        
    Returns:
        float: Debt-to-equity ratio or np.nan if calculation invalid
        
    Example:
        >>> skuldsattningsgrad(3000000, 5000000)
        0.6  # 0.6 debt-to-equity ratio
    """
    if not _validate_positive(stockholders_equity):
        return np.nan
    
    # Allow zero debt (debt-free companies)
    if pd.isna(total_debt):
        total_debt = 0.0
    
    try:
        debt_val = float(total_debt)
        if debt_val < 0:
            return np.nan  # Negative debt doesn't make sense
        return _safe_divide(debt_val, stockholders_equity)
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# KASSAFLÖDE (CASH FLOW) - Cash Generation & Financial Flexibility
# =============================================================================

def rorelseflodesmarginal(operating_cash_flow_ttm: OptionalFinancialValue,
                          total_revenue_ttm: OptionalFinancialValue) -> float:
    """
    Calculate Operating Cash Flow Margin (Rörelseflödesmarginal).
    
    Operating cash flow margin measures how efficiently a company converts
    revenue into actual cash flow from operations.
    
    Formula: Operating Cash Flow TTM / Total Revenue TTM
    
    Args:
        operating_cash_flow_ttm: Operating cash flow for trailing twelve months
        total_revenue_ttm: Total revenue for trailing twelve months
        
    Returns:
        float: Operating cash flow margin or np.nan if calculation invalid
        
    Example:
        >>> rorelseflodesmarginal(1200000, 10000000)
        0.12  # 12% operating cash flow margin
    """
    if not _validate_positive(total_revenue_ttm):
        return np.nan
    
    return _safe_divide(operating_cash_flow_ttm, total_revenue_ttm)


def kassaflode_till_skuld(operating_cash_flow_ttm: OptionalFinancialValue,
                          total_debt: OptionalFinancialValue) -> float:
    """
    Calculate Cash Flow to Debt Ratio (Kassaflöde till skuld).
    
    Cash flow to debt ratio measures a company's ability to service its debt
    with cash flow from operations. Higher values indicate better debt servicing capability.
    
    Formula: Operating Cash Flow TTM / Total Debt
    
    Args:
        operating_cash_flow_ttm: Operating cash flow for trailing twelve months
        total_debt: Total debt
        
    Returns:
        float: Cash flow to debt ratio or np.nan if calculation invalid
        
    Example:
        >>> kassaflode_till_skuld(1200000, 3000000)
        0.4  # 40% cash flow to debt ratio
    """
    # If no debt, ratio is undefined (company is debt-free)
    if pd.isna(total_debt) or float(total_debt) <= MIN_VALID_VALUE:
        return np.nan
    
    if not _validate_positive(total_debt):
        return np.nan
    
    return _safe_divide(operating_cash_flow_ttm, total_debt)


# =============================================================================
# VÄRDERING (VALUATION) - Market Valuation Metrics
# =============================================================================

def pe_ratio(price: OptionalFinancialValue,
             basic_eps_ttm: OptionalFinancialValue) -> float:
    """
    Calculate Price-to-Earnings Ratio (P/E-tal).
    
    P/E ratio measures market valuation relative to earnings.
    Lower values may indicate undervaluation, but context is important.
    
    Formula: Price / Basic EPS TTM
    
    Args:
        price: Stock price (current or historical)
        basic_eps_ttm: Basic earnings per share for trailing twelve months
        
    Returns:
        float: P/E ratio or np.nan if calculation invalid
        
    Example:
        >>> pe_ratio(100, 5)
        20.0  # P/E ratio of 20
    """
    if not _validate_positive(price):
        return np.nan
    
    if not _validate_positive(basic_eps_ttm):
        return np.nan
    
    return _safe_divide(price, basic_eps_ttm)


def pb_ratio(price: OptionalFinancialValue,
             stockholders_equity: OptionalFinancialValue,
             shares_outstanding: OptionalFinancialValue) -> float:
    """
    Calculate Price-to-Book Ratio (P/B-tal).
    
    P/B ratio measures market valuation relative to book value.
    Lower values may indicate undervaluation relative to assets.
    
    Formula: Price / (Stockholders Equity / Shares Outstanding)
    
    Args:
        price: Stock price (current or historical)
        stockholders_equity: Total shareholders' equity
        shares_outstanding: Number of shares outstanding
        
    Returns:
        float: P/B ratio or np.nan if calculation invalid
        
    Example:
        >>> pb_ratio(100, 5000000, 1000000)
        20.0  # P/B ratio of 20 (100 / (5000000/1000000))
    """
    if not _validate_positive(price):
        return np.nan
    
    if not _validate_positive(stockholders_equity):
        return np.nan
    
    if not _validate_positive(shares_outstanding):
        return np.nan
    
    # Calculate book value per share
    book_value_per_share = _safe_divide(stockholders_equity, shares_outstanding)
    if pd.isna(book_value_per_share):
        return np.nan
    
    return _safe_divide(price, book_value_per_share)


def ev_ebitda_ratio(price: OptionalFinancialValue,
                    shares_outstanding: OptionalFinancialValue,
                    total_debt: OptionalFinancialValue,
                    cash_and_equivalents: OptionalFinancialValue,
                    ebitda_ttm: OptionalFinancialValue) -> float:
    """
    Calculate Enterprise Value to EBITDA Ratio (EV/EBITDA).
    
    EV/EBITDA measures company valuation relative to operating performance,
    accounting for debt and cash positions.
    
    Formula: (Market Cap + Total Debt - Cash) / EBITDA TTM
    Where Market Cap = Price * Shares Outstanding
    
    Args:
        price: Stock price (current or historical)
        shares_outstanding: Number of shares outstanding
        total_debt: Total debt
        cash_and_equivalents: Cash and cash equivalents
        ebitda_ttm: EBITDA for trailing twelve months
        
    Returns:
        float: EV/EBITDA ratio or np.nan if calculation invalid
        
    Example:
        >>> ev_ebitda_ratio(100, 1000000, 3000000, 500000, 2000000)
        126.25  # EV/EBITDA of 126.25
    """
    # Validate required positive values
    if not _validate_positive(price):
        return np.nan
    
    if not _validate_positive(shares_outstanding):
        return np.nan
    
    if not _validate_positive(ebitda_ttm):
        return np.nan
    
    # Calculate market cap
    try:
        market_cap = float(price) * float(shares_outstanding)
    except (ValueError, TypeError):
        return np.nan
    
    # Handle missing debt (assume zero if missing)
    if pd.isna(total_debt):
        total_debt = 0.0
    
    # Handle missing cash (assume zero if missing)  
    if pd.isna(cash_and_equivalents):
        cash_and_equivalents = 0.0
    
    try:
        debt_val = float(total_debt)
        cash_val = float(cash_and_equivalents)
        
        # Ensure non-negative values
        if debt_val < 0 or cash_val < 0:
            return np.nan
        
        # Calculate enterprise value
        enterprise_value = market_cap + debt_val - cash_val
        
        return _safe_divide(enterprise_value, ebitda_ttm)
        
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# HELPER FUNCTIONS FOR BOOK VALUE CALCULATIONS
# =============================================================================

def calculate_book_value_per_share(stockholders_equity: OptionalFinancialValue,
                                   shares_outstanding: OptionalFinancialValue) -> float:
    """
    Calculate book value per share.
    
    Helper function for P/B ratio and other book value based calculations.
    
    Args:
        stockholders_equity: Total shareholders' equity
        shares_outstanding: Number of shares outstanding
        
    Returns:
        float: Book value per share or np.nan if calculation invalid
    """
    if not _validate_positive(stockholders_equity):
        return np.nan
    
    if not _validate_positive(shares_outstanding):
        return np.nan
    
    return _safe_divide(stockholders_equity, shares_outstanding)


# =============================================================================
# RATIO FUNCTION REGISTRY
# =============================================================================

# Dictionary mapping function names to actual functions for dynamic access
RATIO_FUNCTIONS = {
    # Quality ratios
    'roe': roe,
    'roic': roic,
    
    # Profitability ratios
    'bruttomarginal': bruttomarginal,
    'rorelsemarginal': rorelsemarginal,
    'vinstmarginal': vinstmarginal,
    
    # Financial health ratios
    'soliditet': soliditet,
    'skuldsattningsgrad': skuldsattningsgrad,
    
    # Cash flow ratios
    'rorelseflodesmarginal': rorelseflodesmarginal,
    'kassaflode_till_skuld': kassaflode_till_skuld,
    
    # Valuation ratios
    'pe_ratio': pe_ratio,
    'pb_ratio': pb_ratio,
    'ev_ebitda_ratio': ev_ebitda_ratio,
    
    # Helper functions
    'calculate_book_value_per_share': calculate_book_value_per_share
}


def get_ratio_function(function_name: str):
    """
    Get a ratio function by name.
    
    Args:
        function_name: Name of the ratio function
        
    Returns:
        callable: The ratio function or None if not found
        
    Example:
        >>> func = get_ratio_function('roe')
        >>> result = func(1000000, 5000000)
    """
    return RATIO_FUNCTIONS.get(function_name)


def list_available_ratios() -> list:
    """
    Get list of all available ratio function names.
    
    Returns:
        list: List of available ratio function names
    """
    return list(RATIO_FUNCTIONS.keys())


# =============================================================================
# VALIDATION AND TESTING HELPERS
# =============================================================================

def validate_ratio_calculation(function_name: str, **kwargs) -> dict:
    """
    Validate a ratio calculation with detailed error reporting.
    
    Args:
        function_name: Name of the ratio function to validate
        **kwargs: Arguments to pass to the ratio function
        
    Returns:
        dict: Validation result with success status and details
        
    Example:
        >>> result = validate_ratio_calculation('roe', net_income_ttm=1000, stockholders_equity=5000)
        >>> print(result['success'])  # True
        >>> print(result['value'])    # 0.2
    """
    result = {
        'success': False,
        'value': np.nan,
        'error': None,
        'warnings': []
    }
    
    # Check if function exists
    func = get_ratio_function(function_name)
    if func is None:
        result['error'] = f"Unknown ratio function: {function_name}"
        return result
    
    try:
        # Attempt calculation
        value = func(**kwargs)
        result['value'] = value
        result['success'] = not pd.isna(value)
        
        if pd.isna(value):
            result['warnings'].append("Calculation returned NaN - check input data quality")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


if __name__ == "__main__":
    # Basic testing when module is run directly
    print("Swedish Ratio Functions Module - Basic Tests")
    print("=" * 50)
    
    # Test each ratio function
    test_cases = [
        ('roe', {'net_income_ttm': 1000000, 'stockholders_equity': 5000000}),
        ('roic', {'ebit_ttm': 2000000, 'tax_provision_ttm': 400000, 'pretax_income_ttm': 1500000, 
                  'total_debt': 3000000, 'stockholders_equity': 5000000}),
        ('bruttomarginal', {'gross_profit_ttm': 3000000, 'total_revenue_ttm': 10000000}),
        ('pe_ratio', {'price': 100, 'basic_eps_ttm': 5}),
        ('soliditet', {'stockholders_equity': 5000000, 'total_assets': 10000000}),
    ]
    
    for func_name, params in test_cases:
        result = validate_ratio_calculation(func_name, **params)
        status = "✓" if result['success'] else "✗"
        print(f"{status} {func_name}: {result['value']:.4f}" if result['success'] else f"{status} {func_name}: {result['error']}")
    
    print(f"\nAvailable ratio functions: {len(list_available_ratios())}")
    print("Module ready for integration with pipeline.")