import numpy as np
import pandas as pd
from financial_utils import calculate_growth_trend_slope


def calculate_all_ratios(
    raw_data: dict,
    ratio_definitions: dict,
    period_type: str = 'annual',
    config: dict = None
) -> dict:
    """
    Calculate all defined financial ratios for each company.
    Calculates ratios based on the cluster configuration, determining which periods to generate
    based on the data source and metric type defined in config['cluster'].
    
    For annual data (data_source: "annual"):
        - Generates ratios for cluster periods with data_source: "annual" (typically long_trend)
    For quarterly data (data_source: "quarterly"):  
        - Generates ratios for cluster periods with data_source: "quarterly" (typically ttm_current, ttm_momentum)
    
    Column naming uses the cluster period names:
        - 'ROE_long_trend_ratioValue' for 4-year trend from annual data
        - 'ROE_ttm_current_ratioValue' for latest TTM value from quarterly data
        - 'ROE_ttm_momentum_ratioValue' for TTM momentum from quarterly data
    
    Also includes historical ratio values for each period, e.g., 'ROE_year_2022', 'ROE_quarter_2025Q1'.

    Args:
        raw_data (dict): Financial data for each ticker.
        ratio_definitions (dict): Definitions of ratios to calculate.
        period_type (str): Type of period for calculations ('annual' or 'quarterly'). Defaults to 'annual'.
        config (dict): Configuration containing cluster definitions. If None, falls back to legacy naming.

    Returns:
        dict: Calculated ratios for each ticker.
    """
    period_suffix = 'quarter' if period_type == 'quarterly' else 'year'
    
    # Determine which cluster periods to generate based on data source
    cluster_periods_to_generate = []
    data_source = 'quarterly' if period_type == 'quarterly' else 'annual'
    
    if config and 'cluster' in config:
        cluster_config = config['cluster']
        if isinstance(cluster_config, dict):
            # New cluster configuration format
            for period_name, period_config in cluster_config.items():
                if period_config.get('data_source') == data_source:
                    cluster_periods_to_generate.append({
                        'name': period_name,
                        'metric_type': period_config.get('metric_type', 'latest')
                    })
        else:
            # Legacy list format - fallback to old behavior
            cluster_periods_to_generate = [
                {'name': 'latest', 'metric_type': 'latest'},
                {'name': 'trend', 'metric_type': 'trend'}
            ]
    else:
        # No config provided - use legacy naming
        cluster_periods_to_generate = [
            {'name': 'latest', 'metric_type': 'latest'},
            {'name': 'trend', 'metric_type': 'trend'}
        ]
    
    calculated_ratios = {}
    for ticker, data in raw_data.items():
        if data is None:
            calculated_ratios[ticker] = {}
            for ratio_name in ratio_definitions:
                for cluster_period in cluster_periods_to_generate:
                    period_name = cluster_period['name']
                    calculated_ratios[ticker][f'{ratio_name}_{period_name}_ratioValue'] = np.nan
            continue
        if data['balance_sheet'].empty or data['income_statement'].empty or data['cash_flow'].empty:
            calculated_ratios[ticker] = {}
            for ratio_name in ratio_definitions:
                for cluster_period in cluster_periods_to_generate:
                    period_name = cluster_period['name']
                    calculated_ratios[ticker][f'{ratio_name}_{period_name}_ratioValue'] = np.nan
            continue
        bs_copy = data['balance_sheet'].copy()
        is_copy = data['income_statement'].copy()
        cf_copy = data['cash_flow'].copy()
        bs_copy.fillna(np.nan, inplace=True)
        is_copy.fillna(np.nan, inplace=True)
        cf_copy.fillna(np.nan, inplace=True)
        periods_is = []
        periods_bs = []
        periods_cf = []
        for idx in is_copy.index:
            try:
                if period_type == 'quarterly':
                    period = f"{pd.to_datetime(idx).year}Q{pd.to_datetime(idx).quarter}"
                else:
                    period = pd.to_datetime(idx).year
                periods_is.append(period)
            except (ValueError, TypeError):
                pass
        for idx in bs_copy.index:
            try:
                if period_type == 'quarterly':
                    period = f"{pd.to_datetime(idx).year}Q{pd.to_datetime(idx).quarter}"
                else:
                    period = pd.to_datetime(idx).year
                periods_bs.append(period)
            except (ValueError, TypeError):
                pass
        for idx in cf_copy.index:
            try:
                if period_type == 'quarterly':
                    period = f"{pd.to_datetime(idx).year}Q{pd.to_datetime(idx).quarter}"
                else:
                    period = pd.to_datetime(idx).year
                periods_cf.append(period)
            except (ValueError, TypeError):
                pass
        periods_is = [p for p in periods_is if p is not None]
        periods_bs = [p for p in periods_bs if p is not None]
        periods_cf = [p for p in periods_cf if p is not None]
        periods = sorted(list(set(periods_is) & set(periods_bs) & set(periods_cf)))
        
        ratios = {}
        # Get historical prices if available
        historical_prices = data.get('historical_prices', {})
        
        # For latest calculations, use the most recent historical price or fall back to current_price
        latest_price = data.get('current_price')
        if historical_prices:
            # Use the most recent historical price
            sorted_dates = sorted(historical_prices.keys(), reverse=False)
            if sorted_dates:
                latest_price = historical_prices[sorted_dates[0]]
        
        # ratios.update(raw_fields)
        for ratio_name, definition in ratio_definitions.items():
            try:
                locals_dict = {
                    'Stockholders_Equity': bs_copy.loc[bs_copy.index[0], 'Stockholders Equity'] if 'Stockholders Equity' in bs_copy.columns else np.nan,
                    'Total_Assets': bs_copy.loc[bs_copy.index[0], 'Total Assets'] if 'Total Assets' in bs_copy.columns else np.nan,
                    'Total_Debt': bs_copy.loc[bs_copy.index[0], 'Total Debt'] if 'Total Debt' in bs_copy.columns else np.nan,
                    'Cash_And_Cash_Equivalents': bs_copy.loc[bs_copy.index[0], 'Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in bs_copy.columns else np.nan,
                    'Net_Income': is_copy.loc[is_copy.index[0], 'Net Income'] if 'Net Income' in is_copy.columns else np.nan,
                    'EBIT': is_copy.loc[is_copy.index[0], 'EBIT'] if 'EBIT' in is_copy.columns else np.nan,
                    'Pretax_Income': is_copy.loc[is_copy.index[0], 'Pretax Income'] if 'Pretax Income' in is_copy.columns else np.nan,
                    'Tax_Provision': is_copy.loc[is_copy.index[0], 'Tax Provision'] if 'Tax Provision' in is_copy.columns else np.nan,
                    'Interest_Expense': is_copy.loc[is_copy.index[0], 'Interest Expense'] if 'Interest Expense' in is_copy.columns else np.nan,
                    'Gross_Profit': is_copy.loc[is_copy.index[0], 'Gross Profit'] if 'Gross Profit' in is_copy.columns else np.nan,
                    'Total_Revenue': is_copy.loc[is_copy.index[0], 'Total Revenue'] if 'Total Revenue' in is_copy.columns else np.nan,
                    'Operating_Income': is_copy.loc[is_copy.index[0], 'Operating Income'] if 'Operating Income' in is_copy.columns else np.nan,
                    'Basic_EPS': is_copy.loc[is_copy.index[0], 'Basic EPS'] if 'Basic EPS' in is_copy.columns else np.nan,
                    'Operating_Cash_Flow': cf_copy.loc[cf_copy.index[0], 'Operating Cash Flow'] if 'Operating Cash Flow' in cf_copy.columns else np.nan,
                    'Free_Cash_Flow': cf_copy.loc[cf_copy.index[0], 'Free Cash Flow'] if 'Free Cash Flow' in cf_copy.columns else np.nan,
                    'EBITDA': is_copy.loc[is_copy.index[0], 'EBITDA'] if 'EBITDA' in is_copy.columns else np.nan,
                    'sharesOutstanding': data['shares_outstanding'],
                    'currentPrice': latest_price,
                    'marketCap': data['market_cap']
                }
                def ensure_list(field):
                    if isinstance(field, str):
                        return [field]
                    return field if isinstance(field, list) else []
                required_fields = (
                    ensure_list(definition.get('source_income', [])) +
                    ensure_list(definition.get('source_bs', [])) +
                    ensure_list(definition.get('source_cf', [])) +
                    ensure_list(definition.get('source_stock', []))
                )
                required_fields = [f.replace(' ', '_') for f in required_fields]
                if any(pd.isna(locals_dict.get(field)) or locals_dict.get(field) == 0 for field in required_fields):
                    # Set NaN for all cluster periods that should be generated
                    for cluster_period in cluster_periods_to_generate:
                        period_name = cluster_period['name']
                        ratios[f'{ratio_name}_{period_name}_ratioValue'] = np.nan
                    for period in periods:
                        ratios[f'{ratio_name}_{period_suffix}_{period}'] = np.nan
                    continue
                    
                latest_value = eval(definition['formula'], globals(), locals_dict)
                
                # Generate columns for each cluster period
                for cluster_period in cluster_periods_to_generate:
                    period_name = cluster_period['name']
                    metric_type = cluster_period['metric_type']
                    
                    if metric_type == 'latest':
                        ratios[f'{ratio_name}_{period_name}_ratioValue'] = latest_value
                    elif metric_type == 'trend':
                        # Calculate trend - will be set after historical analysis
                        pass  # This will be filled in the trend calculation section below
                
                if len(periods) >= 2 and len(is_copy) >= 2 and len(bs_copy) >= 2 and len(cf_copy) >= 2:
                    historical_values = []
                    for i in range(len(periods) - 1, -1, -1):
                        try:
                            if i >= len(bs_copy) or i >= len(is_copy) or i >= len(cf_copy):
                                historical_values.append(np.nan)
                                ratios[f'{ratio_name}_{period_suffix}_{periods[-(i+1)]}'] = np.nan
                                continue
                            # Get historical price for this period if available
                            period_date = pd.to_datetime(bs_copy.index[i]).strftime('%Y-%m-%d')
                            historical_price = historical_prices.get(period_date, data.get('current_price'))
                            
                            locals_dict_hist = {
                                'Stockholders_Equity': bs_copy.loc[bs_copy.index[i], 'Stockholders Equity'] if 'Stockholders Equity' in bs_copy.columns else np.nan,
                                'Total_Assets': bs_copy.loc[bs_copy.index[i], 'Total Assets'] if 'Total Assets' in bs_copy.columns else np.nan,
                                'Total_Debt': bs_copy.loc[bs_copy.index[i], 'Total Debt'] if 'Total Debt' in bs_copy.columns else np.nan,
                                'Cash_And_Cash_Equivalents': bs_copy.loc[bs_copy.index[i], 'Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in bs_copy.columns else np.nan,
                                'Net_Income': is_copy.loc[is_copy.index[i], 'Net Income'] if 'Net Income' in is_copy.columns else np.nan,
                                'EBIT': is_copy.loc[is_copy.index[i], 'EBIT'] if 'EBIT' in is_copy.columns else np.nan,
                                'Pretax_Income': is_copy.loc[is_copy.index[i], 'Pretax Income'] if 'Pretax Income' in is_copy.columns else np.nan,
                                'Tax_Provision': is_copy.loc[is_copy.index[i], 'Tax Provision'] if 'Tax Provision' in is_copy.columns else np.nan,
                                'Interest_Expense': is_copy.loc[is_copy.index[i], 'Interest Expense'] if 'Interest Expense' in is_copy.columns else np.nan,
                                'Gross_Profit': is_copy.loc[is_copy.index[i], 'Gross Profit'] if 'Gross Profit' in is_copy.columns else np.nan,
                                'Total_Revenue': is_copy.loc[is_copy.index[i], 'Total Revenue'] if 'Total Revenue' in is_copy.columns else np.nan,
                                'Operating_Income': is_copy.loc[is_copy.index[i], 'Operating Income'] if 'Operating Income' in is_copy.columns else np.nan,
                                'Basic_EPS': is_copy.loc[is_copy.index[i], 'Basic EPS'] if 'Basic EPS' in is_copy.columns else np.nan,
                                'Operating_Cash_Flow': cf_copy.loc[cf_copy.index[i], 'Operating Cash Flow'] if 'Operating Cash Flow' in cf_copy.columns else np.nan,
                                'Free_Cash_Flow': cf_copy.loc[cf_copy.index[i], 'Free Cash Flow'] if 'Free Cash Flow' in cf_copy.columns else np.nan,
                                'EBITDA': is_copy.loc[is_copy.index[i], 'EBITDA'] if 'EBITDA' in is_copy.columns else np.nan,
                                'sharesOutstanding': data['shares_outstanding'],
                                'currentPrice': historical_price,
                                'marketCap': data['market_cap']
                            }
                            if any(pd.isna(locals_dict_hist.get(field)) or locals_dict_hist.get(field) == 0 for field in required_fields):
                                historical_values.append(np.nan)
                                ratios[f'{ratio_name}_{period_suffix}_{periods[-(i+1)]}'] = np.nan
                                continue
                            value = eval(definition['formula'], globals(), locals_dict_hist)
                            historical_values.append(value)
                            ratios[f'{ratio_name}_{period_suffix}_{periods[-(i+1)]}'] = value
                        except (ZeroDivisionError, TypeError, KeyError):
                            historical_values.append(np.nan)
                            ratios[f'{ratio_name}_{period_suffix}_{periods[-(i+1)]}'] = np.nan
                    
                    trend_value = calculate_growth_trend_slope(historical_values)
                    
                    # Assign trend value to trend cluster periods
                    for cluster_period in cluster_periods_to_generate:
                        period_name = cluster_period['name']
                        metric_type = cluster_period['metric_type']
                        if metric_type == 'trend':
                            ratios[f'{ratio_name}_{period_name}_ratioValue'] = trend_value
                else:
                    # No historical data available - set trend periods to NaN
                    for cluster_period in cluster_periods_to_generate:
                        period_name = cluster_period['name']
                        metric_type = cluster_period['metric_type']
                        if metric_type == 'trend':
                            ratios[f'{ratio_name}_{period_name}_ratioValue'] = np.nan
                    for period in periods:
                        ratios[f'{ratio_name}_{period_suffix}_{period}'] = np.nan
            except (ZeroDivisionError, TypeError, KeyError):
                # Set NaN for all cluster periods that should be generated
                for cluster_period in cluster_periods_to_generate:
                    period_name = cluster_period['name']
                    ratios[f'{ratio_name}_{period_name}_ratioValue'] = np.nan
                for period in periods:
                    ratios[f'{ratio_name}_{period_suffix}_{period}'] = np.nan
        calculated_ratios[ticker] = ratios
    return calculated_ratios
