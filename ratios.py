import numpy as np
import pandas as pd

def calculate_all_ratios(
    raw_data: dict,
    ratio_definitions: dict
) -> dict:
    """
    Calculate all defined financial ratios for each company.

    Args:
        raw_data (dict): Financial data for each ticker.
        ratio_definitions (dict): Definitions of ratios to calculate.

    Returns:
        dict: Calculated ratios for each ticker.
    """
    calculated_ratios = {}
    for ticker, data in raw_data.items():
        if data is None:
            calculated_ratios[ticker] = {f'{ratio_name}_latest_ratioValue': np.nan for ratio_name in ratio_definitions}
            for ratio_name in ratio_definitions:
                calculated_ratios[ticker][f'{ratio_name}_trend_ratioValue'] = np.nan
            continue
        if data['balance_sheet'].empty or data['income_statement'].empty or data['cash_flow'].empty:
            calculated_ratios[ticker] = {f'{ratio_name}_latest_ratioValue': np.nan for ratio_name in ratio_definitions}
            for ratio_name in ratio_definitions:
                calculated_ratios[ticker][f'{ratio_name}_trend_ratioValue'] = np.nan
            continue
        bs_copy = data['balance_sheet'].copy()
        is_copy = data['income_statement'].copy()
        cf_copy = data['cash_flow'].copy()
        bs_copy.fillna(np.nan, inplace=True)
        is_copy.fillna(np.nan, inplace=True)
        cf_copy.fillna(np.nan, inplace=True)
        years_is = []
        years_bs = []
        years_cf = []
        for idx in is_copy.index:
            try:
                year = pd.to_datetime(idx).year
                years_is.append(year)
            except (ValueError, TypeError):
                pass
        for idx in bs_copy.index:
            try:
                year = pd.to_datetime(idx).year
                years_bs.append(year)
            except (ValueError, TypeError):
                pass
        for idx in cf_copy.index:
            try:
                year = pd.to_datetime(idx).year
                years_cf.append(year)
            except (ValueError, TypeError):
                pass
        years_is = [y for y in years_is if y is not None]
        years_bs = [y for y in years_bs if y is not None]
        years_cf = [y for y in years_cf if y is not None]
        years = sorted(list(set(years_is) & set(years_bs) & set(years_cf)))
        ratios = {}
        raw_fields = {
            'Net_Income': is_copy.loc[is_copy.index[0], 'Net Income'] if 'Net Income' in is_copy.columns else np.nan,
            'EBIT': is_copy.loc[is_copy.index[0], 'EBIT'] if 'EBIT' in is_copy.columns else np.nan,
            'Pretax_Income': is_copy.loc[is_copy.index[0], 'Pretax Income'] if 'Pretax Income' in is_copy.columns else np.nan,
            'Tax_Provision': is_copy.loc[is_copy.index[0], 'Tax Provision'] if 'Tax Provision' in is_copy.columns else np.nan,
            'Interest_Expense': is_copy.loc[is_copy.index[0], 'Interest Expense'] if 'Interest Expense' in is_copy.columns else np.nan,
            'Gross_Profit': is_copy.loc[is_copy.index[0], 'Gross Profit'] if 'Gross Profit' in is_copy.columns else np.nan,
            'Total_Revenue': is_copy.loc[is_copy.index[0], 'Total Revenue'] if 'Total Revenue' in is_copy.columns else np.nan,
            'Operating_Income': is_copy.loc[is_copy.index[0], 'Operating Income'] if 'Operating Income' in is_copy.columns else np.nan,
            'Basic_EPS': is_copy.loc[is_copy.index[0], 'Basic EPS'] if 'Basic EPS' in is_copy.columns else np.nan,
            'EBITDA': is_copy.loc[is_copy.index[0], 'EBITDA'] if 'EBITDA' in is_copy.columns else np.nan,
            'Stockholders_Equity': bs_copy.loc[bs_copy.index[0], 'Stockholders Equity'] if 'Stockholders Equity' in bs_copy.columns else np.nan,
            'Total_Assets': bs_copy.loc[bs_copy.index[0], 'Total Assets'] if 'Total Assets' in bs_copy.columns else np.nan,
            'Total_Debt': bs_copy.loc[bs_copy.index[0], 'Total Debt'] if 'Total Debt' in bs_copy.columns else np.nan,
            'Cash_And_Cash_Equivalents': bs_copy.loc[bs_copy.index[0], 'Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in bs_copy.columns else np.nan,
            'Operating_Cash_Flow': cf_copy.loc[cf_copy.index[0], 'Operating Cash Flow'] if 'Operating Cash Flow' in cf_copy.columns else np.nan,
            'Free_Cash_Flow': cf_copy.loc[cf_copy.index[0], 'Free Cash Flow'] if 'Free Cash Flow' in cf_copy.columns else np.nan,
            'sharesOutstanding': data['shares_outstanding'],
            'currentPrice': data['current_price'],
            'marketCap': data['market_cap']
        }
        ratios.update(raw_fields)
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
                    'currentPrice': data['current_price'],
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
                    ratios[f'{ratio_name}_latest_ratioValue'] = np.nan
                    ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                    for year in years:
                        ratios[f'{ratio_name}_year_{year}'] = np.nan
                    continue
                latest_value = eval(definition['formula'], globals(), locals_dict)
                ratios[f'{ratio_name}_latest_ratioValue'] = latest_value
                if len(years) >= 2 and len(is_copy) >= 2 and len(bs_copy) >= 2 and len(cf_copy) >= 2:
                    historical_values = []
                    for i in range(len(years) - 1, -1, -1):
                        try:
                            if i >= len(bs_copy) or i >= len(is_copy) or i >= len(cf_copy):
                                historical_values.append(np.nan)
                                ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = np.nan
                                continue
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
                                'currentPrice': data['current_price'],
                                'marketCap': data['market_cap']
                            }
                            if any(pd.isna(locals_dict_hist.get(field)) or locals_dict_hist.get(field) == 0 for field in required_fields):
                                historical_values.append(np.nan)
                                ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = np.nan
                                continue
                            value = eval(definition['formula'], globals(), locals_dict_hist)
                            historical_values.append(value)
                            ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = value
                        except (ZeroDivisionError, TypeError, KeyError):
                            historical_values.append(np.nan)
                            ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = np.nan
                    x = np.arange(1, len(years) + 1)
                    y = np.array(historical_values)
                    if np.isinf(y).any() or np.isnan(y).all():
                        ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                    else:
                        slope, _ = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)
                        ratios[f'{ratio_name}_trend_ratioValue'] = slope
                else:
                    ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                    for year in years:
                        ratios[f'{ratio_name}_year_{year}'] = np.nan
            except (ZeroDivisionError, TypeError, KeyError):
                ratios[f'{ratio_name}_latest_ratioValue'] = np.nan
                ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                for year in years:
                    ratios[f'{ratio_name}_year_{year}'] = np.nan
        calculated_ratios[ticker] = ratios
    return calculated_ratios
