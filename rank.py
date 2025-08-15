import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import pickle
import datetime
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Ladda .env-filen endast om den finns
if Path('.env').exists():
    load_dotenv()
    
# Bestäm miljön (default till 'local')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
FETCH_DATA = os.getenv('FETCH_DATA', 'Yes')

# Välj CSV-path
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')

# --- Funktionsdefinitioner ---

def load_config(config_file_path):
    """
    Läser in konfiguration från en YAML-fil.
    :param config_file_path: Sökväg till YAML-filen.
    :return: Ett dictionary med konfigurationsinställningar.
    """
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file {config_file_path} was not found.")
        return None

def read_tickers_from_csv(csv_file_path):
    """
    Läser in en lista med tickers från en CSV-fil.
    Antar att tickers finns i en kolumn med namnet 'ticker', 'symbol', eller 'Instrument'.
    :param csv_file_path: Sökväg till CSV-filen.
    :return: En lista med tickers (strängar).
    """
    try:
        df = pd.read_csv(csv_file_path)
        if 'ticker' in df.columns:
            return df['ticker'].tolist()
        elif 'symbol' in df.columns:
            return df['symbol'].tolist()
        elif 'Instrument' in df.columns:
            return df['Instrument'].tolist()
        else:
            print("Error: Could not find a 'ticker', 'symbol' or 'Instrument' column in the CSV file.")
            return []
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return []

def fetch_yfinance_data(ticker, years, period_type="annual"):
    try:
        yf_ticker = f"{ticker}.ST"
        ticker_obj = yf.Ticker(yf_ticker)
        if period_type == "quarterly":
            bs = ticker_obj.quarterly_balance_sheet.transpose()
            is_ = ticker_obj.quarterly_income_stmt.transpose()
            cf = ticker_obj.quarterly_cash_flow.transpose()
            info = ticker_obj.info
        else:
            bs = ticker_obj.balance_sheet.transpose()
            is_ = ticker_obj.income_stmt.transpose()
            cf = ticker_obj.cash_flow.transpose()
            info = ticker_obj.info
            longBusinessSummary = info.get('longBusinessSummary', 'No summary available')
            dividendRate = info.get('dividendRate', None)
            lastDividendDate = info.get('lastDividendDate', None)
            dividends = ticker_obj.dividends

        shares_outstanding = info.get('sharesOutstanding', None)
        current_price = info.get('currentPrice', None)
        market_cap = info.get('marketCap', None)


        # Fetch the date for the latest annual report if available
        latest_report_date = None
        if hasattr(bs, 'index') and len(bs.index) > 0:
            # Try to parse the first index as date
            try:
                latest_report_date = pd.to_datetime(bs.index[0])
            except Exception:
                latest_report_date = str(bs.index[0])

        if not all([bs is not None, is_ is not None, cf is not None, market_cap is not None, info is not None]):
            print(f"Warning: Incomplete data for {ticker}. Skipping.")
            return None

        # Begränsa till angivet antal år och åtgärda varningar
        bs = bs.head(years).copy().infer_objects(copy=False).fillna(0)
        is_ = is_.head(years).copy().infer_objects(copy=False).fillna(0)
        cf = cf.head(years).copy().infer_objects(copy=False).fillna(0)
        
        return {
            'balance_sheet': bs,
            'income_statement': is_,
            'cash_flow': cf,
            'current_price': current_price,
            'shares_outstanding': shares_outstanding,
            'info': info,
            'dividendRate': dividendRate if period_type == "annual" else None,
            'lastDividendDate': lastDividendDate if period_type == "annual" else None,
            'longBusinessSummary': longBusinessSummary if period_type == "annual" else None,
            'market_cap': market_cap,
            'dividends': dividends if period_type == "annual" else None,
            'latest_report_date': latest_report_date if latest_report_date else None,
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_all_ratios(raw_data, ratio_definitions):
    """
    Beräknar alla definierade nyckeltal för varje bolag.
    Inkluderar beräkning av linjär regression för trenden och sparar historiska värden för tillgängliga år.
    Sparar resultaten, inklusive rådata, i en CSV-fil.
    
    Args:
        raw_data (dict): Dictionary med finansiella data för varje ticker, innehållande
                         'balance_sheet', 'income_statement', 'cash_flow', 'shares_outstanding', 'current_price'.
        ratio_definitions (dict): Dictionary med definitioner av nyckeltal från rank-config.yaml.
        output_csv (str): Sökväg till CSV-filen där resultaten sparas (default: 'calculated_ratios.csv').
    
    Returns:
        dict: Dictionary med beräknade nyckeltal för varje ticker.
    """
    calculated_ratios = {}
    
    for ticker, data in raw_data.items():
        if data is None:
            print(f"Varning: Ingen data för {ticker}. Sätter alla nyckeltal till NaN.")
            calculated_ratios[ticker] = {f'{ratio_name}_latest_ratioValue': np.nan for ratio_name in ratio_definitions}
            for ratio_name in ratio_definitions:
                calculated_ratios[ticker][f'{ratio_name}_trend_ratioValue'] = np.nan
            continue

        # Säkerhetskontroll: Kontrollera om DataFrames är tomma
        if data['balance_sheet'].empty or data['income_statement'].empty or data['cash_flow'].empty:
            print(f"Varning: Finansiella data är ofullständig för {ticker}. Kan inte beräkna nyckeltal.")
            calculated_ratios[ticker] = {f'{ratio_name}_latest_ratioValue': np.nan for ratio_name in ratio_definitions}
            for ratio_name in ratio_definitions:
                calculated_ratios[ticker][f'{ratio_name}_trend_ratioValue'] = np.nan
            continue

        # Säkerställ att du jobbar med kopior för att undvika SettingWithCopyWarning
        bs_copy = data['balance_sheet'].copy()
        is_copy = data['income_statement'].copy()
        cf_copy = data['cash_flow'].copy()

        # Fyll i saknade värden med NaN istället för 0
        bs_copy.fillna(np.nan, inplace=True)
        is_copy.fillna(np.nan, inplace=True)
        cf_copy.fillna(np.nan, inplace=True)

        # Hämta år dynamiskt från indexen och hitta gemensamma år
        years_is = []
        years_bs = []
        years_cf = []
        for idx in is_copy.index:
            try:
                year = pd.to_datetime(idx).year
                years_is.append(year)
            except (ValueError, TypeError):
                print(f"Varning: Ogiltigt Dateindex för {ticker} (is_copy): {idx}. Hoppar över.")
        for idx in bs_copy.index:
            try:
                year = pd.to_datetime(idx).year
                years_bs.append(year)
            except (ValueError, TypeError):
                print(f"Varning: Ogiltigt Dateindex för {ticker} (bs_copy): {idx}. Hoppar över.")
        for idx in cf_copy.index:
            try:
                year = pd.to_datetime(idx).year
                years_cf.append(year)
            except (ValueError, TypeError):
                print(f"Varning: Ogiltigt Dateindex för {ticker} (cf_copy): {idx}. Hoppar över.")

        # Filtrera bort ogiltiga år och hitta gemensamma år
        years_is = [y for y in years_is if y is not None]
        years_bs = [y for y in years_bs if y is not None]
        years_cf = [y for y in years_cf if y is not None]
        years = sorted(list(set(years_is) & set(years_bs) & set(years_cf)))  # Gemensamma år i kronologisk ordning

        print(f"År för {ticker}: {years} (is: {len(is_copy)}, bs: {len(bs_copy)}, cf: {len(cf_copy)})")

        ratios = {}

        # Samla rådata från finansiella källor (för att inkludera i CSV)
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

        # Beräkna varje nyckeltal
        for ratio_name, definition in ratio_definitions.items():
            try:
                # Definiera variabler för formeln (senaste året)
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

                # Hantera required_fields för både strängar och listor
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

                # Kontrollera om något värde är NaN eller noll
                if any(pd.isna(locals_dict.get(field)) or locals_dict.get(field) == 0 for field in required_fields):
                    print(f"Varning: Ogiltiga värden (NaN eller 0) för {ticker}'s {ratio_name}: {locals_dict}")
                    ratios[f'{ratio_name}_latest_ratioValue'] = np.nan
                    ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                    for year in years:
                        ratios[f'{ratio_name}_year_{year}'] = np.nan
                    continue

                # Beräkna senaste årets värde
                latest_value = eval(definition['formula'], globals(), locals_dict)
                ratios[f'{ratio_name}_latest_ratioValue'] = latest_value

                # Beräkna historiska värden och trend om tillräckligt med data finns
                if len(years) >= 2 and len(is_copy) >= 2 and len(bs_copy) >= 2 and len(cf_copy) >= 2:
                    historical_values = []
                    for i in range(len(years) - 1, -1, -1):  # Iterera baklänges: äldsta till senaste
                        try:
                            if i >= len(bs_copy) or i >= len(is_copy) or i >= len(cf_copy):
                                print(f"Varning: Index {i} utanför gränsen för {ticker}'s {ratio_name} (bs: {len(bs_copy)}, is: {len(is_copy)}, cf: {len(cf_copy)})")
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
                            # Kontrollera om något värde är NaN eller noll
                            if any(pd.isna(locals_dict_hist.get(field)) or locals_dict_hist.get(field) == 0 for field in required_fields):
                                print(f"Varning: Ogiltiga historiska värden för {ticker}'s {ratio_name} år {years[-(i+1)]}: {locals_dict_hist}")
                                historical_values.append(np.nan)
                                ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = np.nan
                                continue
                            value = eval(definition['formula'], globals(), locals_dict_hist)
                            historical_values.append(value)
                            ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = value
                        except (ZeroDivisionError, TypeError, KeyError) as e:
                            print(f"Varning: Fel vid trendberäkning för {ticker}'s {ratio_name} år {years[-(i+1)]}: {e}")
                            historical_values.append(np.nan)
                            ratios[f'{ratio_name}_year_{years[-(i+1)]}'] = np.nan
                    x = np.arange(1, len(years) + 1)  # Dynamisk x-axel baserat på antal år
                    y = np.array(historical_values)

                    if np.isinf(y).any() or np.isnan(y).all():
                        print(f"Varning: Ogiltiga historiska värden för {ticker}'s {ratio_name}_trend_ratioValue: {y}")
                        ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                    else:
                        slope, _ = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)
                        ratios[f'{ratio_name}_trend_ratioValue'] = slope
                else:
                    print(f"Varning: Otillräcklig data för trendberäkning för {ticker}'s {ratio_name} (bs: {len(bs_copy)}, is: {len(is_copy)}, cf: {len(cf_copy)}, years: {years})")
                    ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                    for year in years:
                        ratios[f'{ratio_name}_year_{year}'] = np.nan

            except (ZeroDivisionError, TypeError, KeyError) as e:
                print(f"Varning: Beräkningsfel för {ticker}'s {ratio_name}: {e}")
                print(f"Variabler: {locals_dict}")
                ratios[f'{ratio_name}_latest_ratioValue'] = np.nan
                ratios[f'{ratio_name}_trend_ratioValue'] = np.nan
                for year in years:
                    ratios[f'{ratio_name}_year_{year}'] = np.nan

        calculated_ratios[ticker] = ratios

    # Konvertera calculated_ratios till en DataFrame och spara till CSV
    """df = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    try:
        df.to_csv(output_csv)
        print(f"Resultat sparade till {output_csv}")
    except Exception as e:
        print(f"Fel vid skrivning till CSV {output_csv}: {e}")"""

    return calculated_ratios

def rank_all_ratios(calculated_ratios, ratio_definitions):
    """
    Rankar varje nyckeltal (senaste år och trend) på en 0-100 skala.
    Använder percentilrankning och hybridmetoden för trend.
    """
    ranked_ratios = {ticker: {} for ticker in calculated_ratios.keys()}
    df = pd.DataFrame.from_dict(calculated_ratios, orient='index')

    for column in df.columns:
        if column.endswith('_latest_ratioValue'):
            ratio_name = column.replace('_latest_ratioValue', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            #ascending = not is_better
            ranked = df[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)  # Fyller NaN med 50 för att representera medelvärde
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_latest_ratioRank'] = rank if not pd.isna(rank) else np.nan
        
        elif column.endswith('_trend_ratioValue'):
            ratio_name = column.replace('_trend_ratioValue', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            
            ranked = df[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)  # Fyller NaN med 50 för att representera medelvärde
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_trend_ratioRank'] = rank if not pd.isna(rank) else np.nan

    return ranked_ratios

def aggregate_category_ranks(ranked_ratios, category_ratios):
    """
    Aggregerar de rankade nyckeltalen till en total poäng för varje kategori.
    Efter beräkning rankas alla aggregerade värden (inklusive totalscore) på en 0-100-skala.
    """
    aggregated_scores = {}
    for ticker, ranks in ranked_ratios.items():
        if not ranks:
            continue
        ticker_scores = {}
        total_latest_score = 0
        total_trend_score = 0
        total_latest_weight = 0
        total_trend_weight = 0
        for category, ratios in category_ratios.items():
            category_score = 0
            num_ratios = 0
            for rank_name, rank_value in ranks.items():
                if rank_name in ratios:
                    if not pd.isna(rank_value):
                        category_score += rank_value
                        num_ratios += ratios[rank_name]
            cat_avg_name = category.replace('_ratioRank', '')
            if num_ratios > 0:
                ticker_scores[f'{cat_avg_name}_catAvg'] = category_score / num_ratios if num_ratios > 0 else np.nan
            else:
                ticker_scores[f'{cat_avg_name}_catAvg'] = np.nan
            # Add to total_latest_score or total_trend_score depending on category name
            if category.endswith('_latest_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_latest_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_latest_weight += 1
            elif category.endswith('_trend_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_trend_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_trend_weight += 1
        aggregated_scores[ticker] = ticker_scores

    # --- Rank all aggregated calculations (0-100) ---
    # Convert to DataFrame for easy ranking
    df_agg = pd.DataFrame.from_dict(aggregated_scores, orient='index')
    for col in df_agg.columns:
        col_name = col.replace('_catAvg', '_catRank') if col.endswith('_catAvg') else col
        if df_agg[col].dtype in [float, int]:
            # Rank so that higher is better (ascending=True)
            ranks = df_agg[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            df_agg[col_name] = ranks

    return df_agg.to_dict(orient='index')

def combine_all_results(calculated_ratios, ranked_ratios, category_scores, cluster_ranks):
    """
    Slår ihop alla resultat till en enda DataFrame.
    """
    df_calculated = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    df_ranked = pd.DataFrame.from_dict(ranked_ratios, orient='index')
    df_scores = pd.DataFrame.from_dict(category_scores, orient='index')
    df_cluster_ranks = pd.DataFrame.from_dict(cluster_ranks, orient='index')
    df_agr = pd.read_csv(CSV_PATH / "agr_results.csv", index_col=0)
    df_agr_dividends = pd.read_csv(CSV_PATH / "agr_dividend_results.csv", index_col=0)
    # Load tickers file as defined in config
    tickers_file = CSV_PATH / config.get("input_ticker_file")
    df_tickers = pd.read_csv(tickers_file, index_col='Instrument')
    df_tickers = df_tickers.rename(columns={'Instrument': 'Ticker'})
    df_latest_report_dates = pd.read_csv(CSV_PATH / "latest_report_dates.csv", index_col='Ticker')
    df_latest_report_dates_quarterly = pd.read_csv(CSV_PATH / "latest_report_dates_quarterly.csv", index_col='Ticker')

    df_last_SMA = pd.read_csv(CSV_PATH / "last_SMA.csv", index_col='Ticker')
    df_long_business_summary = pd.read_csv(CSV_PATH / "longBusinessSummary.csv", index_col='Ticker')
    # Read the quarterly calculated ratios CSV (long format)
    df_calculated_quarterly_long = pd.read_csv(CSV_PATH / "calculated_ratios_quarterly.csv")

    # Pivot to wide format: index = Ticker, columns = Metric, values = Values
    df_calculated_quarterly = df_calculated_quarterly_long.pivot(index='Ticker', columns='Metric', values='Values')

    # Optional: ensure index is string type for consistency
    df_calculated_quarterly.index = df_calculated_quarterly.index.astype(str)

    final_df = pd.concat([df_tickers,df_calculated, df_calculated_quarterly,df_ranked, df_scores, df_last_SMA, df_cluster_ranks,df_agr,df_agr_dividends,df_latest_report_dates,df_latest_report_dates_quarterly, df_long_business_summary], axis=1)

    return final_df#.sort_values(by='Total_Score', ascending=False)

def save_results_to_csv(results_df, file_path):
    """
    Sparar den slutliga DataFrame till en CSV-fil och säkerställer att målkatalogen finns.
    """
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(file_path, index=True)

def save_raw_data_to_csv(raw_data, csv_file_path):
    segments = ['balance_sheet','income_statement','cash_flow']
    df = pd.DataFrame()
    for ticker,data in raw_data.items():
        for segment in segments:
            df_segment_data = data[segment].T
            df_temp = (df_segment_data
                              .reset_index()
                              .rename(columns={'index': 'Metric'}))
            df_long_format = pd.melt(df_temp,id_vars=['Metric'], 
                         var_name='Date', 
                         value_name='Value')
            df_long_format['Ticker'] = ticker
            df_long_format['Segment'] = segment
            df_long_format['Date'] = pd.to_datetime(df_long_format['Date'], errors='coerce')
            df_long_format['Date'] = df_long_format['Date'].dt.date
            df = pd.concat([df, df_long_format], ignore_index=True)

    df.to_csv(csv_file_path, index=False)

def save_longBusinessSummary_to_csv(raw_data, csv_file_path):
    df = pd.DataFrame()
    for ticker, data in raw_data.items():
        if 'longBusinessSummary' in data:
            summary = data['longBusinessSummary']
            df_temp = pd.DataFrame({'Ticker': [ticker], 'LongBusinessSummary': [summary]})
            df = pd.concat([df, df_temp], ignore_index=True)
        else:
            print(f"Warning: No longBusinessSummary for {ticker}. Skipping.")
    
    df.to_csv(csv_file_path, index=False)

def save_latest_report_dates_to_csv(raw_data, csv_file_path,period_type="Y"):
    df = pd.DataFrame()
    for ticker, data in raw_data.items():
        if 'latest_report_date' in data:
            report_date = data['latest_report_date']
            df_temp = pd.DataFrame({'Ticker': [ticker], f'LatestReportDate_{period_type}': [report_date]})
            df = pd.concat([df, df_temp], ignore_index=True)
        else:
            print(f"Warning: No latest_report_date for {ticker}. Skipping.")

    df.to_csv(csv_file_path, index=False)

def save_dividends_to_csv(raw_data, csv_file_path):
    """
    Saves dividend information for each ticker to a CSV file.
    
    Parameters:
        raw_data (dict): Dictionary containing financial data for each ticker, including dividend info.
        csv_file_path (str or Path): Path to the CSV file where dividend data will be saved.
    
    Returns:
        None. Writes dividend data to the specified CSV file.
    """
    rows = []
    for ticker, data in raw_data.items():
        dividend_last_year = None
        last_dividend_year = None
        if 'info' in raw_data[ticker]:
            if 'dividendRate' in raw_data[ticker]['info']:
                dividend_last_year = raw_data[ticker]['info']['dividendRate'] if raw_data[ticker]['info']['dividendRate'] is not None else None
            else:
                dividend_last_year = None
            if 'lastDividendDate' in raw_data[ticker]['info']:
                last_dividend_year = int(datetime.datetime.fromtimestamp(raw_data[ticker]['info']['lastDividendDate']).strftime('%Y'))
            else:
                last_dividend_year = None
        if 'dividends' in data:
            dividends = data['dividends']
            # dividends is a pandas Series: index=date, value=dividend
            if hasattr(dividends, 'items'):
                # Build a DataFrame for aggregation
                div_df = pd.DataFrame(list(dividends.items()), columns=['Date', 'Value'])
                div_df['Date'] = pd.to_datetime(div_df['Date'], errors='coerce')
                div_df['Year'] = div_df['Date'].dt.year
                # Aggregate by year (sum all dividends for the same year)
                yearly = div_df.groupby('Year')['Value'].sum().reset_index()
                for _, row in yearly.iterrows():
                    if last_dividend_year is not None and int(row['Year']) == last_dividend_year and dividend_last_year is not None:
                        rows.append({'Ticker': ticker, 'Year': int(row['Year']), 'Value': dividend_last_year})
                    else:
                        rows.append({'Ticker': ticker, 'Year': int(row['Year']), 'Value': row['Value']})
        else:
            print(f"Warning: No dividends for {ticker}. Skipping.")

    df = pd.DataFrame(rows)
    df.to_csv(csv_file_path, index=False)

def save_calculated_ratios_to_csv(calculated_ratios, csv_file_path, period_type="annual"):
    df = pd.DataFrame()
    for ticker, data in calculated_ratios.items():
        if period_type == "quarterly":
            # Only keep metrics ending with '_latest_ratioValue' and rename to '_TTM'
            filtered_data = {k.replace('_latest_ratioValue', '_TTM'): v for k, v in data.items() if k.endswith('_latest_ratioValue')}
        else:
            filtered_data = data
        df_metrics = pd.DataFrame(filtered_data, index=[ticker]).T
        df_metrics.columns = ['Values']
        df_temp = df_metrics.reset_index().rename(columns={'index': 'Metric'})
        df_temp['Ticker'] = ticker
        df = pd.concat([df, df_temp], ignore_index=False)
    df.to_csv(csv_file_path, index=False)

def save_ranked_ratios_to_csv(ranked_ratios, csv_file_path):
    df = pd.DataFrame()
    for ticker,data in ranked_ratios.items():
        df_metrics = pd.DataFrame(data, index=[ticker]).T
        df_metrics.columns = ['Values']
        df_temp = df_metrics.reset_index().rename(columns={'index': 'Rank'})
        df_temp['Ticker'] = ticker
        df = pd.concat([df, df_temp], ignore_index=False)
    df.to_csv(csv_file_path, index=False)

def save_category_scores_to_csv(category_scores, csv_file_path):
    df = pd.DataFrame()
    for ticker,data in category_scores.items():
        df_metrics = pd.DataFrame(data, index=[ticker]).T
        df_metrics.columns = ['Values']
        df_temp = df_metrics.reset_index().rename(columns={'index': 'Category'})
        df_temp['Ticker'] = ticker
        df = pd.concat([df, df_temp], ignore_index=False)
    df.to_csv(csv_file_path, index=True)

def get_price_data(SMA_short:int, SMA_medium:int, SMA_long:int, tickers:list, data_fetch_years:int, price_data_file_path:str):
    df_complete = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Fetching stock price data", disable=False if ENVIRONMENT == "local" else True):
        try:
            yf_ticker = f"{ticker}.ST"
            stock = yf.Ticker(yf_ticker)
            df_price_data = stock.history(period=f"{data_fetch_years}y")[['Close', 'Volume']]
            df_price_data['Ticker'] = ticker
            if df_price_data.index.tz is not None:
                df_price_data.index = df_price_data.index.tz_localize(None)
            if df_price_data.empty:
                print(f"Ingen data hämtad för {yf_ticker}")
            # Calculate moving averages
            df_price_data['SMA_short'] = df_price_data['Close'].rolling(window=SMA_short).mean()
            df_price_data['SMA_medium'] = df_price_data['Close'].rolling(window=SMA_medium).mean()
            df_price_data['SMA_long'] = df_price_data['Close'].rolling(window=SMA_long).mean()
            # Calculate percent differences
            df_price_data['pct_SMA_medium_vs_SMA_long'] = (((df_price_data['SMA_medium'] - df_price_data['SMA_long']) / df_price_data['SMA_long']) * 100).fillna(0)
            df_price_data['pct_SMA_short_vs_SMA_medium'] = (((df_price_data['SMA_short'] - df_price_data['SMA_medium']) / df_price_data['SMA_medium']) * 100).fillna(0)
            df_price_data['pct_Close_vs_SMA_short'] = (((df_price_data['Close'] - df_price_data['SMA_short']) / df_price_data['SMA_short']) * 100).fillna(0)
            # Calculate percent differences for closing price
            """df_price_data['pct_diff_short'] = (df_price_data['Close'].pct_change(periods=pct_diff_short)* 100).fillna(0)
            df_price_data['pct_diff_medium'] = (df_price_data['Close'].pct_change(periods=pct_diff_medium)* 100).fillna(0)
            df_price_data['pct_diff_long'] = (df_price_data['Close'].pct_change(periods=pct_diff_long)* 100).fillna(0)"""

            df_complete = pd.concat([df_complete, df_price_data])
        except Exception as e:
            print(f"Fel vid hämtning av data för {yf_ticker}: {str(e)}")
    # Om df_complete är tom, skriv ut ett meddelande
    if df_complete.empty:
        print("Ingen prisdata hämtad för några tickers.")
    df_complete.to_csv(price_data_file_path, index=True)

def save_last_SMA_to_csv(read_from, save_to):
    """
    Sparar den senaste SMA-data till en separat CSV-fil.
    Beräknar även CAGR för 'Close' per ticker och lägger till det i resultatet.
    """
    try:
        df = pd.read_csv(read_from, index_col='Date', parse_dates=True)
        if 'SMA_short' in df.columns and 'SMA_medium' in df.columns and 'SMA_long' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            # Get the latest row for each ticker
            last_rows = df.groupby('Ticker').tail(1)[['Ticker', 'pct_Close_vs_SMA_short', 'pct_SMA_short_vs_SMA_medium', 'pct_SMA_medium_vs_SMA_long']]
            # Calculate CAGR for each ticker
            cagr_list = []
            for ticker, group in df.groupby('Ticker'):
                group = group.sort_index()
                if len(group) > 1:
                    start_price = group['Close'].iloc[0]
                    end_price = group['Close'].iloc[-1]
                    num_years = (group.index[-1] - group.index[0]).days / 365.25
                    if start_price > 0 and num_years > 0:
                        cagr = (((end_price / start_price) ** (1 / num_years)) - 1)# * 100  # CAGR in percent
                        cagr_list.append({'Ticker': ticker, 'CAGR': cagr})
                    else:
                        cagr_list.append({'Ticker': ticker, 'CAGR': 0})  # If no valid CAGR can be calculated, set to 0
                else:
                    cagr_list.append({'Ticker': ticker, 'CAGR': 0})  # If no valid CAGR can be calculated, set to 0
            df_cagr = pd.DataFrame(cagr_list).set_index('Ticker')
            last_rows = last_rows.set_index('Ticker')
            last_rows['cagr_close'] = df_cagr['CAGR']
            last_rows.reset_index(inplace=True)
            last_rows.to_csv(save_to, index=False)
            print(f"Senaste SMA-data per ticker sparad i '{save_to}'")
        else:
            print("Fel: Saknar nödvändiga kolumner i prisdata.")
    except Exception as e:
        print(f"Fel vid sparande av senaste SMA-data: {e}")

def aggregate_cluster_ranks(category_ranks):
    """
    Agregerar rankningar för varje cluster baserat på de angivna cluster-ratio.
    """
    results = []
    for ticker, subdict in category_ranks.items():
        latest_vals = [v for k, v in subdict.items() if "latest_catRank" in k]
        trend_vals = [v for k, v in subdict.items() if "trend_catRank" in k]
        latest_avg = sum(latest_vals) / len(latest_vals) if latest_vals else None
        trend_avg = sum(trend_vals) / len(trend_vals) if trend_vals else None
        results.append({
            "Ticker": ticker,
            "Latest_clusterAvg": latest_avg,
            "Trend_clusterAvg": trend_avg
        })
    df = pd.DataFrame(results)
    for col in df.columns:
        col_name = col.replace('_clusterAvg', '_clusterRank') if col.endswith('_clusterAvg') else col
        if df[col].dtype in [float, int]:
            # Rank so that higher is better (ascending=True)
            ranks = df[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            df[col_name] = ranks
    return df.set_index('Ticker').to_dict(orient='index')

# --- AGR Calculation ---
def calculate_agr_for_ticker(csv_path, tickers, dimensions):
    """
    Beräknar genomsnittlig tillväxttakt (AGR) för angivna dimensioner för en ticker.
    Hanterar 0-värden och NaN-värden så att de inte ger division-by-zero eller felaktiga tillväxttal.
    Args:
        df (pd.DataFrame): DataFrame från raw_financial_data.csv
        tickers (list): Lista med tickers att beräkna AGR för
        dimensions (list): Lista med strängar, t.ex. ['Total Revenue', 'Basic EPS', 'Free Cash Flow']
    Returns:
        dict: AGR per dimension för varje ticker, med dimensionnamn utan mellanslag
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
    agr_results = {}
    # Group the DataFrame by 'Ticker' and 'Metric' for efficient lookup
    grouped = df.groupby(['Ticker', 'Metric'])
    for ticker in tickers:
        ticker_agr = {}
        for dim in dimensions:
            try:
                group = grouped.get_group((ticker, dim)).sort_values('Date')
                values = pd.to_numeric(group['Value'].values, errors='coerce')
                years = group['Date'].values
            except KeyError:
                values = np.array([])
                years = np.array([])
            # Beräkna år-till-år tillväxt
            growth_rates = []
            for i in range(1, len(values)):
                prev = values[i - 1]
                curr = values[i]
                if pd.isna(prev) or pd.isna(curr) or prev == 0:
                    continue  # Hoppa över om föregående år är 0 eller NaN
                growth = (curr - prev) / abs(prev)
                growth_rates.append(growth)
            if growth_rates:
                agr = np.mean(growth_rates)
            else:
                agr = np.nan
            dim_key = dim.replace(" ", "_") + "_AvgGrowth"
            ticker_agr[dim_key] = agr
            # Add yearly data to result dict
            dim_data_key = dim.replace(" ", "_")
            for year, value in zip(years, values):
                # year is a datetime.date, convert to string (YYYY)
                year_str = str(year)
                ticker_agr[f"{dim_data_key}_year_{year_str}"] = value
        agr_results[ticker] = ticker_agr

    return agr_results

def calculate_agr_dividend_for_ticker(csv_path, tickers, n_years=4):
    """
    Beräknar genomsnittlig tillväxttakt (AGR) för utdelningar per ticker.
    Hanterar 0-värden och NaN-värden så att de inte ger division-by-zero eller felaktiga tillväxttal.
    Args:
        csv_path (str): Sökväg till CSV-filen med utdelningsdata
        tickers (list): Lista med tickers att beräkna AGR för
    Returns:
        dict: AGR per ticker för utdelningar
    """
    df = pd.read_csv(csv_path, parse_dates=['Year'])
    df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
    agr_results = {}

    for ticker in tickers:
        # Aggregate dividends per year (sum all dividends for the same year)
        group = df[df['Ticker'] == ticker]
        yearly = group.groupby('Year')['Value'].sum().sort_index()
        # Only keep the last n_years
        # Only keep values from the last n_years (e.g., if n_years=4, keep years >= current_year - 4)
        current_year = pd.Timestamp.today().year
        min_year = current_year - n_years
        yearly = yearly[yearly.index >= min_year]
        years = yearly.index.values
        values = yearly.values

        # Beräkna år-till-år tillväxt
        growth_rates = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            if pd.isna(prev) or pd.isna(curr) or prev == 0:
                continue
            growth = (curr - prev) / abs(prev)
            growth_rates.append(growth)
        if growth_rates:
            agr = np.mean(growth_rates)
        else:
            agr = np.nan
        # Build result dict for ticker
        ticker_dict = {'Dividend_AvgGrowth': agr}
        for year, value in zip(years, values):
            ticker_dict[f"Dividend_year_{year}"] = value
        agr_results[ticker] = ticker_dict

    return agr_results


def save_agr_results_to_csv(agr_results, csv_file_path):
    """
    Sparar AGR-resultat till en CSV-fil.
    Args:
        agr_results (dict): AGR-resultat per ticker och dimension
        csv_file_path (str): Sökväg till CSV-filen att spara
    """
    df = pd.DataFrame.from_dict(agr_results, orient='index')
    # Calculate percentage rank for each _AvgGrowth column
    for col in df.columns:
        if col.endswith('_AvgGrowth'):
            # Rank so that higher AGR is better (100=best, 0=worst)
            ranks = df[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            rank_col = col.replace('_AvgGrowth', '_AvgGrowth_Rank')
            df[rank_col] = ranks
    df.to_csv(csv_file_path, index=True)
    print(f"AGR-resultat sparade till {csv_file_path}")

def summarize_quarterly_data_to_yearly(raw_financial_data_quarterly):
    """
    Sammanfattar kvartalsdata till årsdata genom att summera eller medelvärdesbilda beroende på segment.
    Args:
        raw_financial_data_quarterly (dict): Dictionary med kvartalsdata per ticker
    Returns:
        dict: Sammanfattad årsdata per ticker
    """
    # Define which metrics to sum (flow) and which to take latest (stock)
    sum_metrics = [
        'Net Income', 'EBIT', 'Pretax Income', 'Tax Provision', 'Interest Expense',
        'Gross Profit', 'Total Revenue', 'Operating Income', 'Basic EPS', 'EBITDA',
        'Operating Cash Flow', 'Free Cash Flow'
    ]
    latest_metrics = [
        'Stockholders Equity', 'Total Assets', 'Total Debt', 'Cash And Cash Equivalents',
        'sharesOutstanding', 'currentPrice', 'marketCap'
    ]
    summarized_data = {}
    for ticker, data in raw_financial_data_quarterly.items():
        if data is None:
            summarized_data[ticker] = None
            continue
        # Prepare output dict for this ticker
        summarized = {
            'balance_sheet': pd.DataFrame(),
            'income_statement': pd.DataFrame(),
            'cash_flow': pd.DataFrame(),
            'current_price': data.get('current_price'),
            'shares_outstanding': data.get('shares_outstanding'),
            'info': data.get('info'),
            'market_cap': data.get('market_cap'),
        }
        # For each segment, aggregate as needed
        for segment in ['balance_sheet', 'income_statement', 'cash_flow']:
            df = data.get(segment)
            if df is None or df.empty:
                summarized[segment] = pd.DataFrame()
                continue
            # Only keep the last 4 quarters
            df = df.head(4)
            agg_dict = {}
            for col in df.columns:
                if col in sum_metrics:
                    agg_dict[col] = df[col].sum()
                elif col in latest_metrics:
                    agg_dict[col] = df[col].iloc[0]
                else:
                    # Default: take latest
                    agg_dict[col] = df[col].iloc[0]
            # Create a single-row DataFrame with index as latest quarter's date
            latest_idx = df.index[0] if len(df.index) > 0 else None
            summarized[segment] = pd.DataFrame([agg_dict], index=[latest_idx])
        summarized_data[ticker] = summarized
    return summarized_data

def post_processing(final_df, rank_decimals):
    def get_quarter(dt):
        if pd.isnull(dt):
            return None
        return (dt.month - 1) // 3 + 1

    def quarter_diff(row):
        if pd.isnull(row['LatestReportDate_Q']) or pd.isnull(row['LatestReportDate_Y']):
            return None
        y1, q1 = row['LatestReportDate_Y'].year, get_quarter(row['LatestReportDate_Y'])
        y2, q2 = row['LatestReportDate_Q'].year, get_quarter(row['LatestReportDate_Q'])
        return (y2 - y1) * 4 + (q2 - q1)
    
    # Parse dates
    final_df['LatestReportDate_Q'] = pd.to_datetime(final_df['LatestReportDate_Q'], errors='coerce')
    final_df['LatestReportDate_Y'] = pd.to_datetime(final_df['LatestReportDate_Y'], errors='coerce')

    # Calculate quarter difference
    final_df['QuarterDiff'] = final_df.apply(quarter_diff, axis=1)
    
    
    
    all_ratios = []
    for category, ratios in config['kategorier'].items():
        all_ratios.extend(ratios)

    for ratio in all_ratios:
        final_df[f'{ratio}_TTM_diff'] = (final_df[f'{ratio}_TTM'] - final_df[f'{ratio}_latest_ratioValue'])
    # ROE_latest_ratioValue
    # ROE_TTM


    # Force index to string type
    final_df.index = final_df.index.astype(str)
    final_df['Name'] = final_df['Name'].astype(str)
    # Round all columns containing "Rank" to 1 decimal
    for col in final_df.columns:
        if "Rank" in col:
            final_df[col] = final_df[col].round(rank_decimals)
    return final_df

# --- Main Execution ---

if __name__ == "__main__":
    # Load configuration from YAML
    config = load_config("rank-config.yaml")
    if config:
        TICKERS_FILE_NAME = config["input_ticker_file"]
        if not TICKERS_FILE_NAME:
            print("No tickers file name found. Please check your CSV file.")
        else:
            # Step 0: Read tickers from CSV file
            print(f"Reading tickers from {CSV_PATH / TICKERS_FILE_NAME}...")
            tickers = read_tickers_from_csv(CSV_PATH / TICKERS_FILE_NAME)

            # Step 1: Fetch financial data for each ticker
            
            if not tickers:
                print("No tickers found in the file. Exiting.")
                exit(1)

            if FETCH_DATA == "Yes":
                raw_financial_data = {}
                raw_financial_data_quarterly = {}
                print("Fetching financial data...")
                for ticker in tqdm(tickers, desc="Fetching financial data", disable=True if ENVIRONMENT == "remote" else False):
                    raw_financial_data[ticker] = fetch_yfinance_data(ticker, config["data_fetch_years"], period_type="annual")
                    raw_financial_data_quarterly[ticker] = fetch_yfinance_data(ticker, config["data_fetch_quarterly"], period_type="quarterly")

                    # Save raw_financial_data as pickle for fast reload/debug
                    if ENVIRONMENT == "local":
                        with open(CSV_PATH / "raw_financial_data.pkl", "wb") as f:
                            pickle.dump(raw_financial_data, f)
                        with open(CSV_PATH / "raw_financial_data_quarterly.pkl", "wb") as f:
                            pickle.dump(raw_financial_data_quarterly, f)

            else:
                # Load raw financial data from pickle file
                print("Loading raw financial data from pickles...")
                try:
                    with open(CSV_PATH / "raw_financial_data.pkl", "rb") as f:
                        raw_financial_data = pickle.load(f)
                    with open(CSV_PATH / "raw_financial_data_quarterly.pkl", "rb") as f:
                        raw_financial_data_quarterly = pickle.load(f)
                except FileNotFoundError:
                    print("No raw financial data found. Please fetch data first.")
                    exit(1)

            # Remove tickers with no data
            raw_financial_data = {ticker: data for ticker, data in raw_financial_data.items() if data is not None}
            raw_financial_data_quarterly = {ticker: data for ticker, data in raw_financial_data_quarterly.items() if data is not None}

            # Save raw financial data and business summaries
            save_raw_data_to_csv(raw_financial_data, CSV_PATH / "raw_financial_data.csv")
            save_raw_data_to_csv(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.csv")
            save_longBusinessSummary_to_csv(raw_financial_data, CSV_PATH / "longBusinessSummary.csv")
            save_dividends_to_csv(raw_financial_data, CSV_PATH / "dividends.csv")
            save_latest_report_dates_to_csv(raw_financial_data, CSV_PATH / "latest_report_dates.csv", period_type="Y")
            save_latest_report_dates_to_csv(raw_financial_data_quarterly, CSV_PATH / "latest_report_dates_quarterly.csv", period_type="Q")

            # Step 2: Fetch and process stock price data
            
            if FETCH_DATA == "Yes":
                print("Fetching stock price data...")
                get_price_data(config["SMA_short"],config["SMA_medium"], config["SMA_long"],
                           raw_financial_data.keys(),config["price_data_years"],CSV_PATH / config["price_data_file"])
            
            save_last_SMA_to_csv(
                read_from=CSV_PATH / config["price_data_file"],
                save_to=CSV_PATH / "last_SMA.csv"
)

            # Step 3: Calculate ratios and rankings
            raw_financial_data_quarterly_summarized=summarize_quarterly_data_to_yearly(raw_financial_data_quarterly)

            calculated_ratios_quarterly = calculate_all_ratios(raw_financial_data_quarterly_summarized, config["ratio_definitions"])
            save_calculated_ratios_to_csv(calculated_ratios_quarterly, CSV_PATH / "calculated_ratios_quarterly.csv", period_type="quarterly")

            calculated_ratios = calculate_all_ratios(raw_financial_data, config["ratio_definitions"])
            save_calculated_ratios_to_csv(calculated_ratios, CSV_PATH / "calculated_ratios.csv", period_type="annual")

            ranked_ratios = rank_all_ratios(calculated_ratios, config["ratio_definitions"])
            save_ranked_ratios_to_csv(ranked_ratios, CSV_PATH / "ranked_ratios.csv")

            # Step 4: Aggregate category and cluster ranks
            category_ranks = aggregate_category_ranks(ranked_ratios, config["category_ratios"])
            save_category_scores_to_csv(category_ranks, CSV_PATH / "category_ranks.csv")

            cluster_ranks = aggregate_cluster_ranks(category_ranks)
            save_category_scores_to_csv(cluster_ranks, CSV_PATH / "cluster_ranks.csv")

            # Step 5: Calculate AGR results
            agr_results = calculate_agr_for_ticker(CSV_PATH / "raw_financial_data.csv", tickers, config['agr_dimensions'])
            save_agr_results_to_csv(agr_results, CSV_PATH / "agr_results.csv")

            agr_dividend = calculate_agr_dividend_for_ticker(CSV_PATH / "dividends.csv", tickers, config.get('data_fetch_years', 4))
            save_agr_results_to_csv(agr_dividend, CSV_PATH / "agr_dividend_results.csv")

            # Step 6: Combine all results and save final output
            combined_results = combine_all_results(
                calculated_ratios,
                ranked_ratios,
                category_ranks,
                cluster_ranks
            )

            final_results=post_processing(combined_results,config["rank_decimals"])
            save_results_to_csv(final_results, CSV_PATH / config["results_file"])

            print(f"Stock evaluation completed and saved to {CSV_PATH / config['results_file']}")
    else:
        print("Could not load configuration. Exiting.")
