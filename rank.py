import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os
import pickle


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

def fetch_yfinance_data(ticker, years):
    try:
        yf_ticker = f"{ticker}.ST"
        ticker_obj = yf.Ticker(yf_ticker)
        bs = ticker_obj.balance_sheet.transpose()
        is_ = ticker_obj.income_stmt.transpose()
        cf = ticker_obj.cash_flow.transpose()
        info = ticker_obj.info
        shares_outstanding = info.get('sharesOutstanding', None)
        current_price = info.get('currentPrice', None)
        market_cap = info.get('marketCap', None)
        longBusinessSummary = info.get('longBusinessSummary', 'No summary available')

        print(f"Ticker: {ticker}")
        """print(f"Balance sheet columns: {bs.columns.tolist() if not bs.empty else 'Empty'}")
        print(f"Income statement columns: {is_.columns.tolist() if not is_.empty else 'Empty'}")
        print(f"Cash flow columns: {cf.columns.tolist() if not cf.empty else 'Empty'}")
        print(f"Shares outstanding: {shares_outstanding}")
        print(f"Current price: {current_price}")"""

        if not all([bs is not None, is_ is not None, cf is not None, shares_outstanding is not None, current_price is not None]):
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
            'longBusinessSummary': longBusinessSummary,
            'market_cap': market_cap
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_all_ratios(raw_data, ratio_definitions, output_csv="csv-data/calculated_ratios.csv"):
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
            calculated_ratios[ticker] = {f'{ratio_name}_ratio_latest': np.nan for ratio_name in ratio_definitions}
            for ratio_name in ratio_definitions:
                calculated_ratios[ticker][f'{ratio_name}_ratio_trendSlope'] = np.nan
            continue

        # Säkerhetskontroll: Kontrollera om DataFrames är tomma
        if data['balance_sheet'].empty or data['income_statement'].empty or data['cash_flow'].empty:
            print(f"Varning: Finansiella data är ofullständig för {ticker}. Kan inte beräkna nyckeltal.")
            calculated_ratios[ticker] = {f'{ratio_name}_ratio_latest': np.nan for ratio_name in ratio_definitions}
            for ratio_name in ratio_definitions:
                calculated_ratios[ticker][f'{ratio_name}_ratio_trendSlope'] = np.nan
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
                    'sharesOutstanding': data['shares_outstanding'],
                    'currentPrice': data['current_price'],
                    'EBITDA': is_copy.loc[is_copy.index[0], 'EBITDA'] if 'EBITDA' in is_copy.columns else np.nan,
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
                    ratios[f'{ratio_name}_ratio_latest'] = np.nan
                    ratios[f'{ratio_name}_ratio_trendSlope'] = np.nan
                    for year in years:
                        ratios[f'{ratio_name}_year_{year}'] = np.nan
                    continue

                # Beräkna senaste årets värde
                latest_value = eval(definition['formula'], globals(), locals_dict)
                ratios[f'{ratio_name}_ratio_latest'] = latest_value

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
                                'sharesOutstanding': data['shares_outstanding'],
                                'currentPrice': data['current_price'],
                                'EBITDA': is_copy.loc[is_copy.index[i], 'EBITDA'] if 'EBITDA' in is_copy.columns else np.nan,
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
                        print(f"Varning: Ogiltiga historiska värden för {ticker}'s {ratio_name}_ratio_trendSlope: {y}")
                        ratios[f'{ratio_name}_ratio_trendSlope'] = np.nan
                    else:
                        slope, _ = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)
                        ratios[f'{ratio_name}_ratio_trendSlope'] = slope
                else:
                    print(f"Varning: Otillräcklig data för trendberäkning för {ticker}'s {ratio_name} (bs: {len(bs_copy)}, is: {len(is_copy)}, cf: {len(cf_copy)}, years: {years})")
                    ratios[f'{ratio_name}_ratio_trendSlope'] = np.nan
                    for year in years:
                        ratios[f'{ratio_name}_year_{year}'] = np.nan

            except (ZeroDivisionError, TypeError, KeyError) as e:
                print(f"Varning: Beräkningsfel för {ticker}'s {ratio_name}: {e}")
                print(f"Variabler: {locals_dict}")
                ratios[f'{ratio_name}_ratio_latest'] = np.nan
                ratios[f'{ratio_name}_ratio_trendSlope'] = np.nan
                for year in years:
                    ratios[f'{ratio_name}_year_{year}'] = np.nan

        calculated_ratios[ticker] = ratios

    # Konvertera calculated_ratios till en DataFrame och spara till CSV
    df = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    try:
        df.to_csv(output_csv)
        print(f"Resultat sparade till {output_csv}")
    except Exception as e:
        print(f"Fel vid skrivning till CSV {output_csv}: {e}")

    return calculated_ratios

def rank_all_ratios(calculated_ratios, ranking_config, ratio_definitions):
    """
    Rankar varje nyckeltal (senaste år och trend) på en 0-100 skala.
    Använder percentilrankning och hybridmetoden för trend.
    """
    ranked_ratios = {ticker: {} for ticker in calculated_ratios.keys()}
    df = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    #ratio_definitions = ranking_config.get('ratio_definitions', {})

    for column in df.columns:
        if column.endswith('_ratio_latest'):
            ratio_name = column.replace('_ratio_latest', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            #ascending = not is_better
            ranked = df[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)  # Fyller NaN med 50 för att representera medelvärde
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_latest_ratioRank'] = rank if not pd.isna(rank) else np.nan
        
        elif column.endswith('_ratio_trendSlope'):
            ratio_name = column.replace('_ratio_trendSlope', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            
            df_cleaned = df.loc[df[column].notna()]
            slopes = df_cleaned[column]

            if is_better:
                positive_trend_slopes = slopes[slopes > ranking_config.get('ranking_method', {}).get('trend_zero_threshold', 0)]
                negative_trend_slopes = slopes[slopes < -ranking_config.get('ranking_method', {}).get('trend_zero_threshold', 0)]
            else:
                positive_trend_slopes = slopes[slopes < -ranking_config.get('ranking_method', {}).get('trend_zero_threshold', 0)]
                negative_trend_slopes = slopes[slopes > ranking_config.get('ranking_method', {}).get('trend_zero_threshold', 0)]

            if not positive_trend_slopes.empty:
                pos_ranks = positive_trend_slopes.rank(pct=True, ascending=True) * 50 + 50

                for ticker, rank in pos_ranks.items():
                    ranked_ratios[ticker][f'{ratio_name}_trend_ratioRank'] = rank
            
            if not negative_trend_slopes.empty:
                neg_ranks = negative_trend_slopes.rank(pct=True, ascending=True) * 50
                for ticker, rank in neg_ranks.items():
                    ranked_ratios[ticker][f'{ratio_name}_trend_ratioRank'] = rank

            for ticker in df.index:
                if f'{ratio_name}_trend_ratioRank' not in ranked_ratios[ticker]:
                    ranked_ratios[ticker][f'{ratio_name}_trend_ratioRank'] = 50

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

def combine_all_results(calculated_ratios, ranked_ratios, category_scores,cluster_ranks,cagr_results,rank_decimals):
    """
    Slår ihop alla resultat till en enda DataFrame.
    """
    df_calculated = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    df_ranked = pd.DataFrame.from_dict(ranked_ratios, orient='index')
    df_scores = pd.DataFrame.from_dict(category_scores, orient='index')
    df_cluster_ranks = pd.DataFrame.from_dict(cluster_ranks, orient='index')
    df_cagr = pd.DataFrame.from_dict(cagr_results, orient='index')

    # Load tickers file as defined in config
    tickers_file = config.get("input_ticker_file", "tickers/tickers_lists.csv")
    df_tickers = pd.read_csv(tickers_file, index_col='Instrument')
    df_tickers = df_tickers.rename(columns={'Instrument': 'Ticker'})
    df_last_SMA = pd.read_csv("csv-data/last_SMA.csv", index_col='Ticker')

    final_df = pd.concat([df_calculated, df_ranked, df_scores, df_tickers, df_last_SMA, df_cluster_ranks, df_cagr], axis=1)
    # Round all columns containing "Rank" to 1 decimal
    for col in final_df.columns:
        if "Rank" in col:
            final_df[col] = final_df[col].round(rank_decimals)
    final_df
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

def save_calculated_ratios_to_csv(calculated_ratios, csv_file_path):
    df = pd.DataFrame()
    for ticker,data in calculated_ratios.items():
        df_metrics = pd.DataFrame(data, index=[ticker]).T
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
    for ticker in tickers:
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
                        cagr = ((end_price / start_price) ** (1 / num_years)) - 1
                        cagr_list.append({'Ticker': ticker, 'CAGR': cagr})
                    else:
                        cagr_list.append({'Ticker': ticker, 'CAGR': np.nan})
                else:
                    cagr_list.append({'Ticker': ticker, 'CAGR': np.nan})
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
            df[col_name] = ranks
    return df.set_index('Ticker').to_dict(orient='index')

def calculate_cagr(cagr_dimension, raw_financial_data_file, save_values_to_csv_file):
    """
    Beräknar CAGR (Compound Annual Growth Rate) för de angivna dimensionerna.
    Dimensionerna finns i 'Metric'-kolumnen och datum i 'Date'-kolumnen.
    Resultatnyckeln är 'cagr' + dimension med mellanslag ersatt av underscore.
    """
    cagr_results = {}
    raw_financial_data = pd.read_csv(raw_financial_data_file, parse_dates=['Date'])
    tickers = raw_financial_data['Ticker'].unique()
    # Collect all rows used for CAGR calculations
    cagr_raw_rows = []
    for ticker in tickers:
        data_ticker = raw_financial_data[raw_financial_data['Ticker'] == ticker]
        cagr_results[ticker] = {}
        for dimension in cagr_dimension:
            key = f"cagr{dimension.replace(' ', '_')}"
            data_dim = data_ticker[data_ticker['Metric'] == dimension].sort_values('Date')
            if not data_dim.empty:
                # Save all rows used for this ticker/dimension
                cagr_raw_rows.append(data_dim.assign(CAGR_Dimension=dimension))
                start_value = data_dim['Value'].iloc[0]
                end_value = data_dim['Value'].iloc[-1]
                start_date = pd.to_datetime(data_dim['Date'].iloc[0])
                end_date = pd.to_datetime(data_dim['Date'].iloc[-1])
                num_years = (end_date - start_date).days / 365.25
                if start_value > 0 and num_years > 0:
                    cagr = ((end_value / start_value) ** (1 / num_years)) - 1
                    cagr_results[ticker][key] = cagr
                else:
                    cagr_results[ticker][key] = np.nan
            else:
                cagr_results[ticker][key] = np.nan
    # Save all raw rows used for CAGR calculations to the specified CSV file
    if cagr_raw_rows:
        cagr_raw_df = pd.concat(cagr_raw_rows, ignore_index=True)
        cagr_raw_df.to_csv(save_values_to_csv_file, index=False)
    else:
        # If no data, create an empty file with expected columns
        pd.DataFrame(columns=['Ticker','Metric','Date','Value','CAGR_Dimension']).to_csv(save_values_to_csv_file, index=False)
    return cagr_results


# --- Huvudkörning ---

if __name__ == "__main__":
    config = load_config("rank-config.yaml")
    if config:
        tickers = read_tickers_from_csv(config["input_ticker_file"])
        if not tickers:
            print("No tickers to evaluate. Please check your CSV file.")
        else:
            # 1. Hämta eller ladda data
            output_dir = config.get("output_path", "csv-data")
            #pickle_file_path = os.path.join(output_dir, "raw_financial_data.pkl")
            raw_financial_data = {}

            for ticker in tickers:
                raw_financial_data[ticker] = fetch_yfinance_data(ticker, config["data_fetch_years"])
            
            # Filtrera bort tickers som inte har data 
            # TODO: Hantera fall där data är None
            raw_financial_data = {ticker: data for ticker, data in raw_financial_data.items() if data is not None}
            
            save_raw_data_to_csv(raw_financial_data, os.path.join(output_dir, "raw_financial_data.csv"))
            save_longBusinessSummary_to_csv(raw_financial_data, os.path.join(output_dir, "longBusinessSummary.csv"))
            print("läser in stock price data...")
            get_price_data(config["SMA_short"], 
                           config["SMA_medium"], 
                           config["SMA_long"],
                           tickers, 
                           config["price_data_years"], 
                           os.path.join(output_dir, config["price_data_file"]))
            save_last_SMA_to_csv(read_from=os.path.join(output_dir, config["price_data_file"]),
                                 save_to=os.path.join(output_dir, "last_SMA.csv"))

            # 2. Utför alla beräkningar och rankningar med den hämtade/sparade datan
            calculated_ratios = calculate_all_ratios(raw_financial_data, config["ratio_definitions"])
            save_calculated_ratios_to_csv(calculated_ratios, os.path.join(output_dir, "calculated_ratios.csv"))

            ranked_ratios = rank_all_ratios(calculated_ratios, config["ranking_method"],config["ratio_definitions"])
            save_ranked_ratios_to_csv(ranked_ratios, os.path.join(output_dir, "ranked_ratios.csv"))

            category_ranks = aggregate_category_ranks(ranked_ratios, config["category_ratios"])
            save_category_scores_to_csv(category_ranks, os.path.join(output_dir, "category_ranks.csv"))

            cluster_ranks = aggregate_cluster_ranks(category_ranks)
            save_category_scores_to_csv(cluster_ranks, os.path.join(output_dir, "cluster_ranks.csv"))

            cagr_results = calculate_cagr(config['cagr_dimension'], 
                                          os.path.join(output_dir, "raw_financial_data.csv"),
                                          os.path.join(output_dir, "cagr_results.csv"))

            final_results = combine_all_results(calculated_ratios, ranked_ratios, category_ranks, cluster_ranks, cagr_results, config["rank_decimals"])
            save_results_to_csv(final_results, config["output_file_path"])

            print(f"Aktieutvärdering slutförd och sparad i {config['output_file_path']}")
    else:
        print("Kunde inte ladda konfigurationen. Avslutar.")
