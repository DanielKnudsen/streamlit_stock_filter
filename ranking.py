import numpy as np
import pandas as pd

def create_ratios_to_ranks(calculated_ratios,calculated_ratios_quarterly,ratio_definitions,category_ratios):
    ranked_ratios = {ticker: {} for ticker in calculated_ratios.keys()}

    df = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    # only keep columns ending with '_latest_ratioValue' or '_trend_ratioValue'
    df = df[df.columns[df.columns.str.endswith(('_latest_ratioValue', '_trend_ratioValue'))]]
    # rename columns from '_latest_ratioValue' to '_latest' and '_trend_ratioValue' to '_trend'
    df.columns = df.columns.str.replace('_latest_ratioValue', '_latest')
    df.columns = df.columns.str.replace('_trend_ratioValue', '_trend')

    df_quarterly = pd.DataFrame.from_dict(calculated_ratios_quarterly, orient='index')
    # only keep columns ending with '_latest_ratioValue' and rename to '_ttm'
    df_quarterly = df_quarterly[df_quarterly.columns[df_quarterly.columns.str.endswith('_latest_ratioValue')]]
    df_quarterly.columns = df_quarterly.columns.str.replace('_latest_ratioValue', '_ttm')   

    # left join df_quarterly onto df where
    df_merged = df.merge(df_quarterly, how='left', left_index=True, right_index=True, suffixes=('', '_quarterly'))
    # sort column names
    df_merged = df_merged.reindex(sorted(df_merged.columns), axis=1)

    # go through the columns and calculate the diff between columns ending with *'_ttm' and column ending with *'_latest', call them *'_ttm_diff'
    for col in df_merged.columns:
        if col.endswith('_latest'):
            base = col[:-7]  # remove '_latest'
            ttm_col = base + '_ttm'
            if ttm_col in df_merged.columns:
                diff_col = base + '_ttm_diff'
                df_merged[diff_col] = df_merged[ttm_col] - df_merged[col]
    
    # drop columns ending with '_ttm' and rename columns ending with '_ttm_diff' to '_ttm'
    df_merged = df_merged.drop(columns=[col for col in df_merged.columns if col.endswith('_ttm')])
    df_merged = df_merged.rename(columns={col: col.replace('_ttm_diff', '_ttm') for col in df_merged.columns if col.endswith('_ttm_diff')})

    for column in df_merged.columns:
        if column.endswith('_latest'):
            ratio_name = column.replace('_latest', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            ranked = df_merged[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_latest_ratioRank'] = rank if not pd.isna(rank) else np.nan
        elif column.endswith('_trend'):
            ratio_name = column.replace('_trend', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            ranked = df_merged[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_trend_ratioRank'] = rank if not pd.isna(rank) else np.nan
        elif column.endswith('_ttm'):
            ratio_name = column.replace('_ttm', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            ranked = df_merged[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_ttm_ratioRank'] = rank if not pd.isna(rank) else np.nan

    aggregated_ranks = aggregate_category_ranks(ranked_ratios, category_ratios)

    cluster_ranks = aggregate_cluster_ranks(aggregated_ranks)

    # combine ranked_ratios, aggregated_ranks and cluster_ranks
    # Merge all dicts per ticker so each ticker has all its data in a single nested dict
    complete_ranks = {}
    tickers = set(ranked_ratios) | set(aggregated_ranks) | set(cluster_ranks)
    for ticker in tickers:
        complete_ranks[ticker] = {}
        if ticker in ranked_ratios:
            complete_ranks[ticker].update(ranked_ratios[ticker])
        if ticker in aggregated_ranks:
            complete_ranks[ticker].update(aggregated_ranks[ticker])
        if ticker in cluster_ranks:
            complete_ranks[ticker].update(cluster_ranks[ticker])
    return complete_ranks

def aggregate_category_ranks(
    ranked_ratios: dict,
    category_ratios: dict
) -> dict:
    """
    Aggregate ranked ratios into a total score for each category and rank all aggregated values on a 0-100 scale.

    Args:
        ranked_ratios (dict): Ranked ratios per ticker.
        category_ratios (dict): Category definitions.

    Returns:
        dict: Aggregated category scores per ticker.
    """
    aggregated_scores = {}
    for ticker, ranks in ranked_ratios.items():
        if not ranks:
            continue
        ticker_scores = {}
        total_latest_score = 0
        total_trend_score = 0
        total_ttm_score = 0
        total_latest_weight = 0
        total_trend_weight = 0
        total_ttm_weight = 0
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
            if category.endswith('_latest_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_latest_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_latest_weight += 1
            elif category.endswith('_trend_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_trend_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_trend_weight += 1
            elif category.endswith('_ttm_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_ttm_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_ttm_weight += 1
        aggregated_scores[ticker] = ticker_scores
    df_agg = pd.DataFrame.from_dict(aggregated_scores, orient='index')
    for col in df_agg.columns:
        col_name = col.replace('_catAvg', '_catRank') if col.endswith('_catAvg') else col
        if df_agg[col].dtype in [float, int]:
            ranks = df_agg[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            df_agg[col_name] = ranks
    return df_agg.to_dict(orient='index')

def aggregate_cluster_ranks(category_ranks: dict) -> dict:
    """
    Aggregate category ranks into cluster ranks.

    Args:
        category_ranks (dict): Category ranks per ticker.

    Returns:
        dict: Cluster ranks per ticker.
    """
    results = []
    for ticker, subdict in category_ranks.items():
        latest_vals = [v for k, v in subdict.items() if "latest_catRank" in k]
        trend_vals = [v for k, v in subdict.items() if "trend_catRank" in k]
        ttm_vals = [v for k, v in subdict.items() if "ttm_catRank" in k]
        latest_avg = sum(latest_vals) / len(latest_vals) if latest_vals else None
        trend_avg = sum(trend_vals) / len(trend_vals) if trend_vals else None
        ttm_avg = sum(ttm_vals) / len(ttm_vals) if ttm_vals else None
        results.append({
            "Ticker": ticker,
            "Latest_clusterAvg": latest_avg,
            "Trend_clusterAvg": trend_avg,
            "TTM_clusterAvg": ttm_avg
        })
    df = pd.DataFrame(results)
    for col in df.columns:
        col_name = col.replace('_clusterAvg', '_clusterRank') if col.endswith('_clusterAvg') else col
        if df[col].dtype in [float, int]:
            ranks = df[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            df[col_name] = ranks
    return df.set_index('Ticker').to_dict(orient='index')
