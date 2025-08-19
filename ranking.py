import numpy as np
import pandas as pd

def rank_all_ratios(
    calculated_ratios: dict,
    ratio_definitions: dict
) -> dict:
    """
    Rank each ratio (latest year and trend) on a 0-100 scale using percentile ranking and hybrid method for trend.

    Args:
        calculated_ratios (dict): Calculated ratios per ticker.
        ratio_definitions (dict): Ratio definitions.

    Returns:
        dict: Ranked ratios per ticker.
    """
    ranked_ratios = {ticker: {} for ticker in calculated_ratios.keys()}
    df = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    for column in df.columns:
        if column.endswith('_latest_ratioValue'):
            ratio_name = column.replace('_latest_ratioValue', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            ranked = df[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_latest_ratioRank'] = rank if not pd.isna(rank) else np.nan
        elif column.endswith('_trend_ratioValue'):
            ratio_name = column.replace('_trend_ratioValue', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            ranked = df[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_trend_ratioRank'] = rank if not pd.isna(rank) else np.nan
        elif column.endswith('_TTM'):
            ratio_name = column.replace('_TTM', '')
            is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
            ranked = df[column].rank(pct=True, ascending=is_better) * 100
            ranked = ranked.fillna(50)
            for ticker, rank in ranked.items():
                ranked_ratios[ticker][f'{ratio_name}_TTM_ratioRank'] = rank if not pd.isna(rank) else np.nan
    return ranked_ratios

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
            if category.endswith('_latest_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_latest_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_latest_weight += 1
            elif category.endswith('_trend_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_trend_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_trend_weight += 1
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
            ranks = df[col].rank(pct=True, ascending=True) * 100
            ranks = ranks.fillna(50)
            df[col_name] = ranks
    return df.set_index('Ticker').to_dict(orient='index')
