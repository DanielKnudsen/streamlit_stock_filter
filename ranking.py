import numpy as np
import pandas as pd
from collections import OrderedDict
from config_mappings import ConfigMappings


def create_ratios_to_ranks(
    calculated_ratios: dict,
    calculated_ratios_ttm_trends: dict,
    ratio_definitions: dict,
    category_ratios: dict,
    config: dict = None
) -> dict:
    """
    Calculate percentile ranks for financial ratios and temporal views for each ticker based on cluster configuration.
    
    Uses config['cluster'] to determine which periods to process:
    - For data_source: "annual" - processes ratios from calculated_ratios
    - For data_source: "quarterly" - processes ratios from calculated_ratios_ttm_trends
    
    Column naming follows pattern: {ratio}_{period_suffix}_{cluster_period}_ratioValue
    where period_suffix is 'year' for annual data and 'quarter' for quarterly data.

    Then aggregate them by category, and compute cluster-level ranks.

    This function computes percentile-based ranks for each ratio (higher or lower is better depending on the definition), 
    and aggregates these ranks into category and cluster scores. 
    
    The output is a nested dictionary with all computed ranks for each ticker.

    Args:
        calculated_ratios (dict):
            Dictionary mapping ticker symbols to their annual calculated ratios. Each value is a dict of ratio names to values.
        calculated_ratios_ttm_trends (dict):
            Dictionary mapping ticker symbols to their quarterly calculated ratios. Each value is a dict of ratio names to values.
        ratio_definitions (dict):
            Dictionary defining each ratio's properties, including whether a higher value is better (key: 'higher_is_better').
        category_ratios (dict):
            Dictionary mapping category names to lists of ratio names that belong to each category.
        config (dict, optional):
            Configuration containing cluster definitions. If None, uses legacy behavior.

    Returns:
        dict: Nested dictionary where each key is a ticker symbol and each value is a dict containing:
            - Individual ratio ranks (e.g., 'ROE_ttm_current_ratioRank', 'ROE_long_trend_ratioRank')
            - Aggregated category average and rank (e.g., 'Profitability_catAvg', 'Profitability_catRank')
            - Cluster-level average and rank (e.g., 'ttm_current_clusterAvg', 'ttm_current_clusterRank')

    Example:
        >>> create_ratios_to_ranks(
                {'HM-B': {'ROE_year_long_trend_ratioValue': 0.25, ...}},
                {'AAK': {'ROE_quarter_ttm_current_ratioValue': 0.18, 'ROE_quarter_ttm_momentum_ratioValue': 0.02, ...}},
                {'ROE': {'formula': 'Net_Income / Stockholders_Equity', 'higher_is_better': True, ...}},
                {'Kvalitet_ttm_current_ratioRank': {'ROE_ttm_current_ratioRank': 1}, ...},
                config
            )
        {'AAPL': {
            'ROE_ttm_current_ratioRank': 80.0,
            'ROE_long_trend_ratioRank': 75.0,
            'Profitability_catAvg': 77.5,
            'Profitability_catRank': 90.0,
            'ttm_current_clusterAvg': 77.5,
            'ttm_current_clusterRank': 95.0,
            ...
        }}

    Notes:
        - Percentile ranks are on a 0-100 scale, with 100 being best and uses 'higher_is_better' to get the ranking calculation correct.
        - NaN values are filled with 50 for ranking purposes.
        - The function expects ratio keys to follow the naming convention: '{ratio}_{period_suffix}_{cluster_period}_ratioValue'
    """
    # Get cluster configuration and determine which periods to process
    mappings = ConfigMappings(config)
    cluster_periods = []
    if config and 'cluster' in config:
        cluster_config = config['cluster']
        if isinstance(cluster_config, dict):
            cluster_periods = list(cluster_config.keys())
        else:
            # Legacy list format
            cluster_periods = cluster_config
    else:
        # Fallback to new default period names
        cluster_periods = ['ttm_current', 'long_trend', 'ttm_momentum']
    
    ranked_ratios = {ticker: {} for ticker in calculated_ratios.keys()}

    # Create combined DataFrame with all ratio data
    df_annual = pd.DataFrame.from_dict(calculated_ratios, orient='index')
    df_quarterly = pd.DataFrame.from_dict(calculated_ratios_ttm_trends, orient='index')
    
    # Merge annual and quarterly dataframes on index (Ticker)
    df_merged = df_annual.merge(df_quarterly, how='outer', left_index=True, right_index=True)

    # Process each cluster period
    for period_name in cluster_periods:
        # Find columns that match this cluster period pattern
        period_columns = [col for col in df_merged.columns if f'_{period_name}_ratioValue' in col]
        
        for column in period_columns:
            # Extract ratio name from column (e.g., 'ROE_year_long_trend_ratioValue' -> 'ROE')
            if f'_{period_name}_ratioValue' in column:
                # Remove the suffix to get ratio name
                ratio_name = column.replace(f'_{period_name}_ratioValue', '')
                
                # Get ranking direction from ratio definition and handle ignore flag for ttm_current that should be skipped
                # this is because it is not meaningful to calculate ranks for pure values such as Total Revenue
                # it will just clutter the output with meaningless ranks
                is_better = ratio_definitions.get(ratio_name, {}).get('higher_is_better', True)
                ignore_calculate_rank_ttm_current = ratio_definitions.get(ratio_name, {}).get('ignore_calculate_rank_ttm_current', False)
                if period_name == 'ttm_current' and ignore_calculate_rank_ttm_current:
                    ranked = pd.Series([np.nan] * len(df_merged), index=df_merged.index)
                else:
                    # Calculate percentile ranks
                    ranked = df_merged[column].rank(pct=True, ascending=is_better) * 100
                ranked = ranked.fillna(50)
                
                # Store ranks for each ticker
                for ticker, rank in ranked.items():
                    if ticker not in ranked_ratios:
                        ranked_ratios[ticker] = {}
                    ranked_ratios[ticker][f'{ratio_name}_{period_name}_ratioRank'] = rank if not pd.isna(rank) else np.nan

    aggregated_ranks = aggregate_category_ranks(ranked_ratios, category_ratios)

    cluster_ranks = aggregate_cluster_ranks(aggregated_ranks, cluster_periods)


    def calc_total_ranks(cluster_ranks: dict, cluster_columns: list) -> dict:
        """
        Use the cluster columns from mappings to combine cluster ranks into a total rank for each ticker.
        Calculates the average of cluster ranks and then converts to percentile rank (0-100 scale).
        Args:
            cluster_ranks (dict): Cluster ranks per ticker.
            cluster_columns (list): List of cluster period names from config.
        
        Returns:
            dict: Nested dictionary with 'totalRank' key for each ticker.
        """
        # First, calculate averages
        total_rank_avgs = {}
        for ticker, ranks in cluster_ranks.items():
            total_rank = sum(ranks.get(col, 0) for col in cluster_columns) / len(cluster_columns) if cluster_columns else 0
            total_rank_avgs[ticker] = total_rank
        
        # Convert averages to percentile ranks (0-100 scale)
        total_ranks_result = {}
        if total_rank_avgs:
            df_totals = pd.Series(total_rank_avgs)
            total_rank_ranks = df_totals.rank(pct=True, ascending=True) * 100
            total_rank_ranks = total_rank_ranks.fillna(50)
            
            # Convert to nested dict format
            for ticker, rank_value in total_rank_ranks.items():
                total_ranks_result[ticker] = {'totalRank': rank_value}
        
        return total_ranks_result

    total_ranks_dict = calc_total_ranks(cluster_ranks, mappings.cluster_columns)

    # combine ranked_ratios, aggregated_ranks, cluster_ranks and total_ranks
    # Merge all dicts per ticker so each ticker has all its data in a single nested dict
    complete_ranks = {}
    sorted_ranked_ratios = {
        ticker: OrderedDict(sorted(data.items()))
        for ticker, data in ranked_ratios.items()
    }
    ranked_ratios = sorted_ranked_ratios
    sorted_aggregated_ranks = {
        ticker: OrderedDict(sorted(data.items()))
        for ticker, data in aggregated_ranks.items()
    }
    aggregated_ranks = sorted_aggregated_ranks
    sorted_cluster_ranks = {
        ticker: OrderedDict(sorted(data.items()))
        for ticker, data in cluster_ranks.items()
    }
    cluster_ranks = sorted_cluster_ranks
    tickers = set(ranked_ratios) | set(aggregated_ranks) | set(cluster_ranks) | set(total_ranks_dict)
    for ticker in tickers:
        complete_ranks[ticker] = {}
        if ticker in ranked_ratios:
            complete_ranks[ticker].update(ranked_ratios[ticker])
        if ticker in aggregated_ranks:
            complete_ranks[ticker].update(aggregated_ranks[ticker])
        if ticker in cluster_ranks:
            complete_ranks[ticker].update(cluster_ranks[ticker])
        if ticker in total_ranks_dict:
            complete_ranks[ticker].update(total_ranks_dict[ticker])
    """sorted_complete_ranks = {
        ticker: OrderedDict(sorted(data.items()))
        for ticker, data in complete_ranks.items()
    }"""
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
                        num_ratios += 1  # Count each ratio as 1, not ratios[rank_name]
            cat_avg_name = category.replace('_ratioRank', '')
            if num_ratios > 0:
                ticker_scores[f'{cat_avg_name}_catAvg'] = category_score / num_ratios if num_ratios > 0 else np.nan
            else:
                ticker_scores[f'{cat_avg_name}_catAvg'] = np.nan
            if category.endswith('_ttm_current_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_latest_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_latest_weight += 1
            elif category.endswith('_long_trend_ratioRank'):
                if not pd.isna(ticker_scores[f'{cat_avg_name}_catAvg']):
                    total_trend_score += ticker_scores[f'{cat_avg_name}_catAvg']
                    total_trend_weight += 1
            elif category.endswith('_ttm_momentum_ratioRank'):
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

def aggregate_cluster_ranks(category_ranks: dict, cluster_periods: list = None) -> dict:
    """
    Aggregate category ranks into cluster ranks.

    Args:
        category_ranks (dict): Category ranks per ticker.
        cluster_periods (list): List of period names from config.

    Returns:
        dict: Cluster ranks per ticker.
    """
    # Default to legacy periods if not provided
    if cluster_periods is None:
        cluster_periods = ['ttm_current', 'long_trend', 'ttm_momentum']
    
    results = []
    for ticker, subdict in category_ranks.items():
        period_averages = {}
        
        # Calculate averages for each configured period
        for period in cluster_periods:
            period_vals = [v for k, v in subdict.items() if f"{period}_catRank" in k]
            period_avg = sum(period_vals) / len(period_vals) if period_vals else None
            period_averages[f"{period}_clusterAvg"] = period_avg
        
        result_row = {"Ticker": ticker}
        result_row.update(period_averages)
        results.append(result_row)
    
    df = pd.DataFrame(results)
    
    # Convert averages to ranks
    for col in df.columns:
        if col.endswith('_clusterAvg') and col != 'Ticker':
            period_name = col.replace('_clusterAvg', '')
            rank_col_name = f'{period_name}_clusterRank'
            if df[col].dtype in [float, int]:
                ranks = df[col].rank(pct=True, ascending=True) * 100
                ranks = ranks.fillna(50)
                df[rank_col_name] = ranks
    
    return df.set_index('Ticker').to_dict(orient='index')
