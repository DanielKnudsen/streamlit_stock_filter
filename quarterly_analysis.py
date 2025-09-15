"""
Quarterly Changes Analysis
Analyzes stocks that had quarterly report updates and their financial metrics
"""

import sqlite3
import pandas as pd
import logging
from datetime import date, timedelta
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_recent_quarterly_changes(db_path: str, days_back: int = 7) -> pd.DataFrame:
    """Get stocks that had quarterly changes in recent days"""
    
    with sqlite3.connect(db_path) as conn:
        cutoff_date = date.today() - timedelta(days=days_back)
        
        # Get unique quarterly changes from recent captures
        query = """
        SELECT DISTINCT
            ticker,
            date as capture_date,
            quarter_diff,
            capture_type
        FROM ranking_history 
        WHERE date >= ?
        AND capture_type = 'after_quarterly'
        ORDER BY date DESC, ticker
        """
        
        results = pd.read_sql(query, conn, params=(cutoff_date,))
        
        return results

def analyze_quarterly_stocks(csv_path: str, quarterly_tickers: list) -> pd.DataFrame:
    """Analyze the current financial metrics of stocks with quarterly changes"""
    
    # Load the current stock evaluation results
    df = pd.read_csv(csv_path)
    ticker_col = df.columns[0]  # First column is ticker
    
    # Filter to only quarterly-changed stocks
    quarterly_df = df[df[ticker_col].isin(quarterly_tickers)].copy()
    
    if quarterly_df.empty:
        return pd.DataFrame()
    
    # Focus on key metrics that indicate quarterly performance
    key_metrics = [
        ticker_col, 'Name', 'Sektor', 'Lista',
        'QuarterDiff',  # The quarterly difference that triggered capture
        'ROE_ttm_ratioValue', 'ROE_latest_ratioValue', 'ROE_ttm_diff',
        'ROIC_ttm_ratioValue', 'ROIC_latest_ratioValue', 'ROIC_ttm_diff', 
        'Vinstmarginal_ttm_ratioValue', 'Vinstmarginal_latest_ratioValue', 'Vinstmarginal_ttm_diff',
        'PE_tal_ttm_ratioValue', 'PE_tal_latest_ratioValue', 'PE_tal_ttm_diff',
        'Total_Revenue_ttm', 'Total_Revenue_ttm_diff',
        'Basic_EPS_ttm', 'Basic_EPS_ttm_diff',
        'Latest_clusterRank', 'TTM_clusterRank',
        'LatestReportDate_Y', 'LatestReportDate_Q'
    ]
    
    # Select available columns
    available_cols = [col for col in key_metrics if col in quarterly_df.columns]
    analysis_df = quarterly_df[available_cols].copy()
    
    return analysis_df

def generate_quarterly_report(db_path: str, csv_path: str, days_back: int = 7) -> str:
    """Generate a comprehensive quarterly changes report"""
    
    # Get recent quarterly changes
    quarterly_changes = get_recent_quarterly_changes(db_path, days_back)
    
    if quarterly_changes.empty:
        return f"No quarterly changes found in the last {days_back} days."
    
    # Get unique tickers that had quarterly changes
    quarterly_tickers = quarterly_changes['ticker'].unique().tolist()
    
    # Analyze their current financial metrics
    financial_analysis = analyze_quarterly_stocks(csv_path, quarterly_tickers)
    
    # Build report
    report = []
    report.append("🏢 QUARTERLY EARNINGS IMPACT ANALYSIS")
    report.append("="*60)
    report.append(f"📅 Analysis Period: Last {days_back} days")
    report.append(f"📊 Total stocks with quarterly updates: {len(quarterly_tickers)}")
    report.append("")
    
    if not financial_analysis.empty:
        # Sort by TTM cluster ranking (higher score = better performance, 100 is best)
        if 'TTM_clusterRank' in financial_analysis.columns:
            financial_analysis = financial_analysis.sort_values('TTM_clusterRank', ascending=False)
        
        report.append("🎯 TOP QUARTERLY PERFORMERS (by TTM Cluster Ranking):")
        report.append("-" * 50)
        
        for i, (_, row) in enumerate(financial_analysis.head(10).iterrows()):
            ticker = row.iloc[0]  # First column is ticker
            name = row.get('Name', 'N/A')
            sector = row.get('Sektor', 'N/A')
            
            # Key performance indicators
            ttm_rank = row.get('TTM_clusterRank', 999)
            latest_rank = row.get('Latest_clusterRank', 999)
            rank_improvement = latest_rank - ttm_rank  # Positive = rank improved (lower number)
            revenue_ttm_diff = row.get('Total_Revenue_ttm_diff', 0)
            eps_ttm_diff = row.get('Basic_EPS_ttm_diff', 0)
            quarter_diff = row.get('QuarterDiff', 0)
            
            report.append(f"#{i+1:2d}. {ticker} - {name}")
            report.append(f"     🏭 Sector: {sector}")
            report.append(f"     🏆 TTM Cluster Rank: #{ttm_rank:.0f}")
            if rank_improvement != 0:
                improvement_text = f"📈 +{rank_improvement:.0f}" if rank_improvement > 0 else f"📉 {rank_improvement:.0f}"
                report.append(f"     {improvement_text} rank positions vs Latest")
            report.append(f"     💰 Revenue TTM Change: {revenue_ttm_diff:+.0f}")
            report.append(f"     💎 EPS TTM Change: {eps_ttm_diff:+.3f}")
            report.append(f"     📊 Quarter Diff: {quarter_diff}")
            report.append("")
        
        # Show worst performers
        report.append("⚠️  QUARTERLY UNDERPERFORMERS:")
        report.append("-" * 40)
        
        worst_performers = financial_analysis.tail(5)
        for i, (_, row) in enumerate(worst_performers.iterrows()):
            ticker = row.iloc[0]
            name = row.get('Name', 'N/A')
            roe_ttm_diff = row.get('ROE_ttm_diff', 0)
            
            report.append(f"• {ticker} - {name}")
            report.append(f"  📉 ROE TTM Change: {roe_ttm_diff:+.3f}")
        
        report.append("")
        
        # Sector analysis
        if 'Sektor' in financial_analysis.columns and 'ROE_ttm_diff' in financial_analysis.columns:
            ticker_col = financial_analysis.columns[0]  # First column is ticker
            sector_performance = financial_analysis.groupby('Sektor').agg({
                'ROE_ttm_diff': 'mean',
                ticker_col: 'count'
            }).round(3).sort_values('ROE_ttm_diff', ascending=False)
            
            report.append("🏭 SECTOR PERFORMANCE:")
            report.append("-" * 30)
            for sector, data in sector_performance.iterrows():
                avg_roe = data['ROE_ttm_diff']
                count = data[ticker_col]
                report.append(f"• {sector}: {avg_roe:+.3f} avg ROE change ({count} stocks)")
    
    return "\n".join(report)

def main():
    setup_logging()
    
    # Paths
    db_path = Path("data/local/ranking_history.db")
    csv_path = Path("data/remote/stock_evaluation_results.csv")
    
    if not db_path.exists():
        logging.error(f"Database not found: {db_path}")
        return
        
    if not csv_path.exists():
        logging.error(f"CSV file not found: {csv_path}")
        return
    
    # Generate report
    logging.info("Generating quarterly changes analysis...")
    report = generate_quarterly_report(str(db_path), str(csv_path), days_back=7)
    
    # Print to console
    print(report)
    
    # Save to file
    output_file = Path("quarterly_changes_analysis.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info(f"📄 Quarterly analysis saved to: {output_file.absolute()}")

if __name__ == "__main__":
    main()
