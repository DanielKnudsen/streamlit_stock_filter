
import pandas as pd
from financial_metrics import calculate_agr_for_ticker, calculate_agr_dividend_for_ticker, save_agr_results_melted_to_csv, save_agr_results_to_csv

from config_utils import CSV_PATH

def process_agr_results(tickers, config):
    """Process AGR results for given tickers and config"""
    agr_results = calculate_agr_for_ticker(CSV_PATH / "raw_financial_data.csv", tickers, config['agr_dimensions'], period_type="year")
    save_agr_results_to_csv(agr_results, CSV_PATH / "agr_results_yearly.csv")
    save_agr_results_melted_to_csv(agr_results, CSV_PATH / "agr_results_yearly_melted.csv")

    agr_results_quarterly = calculate_agr_for_ticker(CSV_PATH / "raw_financial_data_ttm_summarized.csv", tickers, config['agr_dimensions'], period_type="quarterly")
    save_agr_results_to_csv(agr_results_quarterly, CSV_PATH / "agr_results_quarterly.csv")
    save_agr_results_melted_to_csv(agr_results_quarterly, CSV_PATH / "agr_results_quarterly_melted.csv")

    agr_dividend = calculate_agr_dividend_for_ticker(CSV_PATH / "dividends.csv", tickers, config.get('data_fetch_years', 4))
    save_agr_results_to_csv(agr_dividend, CSV_PATH / "agr_dividend_results.csv")
    save_agr_results_melted_to_csv(agr_dividend, CSV_PATH / "agr_dividend_results_melted.csv")

    # read the three melted csv files and stack them into one file
    df_combined = pd.concat([
        pd.read_csv(CSV_PATH / "agr_results_yearly_melted.csv"),
        pd.read_csv(CSV_PATH / "agr_results_quarterly_melted.csv"),
        pd.read_csv(CSV_PATH / "agr_dividend_results_melted.csv")
    ], ignore_index=True)
    df_combined.to_csv(CSV_PATH / "agr_all_results_melted.csv", index=False)

    