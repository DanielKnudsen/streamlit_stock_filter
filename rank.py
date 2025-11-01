# rank.py
from io_utils import load_pickle, save_pickle
from data_fetcher import read_tickers_from_csv, get_price_data, get_raw_financial_data
from ratios import calculate_all_ratios
from ranking import create_ratios_to_ranks
from config_utils import load_config, CSV_PATH, ENVIRONMENT, FETCH_PRICE_DATA, FETCH_FUNDAMENTAL_DATA
from results_processing import combine_all_results,post_processing, summarize_quarterly_data_to_yearly
from data_io import (save_results_to_csv, save_raw_data_to_csv, 
                     save_longBusinessSummary_to_csv, save_market_cap_to_csv,
                     save_latest_report_dates_to_csv, save_dividends_to_csv,save_calculated_ratios_to_csv,
                     save_dict_of_dicts_to_csv)
from price_utils import add_historical_prices_to_filtered_data
from ttm_utils import combine_quarterly_summaries_for_ttm_trends
from avg_growth_rate import process_agr_results
from config_mappings import ConfigMappings


# --- Main Execution ---

if __name__ == "__main__":
    # Load configuration from YAML
    config = load_config("rank-config.yaml")
    if config:
        mappings = ConfigMappings(config)
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

            if FETCH_FUNDAMENTAL_DATA == "Yes":
                raw_financial_data, raw_financial_data_quarterly, raw_financial_info, raw_financial_data_dividends, valid_tickers = get_raw_financial_data(tickers, config["data_fetch_years"], config["data_fetch_quarterly"])
                if ENVIRONMENT == "local":
                    save_pickle(raw_financial_data, CSV_PATH / "raw_financial_data.pkl")
                    save_pickle(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.pkl")
                    save_pickle(raw_financial_info, CSV_PATH / "raw_financial_info.pkl")
                    save_pickle(raw_financial_data_dividends, CSV_PATH / "raw_financial_data_dividends.pkl")
            else:
                print("Loading raw financial data from pickles...")
                try:
                    raw_financial_data = load_pickle(CSV_PATH / "raw_financial_data.pkl")
                    raw_financial_data_quarterly = load_pickle(CSV_PATH / "raw_financial_data_quarterly.pkl")
                    raw_financial_info = load_pickle(CSV_PATH / "raw_financial_info.pkl")
                    raw_financial_data_dividends = load_pickle(CSV_PATH / "raw_financial_data_dividends.pkl")
                    valid_tickers = list(raw_financial_data.keys())
                except FileNotFoundError:
                    print("No raw financial data found. Please fetch data first.")
                    exit(1)

            print(f"Fetched data for {len(valid_tickers)} valid tickers out of {len(tickers)} total tickers.")

            # Save raw financial data and business summaries
            save_raw_data_to_csv(raw_financial_data, CSV_PATH / "raw_financial_data.csv")
            save_raw_data_to_csv(raw_financial_data_quarterly, CSV_PATH / "raw_financial_data_quarterly.csv")
            # save_info_to_csv(raw_financial_info, CSV_PATH / "raw_financial_info.csv")
            save_longBusinessSummary_to_csv(raw_financial_info, CSV_PATH / "longBusinessSummary.csv")
            save_dividends_to_csv(raw_financial_data_dividends, CSV_PATH / "dividends.csv")
            save_latest_report_dates_to_csv(raw_financial_data, CSV_PATH / "latest_report_dates.csv", period_type="Y")
            save_latest_report_dates_to_csv(raw_financial_data_quarterly, CSV_PATH / "latest_report_dates_quarterly.csv", period_type="Q")
            save_market_cap_to_csv(raw_financial_data, CSV_PATH / "market_cap.csv")
            # Step 2: Fetch and process stock price data
            
            if FETCH_PRICE_DATA == "Yes":
                print("Fetching stock price data...")
                get_price_data(config["SMA_short"],config["SMA_medium"], config["SMA_long"],config['SMA_1_month'],config['SMA_3_month'],
                           valid_tickers,config["price_data_years"],
                           CSV_PATH / config["price_data_file_raw"], 
                           CSV_PATH / config["price_data_file"])

            # Step 3: Calculate ratios and rankings
            # Define which keys are needed for ratio calculations
            ratio_keys = ['balance_sheet', 'income_statement', 'cash_flow', 'shares_outstanding', 'market_cap']
            
            # Filter raw_financial_data to only include needed keys
            filtered_raw_data = {
                ticker: {key: data[key] for key in ratio_keys if key in data}
                for ticker, data in raw_financial_data.items()
            }
            
            # Add historical prices around report dates
            print("Adding historical prices to filtered data...")
            filtered_raw_data_with_prices = add_historical_prices_to_filtered_data(
                filtered_raw_data, 
                CSV_PATH / config["price_data_file"]
            )
            
            # Calculate annual ratios
            calculated_ratios = calculate_all_ratios(filtered_raw_data_with_prices, config["ratio_definitions"], config=config)
            save_calculated_ratios_to_csv(calculated_ratios, CSV_PATH / "calculated_ratios.csv", period_type="annual")

            # Summarize quarterly data to yearly for most recent 4 quarters (0 quarters back)
            quarters_back=0
            raw_financial_data_quarterly_summarized_0 = summarize_quarterly_data_to_yearly(raw_financial_data_quarterly,quarters_back)
            
            # Summarize quarterly data to yearly for 4 quarters back (1 quarter back)
            quarters_back=1
            raw_financial_data_quarterly_summarized_1 = summarize_quarterly_data_to_yearly(raw_financial_data_quarterly,quarters_back)
            
            # Combine the two TTM summaries for trend analysis
            print("Combining TTM summaries for trend analysis...")
            combined_ttm_data = combine_quarterly_summaries_for_ttm_trends(
                raw_financial_data_quarterly_summarized_0,
                raw_financial_data_quarterly_summarized_1
            )

            # save combined TTM data for reference
            save_raw_data_to_csv(combined_ttm_data, CSV_PATH / "raw_financial_data_ttm_summarized.csv")
            
            # Filter combined TTM data
            filtered_combined_ttm = {
                ticker: {key: data[key] for key in ratio_keys if key in data}
                for ticker, data in combined_ttm_data.items()
                if data is not None
            }
            
            # Add historical prices to combined TTM data
            filtered_combined_ttm_with_prices = add_historical_prices_to_filtered_data(
                filtered_combined_ttm, 
                CSV_PATH / config["price_data_file"]
            )
            
            # Calculate TTM trend ratios
            calculated_ratios_ttm_trends = calculate_all_ratios(
                filtered_combined_ttm_with_prices, 
                config["ratio_definitions"],
                period_type="quarterly",
                config=config
            )
            save_calculated_ratios_to_csv(
                calculated_ratios_ttm_trends, 
                CSV_PATH / "calculated_ratios_ttm_trends.csv", 
                period_type="quarterly"
            )

            # Create ratios to ranks 
            complete_ranks = create_ratios_to_ranks(
                calculated_ratios,
                calculated_ratios_ttm_trends,
                config["ratio_definitions"],
                config["category_ratios"],
                config
            )
            save_dict_of_dicts_to_csv(complete_ranks, CSV_PATH / "complete_ranks.csv")

            # Step 4: Process AGR results
            process_agr_results(valid_tickers, config)

            # Step 7: Combine all results and save final output
            combined_results = combine_all_results(valid_tickers,
                calculated_ratios,
                calculated_ratios_ttm_trends,
                complete_ranks
            )

            final_results = post_processing(combined_results, config["rank_decimals"], config['sektor_avg'])
            #final_results_trimmed = trim_unused_columns(final_results, config)  # Trim unused columns
            save_results_to_csv(final_results, CSV_PATH / config["results_file"])

            print(f"Stock evaluation completed and saved to {CSV_PATH / config['results_file']}")
    else:
        print("Could not load configuration. Exiting.")
