import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCOME_DIR = os.path.join(BASE_DIR, "income")
BALANCE_DIR = os.path.join(BASE_DIR, "balance")
CASHFLOW_DIR = os.path.join(BASE_DIR, "cashflow")
OUTPUT_CSV = "fundamentals_trend.csv"

def calculate_cagr(start, end, periods):
    """
    Calculate Compound Annual Growth Rate (CAGR).
    :param start_value: Initial value (e.g., revenue at year 0)
    :param end_value: Final value (e.g., revenue at year N)
    :param periods: Number of periods (years)
    :return: CAGR as a decimal (e.g., 0.15 for 15%)
    """
    if start is None or end is None or periods <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / periods) - 1

def get_metric(row, year):
    return row.get(year, np.nan)

def safe_div(numerator, denominator):
    try:
        if denominator == 0 or pd.isna(denominator):
            return np.nan
        return numerator / denominator
    except Exception:
        return np.nan

def process_ticker(ticker):
    # Load CSVs
    try:
        income = pd.read_csv(os.path.join(INCOME_DIR, f"{ticker}.csv"), index_col=0)
        balance = pd.read_csv(os.path.join(BALANCE_DIR, f"{ticker}.csv"), index_col=0)
        cashflow = pd.read_csv(os.path.join(CASHFLOW_DIR, f"{ticker}.csv"), index_col=0)
    except Exception as e:
        print(f"Missing data for {ticker}: {e}")
        return []

    # Get years (columns) as strings, sorted descending (most recent first)
    years = [col for col in income.columns if col[:4].isdigit()]
    years = sorted(years, reverse=True)

    rows = []
    for year in years:
        # Income
        EBIT = get_metric(income.loc["EBIT"], year) if "EBIT" in income.index else np.nan
        Net_Income = get_metric(income.loc["Net Income"], year) if "Net Income" in income.index else np.nan
        EBITDA = get_metric(income.loc["EBITDA"], year) if "EBITDA" in income.index else np.nan
        Basic_EPS = get_metric(income.loc["Basic EPS"], year) if "Basic EPS" in income.index else np.nan
        Diluted_EPS = get_metric(income.loc["Diluted EPS"], year) if "Diluted EPS" in income.index else np.nan
        Tax_Rate = get_metric(income.loc["Tax Rate For Calcs"], year) if "Tax Rate For Calcs" in income.index else np.nan
        Total_Revenue = get_metric(income.loc["Total Revenue"], year) if "Total Revenue" in income.index else np.nan
        Interest_Expense = get_metric(income.loc["Interest Expense"], year) if "Interest Expense" in income.index else np.nan

        # Balance
        Stockholders_Equity = get_metric(balance.loc["Stockholders Equity"], year) if "Stockholders Equity" in balance.index else np.nan
        Long_Term_Debt = get_metric(balance.loc["Long Term Debt"], year) if "Long Term Debt" in balance.index else 0
        Current_Debt = get_metric(balance.loc["Current Debt And Capital Lease Obligation"], year) if "Current Debt And Capital Lease Obligation" in balance.index else 0
        End_Cash = get_metric(cashflow.loc["End Cash Position"], year) if "End Cash Position" in cashflow.index else 0
        Total_Liabilities = get_metric(balance.loc["Total Liabilities Net Minority Interest"], year) if "Total Liabilities Net Minority Interest" in balance.index else np.nan
        Invested_Capital = get_metric(balance.loc["Invested Capital"], year) if "Invested Capital" in balance.index else np.nan

        # Cashflow
        Operating_CF = get_metric(cashflow.loc["Operating Cash Flow"], year) if "Operating Cash Flow" in cashflow.index else np.nan
        Free_CF = get_metric(cashflow.loc["Free Cash Flow"], year) if "Free Cash Flow" in cashflow.index else np.nan
        CapEx = get_metric(cashflow.loc["Capital Expenditure"], year) if "Capital Expenditure" in cashflow.index else np.nan

        # Calculated metrics
        Net_Debt = (Long_Term_Debt + Current_Debt) - End_Cash
        Debt_Equity = safe_div(Long_Term_Debt + Current_Debt, Stockholders_Equity)
        Net_Debt_EBITDA = safe_div(Net_Debt, EBITDA)
        Interest_Coverage = safe_div(EBIT, Interest_Expense)
        ROIC = safe_div(EBIT * (1 - Tax_Rate), Invested_Capital) if not pd.isna(EBIT) and not pd.isna(Tax_Rate) and not pd.isna(Invested_Capital) else np.nan
        ROE = safe_div(Net_Income, Stockholders_Equity)
        Net_Margin = safe_div(Net_Income, Total_Revenue)
        
        rows.append({
            "Ticker": ticker,
            "Year": year,
            "Total Revenue": Total_Revenue,  # <-- Added here
            "EBIT": EBIT,
            "Net Income": Net_Income,
            "EBITDA": EBITDA,
            "ROIC": ROIC,
            "ROE": ROE,
            "Net Margin": Net_Margin,
            "Stockholders Equity": Stockholders_Equity,
            "Net Debt": Net_Debt,
            "Total Liabilities": Total_Liabilities,
            "Operating Cash Flow": Operating_CF,
            "Free Cash Flow": Free_CF,
            "Capital Expenditure": CapEx,
            "Debt/Equity Ratio": Debt_Equity,
            "Net Debt/EBITDA": Net_Debt_EBITDA,
            "Interest Coverage": Interest_Coverage,
            "Basic EPS": Basic_EPS,
            "Diluted EPS": Diluted_EPS
        })
    return rows

def main():
    tickers = [f.replace(".csv", "") for f in os.listdir(INCOME_DIR) if f.endswith(".csv")]
    all_rows = []
    for ticker in tickers:
        all_rows.extend(process_ticker(ticker))
    df = pd.DataFrame(all_rows)
    # Keep only the 4 most recent years per ticker
    df = df.sort_values(["Ticker", "Year"], ascending=[True, False])
    df = df.groupby("Ticker").head(4)
    df = df.sort_values(["Ticker", "Year"])  # Optional: sort for easier plotting
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved metrics to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()