"""
DESTINASI — Comprehensive Economic & Expenditure Profile Extractor
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Ingests and aggregates:
1. data/raw/comprehensive_expenditure.csv (9,792 rows of international and market expenditure)
2. data/raw/hies_state.csv (Household income, expenditure, poverty, Gini)
3. data/raw/gdp_state_real_supply.csv (Real GDP by state and economic sector)
Outputs structured panels to data/processed/.
"""

from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_economic_profiles")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_expenditure_database() -> pd.DataFrame:
    """Cleans and aggregates comprehensive_expenditure.csv via process_comprehensive_expenditure."""
    from src.etl.process_expenditure_data import process_comprehensive_expenditure
    return process_comprehensive_expenditure(
        raw_path=RAW_DIR / "comprehensive_expenditure.csv",
        output_path=PROCESSED_DIR / "expenditure_market_matrix.csv"
    )


def clean_hies_and_gdp() -> pd.DataFrame:
    """Cleans state poverty, household income, and services GDP supply."""
    hies_path = RAW_DIR / "hies_state.csv"
    gdp_path = RAW_DIR / "gdp_state_real_supply.csv"

    if not hies_path.exists():
        return pd.DataFrame()

    df_hies = pd.read_csv(hies_path)
    # Take latest 2024 records
    latest_hies = df_hies[df_hies["date"] == "2024-01-01"].copy()

    # GDP services sector (p3)
    if gdp_path.exists():
        df_gdp = pd.read_csv(gdp_path)
        # Check latest year in GDP
        latest_year = df_gdp["year"].max() if "year" in df_gdp.columns else None
        if latest_year:
            gdp_sub = df_gdp[df_gdp["year"] == latest_year]
            if "state" in gdp_sub.columns and "services" in gdp_sub.columns:
                latest_hies = latest_hies.merge(gdp_sub[["state", "services"]], on="state", how="left")

    latest_hies.to_csv(PROCESSED_DIR / "state_economic_profile.csv", index=False)
    logger.info("Saved state_economic_profile.csv (%d records)", len(latest_hies))
    return latest_hies


def run_all_economic_extraction():
    logger.info("=== Starting Comprehensive Economic Profile Extraction ===")
    clean_expenditure_database()
    clean_hies_and_gdp()
    logger.info("=== Economic Profile Extraction Complete ===")


if __name__ == "__main__":
    run_all_economic_extraction()
