"""
DESTINASI — Comprehensive Dataset Compilation Pipeline
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Executes full ETL extraction across:
- DOSM Domestic Tourism Survey (DTS 2020–2025)
- Tourism Malaysia International Tourist Profiles
- MAMPU Online Travel Deals & Search Demand
- Google Search Travel Trends
Outputs clean, verified CSVs to data/processed/.
"""

from pathlib import Path
import sys
import logging
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parse_external_data import (
    parse_dts_districts,
    parse_dts_attractions,
    parse_dts_od_matrix,
    parse_dts_spending_components,
    parse_dts_state_visitors_timeseries,
    parse_tourist_profile_summary,
    parse_mampu_deals_search,
    fetch_destination_search_trends
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_comprehensive_datasets")

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_all():
    logger.info("=== Starting Comprehensive Dataset Build ===")

    # 1. DTS 2025 Excel Workbook
    dts_2025_file = RAW_DIR / "JADUAL SURVEI PELANCONGAN DOMESTIK 2025.xlsx"
    if dts_2025_file.exists():
        # Sheet 8B: Districts
        df_districts = parse_dts_districts(dts_2025_file)
        df_districts.to_csv(PROCESSED_DIR / "dts_district_visitation.csv", index=False)
        logger.info("Saved dts_district_visitation.csv (%d rows)", len(df_districts))

        # Sheet 8A: Attractions
        df_attractions = parse_dts_attractions(dts_2025_file)
        df_attractions.to_csv(PROCESSED_DIR / "dts_attractions_visitation.csv", index=False)
        logger.info("Saved dts_attractions_visitation.csv (%d rows)", len(df_attractions))

        # Sheet 10: OD Matrix
        df_od = parse_dts_od_matrix(dts_2025_file)
        df_od.to_csv(PROCESSED_DIR / "dts_origin_destination.csv", index=False)
        logger.info("Saved dts_origin_destination.csv (%d rows)", len(df_od))

        # Sheet 6: Expenditure components
        df_spend = parse_dts_spending_components(dts_2025_file)
        df_spend.to_csv(PROCESSED_DIR / "dts_expenditure_shares.csv", index=False)
        logger.info("Saved dts_expenditure_shares.csv (%d rows)", len(df_spend))

        # Sheet 9: State timeseries 2018-2025
        df_ts = parse_dts_state_visitors_timeseries(dts_2025_file)
        df_ts.to_csv(PROCESSED_DIR / "dts_timeseries_2018_2025.csv", index=False)
        logger.info("Saved dts_timeseries_2018_2025.csv (%d rows)", len(df_ts))

    # 2. International Tourist Profile PDF
    pdf_profile = RAW_DIR / "Tourist Profile 2023_New.pdf"
    if pdf_profile.exists():
        df_intl = parse_tourist_profile_summary(pdf_profile)
        df_intl.to_csv(PROCESSED_DIR / "international_profile_summary.csv", index=False)
        logger.info("Saved international_profile_summary.csv (%d rows)", len(df_intl))

    # 3. MAMPU Online Deals Search Intent
    mampu_file = RAW_DIR / "top_5_deals_packages_by_state_mampu.csv"
    if mampu_file.exists():
        df_mampu = parse_mampu_deals_search(mampu_file)
        df_mampu.to_csv(PROCESSED_DIR / "mampu_online_deals_search.csv", index=False)
        logger.info("Saved mampu_online_deals_search.csv (%d rows)", len(df_mampu))

    # 4. Google Travel Trends
    df_trends = fetch_destination_search_trends()
    df_trends.to_csv(PROCESSED_DIR / "google_travel_trends.csv", index=False)
    logger.info("Saved google_travel_trends.csv (%d rows)", len(df_trends))

    logger.info("=== Comprehensive Dataset Build Completed Successfully ===")


if __name__ == "__main__":
    build_all()
