"""
DESTINASI — Comprehensive Inbound Tourism Expenditure ETL Pipeline
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Ingests data/raw/comprehensive_expenditure.csv (9,790 rows across 38 origin markets
and 12 spending categories) and produces data/processed/expenditure_market_matrix.csv.

Computes exact RM Million expenditure values for all 12 spending categories across
all 38 source markets based on percentage shares and market total expenditure.
"""

from pathlib import Path
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_expenditure_data")

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "comprehensive_expenditure.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "expenditure_market_matrix.csv"

# 12 Authoritative Inbound Spending Categories
INBOUND_CATEGORIES = [
    "Shopping",
    "Accommodation",
    "Food & Beverages",
    "Local Transportation",
    "International Airfares",
    "Domestic Airfares",
    "Entertainment",
    "Medical",
    "Organised Tour",
    "Sports",
    "Fuel",
    "Others"
]


def process_comprehensive_expenditure(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Parses comprehensive_expenditure.csv into an analytical market matrix.
    Computes value_rm_million for each category from percentage shares and market totals.
    """
    input_path = raw_path or RAW_FILE
    dest_path = output_path or OUTPUT_FILE

    if not input_path.exists():
        logger.error("Source raw expenditure file not found at %s", input_path)
        raise FileNotFoundError(f"File not found: {input_path}")

    logger.info("Reading raw comprehensive expenditure from %s", input_path)
    df = pd.read_csv(input_path)

    records = []

    # Process all available years in raw data
    years = sorted(df["year"].dropna().unique().astype(int))

    for year in years:
        # Extract market totals for the year
        # Prefer Visitors audience if available, fallback to Tourists
        tot_sub = df[
            (df["year"] == year) &
            (df["metric_group"] == "Total expenditure") &
            (df["period_type"] == "Full year") &
            (df["value_rm_million"].notna())
        ]

        if tot_sub.empty:
            continue

        # If both Visitors and Tourists exist, prioritize Visitors for national headline numbers
        if "Visitors" in tot_sub["audience"].values:
            tot_sub = tot_sub[tot_sub["audience"] == "Visitors"]
        else:
            tot_sub = tot_sub[tot_sub["audience"] == "Tourists"]

        market_totals = {}
        for _, row in tot_sub.iterrows():
            mkt = row["market"]
            val = float(row["value_rm_million"])
            market_totals[mkt] = val

            # Add Total record ("All spending categories")
            records.append({
                "year": year,
                "market": mkt,
                "spending_category": "All spending categories",
                "value_rm_million": round(val, 2),
                "percentage_share": 100.0
            })

        # Check for expenditure component shares (Jan-Dec) for this year
        share_sub = df[
            (df["year"] == year) &
            (df["selected_period"] == "Jan-Dec") &
            (df["metric_group"] == "Expenditure components share") &
            (df["spending_category"].isin(INBOUND_CATEGORIES))
        ]

        if share_sub.empty:
            continue

        if "Visitors" in share_sub["audience"].values:
            share_sub = share_sub[share_sub["audience"] == "Visitors"]
        else:
            share_sub = share_sub[share_sub["audience"] == "Tourists"]

        for _, row in share_sub.iterrows():
            mkt = row["market"]
            cat = row["spending_category"]

            # Parse percentage
            pct = 0.0
            if pd.notna(row.get("value_numeric")):
                pct = float(row["value_numeric"])
            elif pd.notna(row.get("value_display")):
                v_str = str(row["value_display"]).replace("%", "").strip()
                try:
                    pct = float(v_str)
                except ValueError:
                    pct = 0.0

            mkt_total = market_totals.get(mkt, 0.0)
            val_rm_m = (pct / 100.0) * mkt_total

            records.append({
                "year": year,
                "market": mkt,
                "spending_category": cat,
                "value_rm_million": round(val_rm_m, 3),
                "percentage_share": round(pct, 2)
            })

    result_df = pd.DataFrame(records)

    # Sort deterministically
    result_df = result_df.sort_values(
        by=["year", "market", "spending_category"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(dest_path, index=False)
    logger.info("Successfully saved %d records to %s", len(result_df), dest_path)

    return result_df


if __name__ == "__main__":
    matrix_df = process_comprehensive_expenditure()
    print(f"ETL completed successfully. Total records: {len(matrix_df)}")
    print("Sample 2024 Overall records:")
    print(matrix_df[(matrix_df["year"] == 2024) & (matrix_df["market"] == "Overall")])
