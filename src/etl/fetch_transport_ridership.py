"""
DESTINASI — KTMB & Prasarana Public Transport Ridership Ingestion
Source: data.gov.my Data Catalogue (id=ridership_headline)
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Fetches high-frequency daily transit metrics:
- KTM ETS (Intercity Electric Train Service)
- KTM Komuter Utara (Northern Corridor)
- KTM Komuter Central (Klang Valley / Southern)
- KTM Intercity
- Rapid Penang Bus (rpn)
- Prasarana Rail (LRT, MRT, Monorail)
"""

from pathlib import Path
import json
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_transport_ridership")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.data.gov.my/data-catalogue/?id=ridership_headline&limit=365&sort=-date"


def fetch_ridership_data() -> pd.DataFrame:
    """Queries data.gov.my API for daily transit ridership and computes capacity metrics."""
    logger.info("Fetching KTMB & Prasarana ridership from data.gov.my...")
    cache_path = PROCESSED_DIR / "transit_ridership_headline.csv"

    try:
        resp = requests.get(API_URL, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        logger.info("Successfully fetched %d daily records from data.gov.my.", len(df))
    except Exception as e:
        logger.warning("data.gov.my API query failed (%s). Checking cache...", e)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise e

    # Clean and sort by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Save cleaned time-series
    df.to_csv(cache_path, index=False)
    logger.info("Saved transit_ridership_headline.csv (%d rows)", len(df))
    return df


def get_corridor_transit_kpis(df: pd.DataFrame) -> dict:
    """Computes operational transit headroom for pilot corridors."""
    recent = df.tail(30) # Last 30 days average

    return {
        "ets_daily_avg": round(recent["rail_ets"].dropna().mean()),
        "komuter_utara_daily_avg": round(recent["rail_komuter_utara"].dropna().mean()),
        "komuter_central_daily_avg": round(recent["rail_komuter"].dropna().mean()),
        "intercity_daily_avg": round(recent["rail_intercity"].dropna().mean()),
        "rapid_penang_daily_avg": round(recent["bus_rpn"].dropna().mean()),
        "latest_date": str(df["date"].max().strftime("%Y-%m-%d"))
    }


if __name__ == "__main__":
    df_ridership = fetch_ridership_data()
    kpis = get_corridor_transit_kpis(df_ridership)
    print("Corridor Transit KPIs:", kpis)
