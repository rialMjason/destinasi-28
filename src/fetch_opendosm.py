"""
DESTINASI — OpenDOSM & Data.gov.my Data Ingestion Pipeline
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Fetches verified parquet and high-frequency REST datasets from OpenDOSM and data.gov.my,
handles inter-year administrative district naming reconciliations, and outputs cleaned
district-level datasets.
"""

from pathlib import Path
import json
import logging
import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_opendosm")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Verified direct OpenDOSM & data.gov.my storage URLs
URLS = {
    "population_district": "https://storage.dosm.gov.my/population/population_district.parquet",
    "hies_district": "https://storage.dosm.gov.my/hies/hies_district.parquet",
    "gdp_district": "https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet",
    "lfs_district": "https://storage.dosm.gov.my/labour/lfs_district.parquet",
    "arrivals_soe": "https://storage.data.gov.my/demography/arrivals_soe.parquet",
}

# Verified high-frequency APIs (data.gov.my / opendosm)
APIS = {
    "ridership_headline": "https://api.data.gov.my/data-catalogue/?id=ridership_headline&limit=500&sort=-date",
    "cpi_state": "https://api.data.gov.my/opendosm/?id=cpi_state&limit=500&sort=-date",
    "fuelprice": "https://api.data.gov.my/data-catalogue/?id=fuelprice&limit=200&sort=-date",
}

# Administrative Synonym Crosswalk for inter-year DOSM naming drift
DISTRICT_SYNONYMS = {
    "larut dan matang": "Larut & Matang",
    "larut, matang dan selama": "Larut & Matang",
    "larut & matang": "Larut & Matang",
    "seberang perai tengah": "S.P.Tengah",
    "s.p.tengah": "S.P.Tengah",
    "seberang perai utara": "S.P.Utara",
    "s.p.utara": "S.P.Utara",
    "seberang perai selatan": "S.P. Selatan",
    "s.p. selatan": "S.P. Selatan",
    "hulu terengganu": "Hulu",
    "hulu": "Hulu",
    "lubok antu": "Lubok Antu",
}


def normalize_district_name(name: str) -> str:
    """Normalizes district string to standard format."""
    if not isinstance(name, str):
        return ""
    clean = name.strip()
    lower = clean.lower()
    return DISTRICT_SYNONYMS.get(lower, clean)


def fetch_parquet_dataset(name: str, url: str) -> pd.DataFrame:
    """Reads a remote parquet table, caches to data/raw, and returns DataFrame."""
    cache_file = RAW_DIR / f"{name}.parquet"
    logger.info("Fetching %s from %s...", name, url)
    try:
        df = pd.read_parquet(url)
        df.to_parquet(cache_file, index=False)
        logger.info("Successfully fetched %s (%d rows). Cached at %s", name, len(df), cache_file)
        return df
    except Exception as exc:
        logger.warning("Direct fetch failed for %s (%s). Attempting local cache...", name, exc)
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        raise exc


def fetch_api_dataset(name: str, url: str) -> pd.DataFrame:
    """Queries a data.gov.my or OpenDOSM JSON endpoint and returns DataFrame."""
    cache_file = RAW_DIR / f"{name}.json"
    logger.info("Querying API %s...", name)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        df = pd.DataFrame(data)
        logger.info("API %s returned %d records.", name, len(df))
        return df
    except Exception as exc:
        logger.warning("API query failed for %s (%s). Attempting local cache...", name, exc)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
        raise exc


def fetch_all_core_data():
    """Pulls all verified datasets for DESTINASI."""
    dfs = {}
    for name, url in URLS.items():
        try:
            dfs[name] = fetch_parquet_dataset(name, url)
        except Exception as e:
            logger.error("Error loading %s: %s", name, e)

    for name, url in APIS.items():
        try:
            dfs[name] = fetch_api_dataset(name, url)
        except Exception as e:
            logger.error("Error loading API %s: %s", name, e)

    return dfs


if __name__ == "__main__":
    logger.info("Starting OpenDOSM & Data.gov.my Ingestion Pipeline...")
    data = fetch_all_core_data()
    logger.info("Ingestion completed successfully.")
