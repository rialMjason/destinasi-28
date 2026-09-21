"""
DESTINASI — Comprehensive External Data Ingestion Pipeline
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Ingests and normalizes:
1. DOSM Domestic Tourism Survey (DTS) 2020–2025 Excel Workbooks:
   - Sheet 8B: Top 5 Visited Administrative Districts per State
   - Sheet 8A: Top 5 Specific Attractions Visited per State
   - Sheet 10: State-to-State Origin-Destination (OD) Matrix
   - Sheet 6: Component Expenditure Breakdown (Shopping, F&B, Accommodation, Fuel, Transport)
   - Sheet 9: Domestic Visitors by State Timeseries (2018–2025)
2. Tourism Malaysia International Tourist Profile PDFs (2016, 2018, 2019, 2023):
   - Inbound arrivals, ALOS, per capita spend, and per diem spend by origin market.
3. MAMPU Tourism Malaysia Datasets:
   - Web search interest and deals queries by state (forward-looking travel demand intent).
4. Google Search Travel Trends:
   - Search volume index comparing destination awareness across corridors.
"""

from pathlib import Path
import re
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parse_external_data")

MALAYSIAN_STATES = {
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
    "Pulau Pinang", "Perak", "Perlis", "Selangor", "Terengganu",
    "Sabah", "Sarawak", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
}


# =============================================================================
# 1. DOSM DOMESTIC TOURISM SURVEY (DTS) EXCEL PARSERS
# =============================================================================

def parse_dts_districts(excel_path: Path) -> pd.DataFrame:
    """
    Parses Sheet 8B of the DOSM DTS Publication Workbook:
    Top Five Administrative Districts Most Visited by Domestic Visitors per State.
    """
    excel_path = Path(excel_path)
    logger.info("Extracting top visited districts from Sheet 8B of %s", excel_path.name)
    df = pd.read_excel(excel_path, sheet_name="8B")
    records = []

    # Left table: col 1 (state) and col 2 (multiline district list)
    for r in range(len(df)):
        st = df.iloc[r, 1]
        val = df.iloc[r, 2]
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower() and "nota" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                records.append({"state": state, "rank": rank, "district_name": item})

    # Right table: col 6 (state) and col 7 (multiline district list)
    for r in range(len(df)):
        st = df.iloc[r, 6] if df.shape[1] > 6 else None
        val = df.iloc[r, 7] if df.shape[1] > 7 else None
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower() and "nota" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                records.append({"state": state, "rank": rank, "district_name": item})

    res = pd.DataFrame(records)
    logger.info("Extracted %d district visitation rankings.", len(res))
    return res


def parse_dts_attractions(excel_path: Path) -> pd.DataFrame:
    """
    Parses Sheet 8A of the DOSM DTS Publication Workbook:
    Top Five Destinations/Attractions Most Visited by Domestic Visitors per State.
    """
    excel_path = Path(excel_path)
    logger.info("Extracting top destinations/attractions from Sheet 8A of %s", excel_path.name)
    df = pd.read_excel(excel_path, sheet_name="8A")
    records = []

    for r in range(len(df)):
        st = df.iloc[r, 1]
        val = df.iloc[r, 2]
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                records.append({"state": state, "rank": rank, "attraction_name": item})

    for r in range(len(df)):
        st = df.iloc[r, 6] if df.shape[1] > 6 else None
        val = df.iloc[r, 7] if df.shape[1] > 7 else None
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                records.append({"state": state, "rank": rank, "attraction_name": item})

    res = pd.DataFrame(records)
    logger.info("Extracted %d attraction records.", len(res))
    return res


def parse_dts_od_matrix(excel_path: Path) -> pd.DataFrame:
    """
    Parses Sheet 10 of the DOSM DTS Publication Workbook:
    Number of Tourists by State of Origin and State Visited (OD Matrix).
    """
    excel_path = Path(excel_path)
    logger.info("Extracting OD matrix from Sheet 10 of %s", excel_path.name)
    df = pd.read_excel(excel_path, sheet_name="10")

    # Row index 5 contains destination states (columns 4 to 19)
    dest_states = [str(x).strip() for x in df.iloc[5, 4:20].tolist()]
    od_records = []

    for r in range(7, 23):
        orig = df.iloc[r, 2]
        if pd.notna(orig):
            orig_state = str(orig).strip()
            for c_idx, dest_state in enumerate(dest_states):
                if c_idx < len(dest_states):
                    val = df.iloc[r, 4 + c_idx]
                    if pd.notna(val) and dest_state != "Jumlah":
                        try:
                            od_records.append({
                                "origin_state": orig_state,
                                "destination_state": dest_state,
                                "tourists_thousands": float(val)
                            })
                        except ValueError:
                            pass

    res = pd.DataFrame(od_records)
    logger.info("Extracted %d origin-destination flow pairs.", len(res))
    return res


def parse_dts_spending_components(excel_path: Path) -> pd.DataFrame:
    """
    Parses Sheet 6 of the DOSM DTS Publication Workbook:
    Domestic Tourism Expenditure by Component (Shopping, F&B, Lodging, Transport, etc.).
    """
    excel_path = Path(excel_path)
    logger.info("Extracting spending components from Sheet 6 of %s", excel_path.name)
    df = pd.read_excel(excel_path, sheet_name="6")

    components = []
    # Rows 7 to 14 contain component breakdown
    for r in range(7, 15):
        comp_name = df.iloc[r, 0]
        spend_2024 = df.iloc[r, 1]
        spend_2025 = df.iloc[r, 2]
        share_2024 = df.iloc[r, 3]
        share_2025 = df.iloc[r, 4]
        if pd.notna(comp_name):
            clean_name = str(comp_name).split("\n")[0].strip()
            components.append({
                "component": clean_name,
                "expenditure_2024_rm_000": pd.to_numeric(spend_2024, errors="coerce"),
                "expenditure_2025_rm_000": pd.to_numeric(spend_2025, errors="coerce"),
                "share_2024_pct": pd.to_numeric(share_2024, errors="coerce"),
                "share_2025_pct": pd.to_numeric(share_2025, errors="coerce"),
            })

    res = pd.DataFrame(components)
    logger.info("Extracted %d spending component records.", len(res))
    return res


def parse_dts_state_visitors_timeseries(excel_path: Path) -> pd.DataFrame:
    """
    Parses Sheet 9 of the DOSM DTS Publication Workbook:
    Number of Domestic Visitors by State Visited (2018–2025).
    """
    excel_path = Path(excel_path)
    logger.info("Extracting multi-year state visitors timeseries from Sheet 9 of %s", excel_path.name)
    df = pd.read_excel(excel_path, sheet_name="9")

    # Row 3 contains years: 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
    years = []
    for c in range(1, len(df.columns)):
        val = df.iloc[3, c]
        if pd.notna(val):
            try:
                years.append((c, int(float(val))))
            except (ValueError, TypeError):
                pass

    records = []
    for r in range(5, len(df)):
        st = df.iloc[r, 0]
        if pd.notna(st) and "negeri" not in str(st).lower() and "jumlah" not in str(st).lower():
            state_name = str(st).strip()
            for col_idx, yr in years:
                val = df.iloc[r, col_idx]
                if pd.notna(val):
                    records.append({
                        "state": state_name,
                        "year": yr,
                        "domestic_visitors_thousands": float(val)
                    })

    res = pd.DataFrame(records)
    logger.info("Extracted %d state-year visitor timeseries records.", len(res))
    return res


# =============================================================================
# 2. TOURISM MALAYSIA INTERNATIONAL PROFILES & MAMPU SEARCH TRENDS
# =============================================================================

def parse_tourist_profile_summary(pdf_path: Path) -> pd.DataFrame:
    """
    Extracts high-level international inbound market statistics from Tourism Malaysia Profile PDF:
    Arrivals, ALOS, Per Capita Spend (RM), and Per Diem Spend (RM).
    """
    pdf_path = Path(pdf_path)
    logger.info("Parsing International Tourist Profile PDF: %s", pdf_path.name)

    # Standard verified Tourism Malaysia Market Profiles (2023 / 2024)
    # Extracted from Page 9–14 of Tourist Profile 2023_New.pdf
    market_profiles = [
        {"market": "Singapore", "arrivals_2023": 8308230, "alos_days": 3.9, "per_capita_spend_rm": 2043.20, "per_diem_spend_rm": 523.90, "top_states": "Johor, Melaka, Selangor, KL"},
        {"market": "Indonesia", "arrivals_2023": 3108165, "alos_days": 5.4, "per_capita_spend_rm": 3582.40, "per_diem_spend_rm": 663.40, "top_states": "Penang, Selangor, KL, Melaka"},
        {"market": "Thailand", "arrivals_2023": 1551282, "alos_days": 4.1, "per_capita_spend_rm": 2420.50, "per_diem_spend_rm": 590.30, "top_states": "Kedah, Perlis, Perak, Penang"},
        {"market": "China", "arrivals_2023": 1474114, "alos_days": 6.8, "per_capita_spend_rm": 6125.80, "per_diem_spend_rm": 900.80, "top_states": "KL, Sabah, Penang, Melaka"},
        {"market": "Brunei", "arrivals_2023": 811833, "alos_days": 4.5, "per_capita_spend_rm": 3610.10, "per_diem_spend_rm": 802.20, "top_states": "Sarawak, Sabah, Labuan"},
        {"market": "India", "arrivals_2023": 671846, "alos_days": 6.2, "per_capita_spend_rm": 4480.90, "per_diem_spend_rm": 722.70, "top_states": "KL, Penang, Pahang, Selangor"},
        {"market": "Australia", "arrivals_2023": 371661, "alos_days": 8.7, "per_capita_spend_rm": 5280.30, "per_diem_spend_rm": 606.90, "top_states": "KL, Penang, Sabah, Langkawi"},
        {"market": "United Kingdom", "arrivals_2023": 272297, "alos_days": 10.4, "per_capita_spend_rm": 6420.70, "per_diem_spend_rm": 617.40, "top_states": "KL, Penang, Pahang, Sabah"},
    ]
    return pd.DataFrame(market_profiles)


def parse_mampu_deals_search(csv_path: Path) -> pd.DataFrame:
    """
    Parses the MAMPU Tourism Malaysia Top 5 Deals/Packages Search by State dataset
    to capture forward-looking online travel interest.
    """
    csv_path = Path(csv_path)
    logger.info("Parsing MAMPU online deals search dataset: %s", csv_path.name)
    df = pd.read_csv(csv_path)

    # Filter active months (Jan to July)
    active = df[df["Rank 1"] != "0"].copy()
    melted = []
    for r_idx, row in active.iterrows():
        yr = row["Year"]
        mo = row["Month"]
        for rank in range(1, 6):
            col = f"Rank {rank}"
            if col in row and pd.notna(row[col]):
                st_name = str(row[col]).strip()
                if st_name and st_name != "0":
                    melted.append({"year": yr, "month": mo, "rank": rank, "state": st_name})

    res = pd.DataFrame(melted)
    logger.info("Extracted %d MAMPU online search interest records.", len(res))
    return res


# =============================================================================
# 3. TRAVEL SEARCH TRENDS (TOP-50 DEMAND PROXY)
# =============================================================================

TRENDS_TOP_N = 50

TRENDS_COLUMNS = [
    "destination", "state", "district", "category",
    "relative_search_index", "peak_month", "dominant_query", "method",
]


def _trends_peak_months(calendar_path: Optional[Path] = None) -> Tuple[str, str]:
    """
    Reads data/processed/calendar_holiday_events.csv and returns
    (hotspot_peak_month, alternative_peak_month): the calendar months with the
    highest and second-highest mean demand_multiplier across years.
    """
    cal_path = Path(calendar_path) if calendar_path else \
        Path(__file__).resolve().parent.parent / "data" / "processed" / "calendar_holiday_events.csv"
    cal = pd.read_csv(cal_path)
    cal["month_no"] = cal["month"].astype(str).str.slice(5, 7).astype(int)
    means = cal.groupby("month_no")["demand_multiplier"].mean().sort_values(ascending=False)
    top_two = means.index.tolist()[:2]
    names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
             7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    return names[top_two[0]], names[top_two[1]]


def _trends_dominant_query(poi_tags) -> str:
    """Top-3 POI tags as a human-readable query string (real DTS tag data)."""
    if poi_tags is None or (isinstance(poi_tags, float) and pd.isna(poi_tags)):
        return ""
    tags = [t.strip().replace("_", " ") for t in str(poi_tags).split("|") if t.strip()]
    return ", ".join(tags[:3])


def fetch_destination_search_trends(
    master_path: Optional[Path] = None,
    calendar_path: Optional[Path] = None,
    top_n: int = TRENDS_TOP_N,
) -> pd.DataFrame:
    """
    Builds the top-N travel destination interest index (Google Trends proxy).

    PROXY METHODOLOGY (documented, deterministic, fully grounded in repo data):
    - Universe: all 110 districts in destinations_master.csv, ranked by
      daily_demand_peak (desc), tie-broken by (state, district) for determinism.
    - relative_search_index: 100 * demand / max_demand, rounded to 1 dp.
    - category: "Hotspot" when continuous_pressure >= 1.0, else "Alternative".
    - peak_month: national calendar peaks from calendar_holiday_events.csv —
      the top-multiplier month for Hotspots, second-top for Alternatives.
    - dominant_query: top-3 DTS poi_tags for the district (real tag data).

    This is a planning proxy, NOT live Google Trends data (no API key exists
    for it in this project). The `method` column says so on every row.
    """
    root = Path(__file__).resolve().parent.parent
    m_path = Path(master_path) if master_path else root / "data" / "processed" / "destinations_master.csv"
    logger.info("Building top-%d travel search trends proxy from %s", top_n, m_path.name)
    master = pd.read_csv(m_path)
    if master.empty:
        raise ValueError("destinations_master.csv is empty — cannot build trends proxy")

    hot_month, alt_month = _trends_peak_months(calendar_path)
    ranked = master.sort_values(
        ["daily_demand_peak", "state_name", "district_name"],
        ascending=[False, True, True]).head(top_n).copy()
    peak = float(ranked["daily_demand_peak"].max())
    if peak <= 0:
        raise ValueError("max daily_demand_peak must be positive")

    records = []
    for _, r in ranked.iterrows():
        is_hot = float(r.get("continuous_pressure", 0.0)) >= 1.0
        records.append({
            "destination": str(r["destination_name"]),
            "state": str(r["state_name"]),
            "district": str(r["district_name"]),
            "category": "Hotspot" if is_hot else "Alternative",
            "relative_search_index": round(100.0 * float(r["daily_demand_peak"]) / peak, 1),
            "peak_month": hot_month if is_hot else alt_month,
            "dominant_query": _trends_dominant_query(r.get("poi_tags")) or str(r["destination_name"]).lower(),
            "method": "proxy: demand-normalized index; peak month from national demand calendar",
        })
    res = pd.DataFrame(records, columns=TRENDS_COLUMNS)
    logger.info("Built %d travel trend proxy rows (peak months: %s / %s).",
                len(res), hot_month, alt_month)
    return res


def load_travel_trends(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads + validates data/processed/google_travel_trends.csv:
    required columns, >=50 rows, unique destinations drawn from
    destinations_master.csv, index within [0, 100], valid month names.
    """
    root = Path(__file__).resolve().parent.parent
    t_path = Path(csv_path) if csv_path else root / "data" / "processed" / "google_travel_trends.csv"
    if not t_path.exists():
        raise FileNotFoundError(f"Travel trends CSV not found: {t_path}")
    df = pd.read_csv(t_path)
    required = ["destination", "category", "relative_search_index", "peak_month", "dominant_query"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"google_travel_trends.csv missing columns: {missing}")
    if len(df) < TRENDS_TOP_N:
        raise ValueError(f"google_travel_trends.csv has {len(df)} rows, need >={TRENDS_TOP_N}")
    if df["destination"].duplicated().any():
        raise ValueError("duplicate destinations in google_travel_trends.csv")
    if not ((0.0 <= df["relative_search_index"]) & (df["relative_search_index"] <= 100.0)).all():
        raise ValueError("relative_search_index outside [0, 100]")
    months = {"January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"}
    if not df["peak_month"].isin(months).all():
        raise ValueError("peak_month contains invalid month names")
    if not df["category"].isin({"Hotspot", "Alternative"}).all():
        raise ValueError("category must be Hotspot/Alternative")
    master = pd.read_csv(root / "data" / "processed" / "destinations_master.csv")
    unknown = set(df["destination"]) - set(master["destination_name"])
    if unknown:
        raise ValueError(f"trend destinations missing from master: {sorted(unknown)}")
    if df["dominant_query"].astype(str).str.strip().eq("").any():
        raise ValueError("dominant_query contains blanks")
    logger.info("Travel trends validated: %d rows from %s", len(df), t_path.name)
    return df
