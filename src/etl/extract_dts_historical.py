"""
DESTINASI — Comprehensive Multi-Year DTS Historical Extractor (2020–2025)
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Batch parses all 6 DOSM Domestic Tourism Survey (DTS) Publication Workbooks:
- Table of Publication DTS 2020.xlsx
- Jadual Penerbitan Survei Pelancongan Domestik, 2021.xlsx
- Jadual Penerbitan DTS 2022.xlsx
- Jadual Penerbitan DTS 2023.xlsx
- Jadual Penerbitan DTS 2024.xlsx
- JADUAL SURVEI PELANCONGAN DOMESTIK 2025.xlsx

Extracts:
1. Sheet 8B: Top Visited Administrative Districts per State (Historical Panel)
2. Sheet 8A: Top Visited Destinations & Attractions per State (Historical Panel)
3. Sheet 6: Component Expenditure Breakdown (Shopping, F&B, Lodging, Transport, etc.)
4. Sheet 10: State-to-State Origin-Destination (OD) Travel Matrices
5. Sheet 7: Primary Mode of Transport per State
"""

from pathlib import Path
import re
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_dts_historical")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DTS_FILES = [
    (2020, RAW_DIR / "Table of Publication DTS 2020.xlsx"),
    (2021, RAW_DIR / "Jadual Penerbitan Survei Pelancongan Domestik, 2021.xlsx"),
    (2022, RAW_DIR / "Jadual Penerbitan DTS 2022.xlsx"),
    (2023, RAW_DIR / "Jadual Penerbitan DTS 2023.xlsx"),
    (2024, RAW_DIR / "Jadual Penerbitan DTS 2024.xlsx"),
    (2025, RAW_DIR / "JADUAL SURVEI PELANCONGAN DOMESTIK 2025.xlsx"),
]


def extract_districts_from_sheet_8b(file_path: Path, year: int) -> list:
    """Extracts top visited districts from Sheet 8B."""
    if not file_path.exists():
        return []
    try:
        df = pd.read_excel(file_path, sheet_name="8B")
    except Exception as e:
        logger.warning("Failed reading 8B from %s: %s", file_path.name, e)
        return []

    records = []
    # Left table: col 1 (state) and col 2 (districts)
    for r in range(len(df)):
        st = df.iloc[r, 1] if df.shape[1] > 1 else None
        val = df.iloc[r, 2] if df.shape[1] > 2 else None
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower() and "nota" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                clean_item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
                if clean_item and len(clean_item) > 1:
                    records.append({"year": year, "state": state, "rank": rank, "district_name": clean_item})

    # Right table: col 6 (state) and col 7 (districts)
    for r in range(len(df)):
        st = df.iloc[r, 6] if df.shape[1] > 6 else None
        val = df.iloc[r, 7] if df.shape[1] > 7 else None
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower() and "nota" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                clean_item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
                if clean_item and len(clean_item) > 1:
                    records.append({"year": year, "state": state, "rank": rank, "district_name": clean_item})

    return records


def extract_attractions_from_sheet_8a(file_path: Path, year: int) -> list:
    """Extracts top destinations/attractions from Sheet 8A."""
    if not file_path.exists():
        return []
    try:
        df = pd.read_excel(file_path, sheet_name="8A")
    except Exception as e:
        logger.warning("Failed reading 8A from %s: %s", file_path.name, e)
        return []

    records = []
    # Left table: col 1 and col 2
    for r in range(len(df)):
        st = df.iloc[r, 1] if df.shape[1] > 1 else None
        val = df.iloc[r, 2] if df.shape[1] > 2 else None
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower() and "nota" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                clean_item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
                if clean_item and len(clean_item) > 1:
                    records.append({"year": year, "state": state, "rank": rank, "attraction_name": clean_item})

    # Right table: col 6 and col 7
    for r in range(len(df)):
        st = df.iloc[r, 6] if df.shape[1] > 6 else None
        val = df.iloc[r, 7] if df.shape[1] > 7 else None
        if pd.notna(st) and pd.notna(val) and "negeri" not in str(st).lower() and "table" not in str(st).lower() and "nota" not in str(st).lower():
            state = str(st).strip()
            items = [x.strip() for x in str(val).split("\n") if x.strip()]
            for rank, item in enumerate(items, 1):
                clean_item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
                if clean_item and len(clean_item) > 1:
                    records.append({"year": year, "state": state, "rank": rank, "attraction_name": clean_item})

    return records


def extract_spending_from_sheet_6(file_path: Path, year: int) -> list:
    """Extracts expenditure components from Sheet 6."""
    if not file_path.exists():
        return []
    try:
        df = pd.read_excel(file_path, sheet_name="6")
    except Exception as e:
        logger.warning("Failed reading Sheet 6 from %s: %s", file_path.name, e)
        return []

    records = []
    for r in range(len(df)):
        comp = df.iloc[r, 0]
        if pd.notna(comp):
            c_str = str(comp).strip()
            # Look for expenditure components
            if any(k in c_str.lower() for k in ["beli", "makan", "minum", "penginapan", "pengangkutan", "minyak", "pakej", "aktiviti", "hiburan", "shopping", "food"]):
                clean_comp = c_str.split("\n")[0].strip()
                # Find numerical values in subsequent columns
                nums = []
                for c in range(1, min(len(df.columns), 6)):
                    v = df.iloc[r, c]
                    try:
                        f = float(str(v).replace(",", ""))
                        nums.append(f)
                    except ValueError:
                        pass
                if nums:
                    val_rm = nums[0]
                    share = nums[1] if len(nums) > 1 else None
                    records.append({
                        "year": year,
                        "component": clean_comp,
                        "expenditure_rm_million": val_rm,
                        "share_pct": share
                    })
    return records


def extract_od_matrix_from_sheet_10(file_path: Path, year: int) -> list:
    """Extracts state origin-destination flows from Sheet 10."""
    if not file_path.exists():
        return []
    try:
        df = pd.read_excel(file_path, sheet_name="10")
    except Exception as e:
        logger.warning("Failed reading Sheet 10 from %s: %s", file_path.name, e)
        return []

    records = []
    # Identify row with destination states
    header_row_idx = None
    for r in range(min(12, len(df))):
        row_vals = [str(x).strip() for x in df.iloc[r].tolist() if pd.notna(x)]
        if any("johor" in v.lower() for v in row_vals) and any("perak" in v.lower() for v in row_vals):
            header_row_idx = r
            break

    if header_row_idx is not None:
        dest_cols = []
        for c in range(len(df.columns)):
            v = df.iloc[header_row_idx, c]
            if pd.notna(v) and any(s in str(v).lower() for s in ["johor", "kedah", "kelantan", "melaka", "negeri", "pahang", "perak", "perlis", "pulau", "selangor", "terengganu", "sabah", "sarawak", "kuala lumpur", "labuan", "putrajaya"]):
                dest_cols.append((c, str(v).strip()))

        for r in range(header_row_idx + 1, min(header_row_idx + 25, len(df))):
            orig_candidate = None
            for c in range(min(4, len(df.columns))):
                v = df.iloc[r, c]
                if pd.notna(v) and any(s in str(v).lower() for s in ["johor", "kedah", "kelantan", "melaka", "negeri", "pahang", "perak", "perlis", "pulau", "selangor", "terengganu", "sabah", "sarawak", "kuala lumpur", "labuan", "putrajaya"]):
                    orig_candidate = str(v).strip()
                    break

            if orig_candidate and "jumlah" not in orig_candidate.lower():
                for c_idx, dest_name in dest_cols:
                    val = df.iloc[r, c_idx]
                    if pd.notna(val):
                        try:
                            f_val = float(str(val).replace(",", ""))
                            records.append({
                                "year": year,
                                "origin_state": orig_candidate,
                                "destination_state": dest_name,
                                "tourists_thousands": f_val
                            })
                        except ValueError:
                            pass

    return records


def run_all_dts_extraction():
    logger.info("=== Starting Multi-Year DTS Historical Extraction (2020-2025) ===")
    all_districts = []
    all_attractions = []
    all_spending = []
    all_od = []

    for year, fpath in DTS_FILES:
        logger.info("Processing DTS %d: %s", year, fpath.name)
        dists = extract_districts_from_sheet_8b(fpath, year)
        all_districts.extend(dists)
        logger.info("  -> Extracted %d district rankings for %d", len(dists), year)

        attrs = extract_attractions_from_sheet_8a(fpath, year)
        all_attractions.extend(attrs)
        logger.info("  -> Extracted %d attraction rankings for %d", len(attrs), year)

        spends = extract_spending_from_sheet_6(fpath, year)
        all_spending.extend(spends)
        logger.info("  -> Extracted %d spending components for %d", len(spends), year)

        ods = extract_od_matrix_from_sheet_10(fpath, year)
        all_od.extend(ods)
        logger.info("  -> Extracted %d OD pairs for %d", len(ods), year)

    df_dist = pd.DataFrame(all_districts)
    if not df_dist.empty:
        df_dist.to_csv(PROCESSED_DIR / "dts_all_years_districts.csv", index=False)
        logger.info("Saved %s (%d records)", "dts_all_years_districts.csv", len(df_dist))

    df_attr = pd.DataFrame(all_attractions)
    if not df_attr.empty:
        df_attr.to_csv(PROCESSED_DIR / "dts_all_years_attractions.csv", index=False)
        logger.info("Saved %s (%d records)", "dts_all_years_attractions.csv", len(df_attr))

    df_spend = pd.DataFrame(all_spending)
    if not df_spend.empty:
        df_spend.to_csv(PROCESSED_DIR / "dts_all_years_expenditure.csv", index=False)
        logger.info("Saved %s (%d records)", "dts_all_years_expenditure.csv", len(df_spend))

    df_od = pd.DataFrame(all_od)
    if not df_od.empty:
        df_od.to_csv(PROCESSED_DIR / "dts_all_years_od_matrix.csv", index=False)
        logger.info("Saved %s (%d records)", "dts_all_years_od_matrix.csv", len(df_od))

    logger.info("=== Multi-Year DTS Historical Extraction Complete ===")


if __name__ == "__main__":
    run_all_dts_extraction()
