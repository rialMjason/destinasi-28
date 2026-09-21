"""
DESTINASI — MOTAC PDF Publication Automated Batch Miner (118 PDFs)
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Batch extracts and normalizes:
1. Average Occupancy Rates (AOR) of Hotels (2017 to 2026 Q1)
2. Hotel and Rooms Inventory by State (2022 to 2026 Q1)
3. Hotel Guests (Domestic vs Foreigner) by State (2017 to 2026 Q1)
4. Monthly Visitor, Tourist, and Excursionist Arrivals by Origin Market
"""

from pathlib import Path
import re
import logging
import pdfplumber
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_motac_pdfs")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MALAYSIAN_STATES = [
    "Kuala Lumpur", "Putrajaya", "Selangor", "Pulau Pinang", "Perak",
    "Kedah", "Perlis", "Negeri Sembilan", "Melaka", "Johor",
    "Pahang", "Terengganu", "Kelantan", "Sabah", "Sarawak", "Labuan"
]


def clean_num(val_str: str) -> float:
    """Helper to convert string with commas or brackets to float."""
    if not val_str:
        return None
    s = str(val_str).strip().replace(",", "").replace("%", "")
    # Handle negative in brackets (1.5) -> -1.5
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


MALAYSIAN_STATES = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
    "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor",
    "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
]

STATE_EXTRACTION_ALIASES = {
    "PENANG": "Pulau Pinang",
    "PULAU PINANG": "Pulau Pinang",
    "KUALA LUMPUR": "W.P. Kuala Lumpur",
    "W.P. KUALA LUMPUR": "W.P. Kuala Lumpur",
    "PUTRAJAYA": "W.P. Putrajaya",
    "W.P. PUTRAJAYA": "W.P. Putrajaya",
    "LABUAN": "W.P. Labuan",
    "W.P. LABUAN": "W.P. Labuan",
    "SELANGOR": "Selangor",
    "PERAK": "Perak",
    "KEDAH": "Kedah",
    "PERLIS": "Perlis",
    "NEGERI SEMBILAN": "Negeri Sembilan",
    "MELAKA": "Melaka",
    "JOHOR": "Johor",
    "PAHANG": "Pahang",
    "TERENGGANU": "Terengganu",
    "KELANTAN": "Kelantan",
    "SABAH": "Sabah",
    "SARAWAK": "Sarawak",
}


def parse_all_aor_pdfs() -> pd.DataFrame:
    """Batch parses all AOR_*.pdf files for state occupancy rates across all 16 states (2017–2026)."""
    aor_files = sorted(RAW_DIR.glob("AOR_*.pdf"))
    logger.info("Found %d AOR PDF files to process.", len(aor_files))
    records = []

    for fpath in aor_files:
        fname = fpath.stem
        year_str = re.search(r"\d{4}", fname)
        if not year_str:
            continue
        year = int(year_str.group(0))
        period = "Annual"
        if "Q1" in fname:
            period = "Q1"
        elif "Q2" in fname:
            period = "Q2"
        elif "Q3" in fname:
            period = "Q3"
        elif "Q4" in fname:
            period = "Q4"

        try:
            with pdfplumber.open(fpath) as pdf:
                full_text = ""
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    # Handle two-line split of Negeri Sembilan in MOTAC PDFs
                    txt = re.sub(r"NEGERI\s*\n\s*([\d\.\s\-]+)\n\s*SEMBILAN", r"NEGERI SEMBILAN \1", txt, flags=re.IGNORECASE)
                    txt = re.sub(r"NEGERI\s*\n\s*SEMBILAN", "NEGERI SEMBILAN", txt, flags=re.IGNORECASE)
                    full_text += "\n" + txt

                lines = full_text.split("\n")
                for line in lines:
                    for alias, canonical_state in STATE_EXTRACTION_ALIASES.items():
                        if re.search(rf"\b{re.escape(alias)}\b", line, re.IGNORECASE):
                            tokens = line.split()
                            numbers = []
                            for tok in tokens:
                                num = clean_num(tok)
                                if num is not None:
                                    numbers.append(num)
                            if not numbers:
                                continue

                            # Parse true AOR according to official MOTAC PDF table column layout:
                            # 2017 to 2021: Column 1=Prev Year, Column 2=Current Year, Column 3=Difference
                            if year in [2017, 2018, 2019, 2020, 2021]:
                                aor_val = numbers[1] if len(numbers) >= 2 else numbers[0]
                            # 2022: Column 1=2022, Column 2=2021, Column 3=Difference
                            elif year == 2022:
                                aor_val = numbers[0]
                            # 2023 onwards: Last number in the row is the period/annual AOR
                            else:
                                aor_val = numbers[-1]

                            # Sanity check: ensure positive AOR within [10.0, 100.0]
                            if aor_val is not None and aor_val > 0.0:
                                records.append({
                                    "source_file": fpath.name,
                                    "year": year,
                                    "period": period,
                                    "state": canonical_state,
                                    "average_occupancy_rate_pct": round(float(aor_val), 1)
                                })
                            break
        except Exception as e:
            logger.warning("Error parsing %s: %s", fpath.name, e)

    df = pd.DataFrame(records).drop_duplicates(subset=["year", "period", "state"])
    logger.info("Extracted %d total AOR records.", len(df))
    return df


def parse_all_hotel_inventory_pdfs() -> pd.DataFrame:
    """Batch parses all hotel_inventory_*.pdf files for hotel counts and rooms."""
    inv_files = sorted(RAW_DIR.glob("hotel_inventory_*.pdf"))
    logger.info("Found %d Hotel Inventory PDF files to process.", len(inv_files))
    records = []

    for fpath in inv_files:
        fname = fpath.stem
        year_str = re.search(r"\d{4}", fname)
        if not year_str:
            continue
        year = int(year_str.group(0))
        period = "Annual"
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in fname:
                period = q
                break

        try:
            with pdfplumber.open(fpath) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    lines = text.split("\n")
                    for line in lines:
                        for state in MALAYSIAN_STATES:
                            if re.search(rf"\b{re.escape(state)}\b", line, re.IGNORECASE):
                                tokens = line.split()
                                nums = []
                                for tok in tokens:
                                    # Look for integer numbers (room counts often > 100)
                                    cleaned = tok.replace(",", "").strip()
                                    if cleaned.isdigit():
                                        nums.append(int(cleaned))
                                if len(nums) >= 2:
                                    # Last number is ALWAYS rooms, second-to-last is ALWAYS hotel count
                                    rooms = nums[-1]
                                    hotels = nums[-2]
                                    records.append({
                                        "source_file": fpath.name,
                                        "year": year,
                                        "period": period,
                                        "state": state,
                                        "hotels_count": hotels,
                                        "rooms_count": rooms
                                    })
                                elif len(nums) == 1:
                                    records.append({
                                        "source_file": fpath.name,
                                        "year": year,
                                        "period": period,
                                        "state": state,
                                        "hotels_count": None,
                                        "rooms_count": nums[0]
                                    })
                                break
        except Exception as e:
            logger.warning("Error parsing %s: %s", fpath.name, e)

    df = pd.DataFrame(records).drop_duplicates(subset=["year", "period", "state"])
    logger.info("Extracted %d total Hotel Inventory records.", len(df))
    return df


def parse_all_hotel_guests_pdfs() -> pd.DataFrame:
    """Batch parses all hotel_guests_*.pdf files for domestic vs foreign guests."""
    guest_files = sorted(RAW_DIR.glob("hotel_guests_*.pdf"))
    logger.info("Found %d Hotel Guests PDF files to process.", len(guest_files))
    records = []

    for fpath in guest_files:
        fname = fpath.stem
        year_str = re.search(r"\d{4}", fname)
        if not year_str:
            continue
        year = int(year_str.group(0))
        period = "Annual"
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in fname:
                period = q
                break

        try:
            with pdfplumber.open(fpath) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    lines = text.split("\n")
                    for line in lines:
                        for state in MALAYSIAN_STATES:
                            if re.search(rf"\b{re.escape(state)}\b", line, re.IGNORECASE):
                                tokens = line.split()
                                big_nums = []
                                for tok in tokens:
                                    cleaned = tok.replace(",", "").strip()
                                    if cleaned.isdigit() and int(cleaned) > 1000:
                                        big_nums.append(int(cleaned))
                                if len(big_nums) >= 2:
                                    dom = big_nums[0]
                                    foreigner = big_nums[1]
                                    total = big_nums[2] if len(big_nums) > 2 else (dom + foreigner)
                                    records.append({
                                        "source_file": fpath.name,
                                        "year": year,
                                        "period": period,
                                        "state": state,
                                        "domestic_guests": dom,
                                        "foreign_guests": foreigner,
                                        "total_guests": total
                                    })
                                break
        except Exception as e:
            logger.warning("Error parsing %s: %s", fpath.name, e)

    df = pd.DataFrame(records).drop_duplicates(subset=["year", "period", "state"])
    logger.info("Extracted %d total Hotel Guests records.", len(df))
    return df


def parse_all_arrivals_pdfs() -> pd.DataFrame:
    """Batch parses monthly and annual visitor/tourist/excursionist arrival PDFs."""
    arr_files = sorted(
        list(RAW_DIR.glob("*visitor*.pdf")) +
        list(RAW_DIR.glob("*tourist*.pdf")) +
        list(RAW_DIR.glob("*excursionist*.pdf"))
    )
    logger.info("Found %d Arrival PDF files to process.", len(arr_files))
    records = []

    top_markets = ["SINGAPORE", "INDONESIA", "CHINA", "THAILAND", "BRUNEI", "INDIA", "AUSTRALIA", "UNITED KINGDOM", "JAPAN", "PHILIPPINES"]

    for fpath in arr_files:
        fname = fpath.stem.lower()
        arr_type = "tourist"
        if "excursionist" in fname:
            arr_type = "excursionist"
        elif "visitor" in fname:
            arr_type = "visitor"

        year_m = re.search(r"\d{4}", fname)
        if not year_m:
            continue
        year = int(year_m.group(0))

        month = "annual"
        for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]:
            if m in fname:
                month = m.upper()
                break

        try:
            with pdfplumber.open(fpath) as pdf:
                if len(pdf.pages) > 0:
                    text = pdf.pages[0].extract_text()
                    if text:
                        for line in text.split("\n"):
                            for market in top_markets:
                                if re.search(rf"\b{market}\b", line, re.IGNORECASE):
                                    tokens = line.split()
                                    nums = []
                                    for t in tokens:
                                        c = t.replace(",", "").strip()
                                        if c.isdigit() and int(c) > 500:
                                            nums.append(int(c))
                                    if nums:
                                        records.append({
                                            "source_file": fpath.name,
                                            "year": year,
                                            "month": month,
                                            "arrival_type": arr_type,
                                            "origin_market": market.title(),
                                            "arrivals_count": nums[0]
                                        })
                                    break
        except Exception as e:
            logger.warning("Error parsing arrival %s: %s", fpath.name, e)

    df = pd.DataFrame(records).drop_duplicates(subset=["year", "month", "arrival_type", "origin_market"])
    logger.info("Extracted %d total Arrival records.", len(df))
    return df


def run_all_motac_extraction():
    logger.info("=== Starting MOTAC PDF Publication Batch Extraction ===")

    df_aor = parse_all_aor_pdfs()
    if not df_aor.empty:
        df_aor.to_csv(PROCESSED_DIR / "motac_hotel_occupancy_aor_timeseries.csv", index=False)
        logger.info("Saved motac_hotel_occupancy_aor_timeseries.csv (%d records)", len(df_aor))

    df_inv = parse_all_hotel_inventory_pdfs()
    if not df_inv.empty:
        df_inv.to_csv(PROCESSED_DIR / "motac_hotel_inventory_timeseries.csv", index=False)
        logger.info("Saved motac_hotel_inventory_timeseries.csv (%d records)", len(df_inv))

    df_guests = parse_all_hotel_guests_pdfs()
    if not df_guests.empty:
        df_guests.to_csv(PROCESSED_DIR / "motac_hotel_guests_timeseries.csv", index=False)
        logger.info("Saved motac_hotel_guests_timeseries.csv (%d records)", len(df_guests))

    df_arr = parse_all_arrivals_pdfs()
    if not df_arr.empty:
        df_arr.to_csv(PROCESSED_DIR / "motac_monthly_arrivals_summary.csv", index=False)
        logger.info("Saved motac_monthly_arrivals_summary.csv (%d records)", len(df_arr))

    logger.info("=== MOTAC PDF Publication Batch Extraction Complete ===")


if __name__ == "__main__":
    run_all_motac_extraction()
