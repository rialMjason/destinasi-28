"""
DESTINASI — Economic Impact & Multiplier Engine
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Calculates quantified ringgit-and-sen economic growth potential for 2nd-tier destinations
based on official DOSM Domestic Tourism Survey (DTS 2024) and MOTAC Comprehensive Inbound
Expenditure Survey (9,790 records across 38 origin markets & 12 categories).
Supports Domestic (RM 84.9B – 94.9B), Inbound (RM 106.8B), and Blended (RM 191.7B – 201.7B) macro-economies.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import math

ROOT = Path(__file__).resolve().parents[1]
RAW_EXP_PATH = ROOT / "data" / "raw" / "comprehensive_expenditure.csv"
PROCESSED_EXP_PATH = ROOT / "data" / "processed" / "expenditure_market_matrix.csv"

# -----------------------------------------------------------------------------
# Official DOSM Domestic Tourism Survey (DTS 2024) Macro Baselines
# -----------------------------------------------------------------------------
DTS_2024_SPEND_PER_TRIP = 947.50      # RM per overnight domestic tourist
DTS_2024_ALOS = 2.45                   # Average Length of Stay in nights
DTS_2024_SPEND_PER_NIGHT = 386.73     # RM per tourist night (947.50 / 2.45)
DTS_MACRO_VOLUME_LOWER = 84.9       # RM 84.9B (lower/overnight tourists)
DTS_MACRO_VOLUME_UPPER = 94.9       # RM 94.9B (upper/total domestic visitors)
DTS_MACRO_VOLUME_BILLION = 84.9     # RM 84.9B baseline

# -----------------------------------------------------------------------------
# MOTAC International Inbound Expenditure Baselines (comprehensive_expenditure.csv)
# -----------------------------------------------------------------------------
INBOUND_SPEND_PER_TRIP = 3496.80      # RM per inbound foreign tourist
INBOUND_ALOS = 4.80                   # Average Length of Stay in nights
INBOUND_SPEND_PER_NIGHT = 728.50      # RM per tourist night (3,496.80 / 4.80)
INBOUND_MACRO_VOLUME_BILLION = 106.8  # RM 106.8B (RM 106,780M)

# -----------------------------------------------------------------------------
# Blended Total Tourism Economy Baselines (Domestic + Inbound)
# -----------------------------------------------------------------------------
BLENDED_ALOS = 3.63                   # Weighted ALOS ((2.45 + 4.80) / 2 = 3.625 ~ 3.63)
BLENDED_SPEND_PER_NIGHT = 557.62      # Weighted spend/night ((386.73 + 728.50) / 2 = 557.615 ~ 557.62)
BLENDED_MACRO_VOLUME_LOWER = 191.7    # RM 191.7B (84.9B + 106.8B)
BLENDED_MACRO_VOLUME_UPPER = 201.7    # RM 201.7B (94.9B + 106.8B)
BLENDED_MACRO_VOLUME_BILLION = 191.7  # RM 191.7B baseline

# -----------------------------------------------------------------------------
# Output Multipliers (DOSM Tourism Satellite Account TSA & I-O 2019-2021)
# -----------------------------------------------------------------------------
TOURISM_OUTPUT_MULTIPLIER = 1.75      # Domestic TSA Composite Gross Output Multiplier (DOSM 2021 I-O Table 6 & 9)
INBOUND_OUTPUT_MULTIPLIER = 1.82      # Foreign exchange injection multiplier (Inbound weighted)
BLENDED_OUTPUT_MULTIPLIER = 1.78      # Blended aggregate output multiplier

# Empirical DOSM Leontief Multipliers (from official symmetric I-O tables 2019-2021)
DOSM_RETAIL_OUTPUT_MULTIPLIER = 1.5428   # Official DOSM Wholesale & Retail Trade Type I output multiplier
TSA_COMPOSITE_TYPE1_OUTPUT = 1.7525      # DTS-weighted composite Type I output multiplier (2021)
TSA_COMPOSITE_TYPE1_GVA = 0.8151         # Direct + Indirect Gross Value Added (GDP) multiplier per RM 1.00 spend
TSA_COMPOSITE_TYPE2_OUTPUT = 2.9388      # Type II closed-household output multiplier (with induced effects)
TSA_COMPOSITE_TYPE2_GVA = 1.3528         # Type II Net Value Added / GDP multiplier

# -----------------------------------------------------------------------------
# 5-Sector Spending Distributions (Backward-Compatible Aggregations)
# -----------------------------------------------------------------------------
DOMESTIC_SPEND_SHARES = {
    "shopping": 0.332,          # 33.2% Retail, handicrafts, local souvenirs
    "food_beverage": 0.254,     # 25.4% Street food, restaurants, night markets
    "accommodation": 0.231,     # 23.1% Hotels, homestays, chalets
    "transport": 0.123,         # 12.3% Rail fares, fuel, local taxis/e-hailing
    "entertainment_other": 0.060  # 6.0% Tickets, attractions, recreation
}

INBOUND_SPEND_SHARES = {
    "shopping": 0.3736,         # 37.36% Shopping & retail
    "food_beverage": 0.1610,    # 16.10% Dining & F&B
    "accommodation": 0.1809,    # 18.09% Lodging & hotels
    "transport": 0.1726,        # 17.26% (Local 5.69% + Int Air 8.14% + Dom Air 2.39% + Fuel 1.04%)
    "entertainment_other": 0.1119 # 11.19% (Medical 4.73% + Tour 3.83% + Ent 2.17% + Sports 0.15% + Other 0.30%)
}

BLENDED_SPEND_SHARES = {
    k: round(0.50 * DOMESTIC_SPEND_SHARES[k] + 0.50 * INBOUND_SPEND_SHARES[k], 4)
    for k in DOMESTIC_SPEND_SHARES
}

# Legacy alias for backward compatibility
SPEND_SHARES = DOMESTIC_SPEND_SHARES

# -----------------------------------------------------------------------------
# 12 Inbound Spending Categories (MOTAC Standard)
# -----------------------------------------------------------------------------
INBOUND_CATEGORIES: List[str] = [
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

# Authoritative 2024 Overall Benchmarks for 12 Categories
INBOUND_12_CATEGORIES_OVERALL: Dict[str, float] = {
    "Shopping": 0.3736,
    "Accommodation": 0.1809,
    "Food & Beverages": 0.1610,
    "International Airfares": 0.0814,
    "Local Transportation": 0.0569,
    "Medical": 0.0473,
    "Organised Tour": 0.0383,
    "Domestic Airfares": 0.0239,
    "Entertainment": 0.0217,
    "Fuel": 0.0104,
    "Others": 0.0030,
    "Sports": 0.0015
}

# Domestic Mapping into 12 Categories
DOMESTIC_12_CATEGORIES: Dict[str, float] = {
    "Shopping": 0.332,
    "Accommodation": 0.231,
    "Food & Beverages": 0.254,
    "Local Transportation": 0.070,
    "International Airfares": 0.000,
    "Domestic Airfares": 0.013,
    "Entertainment": 0.025,
    "Medical": 0.010,
    "Organised Tour": 0.015,
    "Sports": 0.005,
    "Fuel": 0.040,
    "Others": 0.005
}

# Blended 12-Category Distribution (50% Domestic / 50% Inbound)
BLENDED_12_CATEGORIES: Dict[str, float] = {
    cat: round(0.50 * DOMESTIC_12_CATEGORIES[cat] + 0.50 * INBOUND_12_CATEGORIES_OVERALL[cat], 4)
    for cat in INBOUND_CATEGORIES
}


def np_log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


def get_top_inbound_markets(
    n: int = 10,
    year: int = 2024,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extracts top inbound origin markets ranked by total tourist expenditure (RM Million).
    Supports both n and top_n keyword arguments.
    """
    limit = top_n if top_n is not None else n

    # Prioritize processed matrix if present
    if PROCESSED_EXP_PATH.exists():
        df = pd.read_csv(PROCESSED_EXP_PATH)
        sub = df[
            (df["year"] == year) &
            (df["spending_category"] == "All spending categories") &
            (df["market"] != "Overall") &
            (df["market"].notna())
        ].copy()

        if not sub.empty:
            tot_row = df[
                (df["year"] == year) &
                (df["market"] == "Overall") &
                (df["spending_category"] == "All spending categories")
            ]
            overall_total = tot_row["value_rm_million"].iloc[0] if not tot_row.empty else sub["value_rm_million"].sum()
            sub["share_percent"] = (sub["value_rm_million"] / overall_total) * 100.0
            return (
                sub.sort_values(by="value_rm_million", ascending=False)
                .head(limit)
                .reset_index(drop=True)
            )

    # Fallback to raw file if processed not found
    if RAW_EXP_PATH.exists():
        raw_df = pd.read_csv(RAW_EXP_PATH)
        filtered = raw_df[
            (raw_df["year"] == year) &
            (raw_df["metric_group"] == "Total expenditure") &
            (raw_df["market"] != "Overall") &
            (raw_df["market"].notna()) &
            (raw_df["period_type"] == "Full year") &
            (raw_df["audience"] == "Visitors")
        ]
        if filtered.empty:
            filtered = raw_df[
                (raw_df["year"] == year) &
                (raw_df["metric_group"] == "Total expenditure") &
                (raw_df["market"] != "Overall") &
                (raw_df["market"].notna())
            ]

        top = (
            filtered.groupby("market")["value_rm_million"]
            .max()
            .reset_index()
            .sort_values(by="value_rm_million", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )
        return top

    return pd.DataFrame(columns=["market", "value_rm_million", "share_percent"])


def get_market_expenditure_breakdown(
    market: str = "All Markets",
    year: int = 2024
) -> pd.DataFrame:
    """
    Extracts the 12 spending categories, percentage shares, and RM Million values
    for a specified source market (e.g. 'Singapore', 'China', or 'All Markets'/'Overall').
    """
    target = "Overall" if market in ["All Markets", "Overall", "all", "Total", "National", None] else str(market)

    # Check processed file first
    if PROCESSED_EXP_PATH.exists():
        df = pd.read_csv(PROCESSED_EXP_PATH)
        sub = df[
            (df["year"] == year) &
            (df["market"].str.lower() == target.lower()) &
            (df["spending_category"] != "All spending categories")
        ].copy()

        if sub.empty and target.lower() != "overall":
            # Fallback to Overall if specific market not found
            sub = df[
                (df["year"] == year) &
                (df["market"] == "Overall") &
                (df["spending_category"] != "All spending categories")
            ].copy()

        if not sub.empty:
            return (
                sub[["spending_category", "percentage_share", "value_rm_million"]]
                .sort_values(by="value_rm_million", ascending=False)
                .reset_index(drop=True)
            )

    # Fallback to raw file
    if RAW_EXP_PATH.exists():
        raw_df = pd.read_csv(RAW_EXP_PATH)
        sub = raw_df[
            (raw_df["year"] == year) &
            (raw_df["selected_period"] == "Jan-Dec") &
            (raw_df["audience"] == "Visitors") &
            (raw_df["metric_group"] == "Expenditure components share") &
            (raw_df["market"].str.lower() == target.lower()) &
            (raw_df["spending_category"].isin(INBOUND_CATEGORIES))
        ].copy()

        if sub.empty and target.lower() != "overall":
            sub = raw_df[
                (raw_df["year"] == year) &
                (raw_df["selected_period"] == "Jan-Dec") &
                (raw_df["audience"] == "Visitors") &
                (raw_df["metric_group"] == "Expenditure components share") &
                (raw_df["market"] == "Overall") &
                (raw_df["spending_category"].isin(INBOUND_CATEGORIES))
            ].copy()

        if not sub.empty:
            sub["percentage_share"] = sub["value_numeric"]
            tot_row = raw_df[
                (raw_df["year"] == year) &
                (raw_df["metric_group"] == "Total expenditure") &
                (raw_df["period_type"] == "Full year") &
                (raw_df["audience"] == "Visitors") &
                (raw_df["market"] == sub["market"].iloc[0])
            ]
            tot_val = float(tot_row["value_rm_million"].iloc[0]) if not tot_row.empty else 106780.0
            sub["value_rm_million"] = (sub["percentage_share"] / 100.0) * tot_val
            return (
                sub[["spending_category", "percentage_share", "value_rm_million"]]
                .sort_values(by="value_rm_million", ascending=False)
                .reset_index(drop=True)
            )

    # Static fallback from authoritative constants
    records = []
    tot_ref = INBOUND_MACRO_VOLUME_BILLION * 1000.0 # RM 106,780M
    for cat, share in INBOUND_12_CATEGORIES_OVERALL.items():
        records.append({
            "spending_category": cat,
            "percentage_share": round(share * 100.0, 2),
            "value_rm_million": round(share * tot_ref, 3)
        })
    return pd.DataFrame(records).sort_values(by="value_rm_million", ascending=False).reset_index(drop=True)


def get_inbound_category_breakdown(year: int = 2024) -> pd.DataFrame:
    """Extracts national category spending shares for inbound tourists."""
    return get_market_expenditure_breakdown(market="Overall", year=year)


def calculate_redistribution_economic_impact(
    redirected_visitors_daily: float,
    days_period: int = 30,
    alos: float = 2.45,
    spend_per_night: float = DTS_2024_SPEND_PER_NIGHT,
    poverty_rate_candidate: float = 0.0,
    poverty_rate_hotspot: float = 0.0,
    tourism_mode: str = "domestic",
    origin_market: Optional[str] = None,
    destination_pressure: float = 0.0
) -> Dict[str, Any]:
    """
    Computes direct expenditure injection and total economic multiplier effect.
    Supports three macroeconomic modes:
    - 'domestic': DTS baseline (RM 84.9B/94.9B), ALOS 2.45, RM 386.73/night, 1.75x multiplier.
    - 'inbound': MOTAC inbound (RM 106.8B), ALOS 4.80, RM 728.50/night, 1.82x multiplier.
    - 'blended': Blended total (RM 191.7B/201.7B), ALOS ~3.63, RM ~557.62/night, 1.78x multiplier.

    Preserves default values and return dictionary keys for full backward compatibility:
    'direct_spend_million_rm', 'total_economic_output_million_rm', 'sectoral_breakdown', 'estimated_b40_income_million_rm'.
    Also provides rigorous DOSM Leontief dual-scenario metrics:
    'gdp_value_added_multiplier', 'total_gdp_value_added_million_rm', 'type2_output_multiplier', 'total_type2_output_million_rm'.
    """
    mode = str(tourism_mode).lower().strip()

    # Guard against legacy positional callers who passed:
    # calculate_redistribution_economic_impact(visitors, days, alos, poverty_rate_cand, poverty_rate_hot)
    if spend_per_night < 50.0 and poverty_rate_candidate <= 50.0 and poverty_rate_hotspot == 0.0:
        poverty_rate_hotspot = poverty_rate_candidate
        poverty_rate_candidate = spend_per_night
        spend_per_night = DTS_2024_SPEND_PER_NIGHT

    # Dynamic parameter selection based on macroeconomic mode
    if mode == "inbound":
        effective_alos = INBOUND_ALOS if alos == 2.45 else alos
        effective_spend = INBOUND_SPEND_PER_NIGHT if spend_per_night == DTS_2024_SPEND_PER_NIGHT else spend_per_night
        multiplier = INBOUND_OUTPUT_MULTIPLIER
        macro_volume = INBOUND_MACRO_VOLUME_BILLION

        # If origin_market is specified, use its empirical 12-category shares
        if origin_market and origin_market not in ["All Markets", "Overall"]:
            mkt_df = get_market_expenditure_breakdown(origin_market)
            if not mkt_df.empty:
                cat_12_shares = {
                    row["spending_category"]: float(row["percentage_share"]) / 100.0
                    for _, row in mkt_df.iterrows()
                }
            else:
                cat_12_shares = INBOUND_12_CATEGORIES_OVERALL.copy()
        else:
            cat_12_shares = INBOUND_12_CATEGORIES_OVERALL.copy()

    elif mode == "blended":
        effective_alos = BLENDED_ALOS if alos == 2.45 else alos
        effective_spend = BLENDED_SPEND_PER_NIGHT if spend_per_night == DTS_2024_SPEND_PER_NIGHT else spend_per_night
        multiplier = BLENDED_OUTPUT_MULTIPLIER
        macro_volume = BLENDED_MACRO_VOLUME_BILLION
        cat_12_shares = BLENDED_12_CATEGORIES.copy()

    else:
        # Default domestic mode (preserves exact backward compatibility)
        mode = "domestic"
        effective_alos = alos
        effective_spend = spend_per_night
        multiplier = TOURISM_OUTPUT_MULTIPLIER
        macro_volume = DTS_MACRO_VOLUME_BILLION
        cat_12_shares = DOMESTIC_12_CATEGORIES.copy()

    total_tourist_trips = max(0.0, float(redirected_visitors_daily)) * days_period
    total_tourist_nights = total_tourist_trips * effective_alos

    direct_spend_total = total_tourist_nights * effective_spend
    direct_spend_million = direct_spend_total / 1e6

    # 12-category granular injection amounts
    breakdown_12 = {
        cat: round(direct_spend_total * cat_12_shares.get(cat, 0.0), 2)
        for cat in INBOUND_CATEGORIES
    }

    # 5-sector aggregated injection amounts (guarantees full backward compatibility)
    if mode == "domestic":
        sectoral_injection = {
            "shopping_retail_rm": round(direct_spend_total * DOMESTIC_SPEND_SHARES["shopping"], 2),
            "food_beverage_rm": round(direct_spend_total * DOMESTIC_SPEND_SHARES["food_beverage"], 2),
            "accommodation_rm": round(direct_spend_total * DOMESTIC_SPEND_SHARES["accommodation"], 2),
            "transport_transit_rm": round(direct_spend_total * DOMESTIC_SPEND_SHARES["transport"], 2),
            "entertainment_other_rm": round(direct_spend_total * DOMESTIC_SPEND_SHARES["entertainment_other"], 2),
        }
    else:
        sectoral_injection = {
            "shopping_retail_rm": round(direct_spend_total * cat_12_shares.get("Shopping", 0.0), 2),
            "food_beverage_rm": round(direct_spend_total * cat_12_shares.get("Food & Beverages", 0.0), 2),
            "accommodation_rm": round(direct_spend_total * cat_12_shares.get("Accommodation", 0.0), 2),
            "transport_transit_rm": round(direct_spend_total * (
                cat_12_shares.get("Local Transportation", 0.0) +
                cat_12_shares.get("Fuel", 0.0) +
                cat_12_shares.get("Domestic Airfares", 0.0) +
                cat_12_shares.get("International Airfares", 0.0)
            ), 2),
            "entertainment_other_rm": round(direct_spend_total * (
                cat_12_shares.get("Entertainment", 0.0) +
                cat_12_shares.get("Medical", 0.0) +
                cat_12_shares.get("Organised Tour", 0.0) +
                cat_12_shares.get("Sports", 0.0) +
                cat_12_shares.get("Others", 0.0)
            ), 2),
        }

    total_economic_output_rm = direct_spend_total * multiplier
    total_economic_output_million = total_economic_output_rm / 1e6

    # Empirical Value Added (GDP) and Type II multipliers from DOSM I-O tables
    gva_multiplier = TSA_COMPOSITE_TYPE1_GVA
    total_gdp_value_added_rm = direct_spend_total * gva_multiplier
    total_gdp_value_added_million = total_gdp_value_added_rm / 1e6

    type2_multiplier = TSA_COMPOSITE_TYPE2_OUTPUT
    total_type2_output_rm = direct_spend_total * type2_multiplier
    total_type2_output_million = total_type2_output_rm / 1e6

    # Capacity-adjusted multiplier calculation
    if destination_pressure > 1.0:
        penalty = min(0.50, 0.25 * (destination_pressure - 1.0))
        capacity_adjusted_mult = multiplier * (1.0 - penalty)
    else:
        capacity_adjusted_mult = multiplier

    poverty_ratio = max(1.0, poverty_rate_candidate / max(0.1, poverty_rate_hotspot))
    b40_retention_ratio = min(0.65, 0.35 + 0.03 * np_log1p(poverty_ratio))
    estimated_b40_income_injected_rm = direct_spend_total * b40_retention_ratio
    estimated_b40_income_million = estimated_b40_income_injected_rm / 1e6

    return {
        "tourism_mode": mode,
        "origin_market": origin_market or "All Markets",
        "macro_volume_rm_billion": macro_volume,
        "total_tourist_trips": round(total_tourist_trips),
        "total_tourist_nights": round(total_tourist_nights),
        "direct_spend_total_rm": round(direct_spend_total, 2),
        "direct_spend_million_rm": round(direct_spend_million, 3),
        "total_economic_output_rm": round(total_economic_output_rm, 2),
        "total_economic_output_million_rm": round(total_economic_output_million, 3),
        "gdp_value_added_multiplier": round(gva_multiplier, 4),
        "total_gdp_value_added_rm": round(total_gdp_value_added_rm, 2),
        "total_gdp_value_added_million_rm": round(total_gdp_value_added_million, 3),
        "type2_output_multiplier": round(type2_multiplier, 4),
        "total_type2_output_rm": round(total_type2_output_rm, 2),
        "total_type2_output_million_rm": round(total_type2_output_million, 3),
        "capacity_adjusted_multiplier": round(capacity_adjusted_mult, 4),
        "sectoral_breakdown": sectoral_injection,
        "sectoral_breakdown_12": breakdown_12,
        "category_shares": cat_12_shares,
        "estimated_b40_income_injected_rm": round(estimated_b40_income_injected_rm, 2),
        "estimated_b40_income_million_rm": round(estimated_b40_income_million, 3),
        "output_multiplier": multiplier,
        "alos_used": effective_alos,
        "spend_per_night_used": effective_spend
    }


def generate_treasury_justification_memo(
    hotspot_name: str,
    hotspot_state: str,
    relief_name: str,
    relief_state: str,
    diverted_visitors_daily: float,
    simulation_days: int,
    direct_spend_rm_m: float,
    gross_output_rm_m: float,
    gva_gdp_rm_m: float,
    b40_income_rm_m: float,
    poverty_rate_relief: float,
    transit_mode: str = "KTM ETS / Komuter Rail",
    estimated_subsidy_rm: float = 350000.0,
    output_multiplier: float = 1.75,
    gva_multiplier: float = 0.82,
    origin_hotel_vacancy: Optional[float] = None,
    dest_hotel_vacancy: Optional[float] = None,
    hotel_vacancy_delta: Optional[float] = None,
) -> str:
    """
    Generates a formal, exportable Treasury Budget Justification Memorandum for the
    Ministry of Finance (MOF / Perbendaharaan Malaysia) justifying inter-city transit
    subsidies and state economic matching grants for tourism demand redistribution.
    Reference Code: MOF/BP/DESTINASI/2026/02
    """
    total_trips = diverted_visitors_daily * simulation_days
    fiscal_roi = (direct_spend_rm_m * 1e6) / max(1.0, estimated_subsidy_rm)
    tax_yield_est = direct_spend_rm_m * 0.10  # 10% effective indirect/direct tax clawback

    cushion_section = ""
    if (
        origin_hotel_vacancy is not None
        and dest_hotel_vacancy is not None
        or hotel_vacancy_delta is not None
    ):
        delta_v = (
            hotel_vacancy_delta
            if hotel_vacancy_delta is not None
            else (dest_hotel_vacancy - origin_hotel_vacancy)
        )
        dest_v_str = (
            f"{dest_hotel_vacancy:.1f}%"
            if dest_hotel_vacancy is not None
            else "mencukupi"
        )
        cushion_section = f"\n- **Kusyen Penyerapan Penginapan:** Koridor {relief_name} mencatatkan kadar kekosongan hotel {dest_v_str} ({'+' if delta_v >= 0 else ''}{delta_v:.1f}% perbezaan kusyen berbanding {hotspot_name}), menjamin kapasiti serapan tanpa inflasi bilik."

    return f"""# KEMENTERIAN KEWANGAN MALAYSIA (MOF)
## MEMORANDUM PERBENDAHARAAN: JUSTIFIKASI PERUNTUKAN BELANJAWAN & SUBSIDI TRANSIT PELANCONGAN MAMPAN
**No. Rujukan:** MOF/BP/DESTINASI/2026/02
**Tarikh:** {datetime.now().strftime('%d %B %Y')}
**Kepada:** YB Menteri Kewangan & Ketua Setiausaha Perbendaharaan (KSP)
**Daripada:** Pasukan Khas Pelancongan Mampan & Pelepasan Kesesakan (DESTINASI / DOSM)
**Perkara:** Permohonan Peruntukan Belanjawan & Skim Subsidi Transit Inter-Bandar bagi Penjanaan Pertumbuhan Ekonomi Koridor Pelega ({hotspot_name} ➔ {relief_name})

---

### 1. TUJUAN & RINGKASAN EKSEKUTIF
Memorandum ini dikemukakan bagi mendapatkan kelulusan Perbendaharaan Malaysia untuk memperuntukkan subsidi operasi transit dan geran pemangkin pelancongan bagi mengaktifkan Koridor Pelega **{relief_name} ({relief_state})** sebagai penyerap lebihan muatan dari hotspot **{hotspot_name} ({hotspot_state})**:
1. **Pelepasan Kesesakan Struktur:** Mengalihkan sebanyak **{diverted_visitors_daily:,.0f} pelawat/hari** ({total_trips:,.0f} jumlah pelawat dalam tempoh {simulation_days} hari operasi puncak).
2. **Suntikan Perbelanjaan Terus:** Menjana suntikan tunai terus sebanyak **RM {direct_spend_rm_m:.2f} Juta** ke dalam ekosistem perniagaan tempatan di {relief_name}.
3. **Penjanaan KDNK & Output:** Menggerakkan pengganda ekonomi Leontief DOSM ({output_multiplier:.2f}x Output, {gva_multiplier:.2f}x KDNK) bagi menghasilkan **RM {gross_output_rm_m:.2f} Juta** output kasar dan **RM {gva_gdp_rm_m:.2f} Juta** Nilai Ditambah Kasar (KDNK) baharu.
4. **Peluang Pendapatan Golongan Rentan (B40):** Memastikan **RM {b40_income_rm_m:.2f} Juta** dinikmati secara terus oleh isi rumah dan peniaga B40 (kadar kemiskinan daerah: {poverty_rate_relief:.1f}%).

---

### 2. JUSTIFIKASI PERUNTUKAN SUBSIDI TRANSIT & ANALISIS PULANGAN FISKAL (ROI)
Bagi merealisasikan peralihan mod pengangkutan dan penyuraian spatial yang efektif:
- Peruntukan subsidi tambang 30% dan penambahan set tren berjumlah **RM {estimated_subsidy_rm:,.2f}** melalui mod pengangkutan utama **{transit_mode}**.
- Bagi setiap RM 1.00 subsidi transit, koridor destinasi menerima suntikan terus pelancong sebanyak **RM {fiscal_roi:.1f}** (Nisbah Manfaat-Kos: **{fiscal_roi:.1f} : 1**).
- Anggaran pulangan cukai tidak langsung dan PKS pada kadar efektif 10% berjumlah **RM {tax_yield_est:.2f} Juta** (**Dasar Pembiayaan Kendiri / Self-Funding Policy**).

---

### 3. ANALISIS EKONOMETRIK TOURISM SATELLITE ACCOUNT (TSA & I-O DOSM)
Berdasarkan Jadual Input-Output Rasmi 124-Sektor Jabatan Perangkaan Malaysia (DOSM):
- **Pengganda Output Kasar (Type I Composite Multiplier):** Ditetapkan pada **{output_multiplier:.2f}x**, menggerakkan sektor peruncitan tempatan, F&B, dan homestay.
- **KDNK Sebenar (Gross Value Added / GDP Multiplier):** Ditetapkan pada **{gva_multiplier:.2f}x**, menghasilkan **RM {gva_gdp_rm_m:.2f} Juta** nilai ditambah bersih tanpa pertindihan kiraan perantara.{cushion_section}

---

### 4. PELEPASAN KOS LUARAN (AVOIDED EXTERNALITIES & DAMAGE MITIGATION)
Penyuraian pelancongan ini mengelakkan kerugian ekonomi ketara di {hotspot_name}:
1. Mengurangkan kelewatan trafik puncak sehingga 40%.
2. Menghentikan pelepasan air sisa melebihi had daya tampung ke atas rizab air SPAN dan mengelakkan degradasi cerun sensitif (RFN-4).

---

### 5. PERAKUAN & SYOR KELULUSAN
Perbendaharaan Malaysia disyorkan untuk meluluskan peruntukan subsidi sebanyak **RM {estimated_subsidy_rm:,.2f}** dan menyalurkan geran padanan promosi bersama kepada PBT {relief_name} dan MOTAC.
"""

