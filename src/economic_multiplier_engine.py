"""
DESTINASI — Econometric Input-Output Multiplier Engine (DOSM 2019, 2020, 2021)
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Empirical derivation of Malaysia's tourism multipliers from official Department of
Statistics Malaysia (DOSM) Symmetric Input-Output Tables:
- Table 6 (Absorption Matrix CXC: 124 commodities)
- Table 9 (Leontief Inverse Matrix CXC)
- Table 10/12 (Activity x Activity AXA tables)

Provides:
1. Type I Open Output Multiplier (Direct + Indirect Gross Output).
2. Type I Value-Added (GDP / GVA) Multiplier (Direct + Indirect GDP generated per RM 1.00 final demand).
3. Type II Closed-Household Multiplier (Direct + Indirect + Induced Household Consumption).
4. Type II Value-Added Multiplier.
5. DTS / TSA Expenditure-Weighted Composite Tourism Multipliers.
6. Destination Capacity-Constrained Multiplier Adjustment.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import openpyxl

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_PATH = PROCESSED_DIR / "dosm_io_multipliers.json"

IO_FILES = {
    2019: RAW_DIR / "Table 1 - 17 2019.xlsx",
    2020: RAW_DIR / "Table 1 - 17 2020.xlsx",
    2021: RAW_DIR / "Jadual Input-Output 2021 (Jadual 1 - 17 ).xlsx",
}

# Tourism-relevant commodities (1-indexed code and 0-indexed column index in 124-commodity matrix)
KEY_TOURISM_COMMODITIES = {
    "retail_wholesale": {"code": 93, "idx": 92, "name": "Wholesale and Retail Trade"},
    "accommodation": {"code": 94, "idx": 93, "name": "Accommodation"},
    "food_beverage": {"code": 95, "idx": 94, "name": "Food and Beverage Services"},
    "land_transport": {"code": 96, "idx": 95, "name": "Land Transport"},
    "water_transport": {"code": 97, "idx": 96, "name": "Water Transport"},
    "air_transport": {"code": 98, "idx": 97, "name": "Air Transport"},
    "transport_services": {"code": 100, "idx": 99, "name": "Supporting Transport Services"},
    "arts_recreation": {"code": 123, "idx": 122, "name": "Arts, Entertainment and Recreation"},
    "refined_petroleum": {"code": 38, "idx": 37, "name": "Refined Petroleum Products"},
}

# Empirical Domestic Tourism Survey (DTS 2024/2025) national expenditure weights
DTS_EXPENDITURE_WEIGHTS = {
    "retail_wholesale": 0.3739 + 0.0389 + (0.0778 * 0.5), # Shopping (37.39%) + Pre-trip (3.89%) + 50% visited HH (3.89%) = 45.17%
    "food_beverage": 0.1625 + (0.0778 * 0.5),             # F&B (16.25%) + 50% visited HH (3.89%) = 20.14%
    "accommodation": 0.1116,                              # Accommodation = 11.16%
    "refined_petroleum": 0.1265,                          # Vehicle fuel = 12.65%
    "land_transport": 0.0673 * 0.80,                      # Land transport = 5.38%
    "air_transport": 0.0673 * 0.20,                       # Air transport = 1.35%
    "arts_recreation": 0.0416                             # Recreation / Other activities = 4.16%
}

_PRECOMPUTED_DATA: Optional[Dict[str, Any]] = None


def extract_io_matrices_for_year(year: int) -> Dict[str, Any]:
    """
    Extracts absorption matrix Z, total output X, GVA V, labor wages W, and
    household consumption C from official DOSM Input-Output Excel workbook.
    Computes Leontief inverses (Type I and Type II) and GVA multipliers.
    """
    file_path = IO_FILES.get(year)
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"DOSM I-O file for year {year} not found at {file_path}")

    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws6 = wb["T6_ABSORP CXC"]

    n = 124
    commodity_names = [str(ws6.cell(r, 2).value).strip().replace("\n", " ") for r in range(3, 3 + n)]

    # Intermediate transaction matrix Z (124 x 124)
    Z = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            Z[i, j] = float(ws6.cell(3 + i, 3 + j).value or 0.0)

    # Row 140: Total Output X
    X = np.array([float(ws6.cell(140, 3 + j).value or 0.0) for j in range(n)], dtype=np.float64)
    # Row 134: Gross Value Added (GVA)
    GVA = np.array([float(ws6.cell(134, 3 + j).value or 0.0) for j in range(n)], dtype=np.float64)
    # Row 135: Compensation of Employees (Wages W)
    W = np.array([float(ws6.cell(135, 3 + j).value or 0.0) for j in range(n)], dtype=np.float64)
    # Col 128: Private Consumption C (Household final demand)
    C = np.array([float(ws6.cell(3 + i, 128).value or 0.0) for i in range(n)], dtype=np.float64)

    X_safe = np.where(X > 0, X, 1.0)

    # Technical coefficients matrix A
    A = Z / X_safe[np.newaxis, :]
    I_mat = np.eye(n, dtype=np.float64)
    L = np.linalg.inv(I_mat - A)

    # Type I Gross Output Multiplier = column sum of L
    type1_output_mult = L.sum(axis=0)

    # Type I Gross Value Added (GDP) Multiplier = v * L
    v = GVA / X_safe
    type1_va_mult = v @ L

    # Type II Augmented Matrix A* (closed households)
    w_coeff = W / X_safe
    tot_w = W.sum()
    c_coeff = C / tot_w if tot_w > 0 else np.zeros(n, dtype=np.float64)

    A_star = np.zeros((n + 1, n + 1), dtype=np.float64)
    A_star[:n, :n] = A
    A_star[:n, n] = c_coeff
    A_star[n, :n] = w_coeff
    A_star[n, n] = 0.0

    I_star = np.eye(n + 1, dtype=np.float64)
    L_star = np.linalg.inv(I_star - A_star)

    type2_output_mult = L_star[:n, :n].sum(axis=0)
    type2_va_mult = v @ L_star[:n, :n]

    # Calculate DTS composite multipliers
    tot_w_dts = sum(DTS_EXPENDITURE_WEIGHTS.values())
    w_m1_out = sum(w * type1_output_mult[KEY_TOURISM_COMMODITIES[k]["idx"]] for k, w in DTS_EXPENDITURE_WEIGHTS.items()) / tot_w_dts
    w_m1_va = sum(w * type1_va_mult[KEY_TOURISM_COMMODITIES[k]["idx"]] for k, w in DTS_EXPENDITURE_WEIGHTS.items()) / tot_w_dts
    w_m2_out = sum(w * type2_output_mult[KEY_TOURISM_COMMODITIES[k]["idx"]] for k, w in DTS_EXPENDITURE_WEIGHTS.items()) / tot_w_dts
    w_m2_va = sum(w * type2_va_mult[KEY_TOURISM_COMMODITIES[k]["idx"]] for k, w in DTS_EXPENDITURE_WEIGHTS.items()) / tot_w_dts

    sector_results = {}
    for key, info in KEY_TOURISM_COMMODITIES.items():
        idx = info["idx"]
        sector_results[key] = {
            "code": info["code"],
            "name": info["name"],
            "dosm_name": commodity_names[idx],
            "type1_output": float(round(type1_output_mult[idx], 4)),
            "type1_gva": float(round(type1_va_mult[idx], 4)),
            "type2_output": float(round(type2_output_mult[idx], 4)),
            "type2_gva": float(round(type2_va_mult[idx], 4)),
        }

    return {
        "year": year,
        "composite_tourism": {
            "type1_output_multiplier": float(round(w_m1_out, 4)),
            "type1_gva_multiplier": float(round(w_m1_va, 4)),
            "type2_output_multiplier": float(round(w_m2_out, 4)),
            "type2_gva_multiplier": float(round(w_m2_va, 4)),
        },
        "sectors": sector_results,
    }


def generate_and_cache_all_multipliers() -> Dict[str, Any]:
    """Generates multipliers across all 3 years and writes to dosm_io_multipliers.json."""
    data = {"metadata": {"description": "Empirical Leontief Multipliers from DOSM Symmetric I-O Tables (2019-2021)"}}
    for yr in [2019, 2020, 2021]:
        try:
            logger.info(f"Computing Leontief I-O multipliers for {yr}...")
            data[str(yr)] = extract_io_matrices_for_year(yr)
        except Exception as e:
            logger.warning(f"Could not compute I-O for {yr}: {e}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved precomputed I-O multipliers to {CACHE_PATH}")
    return data


def load_cached_multipliers() -> Dict[str, Any]:
    """Loads precomputed multipliers from JSON cache (or generates if missing)."""
    global _PRECOMPUTED_DATA
    if _PRECOMPUTED_DATA is not None:
        return _PRECOMPUTED_DATA

    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _PRECOMPUTED_DATA = json.load(f)
                return _PRECOMPUTED_DATA
        except Exception as e:
            logger.warning(f"Failed to read {CACHE_PATH}: {e}")

    _PRECOMPUTED_DATA = generate_and_cache_all_multipliers()
    return _PRECOMPUTED_DATA


def get_dosm_multipliers(year: int = 2021) -> Dict[str, Any]:
    """Returns the empirical multiplier report for a specified year (default 2021)."""
    data = load_cached_multipliers()
    yr_str = str(year)
    if yr_str in data:
        return data[yr_str]
    for fallback_yr in ["2021", "2020", "2019"]:
        if fallback_yr in data:
            return data[fallback_yr]
    return extract_io_matrices_for_year(2021)


def get_sector_multipliers(year: int = 2021) -> Dict[str, Dict[str, float]]:
    """Returns sector-specific multipliers dictionary for the specified year."""
    return get_dosm_multipliers(year).get("sectors", {})


def compute_tsa_composite_multiplier(year: int = 2021) -> Dict[str, float]:
    """Returns the composite tourism multipliers for the specified year."""
    return get_dosm_multipliers(year).get("composite_tourism", {
        "type1_output_multiplier": 1.7525,
        "type1_gva_multiplier": 0.8151,
        "type2_output_multiplier": 2.9388,
        "type2_gva_multiplier": 1.3528
    })


def apply_capacity_constrained_multiplier(
    base_multiplier: float,
    continuous_pressure: float,
    elasticity: float = 0.25
) -> float:
    """
    Applies destination capacity bottleneck dampening:
    When destination capacity pressure ratio >= 1.0 (overtourism/bottlenecks),
    congestion diseconomies and price inflation dampen the realized local economic multiplier:
    M_eff = M_base * (1 - elasticity * max(0, pressure - 1.0))
    Capped at a minimum of 0.50 * M_base.
    """
    excess_pressure = max(0.0, float(continuous_pressure) - 1.0)
    penalty = min(0.50, elasticity * excess_pressure)
    dampened = base_multiplier * (1.0 - penalty)
    return float(round(dampened, 4))


if __name__ == "__main__":
    generate_and_cache_all_multipliers()
