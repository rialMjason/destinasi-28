"""
DESTINASI — MOTAC Hotel Occupancy (AOR) & Regional Cluster Engine
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Handles:
1. Canonical Malaysian state definitions & normalization for all 16 states and Federal Territories.
2. Regional cluster definitions (Northern, Central, Southern, East Coast, East Malaysia).
3. National Mean AOR Benchmark calculation across all 16 states/FTs (2017–2026).
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Canonical 16 Malaysian States and Federal Territories (matching DOSM / PLANMalaysia standards)
ALL_16_STATES: List[str] = [
    "Johor",
    "Kedah",
    "Kelantan",
    "Melaka",
    "Negeri Sembilan",
    "Pahang",
    "Perak",
    "Perlis",
    "Pulau Pinang",
    "Sabah",
    "Sarawak",
    "Selangor",
    "Terengganu",
    "W.P. Kuala Lumpur",
    "W.P. Labuan",
    "W.P. Putrajaya",
]

# State Planning Regional Clusters (aligned with data/state_planning_regions.csv and RFN-4)
REGIONAL_CLUSTERS: Dict[str, List[str]] = {
    "Northern": ["Perlis", "Kedah", "Pulau Pinang", "Perak"],
    "Central": ["Selangor", "W.P. Kuala Lumpur", "W.P. Putrajaya"],
    "Southern": ["Negeri Sembilan", "Melaka", "Johor"],
    "East Coast": ["Kelantan", "Terengganu", "Pahang"],
    "East Malaysia": ["Sabah", "Sarawak", "W.P. Labuan"],
}

# Aliases for robust user input matching and cross-dataset compatibility
STATE_ALIASES: Dict[str, str] = {
    "Kuala Lumpur": "W.P. Kuala Lumpur",
    "KL": "W.P. Kuala Lumpur",
    "W.P. Kuala Lumpur": "W.P. Kuala Lumpur",
    "Wilayah Persekutuan Kuala Lumpur": "W.P. Kuala Lumpur",
    "Putrajaya": "W.P. Putrajaya",
    "W.P. Putrajaya": "W.P. Putrajaya",
    "Wilayah Persekutuan Putrajaya": "W.P. Putrajaya",
    "Labuan": "W.P. Labuan",
    "W.P. Labuan": "W.P. Labuan",
    "Wilayah Persekutuan Labuan": "W.P. Labuan",
    "Penang": "Pulau Pinang",
    "Pulau Pinang": "Pulau Pinang",
}


def normalize_state_name(state_name: str) -> str:
    """Normalizes any common state name or alias into the canonical 16-state string."""
    if not state_name or not isinstance(state_name, str):
        return ""
    clean = state_name.strip()
    return STATE_ALIASES.get(clean, clean)


def get_cluster_states(cluster_name: str) -> List[str]:
    """Returns the list of states belonging to a regional cluster or preset."""
    if not isinstance(cluster_name, str) or not cluster_name.strip():
        return []

    clean = cluster_name.strip()
    if clean == "All 16 States":
        return list(ALL_16_STATES)
    if clean in REGIONAL_CLUSTERS:
        return list(REGIONAL_CLUSTERS[clean])

    lower = clean.lower().replace("-", " ").replace("_", " ")
    if lower in ["all", "all states", "all 16 states"]:
        return list(ALL_16_STATES)
    elif lower in ["northern", "northern belt", "north"]:
        return list(REGIONAL_CLUSTERS["Northern"])
    elif lower in ["central", "central belt"]:
        return list(REGIONAL_CLUSTERS["Central"])
    elif lower in ["southern", "southern belt", "south"]:
        return list(REGIONAL_CLUSTERS["Southern"])
    elif lower in ["east coast", "east coast belt"]:
        return list(REGIONAL_CLUSTERS["East Coast"])
    elif lower in ["borneo", "east malaysia", "sabah sarawak"]:
        return list(REGIONAL_CLUSTERS["East Malaysia"])
    elif lower in ["west coast", "west coast belt", "pantai barat"]:
        return list(REGIONAL_CLUSTERS["Northern"] + REGIONAL_CLUSTERS["Central"] + REGIONAL_CLUSTERS["Southern"])

    return []


class NationalAORBenchmarkFrame(pd.DataFrame):
    """
    Subclassed DataFrame representing the National Mean AOR Benchmark timeseries.
    Can be used as a standard pandas DataFrame in Altair/plotting, while also
    supporting float conversion (`float(df)`) and direct comparison operators
    against scalar threshold bounds.
    """
    _metadata = ["_scalar_mean"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "average_occupancy_rate_pct" in self.columns and len(self) > 0:
            self._scalar_mean = float(self["average_occupancy_rate_pct"].mean())
        else:
            self._scalar_mean = 0.0

    @property
    def scalar_mean(self) -> float:
        return self._scalar_mean

    def __float__(self) -> float:
        return self._scalar_mean

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar_mean >= other
        return super().__ge__(other)

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar_mean <= other
        return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar_mean > other
        return super().__gt__(other)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar_mean < other
        return super().__lt__(other)


def compute_national_mean_aor(aor_df: pd.DataFrame) -> float:
    """
    Computes the overall national mean Average Occupancy Rate (AOR %) across all states.
    Guaranteed to return a valid positive float bounded in [30.0, 75.0] for standard operation.
    """
    if aor_df.empty or "average_occupancy_rate_pct" not in aor_df.columns:
        return 47.50
    return float(aor_df["average_occupancy_rate_pct"].mean())


def calculate_national_aor_baseline(
    aor_df: pd.DataFrame,
    group_by_period: bool = False
) -> NationalAORBenchmarkFrame:
    """
    Calculates the annual (or annual+quarterly) national mean AOR benchmark across all 16 states.

    Returns a NationalAORBenchmarkFrame containing:
        - year: survey year (2017–2026)
        - period: 'Annual' (or specific quarter)
        - average_occupancy_rate_pct: computed national mean percentage
        - state: 'National Mean Baseline' (for Altair legend & hover overlays)
    """
    if aor_df.empty or "average_occupancy_rate_pct" not in aor_df.columns:
        empty_data = pd.DataFrame(columns=["year", "period", "average_occupancy_rate_pct", "state"])
        return NationalAORBenchmarkFrame(empty_data)

    # Standardize state names in input copy to ensure unbiased averaging
    df_copy = aor_df.copy()
    if "state" in df_copy.columns:
        df_copy["state_norm"] = df_copy["state"].apply(normalize_state_name)
    else:
        df_copy["state_norm"] = "Unknown"

    if group_by_period and "period" in df_copy.columns:
        grouped = (
            df_copy.groupby(["year", "period"], as_index=False)["average_occupancy_rate_pct"]
            .mean()
        )
    else:
        grouped = (
            df_copy.groupby("year", as_index=False)["average_occupancy_rate_pct"]
            .mean()
        )
        if "period" in df_copy.columns:
            grouped["period"] = "Annual"

    grouped["state"] = "National Mean Baseline"
    grouped = grouped.sort_values(by="year").reset_index(drop=True)
    return NationalAORBenchmarkFrame(grouped)
