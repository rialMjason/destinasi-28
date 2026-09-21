"""
DESTINASI — Carrying Capacity & Multi-System Bottleneck Engine
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Implements the multi-system dynamic bottleneck carrying capacity framework adapted from
Cifuentes (1992, 1999) and PLANMalaysia destination-planning principles:
- Spatial Physical Carrying Capacity (PCC): A_usable * (V/a) * Rf
- Environmental & Seasonal Real Capacity (RCC): PCC * prod(1 - CF_k) where CF_k = M_l / M_t
- Operational & Municipal Effective Capacity (ECC): RCC * MC_pbt
- Multi-System Bottleneck Sustainable Capacity: min(CC_accom, CC_transport, CC_attraction, CC_water, CC_eco, CC_social)
- Decoupled Metrics: Pressure (D/CC), Excess Demand max(0, D - CC), and Capacity Gap (1 - D/CC).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_physical_carrying_capacity(
    usable_area_m2: float,
    space_per_visitor_m2: float,
    daily_operating_hours: float,
    avg_visit_duration_hours: float
) -> float:
    """
    Computes spatial Physical Carrying Capacity (PCC) using the Cifuentes (1992) formulation:
    PCC = A_usable * (V / a) * R_f
    where R_f = daily_operating_hours / avg_visit_duration_hours.
    """
    if usable_area_m2 <= 0 or space_per_visitor_m2 <= 0 or avg_visit_duration_hours <= 0:
        return 0.0
    rotation_factor = daily_operating_hours / avg_visit_duration_hours
    return float(usable_area_m2 * (1.0 / space_per_visitor_m2) * rotation_factor)


def compute_real_carrying_capacity(
    pcc: float,
    correction_factors: Dict[str, float]
) -> float:
    """
    Computes Real Carrying Capacity (RCC) by applying physical/environmental limiting conditions:
    RCC = PCC * prod(1 - CF_k)
    where CF_k = M_l / M_t (measured limiting magnitude over total magnitude).
    """
    reduction_multiplier = 1.0
    for factor_name, cf_value in correction_factors.items():
        bounded_cf = max(0.0, min(1.0, float(cf_value)))
        reduction_multiplier *= (1.0 - bounded_cf)
    return float(pcc * reduction_multiplier)


def compute_effective_carrying_capacity(
    rcc: float,
    management_capacity_ratio: float
) -> float:
    """
    Computes Effective Carrying Capacity (ECC) factoring in municipal PBT infrastructure:
    ECC = RCC * MC_pbt
    where MC_pbt is the management capacity ratio in [0, 1].
    """
    bounded_mc = max(0.0, min(1.0, float(management_capacity_ratio)))
    return float(rcc * bounded_mc)


def diagnose_multisystem_bottleneck(
    demand: float,
    cc_accommodation: float,
    cc_transport: float,
    cc_attraction: float,
    cc_water_waste: float,
    cc_ecology: float,
    cc_social: float
) -> Dict[str, Any]:
    """
    Evaluates a destination's carrying capacity as a multi-system bottleneck:
    CC = min(CC_accom, CC_transport, CC_attraction, CC_water, CC_eco, CC_social)

    Returns:
        - sustainable_capacity: The binding bottleneck capacity threshold
        - binding_constraint: Name of the limiting sub-system
        - continuous_pressure: Demand / Sustainable Capacity
        - excess_demand: max(0, Demand - Sustainable Capacity)
        - capacity_gap: 1 - (Demand / Sustainable Capacity)
        - is_overcapacity: True if Demand > Sustainable Capacity
    """
    capacities = {
        "Accommodation Supply": max(1.0, float(cc_accommodation)),
        "Road & Transit Throughput": max(1.0, float(cc_transport)),
        "Attraction & Amenity Space": max(1.0, float(cc_attraction)),
        "Municipal Water & Waste Buffer": max(1.0, float(cc_water_waste)),
        "Ecosystem & Slope Tolerance": max(1.0, float(cc_ecology)),
        "Resident & Social Tolerance": max(1.0, float(cc_social)),
    }

    binding_constraint = min(capacities, key=capacities.get)
    sustainable_capacity = capacities[binding_constraint]

    demand = max(0.0, float(demand))
    continuous_pressure = demand / sustainable_capacity
    excess_demand = max(0.0, demand - sustainable_capacity)
    capacity_gap = 1.0 - continuous_pressure

    if continuous_pressure < 0.70:
        management_status = "Underutilized (High Absorption Buffer)"
    elif continuous_pressure <= 0.85:
        management_status = "Healthy Utilization"
    elif continuous_pressure <= 1.00:
        management_status = "High Pressure (Approaching Limit)"
    else:
        management_status = "Overcapacity (Immediate Intervention Required)"

    return {
        "demand": demand,
        "sustainable_capacity": round(sustainable_capacity, 1),
        "binding_constraint": binding_constraint,
        "subsystem_capacities": capacities,
        "continuous_pressure": round(continuous_pressure, 3),
        "pressure_percent": round(continuous_pressure * 100.0, 1),
        "excess_demand": round(excess_demand, 1),
        "capacity_gap": round(capacity_gap, 3),
        "capacity_gap_percent": round(capacity_gap * 100.0, 1),
        "is_overcapacity": bool(excess_demand > 0),
        "management_status": management_status,
    }


def evaluate_destination_row(row: pd.Series) -> Dict[str, Any]:
    """Convenience helper to evaluate a row from pilot_corridors.csv."""
    return diagnose_multisystem_bottleneck(
        demand=row["daily_demand_peak"],
        cc_accommodation=row["cc_accommodation"],
        cc_transport=row["cc_transport"],
        cc_attraction=row["cc_attraction"],
        cc_water_waste=row["cc_water_waste"],
        cc_ecology=row["cc_ecology"],
        cc_social=row["cc_social"]
    )


# -----------------------------------------------------------------------------
# Re-export MOTAC AOR Benchmark & Regional Cluster Definitions
# -----------------------------------------------------------------------------
from src.aor_engine import (
    ALL_16_STATES,
    REGIONAL_CLUSTERS,
    STATE_ALIASES,
    normalize_state_name,
    get_cluster_states,
    calculate_national_aor_baseline,
    compute_national_mean_aor,
    NationalAORBenchmarkFrame,
)
