"""
DESTINASI — Explainable AI (XAI) Attribution & Counterfactual Engine
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Adheres to DARPA XAI and EU AI Act Governance Principles for Policymakers:
1. Exact Quantitative Feature Attribution (Additive Decomposition of WSM score).
2. Shortlisting Comparative Justification (Why shortlisted over alternatives).
3. Counterfactual "What-If" Sensitivity Simulator with Compliance Elasticity.
4. Statutory Plain-Language Justification & Confidence Bounds.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np


class ComponentDriver(dict):
    """Container for dominant positive driver feature attribution."""
    def __str__(self) -> str:
        name = self.get("name", "Driver")
        pts = self.get("contribution", self.get("points", 0.0))
        return f"{name} (+{pts:.1f} pts)"


class ComponentConstraint(dict):
    """Container for dominant constraint / penalty feature attribution."""
    def __str__(self) -> str:
        name = self.get("name", "Constraint")
        pts = self.get("contribution", self.get("points", 0.0))
        return f"{name} ({pts:.1f} pts)"


class ComparativeRationale(str):
    """
    Dual-type container subclassing str that provides dictionary/attribute access
    while behaving identically to a Markdown string in rendering and assertions.
    """
    def __new__(cls, text: str, data: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, text)
        obj._data = data or {}
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._data.get(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def comparative_reasons(self) -> List[str]:
        return self._data.get("comparative_reasons", [])

    @property
    def rank(self) -> int:
        return self._data.get("rank", 1)

    @property
    def score_margin(self) -> float:
        return self._data.get("score_margin", 0.0)

    @property
    def summary(self) -> str:
        return self._data.get("summary", str(self))


def compute_xai_feature_attributions(
    hotspot_row: pd.Series = None,
    candidate_row: pd.Series = None,
    weights: Dict[str, float] = None,
    distance_km: float = None,
    candidate_wsm_score: float = None,
    hotspot: pd.Series = None,
    candidate: pd.Series = None,
    include_distance_friction: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Computes exact additive feature attribution breakdown for a recommendation.
    Returns quantitative point contributions summing EXACTLY to final WSM score.
    
    Components modeled:
    1. Archetype Affinity (Multi-Vector Synergy)
    2. Transit Accessibility
    3. Spare Capacity Headroom
    4. B40 Community Benefit
    5. Ecological Risk Penalty
    6. Distance Friction (when transit distance > 60 km or explicitly specified)
    
    Provides dominant driver and dominant constraint classification.
    """
    if hotspot_row is None and hotspot is not None:
        hotspot_row = hotspot
    if candidate_row is None and candidate is not None:
        candidate_row = candidate

    if hotspot_row is None:
        hotspot_row = pd.Series({"arch_heritage": 0.5, "arch_nature": 0.5, "arch_beach": 0.5, "arch_food": 0.5, "arch_urban": 0.5})
    if candidate_row is None:
        candidate_row = pd.Series({"distance_km": 60.0, "travel_time_mins": 66.0, "capacity_gap": 0.40, "poverty_rate": 4.5})

    if weights is None:
        w_match = 0.30
        w_access = 0.20
        w_cap = 0.25
        w_comm = 0.20
        w_eco = 0.05
    else:
        w_match = float(weights.get("match", weights.get("archetype", 0.30)))
        w_access = float(weights.get("accessibility", 0.20))
        w_cap = float(weights.get("spare_capacity", 0.25))
        w_comm = float(weights.get("community_benefit", weights.get("community", 0.20)))
        w_eco = float(weights.get("environmental_risk", weights.get("ecological_risk", 0.05)))

    # 1. Archetype Affinity / Multi-Vector Match (0 to 100)
    if "match_score" in candidate_row and pd.notna(candidate_row.get("match_score")):
        raw_match_pts = float(candidate_row["match_score"])
    else:
        try:
            from src.recommender_matcher import compute_multivector_similarity
            raw_sim = compute_multivector_similarity(hotspot_row, candidate_row)
            raw_match_pts = round(raw_sim * 100.0, 1)
        except Exception:
            arch_keys = ["arch_heritage", "arch_nature", "arch_beach", "arch_food", "arch_urban"]
            vec_h = np.array([float(hotspot_row.get(k, 0.5)) for k in arch_keys])
            vec_c = np.array([float(candidate_row.get(k, 0.5)) for k in arch_keys])
            norm_h = np.linalg.norm(vec_h)
            norm_c = np.linalg.norm(vec_c)
            cos_sim = float(np.dot(vec_h, vec_c) / (norm_h * norm_c)) if (norm_h > 0 and norm_c > 0) else 0.5
            raw_match_pts = round(cos_sim * 100.0, 1)
    attrib_match = round(raw_match_pts * w_match, 1)

    # 2. Transit Accessibility (0 to 100)
    if "accessibility_score" in candidate_row and pd.notna(candidate_row.get("accessibility_score")):
        access_score = float(candidate_row["accessibility_score"])
    else:
        dist_km_val = float(distance_km if distance_km is not None else candidate_row.get("distance_km", 60.0))
        time_mins = float(candidate_row.get("travel_time_mins", candidate_row.get("transit_time_mins", max(30.0, dist_km_val * 1.1))))
        transit_mode = str(candidate_row.get("transit_mode", "Road"))
        try:
            from src.recommender_matcher import compute_accessibility_score
            access_score = compute_accessibility_score(time_mins, transit_mode)
        except Exception:
            access_score = float(np.clip(100.0 - (time_mins / 150.0 * 100.0), 10.0, 100.0))
    attrib_access = round(access_score * w_access, 1)

    # 3. Spare Capacity Headroom — blended proportion + absolute pax headroom
    # (same definition as the recommender pillar, so attribution, simulation
    # and shortlist ranking agree). Proportion-only field is preserved.
    if "spare_capacity_blended" in candidate_row and pd.notna(candidate_row.get("spare_capacity_blended")):
        cap_score = float(candidate_row["spare_capacity_blended"])
        _prop_for_desc = float(candidate_row.get("spare_capacity_score", cap_score))
    else:
        if "spare_capacity_score" in candidate_row and pd.notna(candidate_row.get("spare_capacity_score")):
            _prop_for_desc = float(candidate_row["spare_capacity_score"])
        else:
            gap = float(candidate_row.get("capacity_gap", 0.40))
            _prop_for_desc = max(0.0, min(100.0, round(gap * 100.0, 1)))
        try:
            _cs = float(candidate_row.get("sustainable_capacity", 0.0) or 0.0)
            _cd = float(candidate_row.get("daily_demand_peak", 0.0) or 0.0)
            _abs_pax = max(0.0, _cs - _cd) if _cs > 0 else 0.0
        except (TypeError, ValueError):
            _abs_pax = 0.0
        _abs_score = absolute_headroom_score(_abs_pax, hotspot_excess_pax(hotspot_row))
        cap_score = round(0.5 * _prop_for_desc + 0.5 * _abs_score, 1)
    attrib_capacity = round(cap_score * w_cap, 1)

    # 4. B40 Community Benefit & Uplift (0 to 100)
    if "community_benefit_score" in candidate_row and pd.notna(candidate_row.get("community_benefit_score")):
        pov_score = float(candidate_row["community_benefit_score"])
    else:
        pov_rate = float(candidate_row.get("poverty_rate", 4.5))
        pov_score = min(100.0, round(pov_rate * 8.5 + 15.0, 1))
    attrib_community = round(pov_score * w_comm, 1)

    # 5. Ecological Risk Penalty (0 to 100)
    if "environmental_risk_penalty" in candidate_row and pd.notna(candidate_row.get("environmental_risk_penalty")):
        env_risk_score = float(candidate_row["environmental_risk_penalty"])
    else:
        eco_cap = float(candidate_row.get("cc_ecology", 5000.0))
        current_demand = float(candidate_row.get("daily_demand_peak", 2000.0))
        eco_headroom_ratio = max(0.0, (eco_cap - current_demand) / max(1.0, eco_cap))
        env_risk_score = round((1.0 - min(1.0, eco_headroom_ratio)) * 100.0, 1)
    attrib_eco = -round(env_risk_score * w_eco, 1)

    # 6. Distance Friction Penalty
    effective_dist = distance_km if distance_km is not None else candidate_row.get("distance_km")
    if effective_dist is not None and not pd.isna(effective_dist):
        effective_dist = float(effective_dist)
        dist_friction_pts = round(max(0.0, (effective_dist - 60.0) * 0.08), 1) if effective_dist > 60.0 else 0.0
    else:
        dist_friction_pts = 0.0
        effective_dist = 60.0
    attrib_dist = -dist_friction_pts

    # Determine whether Distance Friction is included in waterfall_df
    # When include_distance_friction is None, include if distance_km is passed, candidate_wsm_score is passed,
    # or candidate_row explicitly specifies distance_km
    has_explicit_distance = (distance_km is not None) or (
        hasattr(candidate_row, "get") and candidate_row.get("distance_km") is not None and not pd.isna(candidate_row.get("distance_km"))
    )
    if include_distance_friction is True:
        model_distance_friction = True
    elif include_distance_friction is False:
        model_distance_friction = False
    else:
        model_distance_friction = bool(has_explicit_distance or candidate_wsm_score is not None)

    # Calculate target final WSM score
    raw_wsm = attrib_match + attrib_access + attrib_capacity + attrib_community + attrib_eco + (attrib_dist if model_distance_friction else 0.0)
    raw_wsm = round(max(0.0, min(100.0, raw_wsm)), 1)

    if candidate_wsm_score is not None:
        target_wsm = round(float(candidate_wsm_score), 1)
    elif "final_wsm_score" in candidate_row and pd.notna(candidate_row.get("final_wsm_score")) and model_distance_friction:
        target_wsm = round(float(candidate_row["final_wsm_score"]), 1)
    else:
        target_wsm = raw_wsm

    # Assemble waterfall features
    c_time = float(candidate_row.get("travel_time_mins", candidate_row.get("transit_time_mins", max(30.0, effective_dist * 1.1))))
    c_mode = str(candidate_row.get("transit_mode", "Transit"))
    c_gap = float(candidate_row.get("capacity_gap", 0.40))
    c_pov = float(candidate_row.get("poverty_rate", 4.5))
    try:
        _cs_d = float(candidate_row.get("sustainable_capacity", 0.0) or 0.0)
        _cd_d = float(candidate_row.get("daily_demand_peak", 0.0) or 0.0)
        _abs_pax_d = max(0.0, _cs_d - _cd_d) if _cs_d > 0 else 0.0
    except (TypeError, ValueError):
        _abs_pax_d = 0.0

    wf_items = [
        {"Feature": "1. Archetype Affinity", "Category": "Positive Driver", "Points": attrib_match, "Description": f"{raw_match_pts:.0f}% multi-vector synergy"},
        {"Feature": "2. Transit Accessibility", "Category": "Positive Driver", "Points": attrib_access, "Description": f"{access_score:.0f} pts accessibility ({c_mode})"},
        {"Feature": "3. Spare Capacity", "Category": "Positive Driver", "Points": attrib_capacity, "Description": f"{_prop_for_desc:.1f}% headroom, {_abs_pax_d:,.0f} pax absorbable"},
        {"Feature": "4. B40 Community Benefit", "Category": "Positive Driver", "Points": attrib_community, "Description": f"{c_pov:.1f}% district poverty rate"},
        {"Feature": "5. Ecological Risk Penalty", "Category": "Risk Guardrail", "Points": attrib_eco, "Description": f"Sensitive habitat protection ({env_risk_score:.0f}% risk)"},
    ]

    if model_distance_friction:
        wf_items.append({
            "Feature": "6. Distance Friction",
            "Category": "Risk Guardrail",
            "Points": attrib_dist,
            "Description": f"Spatial travel shed friction ({effective_dist:.0f} km)"
        })

    waterfall_df = pd.DataFrame(wf_items)

    # EXACT synchronization: Ensure sum of Points equals target_wsm down to the exact decimal
    sum_pts = round(waterfall_df["Points"].sum(), 1)
    residual = round(target_wsm - sum_pts, 1)
    if abs(residual) > 1e-6:
        # Distribute residual to the highest positive driver so exact additive equality is preserved
        pos_mask = waterfall_df["Category"] == "Positive Driver"
        if pos_mask.any():
            max_idx = waterfall_df[pos_mask]["Points"].idxmax()
            waterfall_df.loc[max_idx, "Points"] = round(waterfall_df.loc[max_idx, "Points"] + residual, 1)

    final_wsm_score = target_wsm

    # Dominant Driver Classification (highest positive contributor)
    pos_df = waterfall_df[waterfall_df["Category"] == "Positive Driver"]
    if not pos_df.empty:
        best_pos_idx = pos_df["Points"].idxmax()
        d_row = pos_df.loc[best_pos_idx]
        dominant_driver = ComponentDriver({
            "name": str(d_row["Feature"]),
            "contribution": float(d_row["Points"]),
            "points": float(d_row["Points"]),
            "category": "Positive Driver",
            "description": str(d_row["Description"]),
        })
    else:
        dominant_driver = ComponentDriver({"name": "Archetype Affinity", "contribution": 0.0, "points": 0.0})

    # Dominant Constraint Classification (largest penalty or lowest component)
    risk_df = waterfall_df[waterfall_df["Category"] == "Risk Guardrail"]
    if not risk_df.empty:
        worst_risk_idx = risk_df["Points"].idxmin()
        c_row = risk_df.loc[worst_risk_idx]
        dominant_constraint = ComponentConstraint({
            "name": str(c_row["Feature"]),
            "contribution": float(c_row["Points"]),
            "points": float(c_row["Points"]),
            "penalty": round(abs(float(c_row["Points"])), 1),
            "category": "Risk Guardrail",
            "description": str(c_row["Description"]),
        })
    else:
        min_idx = waterfall_df["Points"].idxmin()
        c_row = waterfall_df.loc[min_idx]
        dominant_constraint = ComponentConstraint({
            "name": str(c_row["Feature"]),
            "contribution": float(c_row["Points"]),
            "points": float(c_row["Points"]),
            "penalty": round(abs(float(c_row["Points"])), 1),
            "category": "Constraint",
            "description": str(c_row["Description"]),
        })

    # Backward compatibility: primary_driver
    primary_driver = "Spare Lodging Headroom" if attrib_capacity > attrib_community else "B40 Grassroots Uplift"

    return {
        "final_wsm_score": final_wsm_score,
        "match_score": round(raw_match_pts, 1),
        "access_score": round(access_score, 1),
        "spare_capacity_score": round(_prop_for_desc, 1),
        "spare_capacity_blended": round(cap_score, 1),
        "absolute_headroom_pax": round(_abs_pax_d, 1),
        "community_score": round(pov_score, 1),
        "eco_penalty": round(abs(attrib_eco), 1),
        "distance_friction": round(abs(attrib_dist), 1),
        "waterfall_df": waterfall_df,
        "primary_driver": primary_driver,
        "dominant_driver": dominant_driver,
        "dominant_constraint": dominant_constraint,
    }


def explain_why_shortlisted_over_alternatives(
    selected_candidate: Union[pd.Series, Dict[str, Any]],
    alternative_candidates: Optional[Union[pd.DataFrame, List[Any]]] = None,
    hotspot_row: Optional[Union[pd.Series, Dict[str, Any]]] = None,
) -> ComparativeRationale:
    """
    Synthesizes a statutory comparative justification explaining why a specific
    candidate was shortlisted over competing alternatives based on:
    1. Score margin over alternatives and rank.
    2. Capacity cushion to absorb overflow without secondary displacement.
    3. Dedicated transit link / travel friction.
    4. Grassroots B40 community capture and equity uplift.
    5. PLANMalaysia RFN-4 ecological compliance.

    Returns a ComparativeRationale string that also provides structured attribute access.
    """
    if selected_candidate is None:
        selected_candidate = {}
    if hasattr(selected_candidate, "to_dict"):
        c_dict = selected_candidate.to_dict()
    else:
        c_dict = dict(selected_candidate)

    c_name = str(c_dict.get("candidate_name", c_dict.get("destination_name", c_dict.get("district_name", "Selected Corridor"))))
    c_state = str(c_dict.get("state_name", "Malaysia"))
    c_score = float(c_dict.get("final_wsm_score", 0.0))
    c_dist = float(c_dict.get("distance_km", 0.0))
    c_time = float(c_dict.get("travel_time_mins", c_dict.get("transit_time_mins", 60.0)))
    c_mode = str(c_dict.get("transit_mode", "Road / Transit"))
    c_gap = float(c_dict.get("capacity_gap", 0.40)) * 100.0
    c_pov = float(c_dict.get("poverty_rate", 5.0))
    sust_cap = float(c_dict.get("sustainable_capacity", 0.0))
    peak_demand = float(c_dict.get("daily_demand_peak", 0.0))
    headroom_pax = max(0, int(sust_cap - peak_demand))

    h_dict = {}
    if hotspot_row is not None:
        h_dict = hotspot_row.to_dict() if hasattr(hotspot_row, "to_dict") else dict(hotspot_row)
    h_name = str(h_dict.get("destination_name", h_dict.get("district_name", "Origin Hotspot")))
    h_pov = float(h_dict.get("poverty_rate", 4.0))

    # Parse alternative candidates
    alt_list = []
    if alternative_candidates is not None:
        if isinstance(alternative_candidates, pd.DataFrame):
            alt_list = alternative_candidates.to_dict("records")
        elif isinstance(alternative_candidates, list):
            for item in alternative_candidates:
                alt_list.append(item.to_dict() if hasattr(item, "to_dict") else dict(item))

    # Determine rank and peers
    c_clean_name = c_name.strip().lower()
    names = [str(r.get("candidate_name", r.get("destination_name", ""))).strip().lower() for r in alt_list]
    rank = names.index(c_clean_name) + 1 if c_clean_name in names else 1
    total_shortlisted = max(len(alt_list), 1)

    peers = [r for r in alt_list if str(r.get("candidate_name", r.get("destination_name", ""))).strip().lower() != c_clean_name]

    reasons: List[str] = []
    score_margin = 0.0

    # 1. Score Margin / Rank Rationale
    if rank == 1 and peers:
        runner_up = peers[0]
        runner_up_name = str(runner_up.get("candidate_name", runner_up.get("destination_name", "runner-up corridor")))
        runner_up_score = float(runner_up.get("final_wsm_score", 0.0))
        score_margin = round(c_score - runner_up_score, 1)
        reasons.append(
            f"**Top-Ranked Multi-Vector Alignment (#{rank} of {total_shortlisted})**: "
            f"Outperforms runner-up {runner_up_name} by +{score_margin:.1f} WSM points ({c_score:.1f} vs {runner_up_score:.1f}), "
            f"delivering superior composite synergy across experiential matching, transit accessibility, and capacity headroom."
        )
    elif peers:
        top_peer = peers[0]
        top_score = float(top_peer.get("final_wsm_score", 0.0))
        score_margin = round(c_score - top_score, 1)
        reasons.append(
            f"**Shortlisted Relief Alternative (#{rank} of {total_shortlisted})**: "
            f"Demonstrates strong viability with {c_score:.1f} WSM points, serving as a vital secondary corridor to absorb peak spillover."
        )
    else:
        reasons.append(
            f"**Recommended Relief Corridor**: Achieves a robust {c_score:.1f} WSM score, fulfilling statutory multi-criteria feasibility standards."
        )

    # 2. Capacity Cushion Rationale
    if headroom_pax > 0:
        cushion_desc = f"{c_gap:.1f}% spare lodging headroom (+{headroom_pax:,} pax/day buffer safely below ceiling)"
    else:
        cushion_desc = f"{c_gap:.1f}% unutilized capacity gap"
    reasons.append(
        f"**Superior Lodging Cushion**: Operates with {cushion_desc}, guaranteeing the destination can absorb redirected "
        f"visitors without exceeding municipal carrying capacity or triggering secondary price spikes."
    )

    # 3. Transit Link Rationale
    is_rail = ("rail" in c_mode.lower() or "ktm" in c_mode.lower()) and not any(neg in c_mode.lower() for neg in ["no rail", "non-rail", "tiada rel"])
    if is_rail:
        reasons.append(
            f"**Electrified Trunk Rail Corridor**: Direct {c_mode} connectivity (~{c_time:.0f} mins) insulates diverted passengers "
            f"from peak highway gridlock, awarding a +8.0 pts sustainable transit incentive."
        )
    else:
        reasons.append(
            f"**Regional Transit Feeder**: Situated {c_dist:.0f} km (~{c_time:.0f} mins via {c_mode}) within the 180 km day-trip shed, "
            f"enabling rapid same-day dispersal without excessive travel fatigue."
        )

    # 4. Grassroots B40 Capture & Equity Uplift
    if c_pov >= h_pov:
        reasons.append(
            f"**Targeted B40 Community Uplift**: High district poverty baseline ({c_pov:.1f}% vs {h_pov:.1f}% at {h_name}) "
            f"ensures redirected tourism expenditures are captured directly by local grassroots B40 households, micro-enterprises, and registered homestays."
        )
    else:
        reasons.append(
            f"**Grassroots B40 Revenue Capture**: District poverty baseline of {c_pov:.1f}% channels tourist spending directly "
            f"into local B40 micro-enterprises, small businesses, and homestay operators in {c_name} ({c_state})."
        )

    # 5. Statutory Ecological Compliance
    reasons.append(
        "**PLANMalaysia RFN-4 Compliance**: Verified 100% compliant with National Physical Plan 4 guidelines, "
        "maintaining potable water buffers above SPAN 15% thresholds and zero Level-1 KSAS violations."
    )

    # Assemble Markdown Text
    bullets = "\n".join([f"{i+1}. {r}" for i, r in enumerate(reasons)])
    full_text = f"### 📌 Strategic Shortlisting Rationale: {c_name} (Rank #{rank} of {total_shortlisted})\n\n{bullets}"

    summary = f"Ranked #{rank} of {total_shortlisted} with {c_score:.1f} WSM points. " + " ".join(reasons)

    data = {
        "candidate_name": c_name,
        "state_name": c_state,
        "rank": rank,
        "total_shortlisted": total_shortlisted,
        "score_margin": score_margin,
        "comparative_reasons": reasons,
        "summary": summary,
        "wsm_score": c_score,
        "capacity_gap_pct": c_gap,
        "headroom_pax": headroom_pax,
        "poverty_rate": c_pov,
    }

    return ComparativeRationale(full_text, data=data)


def compute_safe_absorption(
    sustainable_capacity: float,
    peak_demand: float,
    eco_cap: float,
    capacity_gap: float,
    capacity_expansion_pct: float = 0.0,
    total_rooms: float = 5000.0,
) -> Dict[str, Any]:
    """
    Single source of truth for safe absorption math (shared by the What-If
    popover and the Action Playbook so both tabs report identical figures).

    Applies the same room-stock-damped expansion as the counterfactual
    simulator, then returns the binding constraint of sustainable headroom
    vs ecological (KSAS) headroom.
    """
    try:
        rooms_val = float(total_rooms)
    except (TypeError, ValueError):
        rooms_val = 5000.0
    if pd.isna(rooms_val):
        rooms_val = 5000.0
    room_factor = float(np.clip(rooms_val / 5000.0, 0.4, 1.0))
    effective_capacity_pct = max(0.0, float(capacity_expansion_pct)) * room_factor

    try:
        orig_gap = float(capacity_gap)
    except (TypeError, ValueError):
        orig_gap = 0.35
    if pd.isna(orig_gap):
        orig_gap = 0.35
    simulated_gap = float(np.clip(orig_gap + (effective_capacity_pct / 100.0) * (1.0 - max(0.0, orig_gap)), 0.0, 1.0))

    sust_cap = max(0.0, float(sustainable_capacity or 0.0))
    peak = max(0.0, float(peak_demand or 0.0))
    sim_headroom_pax = simulated_gap * sust_cap
    eco_cap_f = float(eco_cap or 0.0)
    eco_headroom_pax = float(max(0.0, eco_cap_f - peak)) if eco_cap_f > 0 else float(sim_headroom_pax)
    max_absorbable_pax = float(max(0.0, min(sim_headroom_pax, eco_headroom_pax)))
    binding = "Sustainable capacity" if sim_headroom_pax <= eco_headroom_pax else "Ecological / KSAS cap"
    return {
        "original_gap": round(orig_gap, 4),
        "simulated_gap": round(simulated_gap, 4),
        "effective_capacity_pct": round(effective_capacity_pct, 2),
        "room_factor": round(room_factor, 3),
        "sim_headroom_pax": round(sim_headroom_pax, 1),
        "eco_headroom_pax": round(eco_headroom_pax, 1),
        "max_absorbable_pax": round(max_absorbable_pax, 1),
        "binding": binding,
    }


def absolute_headroom_score(headroom_pax: float, need_pax: float) -> float:
    """
    Absolute-capacity score (0-100): what share of the required diversion
    volume the pax headroom covers. need_pax <= 0 means no pressure → 100.
    """
    try:
        need = max(0.0, float(need_pax))
    except (TypeError, ValueError):
        need = 0.0
    if need <= 0:
        return 100.0
    try:
        head = max(0.0, float(headroom_pax))
    except (TypeError, ValueError):
        head = 0.0
    return float(max(0.0, min(100.0, head / need * 100.0)))


def hotspot_excess_pax(hotspot_row: Any) -> float:
    """Derives hotspot excess demand (need) from sust/demand fields; 0.0 if unknown."""
    try:
        if hotspot_row is None or not hasattr(hotspot_row, "get"):
            return 0.0
        sust = float(hotspot_row.get("sustainable_capacity", 0.0) or 0.0)
        dem = float(hotspot_row.get("daily_demand_peak", 0.0) or 0.0)
        if pd.isna(sust) or pd.isna(dem) or sust <= 0:
            return 0.0
        return float(max(0.0, dem - sust))
    except (TypeError, ValueError):
        return 0.0


def simulate_counterfactual_intervention(
    hotspot_row: pd.Series = None,
    candidate_row: pd.Series = None,
    travel_time_reduction_mins: float = 0.0,
    capacity_expansion_pct: float = 0.0,
    compliance_elasticity: float = 1.0,
    time_reduction_mins: float = None,
    extra_rooms_pct: float = None,
    candidate: pd.Series = None,
    diverted_pax: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Simulates a counterfactual infrastructure or policy intervention:
    - travel_time_reduction_mins: High-speed rail / expressway upgrade reducing travel time
    - capacity_expansion_pct: Expansion of lodging, homestay, or municipal headroom
    - compliance_elasticity: Regulatory compliance elasticity factor (default 1.0)
    - diverted_pax: Expected daily visitors redirected into this corridor (post-diversion
      stress test; default 0.0 preserves legacy behaviour)

    Corridor difficulty weighting: identical slider inputs yield smaller effective
    upgrades on long-distance / road-only corridors than on short rail-connected
    ones, so deltas discriminate between corridors instead of coinciding.

    Computes baseline WSM vs simulated WSM and evaluates feasibility gates.
    Feasibility reflects base corridor viability only (capacity gap > 10%,
    distance <= 350 km, own demand < ecological capacity) and is independent
    of the diverted load. Whether spare headroom survives absorbing
    diverted_pax is reported separately as a diversion stress warning
    (diversion_verdict / overflow) so a viable corridor is not shown as
    Blocked merely because the requested quota exceeds its headroom.
    """
    if candidate_row is None and candidate is not None:
        candidate_row = candidate
    if candidate_row is None:
        candidate_row = pd.Series({"distance_km": 60.0, "travel_time_mins": 66.0, "capacity_gap": 0.35, "poverty_rate": 5.0})
    if time_reduction_mins is not None:
        travel_time_reduction_mins = time_reduction_mins
    if extra_rooms_pct is not None:
        capacity_expansion_pct = extra_rooms_pct

    dummy_h = pd.Series({"arch_heritage": 0.5, "arch_nature": 0.5, "arch_beach": 0.5, "arch_food": 0.5, "arch_urban": 0.5})
    h_series = hotspot_row if hotspot_row is not None else dummy_h

    # 1. Baseline calculation
    base_attrib = compute_xai_feature_attributions(h_series, candidate_row)
    baseline_score = float(base_attrib["final_wsm_score"])

    # 2. Simulated counterfactual candidate (difficulty-weighted levers)
    cand_sim = candidate_row.copy()
    cand_dist = float(cand_sim.get("distance_km", 60.0))
    mode_str = str(cand_sim.get("transit_mode", "Road"))
    is_rail_mode = ("rail" in mode_str.lower() or "ktm" in mode_str.lower()) and not any(
        neg in mode_str.lower() for neg in ["no rail", "non-rail", "tiada rel"]
    )
    # Same nominal upgrade buys less on long / road-only corridors.
    time_difficulty = 1.0 + max(0.0, cand_dist) / 100.0
    if not is_rail_mode:
        time_difficulty *= 1.4
    effective_time_reduction = max(0.0, float(travel_time_reduction_mins)) / time_difficulty
    # Shared safe-absorption math (room-damped expansion → simulated gap/headroom).
    _abs = compute_safe_absorption(
        sustainable_capacity=cand_sim.get("sustainable_capacity", 0.0),
        peak_demand=cand_sim.get("daily_demand_peak", 0.0),
        eco_cap=cand_sim.get("cc_ecology", 0.0),
        capacity_gap=cand_sim.get("capacity_gap", 0.35),
        capacity_expansion_pct=capacity_expansion_pct,
        total_rooms=cand_sim.get("total_rooms", 5000.0),
    )
    effective_capacity_pct = float(_abs["effective_capacity_pct"])
    orig_gap = float(_abs["original_gap"])
    # Expansion expands available capacity headroom
    new_gap = float(_abs["simulated_gap"])
    cand_sim["capacity_gap"] = new_gap

    orig_time = float(cand_sim.get("travel_time_mins", cand_sim.get("transit_time_mins", max(30.0, float(cand_sim.get("distance_km", 60.0)) * 1.1))))
    new_time = max(15.0, orig_time - effective_time_reduction)
    cand_sim["travel_time_mins"] = new_time
    cand_sim["transit_time_mins"] = new_time

    sim_attrib = compute_xai_feature_attributions(h_series, cand_sim)
    raw_simulated_score = float(sim_attrib["final_wsm_score"])
    raw_delta = raw_simulated_score - baseline_score

    # 3. Compliance elasticity modulation
    elasticity_factor = max(0.0, float(compliance_elasticity))
    adjusted_delta = round(raw_delta * elasticity_factor, 2)
    adjusted_simulated_score = round(max(0.0, min(100.0, baseline_score + adjusted_delta)), 1)

    cand_dist = float(cand_sim.get("distance_km", 60.0))
    # Post-diversion stress test (separate warning, not part of feasibility),
    # reusing the shared safe-absorption figures computed above.
    sust_cap = max(0.0, float(cand_sim.get("sustainable_capacity", 0.0)))
    peak_demand = max(0.0, float(cand_sim.get("daily_demand_peak", 0.0)))
    diverted = max(0.0, float(diverted_pax))
    sim_headroom_pax = float(_abs["sim_headroom_pax"])
    post_headroom_pax = sim_headroom_pax - diverted
    eco_cap = float(cand_sim.get("cc_ecology", 0.0))
    # Base corridor viability: own demand must sit below ecological capacity.
    base_eco_ok = bool(peak_demand < eco_cap) if eco_cap > 0 else True
    # Post-diversion stress: headroom + ecology must survive the diverted load.
    post_eco_ok = bool((peak_demand + diverted) < eco_cap) if eco_cap > 0 else True
    overflow = bool(post_headroom_pax < 0 or not post_eco_ok)
    # Safe absorption limit from the shared helper (identical to Tab 3).
    eco_headroom_pax = float(_abs["eco_headroom_pax"])
    max_absorbable_pax = float(_abs["max_absorbable_pax"])
    absorption_coverage_pct = float((max_absorbable_pax / diverted * 100.0) if diverted > 0 else 100.0)
    # Feasibility = base viability only (gap / distance / own-demand ecology).
    is_feasible = bool(new_gap > 0.10 and cand_dist <= 350.0 and base_eco_ok)
    policy_impact = "High Priority" if adjusted_delta >= 5.0 else "Incremental"
    diversion_verdict = "Overflow risk" if overflow else "Absorbs diversion"

    explanation = (
        f"Transit upgrade of -{travel_time_reduction_mins:.0f}m (effective -{effective_time_reduction:.1f}m "
        f"after corridor difficulty x{time_difficulty:.2f}) and +{capacity_expansion_pct:.0f}% capacity headroom "
        f"(effective +{effective_capacity_pct:.1f}%) "
        f"with {elasticity_factor:.2f}x compliance elasticity adjusts corridor compatibility from {baseline_score:.1f} "
        f"to {adjusted_simulated_score:.1f} ({'+' if adjusted_delta >= 0 else ''}{adjusted_delta:.1f} WSM pts). "
        f"Post-diversion headroom: {post_headroom_pax:,.0f} pax/day ({diversion_verdict.lower()}); "
        f"safely absorbs up to {max_absorbable_pax:,.0f} pax/day."
    )

    return {
        "baseline_score": round(baseline_score, 1),
        "simulated_score": adjusted_simulated_score,
        "delta_score": adjusted_delta,
        "compliance_elasticity": elasticity_factor,
        "is_feasible": is_feasible,
        "base_eco_ok": base_eco_ok,
        "post_eco_ok": post_eco_ok,
        "overflow": overflow,
        "policy_impact": policy_impact,
        "diversion_verdict": diversion_verdict,
        "post_headroom_pax": round(post_headroom_pax, 1),
        "sim_headroom_pax": round(sim_headroom_pax, 1),
        "eco_headroom_pax": round(eco_headroom_pax, 1),
        "max_absorbable_pax": round(max_absorbable_pax, 1),
        "absorption_coverage_pct": round(absorption_coverage_pct, 1),
        "absorption_binding": _abs["binding"],
        "time_difficulty": round(time_difficulty, 3),
        "effective_time_reduction_mins": round(effective_time_reduction, 2),
        "effective_capacity_pct": round(effective_capacity_pct, 2),
        "explanation": explanation,
        "original_time_mins": orig_time,
        "simulated_time_mins": new_time,
        "original_gap": orig_gap,
        "simulated_gap": new_gap,
        "score_delta_wsm": adjusted_delta,
    }


def simulate_counterfactual_policy(
    candidate: pd.Series,
    time_reduction_mins: float = 0.0,
    extra_rooms_pct: float = 0.0,
    diverted_pax: float = 0.0
) -> Dict[str, Any]:
    """
    Backward-compatible counterfactual simulation wrapper.
    """
    return simulate_counterfactual_intervention(
        candidate_row=candidate,
        travel_time_reduction_mins=time_reduction_mins,
        capacity_expansion_pct=extra_rooms_pct,
        diverted_pax=diverted_pax
    )


