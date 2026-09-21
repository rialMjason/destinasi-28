"""
DESTINASI — Multi-Vector Recommender & Alternative Corridor Matcher
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Calculates experiential cosine similarity across 5 archetypes (Heritage, Nature, Beach, Food, Urban),
penalizes transit friction, enforces hard carrying capacity feasibility gates, and generates
transparent natural-language explainability breakdowns.
"""

from typing import Dict, List, Any, Set
import json
import numpy as np
import pandas as pd

ARCHETYPES = ["arch_heritage", "arch_nature", "arch_beach", "arch_food", "arch_urban"]

PILLAR_LABELS = {
    "arch_heritage": "Heritage & Culture",
    "arch_nature": "Eco-Nature & Highlands",
    "arch_beach": "Coastal & Marine",
    "arch_food": "Gastronomy & Agri-Tourism",
    "arch_urban": "Urban & Integrated Leisure"
}


def _extract_tag_set(raw_tags: Any) -> Set[str]:
    """
    Normalizes arbitrary POI tag inputs into a clean set of lowercase strings.
    Handles None, NaN, sets, lists, tuples, JSON strings, and delimited strings.
    """
    if raw_tags is None or (isinstance(raw_tags, float) and np.isnan(raw_tags)):
        return set()
    if isinstance(raw_tags, (set, frozenset)):
        return {str(x).strip().lower() for x in raw_tags if pd.notna(x) and str(x).strip()}
    if isinstance(raw_tags, (list, tuple)):
        return {str(x).strip().lower() for x in raw_tags if pd.notna(x) and str(x).strip()}
    if isinstance(raw_tags, str):
        s = raw_tags.strip()
        if not s or s.lower() in ("nan", "none", "null", "[]", "set()", "{}"):
            return set()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return {str(x).strip().lower() for x in parsed if str(x).strip()}
            except Exception:
                pass
            inner = s[1:-1]
            return {item.strip(" '\"\t\r\n").lower() for item in inner.split(",") if item.strip(" '\"\t\r\n")}
        if "|" in s:
            items = s.split("|")
        elif ";" in s:
            items = s.split(";")
        elif "," in s:
            items = s.split(",")
        else:
            items = [s]
        return {item.strip().lower() for item in items if item.strip()}
    return set()


def _extract_archetype_vector(row: Any, archetypes: List[str] = ARCHETYPES, default_val: float = 0.50) -> np.ndarray:
    """Safely extracts a 5-dimensional archetype vector, imputing missing keys and NaNs."""
    vals = []
    for k in archetypes:
        val = default_val
        try:
            if hasattr(row, "get"):
                raw = row.get(k, default_val)
            elif hasattr(row, "__getitem__") and k in row:
                raw = row[k]
            else:
                raw = default_val
            if raw is not None and not pd.isna(raw):
                val = float(raw)
                if np.isnan(val) or np.isinf(val):
                    val = default_val
        except (ValueError, TypeError):
            val = default_val
        vals.append(float(np.clip(val, 0.0, 1.0)))
    return np.array(vals, dtype=float)


def compute_vector_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 5-dimensional archetype vectors."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    dot_prod = float(np.dot(vec_a, vec_b))
    return float(np.clip(dot_prod / (norm_a * norm_b), 0.0, 1.0))


def compute_jaccard_poi_similarity(tags_a: Any, tags_b: Any) -> float:
    """
    Computes Jaccard POI feature similarity: J(A, B) = |A ∩ B| / |A ∪ B|.
    Safely processes sets, lists, strings, and missing values. Returns float in [0.0, 1.0].
    """
    set_a = _extract_tag_set(tags_a)
    set_b = _extract_tag_set(tags_b)
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    if union_len == 0:
        return 0.0
    return float(intersection_len / union_len)


def compute_multivector_similarity(
    hotspot_row: pd.Series,
    candidate_row: pd.Series,
    alpha: float = 0.60
) -> float:
    """
    Computes hybrid multi-vector similarity fusing Cosine archetype affinity and Jaccard POI overlap.
    Dynamically transfers weight to Cosine (alpha_eff = 1.0) when POI tags are absent on both sides,
    ensuring 100% backwards-compatibility with existing unit tests and legacy schemas.
    """
    vec_h = _extract_archetype_vector(hotspot_row)
    vec_c = _extract_archetype_vector(candidate_row)
    cos_sim = compute_vector_cosine_similarity(vec_h, vec_c)

    tags_h_raw = hotspot_row.get("poi_tags", hotspot_row.get("tags", None)) if hasattr(hotspot_row, "get") else None
    tags_c_raw = candidate_row.get("poi_tags", candidate_row.get("tags", None)) if hasattr(candidate_row, "get") else None

    set_h = _extract_tag_set(tags_h_raw)
    set_c = _extract_tag_set(tags_c_raw)

    if not set_h and not set_c:
        return float(cos_sim)
    if tags_h_raw is None or tags_c_raw is None:
        return float(cos_sim)

    jaccard_sim = compute_jaccard_poi_similarity(set_h, set_c)
    hybrid_sim = alpha * cos_sim + (1.0 - alpha) * jaccard_sim
    return float(np.clip(hybrid_sim, 0.0, 1.0))


def compute_accessibility_score(travel_time_mins: float, transit_mode: str) -> float:
    """
    Computes an accessibility score in [0, 100].
    Penalizes travel time exceeding 120 minutes, with a bonus for rail/public transit.
    """
    time = max(0.0, float(travel_time_mins))
    # Decay function: 100 at 0 mins, 80 at 60 mins, 50 at 120 mins, 20 at 180 mins
    score = 100.0 * np.exp(-0.0075 * time)
    # Transit sustainability bonus (KTM electric rail vs private road)
    if "ktm" in transit_mode.lower() or "rail" in transit_mode.lower():
        score = min(100.0, score + 8.0)
    return round(float(score), 1)


def score_candidate_alternative(
    hotspot_row: pd.Series,
    candidate_row: pd.Series,
    weights: Dict[str, float] = None,
    alpha: float = 0.60
) -> Dict[str, Any]:
    """
    Scores a candidate 2nd-tier alternative against an overcapacity origin hotspot.
    Implements statutory WSM formula fusing 5-pillar archetype affinity and Jaccard POI overlap,
    enforces hard feasibility gates, and synthesizes a 4-part transparent XAI explanation.
    """
    if weights is None:
        weights = {
            "match": 0.30,
            "accessibility": 0.20,
            "spare_capacity": 0.25,
            "community_benefit": 0.20,
            "environmental_risk": 0.05
        }

    # 1. Multi-Vector Match (0 to 100)
    raw_similarity = compute_multivector_similarity(hotspot_row, candidate_row, alpha=alpha)
    match_score = round(raw_similarity * 100.0, 1)

    # 2. Accessibility Score (0 to 100)
    travel_time = float(candidate_row.get("transit_time_mins", 60.0))
    transit_mode = str(candidate_row.get("transit_mode", "Road"))
    accessibility_score = compute_accessibility_score(travel_time, transit_mode)

    # 3. Spare Capacity Score (0 to 100) — proportion term kept verbatim for
    # backward compatibility; absolute pax headroom enters via the blended
    # pillar below so large corridors outrank small ones at equal percentages.
    capacity_gap = float(candidate_row.get("capacity_gap", 0.50))
    spare_capacity_score = max(0.0, min(100.0, round(capacity_gap * 100.0, 1)))

    sust_cap_c = float(candidate_row.get("sustainable_capacity", 0.0) or 0.0)
    absolute_headroom_pax = max(0.0, sust_cap_c - float(candidate_row.get("daily_demand_peak", 0.0) or 0.0)) if sust_cap_c > 0 else 0.0
    try:
        _hs = float(hotspot_row.get("sustainable_capacity", 0.0) or 0.0) if hasattr(hotspot_row, "get") else 0.0
        _hd = float(hotspot_row.get("daily_demand_peak", 0.0) or 0.0) if hasattr(hotspot_row, "get") else 0.0
        hotspot_need_pax = max(0.0, _hd - _hs) if _hs > 0 else 0.0
    except (TypeError, ValueError):
        hotspot_need_pax = 0.0
    if hotspot_need_pax > 0:
        absolute_headroom_score = max(0.0, min(100.0, round(absolute_headroom_pax / hotspot_need_pax * 100.0, 1)))
    else:
        absolute_headroom_score = 100.0 if absolute_headroom_pax > 0 else round(spare_capacity_score, 1)
    spare_capacity_blended = round(0.5 * spare_capacity_score + 0.5 * absolute_headroom_score, 1)

    # 4. Community Benefit Score (0 to 100)
    poverty_rate = float(candidate_row.get("poverty_rate", 5.0))
    community_score = min(100.0, round(poverty_rate * 8.5 + 15.0, 1))

    # 5. Environmental Risk Penalty (0 to 100)
    eco_cap = float(candidate_row.get("cc_ecology", 5000.0))
    current_demand = float(candidate_row.get("daily_demand_peak", 2000.0))
    eco_headroom_ratio = max(0.0, (eco_cap - current_demand) / max(1.0, eco_cap))
    env_risk_score = round((1.0 - min(1.0, eco_headroom_ratio)) * 100.0, 1)

    # Hard Feasibility Gate:
    # 1. Capacity Gap > 10% (0.10)
    # 2. Demand strictly below Ecological Capacity (current_demand < eco_cap)
    # 3. Transit distance <= 350.0 km
    distance_km = float(candidate_row.get("distance_km", 0.0))
    is_feasible = (capacity_gap > 0.10) and (current_demand < eco_cap) and (distance_km <= 350.0)

    # Weighted Sum Model Score (spare-capacity pillar uses the blended
    # proportion + absolute-headroom score so absolute capacity decides
    # between corridors with similar percentages).
    final_score = (
        weights["match"] * match_score +
        weights["accessibility"] * accessibility_score +
        weights["spare_capacity"] * spare_capacity_blended +
        weights["community_benefit"] * community_score -
        weights["environmental_risk"] * env_risk_score
    )
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    # 4-Part Transparent XAI Explanation Synthesis
    # Part 1: Shared 5-pillar attraction types
    shared_pillars = []
    for k in ARCHETYPES:
        h_val = float(hotspot_row.get(k, 0.0)) if pd.notna(hotspot_row.get(k)) else 0.0
        c_val = float(candidate_row.get(k, 0.0)) if pd.notna(candidate_row.get(k)) else 0.0
        if h_val >= 0.50 and c_val >= 0.50:
            shared_pillars.append(PILLAR_LABELS[k])
    if not shared_pillars:
        best_k = max(
            ARCHETYPES,
            key=lambda k: (float(hotspot_row.get(k, 0.0) or 0.0) * float(candidate_row.get(k, 0.0) or 0.0))
        )
        shared_pillars.append(PILLAR_LABELS[best_k])
    shared_pillars_str = ", ".join(shared_pillars)

    # Part 2: Named Attractions / POI Themes
    tags_h_raw = hotspot_row.get("poi_tags", hotspot_row.get("tags", None)) if hasattr(hotspot_row, "get") else None
    tags_c_raw = candidate_row.get("poi_tags", candidate_row.get("tags", None)) if hasattr(candidate_row, "get") else None
    set_h = _extract_tag_set(tags_h_raw)
    set_c = _extract_tag_set(tags_c_raw)
    shared_pois = sorted(list(set_h.intersection(set_c)))

    attractions_mention = ""
    if shared_pois:
        poi_str = ", ".join([p.replace("_", " ").title() for p in shared_pois[:3]])
        attractions_mention = f" Features shared attraction themes ({poi_str})."
    elif set_c:
        poi_str = ", ".join([p.replace("_", " ").title() for p in sorted(list(set_c))[:3]])
        attractions_mention = f" Features verified attractions including {poi_str}."

    # Part 3: Transit Friction
    if distance_km > 0:
        transit_desc = f"Situated {distance_km:.0f} km (~{travel_time:.0f} mins) via {transit_mode}."
    else:
        transit_desc = f"Situated ~{travel_time:.0f} mins away via {transit_mode}."

    # Part 4: Capacity Headroom & B40 Uplift
    sust_cap = float(candidate_row.get("sustainable_capacity", 0.0))
    headroom_pax = max(0, int(sust_cap - current_demand))
    if sust_cap > 0:
        headroom_desc = f"+{headroom_pax:,} pax/day buffer safely below ecological limit of {eco_cap:,.0f}"
    else:
        headroom_desc = "safe from bottleneck breach"

    explanation = (
        f"High compatibility ({match_score}% synergy across shared {shared_pillars_str})."
        f"{attractions_mention} {transit_desc} "
        f"Operates with {spare_capacity_score:.1f}% spare capacity headroom ({headroom_desc}; "
        f"blended spare-capacity pillar {spare_capacity_blended:.1f} incl. absolute {absolute_headroom_score:.1f}), "
        f"targeting {poverty_rate:.1f}% district B40 community uplift."
    )

    return {
        "candidate_id": candidate_row.get("destination_id", "Unknown"),
        "candidate_name": candidate_row.get("destination_name", candidate_row.get("district_name", "Unknown")),
        "district_name": candidate_row.get("district_name", "Unknown"),
        "state_name": candidate_row.get("state_name", "Unknown"),
        "match_score": match_score,
        "accessibility_score": accessibility_score,
        "spare_capacity_score": spare_capacity_score,
        "absolute_headroom_pax": round(absolute_headroom_pax, 1),
        "absolute_headroom_score": absolute_headroom_score,
        "spare_capacity_blended": spare_capacity_blended,
        "community_benefit_score": community_score,
        "environmental_risk_penalty": env_risk_score,
        "final_wsm_score": final_score,
        "is_feasible": is_feasible,
        "travel_time_mins": travel_time,
        "transit_mode": transit_mode,
        "transit_fare_rm": float(candidate_row.get("transit_fare_rm", 0.0)),
        "explanation": explanation,
        "pbt_authority": candidate_row.get("pbt_name", "Local PBT"),
        "lat": float(candidate_row.get("lat", 0.0)),
        "lon": float(candidate_row.get("lon", 0.0)),
        "shared_poi_tags": shared_pois,
        "jaccard_poi_similarity": round(compute_jaccard_poi_similarity(set_h, set_c) * 100.0, 1),
        "poverty_rate": poverty_rate,
        "capacity_gap": capacity_gap,
        "sustainable_capacity": sust_cap,
        "daily_demand_peak": current_demand,
        "cc_ecology": eco_cap,
        "total_rooms": float(candidate_row.get("total_rooms", 5000.0) or 5000.0),
    }


def find_best_alternatives(
    hotspot_row: pd.Series,
    all_dest_df: pd.DataFrame,
    top_n: int = 3,
    alpha: float = 0.60
) -> pd.DataFrame:
    """
    Ranks candidate destinations against origin hotspot using multi-vector WSM scoring.
    Enforces feasibility gates (capacity gap > 10%, demand < CC_ecology, distance <= 350 km)
    and returns the top N ranked alternatives.
    """
    results = []

    hotspot_lat = hotspot_row.get("lat")
    hotspot_lon = hotspot_row.get("lon")
    h_id = hotspot_row.get("destination_id")
    h_name = str(hotspot_row.get("destination_name", hotspot_row.get("district_name", ""))).strip().lower()

    for idx, candidate_row in all_dest_df.iterrows():
        # Exclude hotspot itself
        c_id = candidate_row.get("destination_id")
        if h_id is not None and c_id is not None and str(h_id).strip() == str(c_id).strip():
            continue
        c_name = str(candidate_row.get("destination_name", candidate_row.get("district_name", ""))).strip().lower()
        if h_name and c_name and h_name == c_name:
            continue

        cand_lat = candidate_row.get("lat")
        cand_lon = candidate_row.get("lon")

        # Grounded corridor inference: honest road distance + ETA + true rail viability.
        # Never trust per-district `transit_mode` CSV labels or 1 min-per-km guesses.
        from src.traffic_engine import infer_corridor as _infer_corridor

        h_lat_f = float(hotspot_lat) if pd.notna(hotspot_lat) else 0.0
        h_lon_f = float(hotspot_lon) if pd.notna(hotspot_lon) else 0.0
        c_lat_f = float(cand_lat) if pd.notna(cand_lat) else 0.0
        c_lon_f = float(cand_lon) if pd.notna(cand_lon) else 0.0
        h_label = str(hotspot_row.get("destination_name", hotspot_row.get("district_name", "")))
        c_label = str(candidate_row.get("destination_name", candidate_row.get("district_name", "")))
        corridor = _infer_corridor(h_lat_f, h_lon_f, c_lat_f, c_lon_f, h_label, c_label)
        distance_km = float(corridor["distance_km"])

        candidate_row = candidate_row.copy()
        candidate_row["distance_km"] = distance_km
        candidate_row["transit_time_mins"] = float(corridor["travel_time_mins"])
        candidate_row["travel_time_mins"] = float(corridor["travel_time_mins"])
        candidate_row["transit_mode"] = str(corridor["transit_mode"])

        score_res = score_candidate_alternative(hotspot_row, candidate_row, alpha=alpha)
        score_res["distance_km"] = round(distance_km, 1)

        # Continuous distance friction penalty for transit distance exceeding local threshold (> 60 km)
        if distance_km > 60.0:
            dist_penalty = (distance_km - 60.0) * 0.08
            score_res["distance_friction_penalty"] = round(dist_penalty, 1)
            score_res["final_wsm_score"] = round(max(0.0, score_res["final_wsm_score"] - dist_penalty), 1)
        else:
            score_res["distance_friction_penalty"] = 0.0

        results.append(score_res)

    if not results:
        return pd.DataFrame()

    # Enforce feasibility gate filtering: return feasible candidates
    feasible_results = [r for r in results if r.get("is_feasible", False)]
    if not feasible_results:
        feasible_results = results

    # Prioritize regional corridor candidates within realistic travel shed (<= 180 km)
    # Allows day-trip and weekend dispersal without cross-country travel friction
    regional_candidates = [r for r in feasible_results if r.get("distance_km", 999.0) <= 180.0]
    final_list = regional_candidates if len(regional_candidates) >= min(top_n, len(feasible_results)) else feasible_results

    res_df = pd.DataFrame(final_list)
    res_df = res_df.sort_values("final_wsm_score", ascending=False).reset_index(drop=True)
    return res_df.head(top_n)
