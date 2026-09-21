"""
DESTINASI — Highway Traffic & Route Grounding Engine (OSRM / Valhalla / Google)
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Route-finding tiers (first hit wins):
  Tier 0 (realtime, keyed) .... Google Maps Platform Directions API
      `duration_in_traffic` vs `duration`. Only path that knows current
      congestion. Use when agency needs live ops.
  Tier 1 (non-realtime ETA, free, no key) ... OSRM public demo
      Returns real road geometry + `distance` + `duration` (free-flow ETA,
      no live traffic). Ideal default for planning / datathon offline demo.
  Tier 2 (non-realtime ETA, free, no key) ... Valhalla public demo
      Same contract as OSRM, different routing graph. Used when OSRM is
      unreachable. Also yields distance + ETA + decoded polyline.
  Tier 3 (non-realtime ETA, keyed, optional) ... OpenRouteService / GraphHopper
      Enabled only if ORS_API_KEY / GRAPHHOPPER_API_KEY env is set.
  Tier 4 (offline fallback) ... heuristic circuity model + LLM/PLUS sensor
      colours. Never a straight line, always labelled as estimated.

  2. Lembaga Lebuhraya Malaysia (LLM) & PLUS Expressways Dynamic Sensor Model:
     Evaluates local clock rush-hours (07:30-09:30, 17:00-19:30), weekend holiday surges,
     and known highway bottleneck nodes (Penang Bridge E36, Menora Tunnel E1, Genting Sempah E8, Skudai Toll E2).
     This colours the route (green/amber/red) when no live-Google colour exists.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import math
import os
import sys
import urllib.parse
import urllib.request
import json
import logging

logger = logging.getLogger("traffic_engine")

__all__ = [
    "get_corridor_traffic_profile",
    "decode_polyline",
    "query_google_maps_traffic",
    "calculate_llm_plus_traffic_congestion",
    "query_osrm_route",
    "query_osrm_route_full",
    "query_valhalla_route",
    "query_openrouteservice_route",
    "generate_dense_corridor_curve",
    "haversine_km",
    "nearest_rail_station",
    "is_rail_viable",
    "estimate_road_distance_time",
    "infer_corridor",
    "build_rail_path_coords",
    "BOTTLENECK_NODES",
    "KTM_ETS_STATIONS",
    "KTM_JUNGLE_STATIONS",
]

BOTTLENECK_NODES = {
    "penang_bridge": {
        "highway": "PLUS E36 / Penang Bridge",
        "normal_speed_kmh": 80,
        "peak_speed_kmh": 35,
        "delay_multiplier": 1.45,
        "description": "PLUS E36 Penang Bridge westbound/eastbound bottleneck"
    },
    "menora_tunnel": {
        "highway": "PLUS E1 (Ipoh - Kuala Kangsar / Menora Tunnel)",
        "normal_speed_kmh": 90,
        "peak_speed_kmh": 40,
        "delay_multiplier": 1.50,
        "description": "PLUS E1 KM260 Menora Tunnel steep gradient congestion"
    },
    "genting_sempah": {
        "highway": "KL-Karak Expressway (E8 / Genting Sempah Tunnel)",
        "normal_speed_kmh": 80,
        "peak_speed_kmh": 30,
        "delay_multiplier": 1.65,
        "description": "LPT / KL-Karak Genting Sempah weekend holiday crawl"
    },
    "skudai_toll": {
        "highway": "PLUS E2 (Skudai - Senai - JB)",
        "normal_speed_kmh": 90,
        "peak_speed_kmh": 45,
        "delay_multiplier": 1.35,
        "description": "PLUS E2 southern industrial & cross-border commuter peak"
    },
    "rawang_bypass": {
        "highway": "Federal Route 1 / Rawang Bypass",
        "normal_speed_kmh": 70,
        "peak_speed_kmh": 38,
        "delay_multiplier": 1.30,
        "description": "Northern Klang Valley arterial congestion"
    }
}


def decode_polyline(polyline_str: str) -> List[List[float]]:
    """
    Decodes a Google Maps encoded polyline into a list of [lat, lon] coordinates.
    Pure Python implementation of Google's Encoded Polyline Algorithm Format.
    """
    if not polyline_str:
        return []

    coordinates = []
    index = 0
    lat = 0
    lng = 0
    length = len(polyline_str)

    while index < length:
        shift = 0
        result = 0
        while index < length:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while index < length:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append([round(lat / 1e5, 5), round(lng / 1e5, 5)])

    return coordinates


def query_google_maps_traffic(
    origin_coords: List[float],
    dest_coords: List[float],
    api_key: str
) -> Optional[Dict[str, Any]]:
    if not api_key or api_key.strip() == "":
        return None

    try:
        orig_str = f"{origin_coords[0]},{origin_coords[1]}"
        dest_str = f"{dest_coords[0]},{dest_coords[1]}"
        url = (
            f"https://maps.googleapis.com/maps/api/directions/json?"
            f"origin={orig_str}&destination={dest_str}&departure_time=now&traffic_model=best_guess&key={api_key}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "DESTINASI-Tourism-System/1.0"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") == "OK" and len(data.get("routes", [])) > 0:
            route = data["routes"][0]
            leg = route["legs"][0]
            norm_dur = leg.get("duration", {}).get("value", 3600)
            traffic_dur = leg.get("duration_in_traffic", {}).get("value", norm_dur)
            delay_factor = traffic_dur / max(1, norm_dur)
            delay_mins = max(0, (traffic_dur - norm_dur) // 60)

            overview_polyline = route.get("overview_polyline", {}).get("points", "")
            polyline_coords = decode_polyline(overview_polyline)

            if delay_factor >= 1.35:
                status = "Severe Gridlock"
                color = [239, 68, 68, 240]
            elif delay_factor >= 1.15:
                status = "Moderate Congestion"
                color = [245, 158, 11, 240]
            else:
                status = "Free-Flow Traffic"
                color = [16, 185, 129, 240]

            return {
                "source": "Google Maps Platform (Live API)",
                "delay_factor": round(delay_factor, 2),
                "delay_mins": delay_mins,
                "status": status,
                "color_rgb": color,
                "duration_mins": traffic_dur // 60,
                "distance_km": round(leg.get("distance", {}).get("value", 50000) / 1000.0, 1),
                "polyline_coords": polyline_coords,
                "summary": f"Google Live: {delay_mins}m delay ({status})"
            }
    except Exception as e:
        logger.warning("Google Maps traffic query failed: %s. Using LLM/PLUS fallback.", e)

    return None


def calculate_llm_plus_traffic_congestion(
    origin_name: str,
    dest_name: str,
    travel_time_nominal_mins: float,
    current_dt: Optional[datetime] = None
) -> Dict[str, Any]:
    if current_dt is None:
        current_dt = datetime.now()

    hour = current_dt.hour + current_dt.minute / 60.0
    weekday = current_dt.weekday()

    is_morning_rush = (7.5 <= hour <= 9.5)
    is_evening_rush = (17.0 <= hour <= 19.5)
    is_friday_outflow = (weekday == 4 and hour >= 16.5)
    is_sunday_return = (weekday == 6 and 15.0 <= hour <= 21.0)

    base_multiplier = 1.00
    if is_morning_rush:
        base_multiplier = 1.28
    elif is_evening_rush:
        base_multiplier = 1.34
    elif is_friday_outflow:
        base_multiplier = 1.42
    elif is_sunday_return:
        base_multiplier = 1.40
    elif 11.0 <= hour <= 15.0:
        base_multiplier = 1.08

    pair_str = f"{origin_name.lower()} {dest_name.lower()}"
    matched_bottleneck = None
    bottleneck_boost = 0.0

    if "timur laut" in pair_str or "george town" in pair_str or "seberang" in pair_str:
        matched_bottleneck = BOTTLENECK_NODES["penang_bridge"]
        bottleneck_boost = 0.18 if (is_morning_rush or is_evening_rush or is_friday_outflow) else 0.06
    elif "cameron" in pair_str or "lipis" in pair_str or "raub" in pair_str or "bentong" in pair_str:
        matched_bottleneck = BOTTLENECK_NODES["genting_sempah"]
        bottleneck_boost = 0.25 if (weekday in [4, 5, 6]) else 0.05
    elif "kinta" in pair_str or "ipoh" in pair_str or "kuala kangsar" in pair_str or "taiping" in pair_str:
        matched_bottleneck = BOTTLENECK_NODES["menora_tunnel"]
        bottleneck_boost = 0.16 if (is_evening_rush or is_sunday_return) else 0.04
    elif "johor" in pair_str or "kulai" in pair_str or "singapore" in pair_str:
        matched_bottleneck = BOTTLENECK_NODES["skudai_toll"]
        bottleneck_boost = 0.15 if (is_morning_rush or is_evening_rush) else 0.04

    effective_delay_factor = min(1.85, base_multiplier + bottleneck_boost)
    delay_mins = int(round(travel_time_nominal_mins * (effective_delay_factor - 1.0)))

    if effective_delay_factor >= 1.32:
        status = "Severe Gridlock"
        color = [239, 68, 68, 240]
    elif effective_delay_factor >= 1.14:
        status = "Moderate Congestion"
        color = [245, 158, 11, 240]
    else:
        status = "Free-Flow Traffic"
        color = [16, 185, 129, 240]

    b_desc = matched_bottleneck["description"] if matched_bottleneck else "Federal Highway Trunk"

    return {
        "source": "LLM / PLUS Expressway Sensor Model",
        "delay_factor": round(effective_delay_factor, 2),
        "delay_mins": max(0, delay_mins),
        "status": status,
        "color_rgb": color,
        "bottleneck_node": b_desc,
        "duration_mins": int(round(travel_time_nominal_mins * effective_delay_factor)),
        "summary": f"{status} (+{delay_mins}m via {b_desc})"
    }


_ROUTE_GEOMETRY_CACHE: Dict[str, List[List[float]]] = {}
_ROUTE_FULL_CACHE: Dict[str, Dict[str, Any]] = {}


def _offline_test_guard() -> bool:
    """True when running under unit tests without explicit force flag."""
    return os.environ.get("DESTINASI_TEST_MODE") == "1" or (
        "unittest" in sys.modules and os.environ.get("DESTINASI_FORCE_OSRM") != "1"
    )


def query_osrm_route_full(
    origin_coords: List[float],
    dest_coords: List[float],
    timeout_sec: float = 2.5,
) -> Optional[Dict[str, Any]]:
    """
    OSRM driving route with real geometry + non-realtime distance/ETA.
    Free, no key. ETA is free-flow (no live traffic) — for live ops use Google.
    Returns {coords, distance_km, duration_mins, source} or None.
    """
    if not origin_coords or not dest_coords or len(origin_coords) < 2 or len(dest_coords) < 2:
        return None
    orig_lat, orig_lon = float(origin_coords[0]), float(origin_coords[1])
    dest_lat, dest_lon = float(dest_coords[0]), float(dest_coords[1])
    cache_key = f"osrm:{round(orig_lat, 4)},{round(orig_lon, 4)}->{round(dest_lat, 4)},{round(dest_lon, 4)}"
    if cache_key in _ROUTE_FULL_CACHE:
        return _ROUTE_FULL_CACHE[cache_key]
    if _offline_test_guard():
        return None
    try:
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{orig_lon:.5f},{orig_lat:.5f};{dest_lon:.5f},{dest_lat:.5f}"
            f"?overview=full&geometries=geojson"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "DESTINASI-Tourism-Routing/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            routes = data.get("routes") or []
            if routes:
                r0 = routes[0]
                coords = r0["geometry"]["coordinates"]
                lat_lon = [[round(float(c[1]), 5), round(float(c[0]), 5)] for c in coords]
                if len(lat_lon) >= 2:
                    res = {
                        "coords": lat_lon,
                        "distance_km": round(float(r0.get("distance", 0)) / 1000.0, 1),
                        "duration_mins": max(1, int(round(float(r0.get("duration", 0)) / 60.0))),
                        "source": "OSRM (non-realtime road ETA)",
                    }
                    _ROUTE_FULL_CACHE[cache_key] = res
                    _ROUTE_GEOMETRY_CACHE[f"{round(orig_lat, 4)},{round(orig_lon, 4)}->{round(dest_lat, 4)},{round(dest_lon, 4)}"] = lat_lon
                    return res
    except Exception as e:
        logger.debug("OSRM route query skipped or offline: %s", e)
    return None


def query_osrm_route(
    origin_coords: List[float],
    dest_coords: List[float],
    timeout_sec: float = 2.5
) -> Optional[List[List[float]]]:
    """
    Backwards-compatible geometry-only wrapper around query_osrm_route_full.
    Queries Open Source Routing Machine (OSRM) driving API to retrieve genuine
    road polylines following real Malaysian expressway and arterial alignments.
    Returns list of [lat, lon] coordinates.
    Cached in memory for instantaneous sub-millisecond execution.
    """
    res = query_osrm_route_full(origin_coords, dest_coords, timeout_sec=timeout_sec)
    return res["coords"] if res else None


def query_valhalla_route(
    origin_coords: List[float],
    dest_coords: List[float],
    timeout_sec: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """
    Valhalla public demo (OSM graph) — free, no key, non-realtime ETA.
    Second fallback when OSRM is unreachable. Returns same contract as OSRM.
    """
    if not origin_coords or not dest_coords or len(origin_coords) < 2 or len(dest_coords) < 2:
        return None
    if _offline_test_guard():
        return None
    try:
        o_lat, o_lon = float(origin_coords[0]), float(origin_coords[1])
        d_lat, d_lon = float(dest_coords[0]), float(dest_coords[1])
        payload = json.dumps({
            "locations": [
                {"lat": o_lat, "lon": o_lon},
                {"lat": d_lat, "lon": d_lon},
            ],
            "costing": "auto",
            "directions_options": {"units": "kilometers"},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://valhalla1.openstreetmap.de/route",
            data=payload,
            headers={"User-Agent": "DESTINASI-Tourism-Routing/1.0", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        trip = data.get("trip") or {}
        legs = trip.get("legs") or []
        if not legs:
            return None
        shape = trip.get("shape") or legs[0].get("shape") or ""
        coords = decode_polyline(shape) if shape else []
        summary = trip.get("summary") or {}
        dist = float(summary.get("length", 0))  # km already
        dur_mins = int(round(float(summary.get("time", 0)) / 60.0))
        if not coords or len(coords) < 2:
            return None
        return {
            "coords": coords,
            "distance_km": round(dist, 1),
            "duration_mins": max(1, dur_mins),
            "source": "Valhalla (non-realtime road ETA)",
        }
    except Exception as e:
        logger.debug("Valhalla route query skipped or offline: %s", e)
    return None


def query_openrouteservice_route(
    origin_coords: List[float],
    dest_coords: List[float],
    api_key: Optional[str] = None,
    timeout_sec: float = 4.0,
) -> Optional[Dict[str, Any]]:
    """
    OpenRouteService driving route — non-realtime ETA, requires ORS_API_KEY.
    Enabled only when key is present (env ORS_API_KEY or passed explicitly).
    """
    key = (api_key or os.environ.get("ORS_API_KEY") or "").strip()
    if not key:
        return None
    if not origin_coords or not dest_coords or len(origin_coords) < 2 or len(dest_coords) < 2:
        return None
    try:
        o_lat, o_lon = float(origin_coords[0]), float(origin_coords[1])
        d_lat, d_lon = float(dest_coords[0]), float(dest_coords[1])
        payload = json.dumps({"coordinates": [[o_lon, o_lat], [d_lon, d_lat]]}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openrouteservice.org/v2/directions/driving-car",
            data=payload,
            headers={"Authorization": key, "Content-Type": "application/json",
                     "User-Agent": "DESTINASI-Tourism-Routing/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        routes = data.get("routes") or []
        if not routes:
            return None
        r0 = routes[0]
        summary = r0.get("summary") or {}
        geom = (r0.get("geometry") or "")
        coords = decode_polyline(geom) if isinstance(geom, str) and geom else []
        if not coords or len(coords) < 2:
            return None
        return {
            "coords": coords,
            "distance_km": round(float(summary.get("distance", 0)) / 1000.0, 1),
            "duration_mins": max(1, int(round(float(summary.get("duration", 0)) / 60.0))),
            "source": "OpenRouteService (non-realtime road ETA)",
        }
    except Exception as e:
        logger.debug("ORS route query failed: %s", e)
    return None


def generate_dense_corridor_curve(
    origin_coords: List[float],
    dest_coords: List[float],
    num_steps: int = 25
) -> List[List[float]]:
    """
    Generates a realistic smooth terrain-following curve when offline or when
    API routing is unavailable, replacing crude straight-line cuts.
    """
    o_lat, o_lon = float(origin_coords[0]), float(origin_coords[1])
    d_lat, d_lon = float(dest_coords[0]), float(dest_coords[1])

    # Add gentle geographic curvature bowing towards the coastal arterial spine
    mid_lat = (o_lat + d_lat) / 2.0
    mid_lon = (o_lon + d_lon) / 2.0
    dx = d_lon - o_lon
    dy = d_lat - o_lat
    # Normal vector perpendicular to chord
    length = max(1e-4, (dx**2 + dy**2)**0.5)
    offset = 0.08 * (1.0 if dx >= 0 else -1.0)
    ctrl_lat = mid_lat - (dx / length) * offset
    ctrl_lon = mid_lon + (dy / length) * offset

    # Quadratic Bezier interpolation
    points = []
    for i in range(num_steps + 1):
        t = i / float(num_steps)
        lat = (1 - t)**2 * o_lat + 2 * (1 - t) * t * ctrl_lat + t**2 * d_lat
        lon = (1 - t)**2 * o_lon + 2 * (1 - t) * t * ctrl_lon + t**2 * d_lon
        points.append([round(lat, 5), round(lon, 5)])
    return points


def get_corridor_traffic_profile(
    origin_coords: List[float],
    dest_coords: List[float],
    origin_name: str,
    dest_name: str,
    travel_time_nominal_mins: float = 60.0,
    google_maps_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tiered route + traffic resolution:
      Tier 0 — Google Maps Directions (realtime duration_in_traffic). Only when
        google_maps_api_key is supplied. This is the live-ops path.
      Tier 1 — OSRM free road geometry + non-realtime ETA (distance/duration).
      Tier 2 — Valhalla free fallback with same contract.
      Tier 3 — OpenRouteService when ORS_API_KEY is set.
      Tier 4 — Heuristic circuity estimate + LLM/PLUS congestion colour.
    Non-Google tiers combine the router's free-flow ETA with the LLM/PLUS
    delay factor for colouring, and label eta_source explicitly so the UI can
    say "non-realtime" vs "live".
    """
    # Tier 0. Google Maps Platform Live API (realtime)
    if google_maps_api_key:
        live_res = query_google_maps_traffic(origin_coords, dest_coords, google_maps_api_key)
        if live_res is not None:
            live_res["eta_source"] = "live"
            live_res["eta_label"] = "Google live (realtime)"
            return live_res

    # Tiers 1-3. Free/keyed routers for geometry + non-realtime base ETA
    routed = (
        query_osrm_route_full(origin_coords, dest_coords)
        or query_valhalla_route(origin_coords, dest_coords)
        or query_openrouteservice_route(origin_coords, dest_coords)
    )
    base_mins = float(routed["duration_mins"]) if routed else float(travel_time_nominal_mins)
    base_km = float(routed["distance_km"]) if routed else None

    # Dynamic LLM / PLUS Expressway Sensor Model (colour + delay on top of base)
    profile = calculate_llm_plus_traffic_congestion(
        origin_name=origin_name,
        dest_name=dest_name,
        travel_time_nominal_mins=base_mins
    )
    profile["eta_source"] = "non-realtime" if routed else "estimated"
    profile["eta_label"] = (
        f"{routed['source']}" if routed
        else "Heuristic estimate (offline fallback)"
    )
    if base_km is not None:
        profile["distance_km"] = base_km

    if routed and routed.get("coords") and len(routed["coords"]) >= 2:
        profile["polyline_coords"] = routed["coords"]
        profile["source"] = f"{profile['source']} + {routed['source']}"
    else:
        # Fallback to smooth geographic curve
        profile["polyline_coords"] = generate_dense_corridor_curve(origin_coords, dest_coords)

    return profile


# -----------------------------------------------------------------------------
# Grounded corridor inference: honest road distance / ETA + true rail viability
# -----------------------------------------------------------------------------
# Real KTM station coordinates (ETS West Coast + Jungle Line). Used to decide
# whether a hotspot->relief pair can honestly claim "via KTM Rail".
KTM_ETS_STATIONS: List[Tuple[str, float, float]] = [
    ("Padang Besar", 6.65, 100.20),
    ("Arau", 6.43, 100.27),
    ("Alor Setar", 6.12, 100.37),
    ("Gurun", 5.82, 100.48),
    ("Sungai Petani", 5.64, 100.49),
    ("Butterworth", 5.39, 100.37),
    ("Bukit Mertajam", 5.36, 100.46),
    ("Nibong Tebal", 5.17, 100.48),
    ("Taiping", 4.85, 100.73),
    ("Kuala Kangsar", 4.77, 100.94),
    ("Ipoh", 4.60, 101.09),
    ("Batu Gajah", 4.48, 101.04),
    ("Kampar", 4.31, 101.15),
    ("Tapah Road", 4.20, 101.27),
    ("Tanjung Malim", 3.68, 101.52),
    ("Rawang", 3.32, 101.58),
    ("KL Sentral", 3.14, 101.69),
    ("Kajang", 3.00, 101.79),
    ("Seremban", 2.73, 101.94),
    ("Tampin", 2.47, 102.23),
    ("Gemas", 2.58, 102.61),
]

KTM_JUNGLE_STATIONS: List[Tuple[str, float, float]] = [
    ("Gemas", 2.58, 102.61),
    ("Bahau", 2.81, 102.41),
    ("Triang", 3.23, 102.43),
    ("Mentakab", 3.48, 102.35),
    ("Jerantut", 3.93, 102.36),
    ("Kuala Lipis", 4.18, 102.05),
    ("Gua Musang", 4.88, 101.97),
    ("Dabong", 5.38, 102.01),
    ("Kuala Krai", 5.53, 102.20),
    ("Tanah Merah", 5.81, 102.15),
    ("Pasir Mas", 6.04, 102.14),
    ("Wakaf Bharu", 6.16, 102.23),
    ("Tumpat", 6.20, 102.17),
]

# Districts with no rail access: winding mountain roads, much slower speeds.
MOUNTAIN_KEYWORDS = (
    "cameron", "genting", "fraser", "kundasang", "ranau",
    "kinabalu", "crocker", "banjaran titiwangsa",
)

RAIL_ACCESS_RADIUS_KM = 25.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return float(r * 2.0 * math.asin(min(1.0, max(0.0, math.sqrt(a)))))


def _is_mountain(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in MOUNTAIN_KEYWORDS)


def nearest_rail_station(
    lat: float, lon: float
) -> Optional[Dict[str, Any]]:
    """Finds nearest KTM station across both lines. Returns None if beyond access radius."""
    best = None
    for line, stations in (("ets", KTM_ETS_STATIONS), ("jungle", KTM_JUNGLE_STATIONS)):
        for s_name, s_lat, s_lon in stations:
            d = haversine_km(lat, lon, s_lat, s_lon)
            if best is None or d < best["dist_km"]:
                best = {
                    "station_name": s_name,
                    "station_lat": s_lat,
                    "station_lon": s_lon,
                    "dist_km": d,
                    "line": line,
                }
    if best is None or best["dist_km"] > RAIL_ACCESS_RADIUS_KM:
        return None
    return best


def is_rail_viable(
    o_lat: float, o_lon: float, d_lat: float, d_lon: float,
    o_name: str = "", d_name: str = "",
    min_pair_km: float = 15.0,
) -> Dict[str, Any]:
    """
    Determines whether a corridor can honestly be labelled as rail.
    Requires BOTH ends within RAIL_ACCESS_RADIUS_KM of a station on a
    compatible line. Cameron Highlands / Genting (mountain, no station
    nearby) therefore correctly resolves to road, not rail.
    """
    o_st = nearest_rail_station(o_lat, o_lon)
    d_st = nearest_rail_station(d_lat, d_lon)
    straight = haversine_km(o_lat, o_lon, d_lat, d_lon)
    if o_st is None or d_st is None:
        return {"viable": False, "reason": "one or both ends beyond rail access radius",
                "origin_station": o_st, "dest_station": d_st, "straight_km": straight}
    if straight < min_pair_km:
        return {"viable": False, "reason": "pair too short for intercity rail claim",
                "origin_station": o_st, "dest_station": d_st, "straight_km": straight}
    # Same line, or both touch Gemas interchange (ETS<->Jungle transfer)
    same_line = o_st["line"] == d_st["line"]
    via_gemas = "gemas" in (o_st["station_name"].lower(), d_st["station_name"].lower()) or (
        {o_st["line"], d_st["line"]} == {"ets", "jungle"}
    )
    # Allow cross-line only via Gemas with explicit transfer label; still viable
    # but slower. Mountain endpoints already excluded by radius check above.
    if not (same_line or via_gemas):
        return {"viable": False, "reason": "incompatible rail lines",
                "origin_station": o_st, "dest_station": d_st, "straight_km": straight}
    return {"viable": True, "reason": "both ends rail-accessible",
            "origin_station": o_st, "dest_station": d_st, "straight_km": straight,
            "via_gemas_transfer": (not same_line)}


def estimate_road_distance_time(
    o_lat: float, o_lon: float, d_lat: float, d_lon: float,
    o_name: str = "", d_name: str = "",
) -> Dict[str, Any]:
    """
    Honest road estimates: Haversian straight line is NEVER shown as driving
    distance. Applies circuity + terrain-aware speed so mountain corridors
    (e.g. Cameron Highlands) report ~55-60 km / ~95-110 min instead of
    the old bogus '33 km / 33 min'.
    """
    straight = haversine_km(o_lat, o_lon, d_lat, d_lon)
    mountain = _is_mountain(o_name) or _is_mountain(d_name)
    if mountain:
        circuity, speed = 1.75, 38.0
        road_label = "Federal Route 59 / Mountain Road"
    elif straight > 120:
        circuity, speed = 1.32, 70.0
        road_label = "PLUS Expressway & Federal Road"
    elif straight > 60:
        circuity, speed = 1.35, 60.0
        road_label = "Federal Road & Express Coach"
    elif straight > 25:
        circuity, speed = 1.30, 50.0
        road_label = "Federal Road & Express Coach"
    else:
        circuity, speed = 1.25, 45.0
        road_label = "Federal / State Road"
    road_km = straight * circuity
    mins = road_km / speed * 60.0 + 8.0  # towns, junctions, lights
    return {
        "straight_km": round(straight, 1),
        "road_km": round(road_km, 1),
        "time_mins": max(15.0, round(mins)),
        "speed_kmh": speed,
        "road_label": road_label,
        "mountain": mountain,
    }


def infer_corridor(
    o_lat: float, o_lon: float, d_lat: float, d_lon: float,
    o_name: str = "", d_name: str = "",
) -> Dict[str, Any]:
    """
    Single source of truth for corridor cards + map tooltips.
    Returns road distance, honest ETA, truthful mode label, and rail detail.
    """
    road = estimate_road_distance_time(o_lat, o_lon, d_lat, d_lon, o_name, d_name)
    rail_check = is_rail_viable(o_lat, o_lon, d_lat, d_lon, o_name, d_name)
    if rail_check["viable"]:
        track_km = rail_check["straight_km"] * 1.25  # rail straighter than road
        rail_mins = track_km / 75.0 * 60.0 + 18.0  # access + stops
        if rail_check.get("via_gemas_transfer"):
            rail_mins += 30.0
            mode = "KTM ETS + Jungle Rail (via Gemas transfer)"
        else:
            mode = "KTM ETS / Komuter Rail" if rail_check["origin_station"]["line"] == "ets" else "KTM Jungle Railway"
        return {
            "distance_km": road["road_km"],
            "travel_time_mins": max(20.0, round(rail_mins)),
            "transit_mode": mode,
            "rail_viable": True,
            "rail_detail": rail_check,
            "road_detail": road,
        }
    mountain_road_modes = {
        True: "Federal Route 59 / Mountain Road (no rail link)",
        False: road["road_label"],
    }
    return {
        "distance_km": road["road_km"],
        "travel_time_mins": road["time_mins"],
        "transit_mode": mountain_road_modes[road["mountain"]] if road["mountain"] else road["road_label"],
        "rail_viable": False,
        "rail_detail": rail_check,
        "road_detail": road,
    }


def build_rail_path_coords(
    o_lat: float, o_lon: float, d_lat: float, d_lon: float,
    rail_detail: Optional[Dict[str, Any]] = None,
) -> Optional[List[List[float]]]:
    """
    Builds a rail-shaped polyline: origin -> origin station -> intermediate
    track stations -> dest station -> destination. Follows the actual track
    alignment instead of a straight chord.
    """
    if rail_detail is None:
        rail_check = is_rail_viable(o_lat, o_lon, d_lat, d_lon)
        if not rail_check["viable"]:
            return None
        rail_detail = rail_check
    elif not rail_detail.get("viable"):
        return None
    o_st = rail_detail["origin_station"]
    d_st = rail_detail["dest_station"]
    line = o_st["line"] if o_st["line"] == d_st["line"] else None
    if line is None:
        # Cross-line via Gemas: concatenate ETS leg + Jungle leg through Gemas
        ets = {s[0]: s for s in KTM_ETS_STATIONS}
        jgl = {s[0]: s for s in KTM_JUNGLE_STATIONS}
        path = [[o_lat, o_lon], [o_st["station_lat"], o_st["station_lon"]]]
        if o_st["line"] == "ets":
            seq_a, seq_b = KTM_ETS_STATIONS, KTM_JUNGLE_STATIONS
            a_from, b_to = o_st["station_name"], d_st["station_name"]
        else:
            seq_a, seq_b = KTM_JUNGLE_STATIONS, KTM_ETS_STATIONS
            a_from, b_to = o_st["station_name"], d_st["station_name"]
        names_a = [s[0] for s in seq_a]
        names_b = [s[0] for s in seq_b]
        try:
            i0, i1 = names_a.index(a_from), names_a.index("Gemas")
            step = 1 if i1 >= i0 else -1
            for n in names_a[i0 + step:i1 + step:step]:
                s = seq_a[names_a.index(n)]
                path.append([s[1], s[2]])
            j0, j1 = names_b.index("Gemas"), names_b.index(b_to)
            step = 1 if j1 >= j0 else -1
            for n in names_b[j0 + step:j1 + step:step] if j0 != j1 else []:
                s = seq_b[names_b.index(n)]
                path.append([s[1], s[2]])
        except ValueError:
            pass
        path.append([d_st["station_lat"], d_st["station_lon"]])
        path.append([d_lat, d_lon])
        return path
    stations = KTM_ETS_STATIONS if line == "ets" else KTM_JUNGLE_STATIONS
    names = [s[0] for s in stations]
    try:
        i0, i1 = names.index(o_st["station_name"]), names.index(d_st["station_name"])
    except ValueError:
        return None
    step = 1 if i1 >= i0 else -1
    path = [[round(o_lat, 5), round(o_lon, 5)]]
    path.append([o_st["station_lat"], o_st["station_lon"]])
    for n in names[i0 + step:i1 + step:step]:
        s = stations[names.index(n)]
        # Skip duplicating endpoints already added
        if [s[1], s[2]] != path[-1]:
            path.append([s[1], s[2]])
    if [round(d_lat, 5), round(d_lon, 5)] != path[-1]:
        if [d_st["station_lat"], d_st["station_lon"]] != path[-1] and [d_st["station_lat"], d_st["station_lon"]] != [round(d_lat, 5), round(d_lon, 5)]:
            path.append([d_st["station_lat"], d_st["station_lon"]])
        path.append([round(d_lat, 5), round(d_lon, 5)])
    # Remove consecutive duplicates (e.g. district centroid == station)
    deduped = [path[0]]
    for pt in path[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped

