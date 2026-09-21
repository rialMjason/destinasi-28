"""
DESTINASI — 3D WebGL Elevation Cockpit & Ecological Map Engine
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Features:
1. 3D WebGL Elevation Cockpit (PyDeck): High-immersion 3D terrain and district
   elevation visualization.
2. Realistic highway and railway waypoint routing across all Peninsular & East Malaysia corridors.
3. KSAS Ecological Buffer Zones rendered natively in 3D (PLANMalaysia / DOE
   Class 1 slope & catchment protection) as flat translucent disks + labels.
4. Dynamic Level-of-Detail (LOD) aggregation across 110 districts and 16 state centroids.
"""

from pathlib import Path
import json
import math
import re
import warnings
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import pydeck as pdk
import sys
import base64

import src.traffic_engine as traffic_engine
from src.traffic_engine import (
    get_corridor_traffic_profile,
    query_osrm_route,
    generate_dense_corridor_curve,
    build_rail_path_coords,
    is_rail_viable,
    infer_corridor,
)

def get_3d_deck_legend_html(visible_layers: Optional[Any] = None, alert_zones: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Returns HTML for the 3D PyDeck Elevation Cockpit interactive legend HUD.
    Reflects active layers, load thresholds, and KSAS ecological alerts.
    """
    has_alerts = bool(alert_zones and len(alert_zones) > 0)
    alert_badge = (
        '<div style="display: flex; align-items: center; gap: 6px;">'
        '<span style="display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 50%; box-shadow: 0 0 6px #ef4444;"></span>'
        '<span style="color: #fca5a5; font-weight: 700;">KSAS Ecological Alert</span>'
        '</div>'
    ) if has_alerts else (
        '<div style="display: flex; align-items: center; gap: 6px;">'
        '<span style="display: inline-block; width: 10px; height: 10px; background: #2dd4bf; border-radius: 50%;"></span>'
        '<span style="color: #cbd5e1;">KSAS Eco Buffer</span>'
        '</div>'
    )
    return f"""
    <div style="background: rgba(15, 23, 42, 0.90); border: 1px solid #334155; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 11px; color: #e2e8f0; display: flex; flex-wrap: wrap; gap: 14px; align-items: center; justify-content: space-between; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 2px; box-shadow: 0 0 6px #ef4444;"></span>
            <span style="color: #fca5a5; font-weight: 600;">Active Hotspot</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; background: #10b981; border-radius: 2px;"></span>
            <span>#1 Relief Corridor</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; background: #0ea5e9; border-radius: 2px;"></span>
            <span>#2 Relief Corridor</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; background: #f59e0b; border-radius: 2px;"></span>
            <span>#3 Relief Corridor</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px; border-left: 1px solid #475569; padding-left: 10px;">
            <span style="color: #94a3b8;">Load:</span>
            <span><span style="color: #f97316; font-weight: bold;">●</span> Acute (&ge;100%)</span>
            <span><span style="color: #fbbf24; font-weight: bold;">●</span> Stressed (80-100%)</span>
            <span><span style="color: #f0f9ff; font-weight: bold;">●</span> Headroom (&lt;80%)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px; border-left: 1px solid #475569; padding-left: 10px;">
            <span style="color: #cbd5e1;">Real-World Route</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px; border-left: 1px solid #475569; padding-left: 10px;">
            {alert_badge}
        </div>
    </div>
    """


__all__ = [
    "render_pydeck_3d_elevation_map",
    "aggregate_destinations_by_state",
    "get_3d_deck_legend_html",
    "get_ksas_ecological_zones",
    "get_ksas_watchlist",
    "build_ksas_pydeck_layers",
    "load_ksas_reference",
    "get_ksas_overlay_image_uri",
    "get_ksas_borneo_image_uri",
    "HIGHWAY_NETWORK",
    "KSAS_ECOLOGICAL_ZONES",
    "KSAS_TAHAP_COLORS",
    "MAP_TOGGLE_LAYERS",
    "AUTO_MACRO_ZOOM_MAX",
    "CYLINDER_HIDE_ZOOM",
]


# KSAS Ecological Buffer Zones — grounded visualization proxies, NOT cadastral boundaries.
#
# Honest methodology (see tests/test_ksas_grounding.py for enforcement):
# - True KSAS limits are polygons from PLANMalaysia RFN / state structure plans and
#   local plans (e.g. Cameron Highlands District Local Plan 2030), gazetted reserve
#   notifications, and Sabah Parks / PERHILITAN / Perak Forestry Dept management plans.
#   No open polygon shapefile ships in this repo, so each zone renders as a circular
#   node disk centred on a VERIFIED anchor coordinate (town / published study point /
#   summit / park gateway — never invented).
# - radius_m = circle-equivalent of the zone's published gazetted/reference area,
#   r = sqrt(area_km2 / pi) * 1000, EXCEPT Taman Negara where the disk marks the
#   Kuala Tahan gateway catchment node (whole-park equiv ~37 km would swallow the map).
# - Every zone carries legal/source/gazetted_area_km2/radius_basis/representation fields
#   so any consumer can audit the claim. Area-derived disks must match their gazetted
#   area within 10% (pi*r^2 vs gazetted_area_km2).
KSAS_ECOLOGICAL_ZONES = [
    {
        "key": "cameron_highlands",
        "name": "Cameron Highlands KSAS Class 1 (Mossy Forest & Slopes >25°)",
        "short_label": "Cameron KSAS",
        "lat": 4.4735,  # Tanah Rata town anchor (DST_043; town sits ~4.4727, 101.3772)
        "lon": 101.3789,
        "radius_m": 15000,  # equiv of ~712 km2 district-scale highland zone (sqrt(712/pi)*1000 ≈ 15050)
        "gazetted_area_km2": 712.0,
        "area_derived": True,
        "radius_basis": "circle-equivalent of district-scale highland zone (~712 km2)",
        "legal": "Cameron Highlands District Local Plan 2030 + Pahang Structure Plan; "
                 "highland >1,000 m & slopes >25 deg KSAS rationale (PLANMalaysia RFN / Act 172 framework)",
        "source": "DST_043 anchor (Tanah Rata town); Cameron Highlands district area ~712 km2",
        "representation": "district-scale circular proxy — NOT the gazetted KSAS polygon",
    },
    {
        "key": "matang_mangrove",
        "name": "Matang Mangrove Forest Reserve (Perak) — Permanent Forest Reserve",
        "short_label": "Matang KSAS",
        "lat": 4.8342,  # published Virgin Jungle Reserve study point 4°50'3"N (PLOS ONE)
        "lon": 100.6167,  # published study point 100°37'0"E (PLOS ONE)
        "radius_m": 11500,  # equiv of 402.9 km2 reserve (sqrt(402.9/pi)*1000 ≈ 11330)
        "gazetted_area_km2": 402.9,
        "area_derived": True,
        "radius_basis": "circle-equivalent of 40,288 ha reserve",
        "legal": "Gazetted 1906 Permanent Forest Reserve (reservation from 1902); "
                 "National Forestry Act 1984; Perak Forestry Dept working plans",
        "source": "PLOS ONE (MMFR 40,288 ha at 4°45'N 100°35'E); NASA Earth Observatory (~40,000 ha)",
        "representation": "reserve-scale circular proxy of a crescent-shaped reserve "
                          "(51.5 km coastline Kuala Gula–Bagan Panchor) — NOT the reserve boundary",
    },
    {
        "key": "kinabalu_geopark",
        "name": "Kinabalu Park UNESCO World Heritage Site (Sabah Parks)",
        "short_label": "Kinabalu KSAS",
        "lat": 6.0750,  # Mount Kinabalu summit anchor (4,095 m; 6.075N 116.559E)
        "lon": 116.5583,
        "radius_m": 15500,  # equiv of 753.7 km2 park (sqrt(753.7/pi)*1000 ≈ 15490)
        "gazetted_area_km2": 753.7,
        "area_derived": True,
        "radius_basis": "circle-equivalent of 75,370 ha World Heritage property",
        "legal": "Gazetted 1964; Sabah Parks Enactment 1984 (am. 1996); "
                 "UNESCO WHS 2000 (criteria ix, x; ID 1012)",
        "source": "UNESCO WHC listing 1012 (75,370 ha; N6°08' E116°38'); Sabah Parks (754 sq km)",
        "representation": "park-scale circular proxy centred on the summit — does NOT depict "
                          "the 4,750 km2 Kinabalu UNESCO Global Geopark territory",
    },
    {
        "key": "taman_negara",
        "name": "Taman Negara Gateway Catchment Node (Kuala Tahan, Pahang)",
        "short_label": "Taman Negara KSAS",
        "lat": 4.3833,  # Kuala Tahan southern gateway anchor (~4°23'N 102°24'E)
        "lon": 102.4000,
        "radius_m": 16000,  # gateway catchment node buffer (NOT whole-park equiv ~37.2 km)
        "gazetted_area_km2": 4343.0,
        "area_derived": False,
        "radius_basis": "gateway catchment node buffer; whole-park circle-equiv (~37.2 km) "
                        "deliberately NOT used — the disk marks the Kuala Tahan gateway, "
                        "not the 4,343 km2 tri-state boundary",
        "legal": "Taman Negara Enactments Kelantan 1938 / Pahang & Terengganu 1939; "
                 "PERHILITAN; UNESCO tentative list 5927",
        "source": "UNESCO WHC tentative list 5927 (434,351 ha; Pahang 57%, Kelantan 24%, "
                  "Terengganu 19%); park coords 4°42'N 102°28'E",
        "representation": "gateway-node circular proxy — does NOT depict the tri-state park boundary",
    },
]


def _fallback_ksas_zones(hotspot_name: str = "") -> List[Dict[str, Any]]:
    """Offline fallback: the original 4 grounded node disks (teal/red styling)."""
    h_lower = (hotspot_name or "").lower()
    zones = []
    for z in KSAS_ECOLOGICAL_ZONES:
        is_alert = z["key"].split("_")[0] in h_lower if h_lower else False
        # "cameron" in "cameron highlands" -> True; other zones match similarly.
        zones.append({
            **z,
            "is_alert": bool(is_alert),
            "color": [255, 0, 96, 215] if is_alert else [45, 212, 191, 60],
            "line_color": [255, 255, 255, 255] if is_alert else [45, 212, 191, 140],
            "edge_w": 5 if is_alert else 1,
            # Alert disks float 350 m above the ground plane: clears PathLayer/
            # ArcLayer and ground WMS overlay z-fighting so the anomaly reads on first sight.
            "elev_m": 350 if is_alert else 0,
            "dist_m": 0.0 if is_alert else float("inf"),
        })
    _keep_single_alert(zones)
    return zones


# PPP KSAS Rajah 6 (m.s 32): official Tahap colour codes for KSAS plans.
KSAS_TAHAP_COLORS: Dict[str, List[int]] = {
    "Tahap 1": [64, 90, 0],
    "Tahap 2": [123, 181, 5],
    "Tahap 3": [206, 255, 111],
}

# Generic Malay geographic tokens excluded from hotspot<->zone name matching
# (they would otherwise link unrelated sites, e.g. "Taman Negara" vs "Taman Laut").
_KSAS_STOPWORDS = {"taman", "sungai", "bukit", "pulau", "tasik", "kuala", "teluk", "tanjung"}


def _ksas_match_words(nama: str, daerah: str = "") -> set:
    """Significant tokens (len>4, alpha) from a zone's paren-stripped name + district."""
    core = (nama or "").split("(")[0]
    words = re.findall(r"[a-z]+", f"{core} {daerah or ''}".lower())
    return {w for w in words if len(w) > 4 and w not in _KSAS_STOPWORDS}


def _hotspot_words(name: str = "") -> set:
    return {w for w in re.findall(r"[a-z]+", (name or "").lower()) if len(w) > 4}


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return float(r * 2.0 * math.asin(min(1.0, max(0.0, math.sqrt(a)))))


def _destination_point(lat: float, lon: float, bearing_deg: float,
                       distance_m: float) -> List[float]:
    """Great-circle destination point; returns [lon, lat] in degrees."""
    r = 6371000.0
    d = float(distance_m) / r
    p1 = math.radians(float(lat))
    l1 = math.radians(float(lon))
    b = math.radians(float(bearing_deg))
    p2 = math.asin(math.sin(p1) * math.cos(d)
                   + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    return [math.degrees(l2), math.degrees(p2)]


KSAS_POLYGON_SEGMENTS = 64


def _circle_polygon_geojson(lat: float, lon: float, radius_m: float,
                            segments: int = KSAS_POLYGON_SEGMENTS) -> Dict[str, Any]:
    """
    Exact circular polygon (GeoJSON, degrees) for a KSAS zone.

    Vertices are computed in Python with the great-circle destination formula,
    so the footprint is true to radius_m on the sphere — the renderer only
    joins explicit vertices and cannot mis-scale the area through unit or
    pixel-radius handling.
    """
    ring = [_destination_point(lat, lon, 360.0 * i / segments, radius_m)
            for i in range(segments)]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


_KSAS_REFERENCE_CACHE: Optional[pd.DataFrame] = None


def load_ksas_reference(csv_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Loads data/processed/ksas_reference.csv through the PPP KSAS ETL parser
    (clean + Jadual 4/7 validation). Cached in memory. Returns None when the
    file or the ETL module is unavailable — callers must fall back.
    """
    global _KSAS_REFERENCE_CACHE
    if _KSAS_REFERENCE_CACHE is not None:
        return _KSAS_REFERENCE_CACHE
    try:
        from src.etl.extract_ksas import parse_ksas_reference
    except Exception:
        return None
    try:
        path = Path(csv_path) if csv_path else PROCESSED_DIR / "ksas_reference.csv"
        df, _report = parse_ksas_reference(path)
        _KSAS_REFERENCE_CACHE = df
        return df
    except Exception:
        return None


def _row_to_zone(row: pd.Series, is_alert: bool) -> Dict[str, Any]:
    """Projects one ksas_reference.csv row onto the 3D layer zone contract."""
    tahap = str(row["tahap"])
    base = KSAS_TAHAP_COLORS.get(tahap, [45, 212, 191])
    luas = row.get("luas_h")
    return {
        "key": str(row["key"]),
        "ksas_code": str(row["ksas_code"]),
        "jenis_ksas": str(row["jenis_ksas"]),
        "tahap": tahap,
        "name": str(row["nama"]),
        "short_label": str(row["label_peta"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "radius_m": int(row["radius_m"]),
        "gazetted_area_km2": (float(luas) / 100.0) if pd.notna(luas) else None,
        "area_derived": str(row.get("area_derived", "")).strip().lower() == "true",
        "radius_basis": str(row.get("radius_basis", "")),
        "legal": f"{row['ksas_code']} {row['jenis_ksas']} ({tahap}); PPP KSAS Jadual 4",
        "source": str(row.get("sumber", "")),
        "representation": str(row.get("representation", "")),
        "keyakinan": str(row.get("keyakinan", "")),
        "is_alert": bool(is_alert),
        "color": [255, 0, 96, 215] if is_alert else [*base, 60],
        "line_color": [255, 255, 255, 255] if is_alert else [*base, 140],
        "edge_w": 5 if is_alert else 1,
        # Alert disks float 350 m above the ground plane: clears PathLayer/
        # ArcLayer and ground WMS overlay z-fighting so the anomaly reads on first sight.
        "elev_m": 350 if is_alert else 0,
    }


# Max hotspot→zone distance (m) for a KSAS disk to render on the 3D map.
# Rationale: all 22 national zones rendered at once stack into a solid mass
# over the dense Peninsular cluster (opacity overdraw = GPU stutter) and bury
# the district cylinders. The hotspot's own zones (alerts) always render;
# the rest are context limited to a ~150 km planning catchment.
KSAS_DISPLAY_RADIUS_M = 150000

# A district at or above this capacity pressure counts as acute for the KSAS
# landing watchlist (same >= 1.0 cut used for tangerine "Acute" cylinders).
KSAS_WATCH_PRESSURE_MIN = 1.0

# Manual-LOD zoom thresholds (deck.gl zoom units).
# The dashboard radio switch selects the 16-state macro or the 110-district
# micro view explicitly. Whole-Malaysia framing sits at zoom ~5.8; the manual
# "Auto" option resolves to macro at or below the macro max.
# Cylinders auto-hide past street zoom via ColumnLayer maxZoom so labels take over.
AUTO_MACRO_ZOOM_MAX = 6.2
CYLINDER_HIDE_ZOOM = 13.0


def _apply_alert_styling(zone: Dict[str, Any]) -> None:
    """Paints a zone high-contrast electric crimson with white hazard border (in place)."""
    zone["is_alert"] = True
    zone["color"] = [255, 0, 96, 215]
    zone["line_color"] = [255, 255, 255, 255]
    zone["edge_w"] = 5
    zone["elev_m"] = 350


def get_ksas_watchlist(all_destinations_df) -> List[Dict[str, Any]]:
    """
    Landing-persistent emergence: every KSAS zone containing at least one
    ACUTE district (capacity pressure >= 1.0) renders red on every map paint —
    before any click. The landing default (Langkawi) sits in no KSAS zone, so
    without this the first paint never shows red.

    Returns alert-styled zone dicts with a `watch_districts` field naming the
    acute districts inside (for legend honesty). Cheap: 22 zones × N districts.
    """
    df = load_ksas_reference()
    if df is None or all_destinations_df is None or len(all_destinations_df) == 0:
        return []
    try:
        acute = all_destinations_df[
            all_destinations_df["continuous_pressure"].astype(float)
            >= KSAS_WATCH_PRESSURE_MIN]
    except (KeyError, TypeError, ValueError):
        return []
    if acute.empty:
        return []
    watch = []
    for _, row in df.iterrows():
        inside = acute[
            acute.apply(lambda r: _haversine_m(
                float(r["lat"]), float(r["lon"]),
                float(row["lat"]), float(row["lon"])) <= float(row["radius_m"]),
                axis=1)]
        if inside.empty:
            continue
        zone = _row_to_zone(row, True)
        zone["watch_districts"] = sorted(set(
            str(r["district_name"]) for _, r in inside.iterrows()))
        watch.append(zone)
    return watch


def get_ksas_ecological_zones(hotspot_name: str = "",
                              hotspot_lat: Optional[float] = None,
                              hotspot_lon: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Returns KSAS zones with per-zone alert styling (red when the hotspot falls
    inside). Primary source: data/processed/ksas_reference.csv (all 15 PPP KSAS
    jenis, colours per Rajah 6 Tahap codes). Falls back to the 4-zone offline
    constant when the CSV/ETL is unavailable.

    Alert triggers: significant name-token overlap with the zone (nama+daerah),
    or — when hotspot coordinates are supplied — great-circle proximity within
    the zone's radius_m.

    Exactly ONE zone may alert: when several zones trigger (overlapping
    catchments), the nearest to the hotspot keeps the alert and the rest render
    in Tahap colours. Agencies see a single red anomaly, matching the singular
    legend chip.
    """
    df = load_ksas_reference()
    if df is None:
        return _fallback_ksas_zones(hotspot_name)
    hw = _hotspot_words(hotspot_name)
    have_coords = hotspot_lat is not None and hotspot_lon is not None
    try:
        hlats, hlons = float(hotspot_lat), float(hotspot_lon)
    except (TypeError, ValueError):
        have_coords = False
    zones = []
    for _, row in df.iterrows():
        token_hit = bool(hw & _ksas_match_words(str(row["nama"]), str(row.get("daerah", ""))))
        dist = (_haversine_m(hlats, hlons, float(row["lat"]), float(row["lon"]))
                if have_coords else float("inf"))
        prox_hit = have_coords and dist <= float(row["radius_m"])
        zone = _row_to_zone(row, token_hit or bool(prox_hit))
        zone["dist_m"] = dist
        zones.append(zone)
    _keep_single_alert(zones)
    return zones


def _keep_single_alert(zones: List[Dict[str, Any]]) -> None:
    """Demotes all but the nearest alerting zone to Tahap styling (in place)."""
    alerting = [z for z in zones if z["is_alert"]]
    if len(alerting) <= 1:
        return
    alerting.sort(key=lambda z: (z.get("dist_m", float("inf")), z["key"]))
    for z in alerting[1:]:
        base = KSAS_TAHAP_COLORS.get(z.get("tahap", ""), [45, 212, 191])
        z["is_alert"] = False
        z["color"] = [*base, 60]
        z["line_color"] = [*base, 140]
        z["edge_w"] = 1
        z["elev_m"] = 0


def build_ksas_pydeck_layers(hotspot_name: str = "",
                             hotspot_lat: Optional[float] = None,
                             hotspot_lon: Optional[float] = None,
                             max_display_m: float = KSAS_DISPLAY_RADIUS_M,
                             watch_keys=None) -> List[pdk.Layer]:
    """
    Builds the KSAS alert disks: one exact GeoJson polygon per alerting zone
    (hotspot's own zone + acute watch zones) — red on the inside with white
    borders, and that's it. No poles, labels, pins or markers.

    True size by construction: each footprint is an explicit 64-vertex ring in
    degrees computed in Python from radius_m, so the renderer cannot distort
    the area through radius-unit or pixel-radius handling. Radii are clamped
    to [6000, 40000] m as a final guard and lightly staggered in elevation so
    overlapping disks never z-fight. pickable=False so district/state
    cylinders keep exclusive click-to-select.

    watch_keys: optional set of zone keys from get_ksas_watchlist() — zones
    containing acute districts render alongside (not instead of) the
    hotspot's single alert. Watch zones bypass the 150 km catchment filter.
    """
    zones = get_ksas_ecological_zones(hotspot_name, hotspot_lat, hotspot_lon)
    try:
        hlats, hlons = float(hotspot_lat), float(hotspot_lon)
        have_coords = True
    except (TypeError, ValueError):
        have_coords = False
    _watch = set(watch_keys) if watch_keys else set()
    if have_coords:
        zones = [z for z in zones if z["is_alert"] or z["key"] in _watch
                 or _haversine_m(
                     hlats, hlons, z["lat"], z["lon"]) <= max_display_m]
    for z in zones:
        if z["key"] in _watch and not z["is_alert"]:
            _apply_alert_styling(z)
    alerting_zones = [z for z in zones if z["is_alert"]]
    poly_rows = []
    for i, z in enumerate(alerting_zones):
        radius = max(6000, min(40000, int(z["radius_m"])))
        poly_rows.append({
            "key": z["key"],
            "geometry": _circle_polygon_geojson(z["lat"], z["lon"], radius),
            "fill": [255, 0, 96, 215],
            "line": [255, 255, 255, 255],
            "lw": 5,
            "elev": 350.0 + 25.0 * i,
            "is_alert": True,
            "radius_m": radius,
        })
    poly_df = pd.DataFrame(poly_rows, columns=["key", "geometry", "fill", "line", "lw", "elev", "is_alert", "radius_m"])
    polygons = pdk.Layer(
        "GeoJsonLayer",
        data=poly_df,
        id="ksas_buffers",
        get_fill_color="fill",
        get_line_color="line",
        get_line_width="lw",
        line_width_units="pixels",
        line_width_min_pixels=3,
        get_elevation="elev",
        elevation_scale=1,
        extruded=True,
        filled=True,
        stroked=True,
        wireframe=False,
        opacity=1.0,
        pickable=False,
        auto_highlight=False,
    )
    return [polygons]

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# Key transit network junctions across Malaysia for realistic multi-point routing
HIGHWAY_NETWORK = {
    "plus_north": [
        (6.52, 100.42), # Bukit Kayu Hitam
        (6.12, 100.37), # Alor Setar
        (5.64, 100.49), # Sungai Petani
        (5.37, 100.40), # Butterworth / Penang Bridge
        (5.17, 100.48), # Nibong Tebal
        (4.85, 100.73), # Taiping
        (4.77, 100.94), # Kuala Kangsar
        (4.60, 101.09), # Ipoh
        (4.20, 101.27), # Tapah
        (3.68, 101.52), # Tanjung Malim
        (3.32, 101.58), # Rawang
        (3.14, 101.69), # Kuala Lumpur
    ],
    "plus_south": [
        (3.14, 101.69), # Kuala Lumpur
        (2.93, 101.70), # Putrajaya / Bangi
        (2.73, 101.94), # Seremban
        (2.28, 102.28), # Ayer Keroh / Melaka
        (2.27, 102.55), # Tangkak
        (2.01, 103.07), # Yong Peng / Batu Pahat
        (1.66, 103.60), # Kulai
        (1.49, 103.74), # Johor Bahru
    ],
    "east_coast_e8": [
        (3.14, 101.69), # KL
        (3.35, 101.78), # Genting Sempah
        (3.52, 101.91), # Bentong
        (3.45, 102.42), # Temerloh
        (3.81, 103.33), # Kuantan
        (4.23, 103.33), # Kemaman
        (4.77, 103.42), # Dungun
        (5.33, 103.14), # Kuala Terengganu
    ],
    "central_spine_road": [
        (4.47, 101.38), # Cameron Highlands
        (4.35, 101.60), # Sungai Koyan
        (4.25, 101.85), # Padang Tengku
        (4.18, 102.05), # Kuala Lipis
        (3.79, 101.86), # Raub
        (3.52, 101.91), # Bentong
    ],
    "coastal_route_5": [
        (3.04, 101.45), # Klang
        (2.82, 101.48), # Banting
        (2.52, 101.80), # Port Dickson
        (2.20, 102.25), # Melaka Tengah
        (2.04, 102.57), # Muar
        (1.85, 102.93), # Batu Pahat
        (1.49, 103.39), # Pontian
    ],
    "sabah_pan_borneo": [
        (5.98, 116.07), # Kota Kinabalu
        (6.18, 116.23), # Tuaran
        (5.95, 116.66), # Kundasang / Ranau
        (5.63, 117.12), # Telupid
        (5.84, 118.12), # Sandakan
        (5.03, 118.33), # Lahad Datu
        (4.25, 117.89), # Tawau
    ],
    "sarawak_pan_borneo": [
        (1.55, 110.34), # Kuching
        (1.17, 110.57), # Serian
        (1.24, 111.46), # Sri Aman
        (2.12, 111.52), # Sarikei
        (2.29, 111.83), # Sibu
        (3.17, 113.04), # Bintulu
        (4.41, 114.01), # Miri
    ],
    "ktm_ets": [
        (6.65, 100.20), # Padang Besar
        (6.43, 100.27), # Arau
        (6.12, 100.37), # Alor Setar
        (5.82, 100.48), # Gurun
        (5.64, 100.49), # Sungai Petani
        (5.39, 100.37), # Butterworth
        (5.36, 100.46), # Bukit Mertajam
        (5.17, 100.48), # Nibong Tebal
        (4.85, 100.73), # Taiping
        (4.77, 100.94), # Kuala Kangsar
        (4.60, 101.09), # Ipoh
        (4.48, 101.04), # Batu Gajah
        (4.31, 101.15), # Kampar
        (3.99, 101.31), # Sungkai
        (3.68, 101.52), # Tanjung Malim
        (3.32, 101.58), # Rawang
        (3.14, 101.69), # Kuala Lumpur / KL Sentral
        (3.00, 101.79), # Kajang
        (2.73, 101.94), # Seremban
        (2.47, 102.23), # Tampin
        (2.58, 102.61), # Gemas
    ],
    "ktm_jungle_rail": [
        (2.58, 102.61), # Gemas
        (2.81, 102.41), # Bahau
        (3.23, 102.43), # Triang
        (3.48, 102.35), # Mentakab
        (3.93, 102.36), # Jerantut
        (4.18, 102.05), # Kuala Lipis
        (4.88, 101.97), # Gua Musang
        (5.38, 102.01), # Dabong
        (5.53, 102.20), # Kuala Krai
        (5.81, 102.15), # Tanah Merah
        (6.04, 102.14), # Pasir Mas
        (6.16, 102.23), # Wakaf Bharu (Kota Bharu)
        (6.20, 102.17), # Tumpat
    ]
}



_KSAS_OVERLAY_PATH: Optional[str] = None
_KSAS_BORNEO_OVERLAY_PATH: Optional[str] = None


def get_ksas_overlay_image_uri() -> str:
    """Returns local file path or live WMS URL for Peninsular PyDeck BitmapLayer.
    
    IMPORTANT: PyDeck inspects the image argument. If given a file path ending in .png,
    PyDeck converts it cleanly to base64 data URI in JSON without prepending '@@='.
    If passed a raw data:image URI string, PyDeck treats it as a JS expression and prepends '@@=',
    breaking DeckGlJsonChart with 'Unexpected \":\" at character 4'.
    """
    global _KSAS_OVERLAY_PATH
    if _KSAS_OVERLAY_PATH is not None:
        return _KSAS_OVERLAY_PATH
    p = (ROOT / "data" / "processed" / "planmalaysia_ksas_peninsular.png").resolve()
    if p.exists():
        _KSAS_OVERLAY_PATH = str(p)
        return _KSAS_OVERLAY_PATH
    _KSAS_OVERLAY_PATH = (
        "https://iplan.planmalaysia.gov.my/geopro/iplan/wms?"
        "service=WMS&version=1.1.1&request=GetMap&layers=ksas&styles="
        "&bbox=99.6,1.2,104.6,6.8&width=1600&height=2000&srs=EPSG:4326&format=image/png&transparent=true"
    )
    return _KSAS_OVERLAY_PATH


def get_ksas_borneo_image_uri() -> str:
    """Returns local file path or live WMS URL for Sabah and Sarawak KSAS overlay in PyDeck BitmapLayer."""
    global _KSAS_BORNEO_OVERLAY_PATH
    if _KSAS_BORNEO_OVERLAY_PATH is not None:
        return _KSAS_BORNEO_OVERLAY_PATH
    p = (ROOT / "data" / "processed" / "planmalaysia_ksas_borneo.png").resolve()
    if p.exists():
        _KSAS_BORNEO_OVERLAY_PATH = str(p)
        return _KSAS_BORNEO_OVERLAY_PATH
    _KSAS_BORNEO_OVERLAY_PATH = str(p)
    return _KSAS_BORNEO_OVERLAY_PATH


# Overlay layers the user can show/hide (Plotly-legend-style toggles).
# District/state cylinders are always on (base layer + click targets).
# The statutory PLANMalaysia WMS overlay covers the entire country; KSAS
# alerts render as red disks with white borders (nothing else).
MAP_TOGGLE_LAYERS: List[Tuple[str, str]] = [
    ("redistribution_arcs", "Relief arcs"),
    ("ground_transit_routes", "Transit routes"),
    ("ksas_overlay", "PLANMalaysia KSAS map"),
    ("ksas_buffers", "KSAS alert zones"),
]


def get_3d_deck_legend_html(visible_layers=None,
                             alert_zones=None) -> str:
    """
    Compact dark HUD legend for the 3D PyDeck Elevation Map.
    - visible_layers: optional set of layer ids currently shown; chips for
      hidden layers are omitted so the legend mirrors the map (Plotly-style).
      Defaults to everything visible. Legacy id "ksas_markers" is treated
      as an alias of "ksas_buffers".
    - alert_zones: accepted for backwards compatibility but ignored — the
      legend shows a static KSAS chip; details live on the map disks.
    Route colours are traffic-aware: green = free-flow, amber = moderate,
    red = severe gridlock, blue = rail.
    """
    visible = None if visible_layers is None else set(visible_layers)
    # Legacy alias: old callers pass ksas_markers.
    if visible is not None and "ksas_markers" in visible:
        visible = set(visible) | {"ksas_buffers"}

    def _chip(swatch: str, label: str) -> str:
        return (
            '<div style="display: flex; align-items: center; gap: 6px;">'
            f"{swatch}"
            f'<span style="color: #cbd5e1;">{label}</span>'
            "</div>"
        )

    def _dot(color: str, glow: str = "") -> str:
        return (
            f'<span style="display: inline-block; width: 11px; height: 11px; '
            f'background: {color}; border-radius: 3px;{glow}"></span>'
        )

    def _line(color: str) -> str:
        return (
            f'<span style="display: inline-block; width: 16px; height: 3px; '
            f'background: {color}; border-radius: 2px;"></span>'
        )

    chips = [
        '<div style="display: flex; align-items: center; gap: 6px;">'
        '<span style="display: inline-block; width: 11px; height: 11px; background: #ef4444; border-radius: 3px; box-shadow: 0 0 6px #ef4444;"></span>'
        '<span style="font-weight: 600; color: #f8fafc;">Active Hotspot</span> <span style="color: #94a3b8;">(&ge;100%)</span>'
        "</div>",
        _chip(_dot("#10b981", " box-shadow: 0 0 6px #10b981;"),
              '<span style="font-weight: 600; color: #f8fafc;">#1 Relief Corridor</span>'),
        _chip(_dot("#06b6d4", " box-shadow: 0 0 6px #06b6d4;"),
              '<span style="font-weight: 600; color: #f8fafc;">#2 Relief Corridor</span>'),
        _chip(_dot("#f59e0b", " box-shadow: 0 0 6px #f59e0b;"),
              '<span style="font-weight: 600; color: #f8fafc;">#3 Relief Corridor</span>'),
        _chip(_dot("#f97316"), "Acute (&ge;100%)"),
        _chip(_dot("#fbbf24"), "Stressed (80–100%)"),
        _chip('<span style="display: inline-block; width: 11px; height: 11px; background: #f0f9ff; border-radius: 3px; border: 1px solid #94a3b8;"></span>',
              "Headroom (&lt;80%)"),
    ]

    show_routes = visible is None or "ground_transit_routes" in visible
    if show_routes:
        chips += [
            _chip(_line("#10b981"), "Real-World Route: free-flow"),
            _chip(_line("#f59e0b"), "Moderate congestion"),
            _chip(_line("#ef4444"), "Severe gridlock"),
            _chip(_line("#38bdf8"), "Rail (track-following)"),
        ]

    show_overlay = visible is None or "ksas_overlay" in visible
    if show_overlay:
        chips += [
            _chip(_dot("#8B4513"), "PLANMalaysia RFN KSAS Tahap 1 (Rank 1 Critical)"),
            _chip(_dot("#F4A460"), "KSAS Tahap 2 (Rank 2 Buffer)"),
            _chip(_dot("#FFF8DC", " border: 1px solid #94a3b8;"), "KSAS Tahap 3 (Rank 3 Controlled)"),
        ]

    show_buffers = visible is None or "ksas_buffers" in visible or "ksas_markers" in visible or "ksas_alert" in visible
    if show_buffers:
        chips.append(
            '<div style="display: flex; align-items: center; gap: 6px;">'
            '<span style="display: inline-block; width: 12px; height: 12px; background: #ff0060; border: 2px solid #ffffff; border-radius: 3px; box-shadow: 0 0 8px #ff0060;"></span>'
            '<span style="color: #cbd5e1;">KSAS alert zone</span>'
            "</div>"
        )

    return (
        '<div style="background: rgba(15, 23, 42, 0.90); border: 1px solid #334155; '
        'border-radius: 8px; padding: 6px 12px; margin-bottom: 8px; '
        'font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; '
        'font-size: 11px; color: #e2e8f0; display: flex; flex-wrap: wrap; gap: 12px; '
        'align-items: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">'
        + "".join(chips) +
        "</div>"
    )


def _resolve_rail_viability(orig_lat: float, orig_lon: float, dest_lat: float, dest_lon: float,
                            orig_name: str = "", dest_name: str = "",
                            claimed_is_rail: bool = False) -> Tuple[bool, dict]:
    """Single gate for rail claims: verifies both ends are truly rail-accessible."""
    try:
        rail_check = traffic_engine.is_rail_viable(orig_lat, orig_lon, dest_lat, dest_lon, orig_name, dest_name)
    except AttributeError:
        return False, {"viable": False}
    if claimed_is_rail and not rail_check.get("viable"):
        return False, rail_check
    return bool(rail_check.get("viable")), rail_check


def _truthful_corridor(h_lat: float, h_lon: float, a_lat: float, a_lon: float,
                       h_name: str, a_name: str, alt: dict) -> Tuple[bool, dict, float, str]:
    """
    Resolves the honest corridor for rendering: verifies rail claims against
    real station proximity and returns (is_rail_actual, corridor, nominal_mins, display_mode).
    Falls back to the alt's claimed mode only if inference is unavailable.
    """
    claimed_mode = str(alt.get("transit_mode", "Road"))
    claimed_is_rail = any(k in claimed_mode.lower() for k in ["rail", "ets", "komuter", "train", "ktm"])
    try:
        corridor = traffic_engine.infer_corridor(h_lat, h_lon, a_lat, a_lon, h_name, a_name)
        is_rail_actual = bool(corridor.get("rail_viable"))
        # If recommender already grounded the mode, keep it; otherwise use inferred truth.
        display_mode = str(corridor.get("transit_mode", claimed_mode))
        nominal = float(corridor.get("travel_time_mins", alt.get("travel_time_mins", 60.0)))
        return is_rail_actual, corridor, nominal, display_mode
    except AttributeError:
        nominal = float(alt.get("travel_time_mins", alt.get("transit_time_mins", 60.0)))
        return claimed_is_rail, {}, nominal, claimed_mode


def generate_realistic_transit_waypoints(orig_lat: float, orig_lon: float, dest_lat: float, dest_lon: float, is_rail: bool = False, orig_name: str = "", dest_name: str = "") -> list:
    """
    Constructs authentic road or rail waypoints following genuine transportation infrastructure:
    1. For road corridors: Queries OSRM Real-World Routing Engine (hundreds of exact highway points).
    2. For rail corridors: Follows KTM ETS or Jungle Railway track alignments via
       station-to-station polylines (never a straight chord). If either end lacks
       a nearby station (e.g. Cameron Highlands), rail is rejected and road is used.
    3. If offline: Generates a dense 25-point smooth highway curvature spline.
    Guarantees no straight Euclidean cuts across mountains or open water.
    """
    if not is_rail:
        osrm_path = query_osrm_route([orig_lat, orig_lon], [dest_lat, dest_lon])
        if osrm_path and len(osrm_path) >= 2:
            return osrm_path

    best_waypoints = []

    if is_rail:
        viable, rail_check = _resolve_rail_viability(orig_lat, orig_lon, dest_lat, dest_lon,
                                                     orig_name, dest_name, claimed_is_rail=True)
        if viable:
            try:
                rail_path = traffic_engine.build_rail_path_coords(
                    orig_lat, orig_lon, dest_lat, dest_lon, rail_detail=rail_check)
                if rail_path and len(rail_path) >= 2:
                    return rail_path
            except AttributeError:
                pass
        # Rail claimed but not viable (e.g. Cameron Highlands has no station):
        # fall through to road routing below instead of drawing a fake straight rail line.
        is_rail = False
        osrm_path = query_osrm_route([orig_lat, orig_lon], [dest_lat, dest_lon])
        if osrm_path and len(osrm_path) >= 2:
            return osrm_path
        # Skip rail-node stitching when rail was rejected: continue to road fallback.
        if not viable:
            pass
        elif abs(orig_lon - dest_lon) < 1.8 and orig_lon < 103.0 and dest_lon < 103.0:
            rail_nodes = HIGHWAY_NETWORK["ktm_ets"]
            min_lat, max_lat = min(orig_lat, dest_lat), max(orig_lat, dest_lat)
            segment = [node for node in rail_nodes if min_lat - 0.25 <= node[0] <= max_lat + 0.25]
            if orig_lat > dest_lat:
                segment = sorted(segment, key=lambda x: x[0], reverse=True)
            else:
                segment = sorted(segment, key=lambda x: x[0])
            if len(segment) >= 2:
                best_waypoints = [[orig_lat, orig_lon]] + [[n[0], n[1]] for n in segment] + [[dest_lat, dest_lon]]

        # Check if corridor connects via East Coast Jungle Railway (Pahang/Kelantan)
        elif (orig_lon > 101.5 or dest_lon > 101.5) and (orig_lat > 3.0 or dest_lat > 3.0) and orig_lat < 6.5 and dest_lat < 6.5:
            rail_nodes = HIGHWAY_NETWORK["ktm_jungle_rail"]
            min_lat, max_lat = min(orig_lat, dest_lat), max(orig_lat, dest_lat)
            segment = [node for node in rail_nodes if min_lat - 0.25 <= node[0] <= max_lat + 0.25]
            if orig_lat > dest_lat:
                segment = sorted(segment, key=lambda x: x[0], reverse=True)
            else:
                segment = sorted(segment, key=lambda x: x[0])
            if len(segment) >= 2:
                best_waypoints = [[orig_lat, orig_lon]] + [[n[0], n[1]] for n in segment] + [[dest_lat, dest_lon]]

    if not best_waypoints:
        # Check Peninsular West Coast expressway corridor (PLUS North / South)
        if abs(orig_lon - dest_lon) < 1.8 and orig_lon < 104.0 and dest_lon < 104.0:
            all_nodes = HIGHWAY_NETWORK["plus_north"] + HIGHWAY_NETWORK["plus_south"][1:]
            min_lat, max_lat = min(orig_lat, dest_lat), max(orig_lat, dest_lat)
            segment = [node for node in all_nodes if min_lat - 0.2 <= node[0] <= max_lat + 0.2]
            if orig_lat > dest_lat:
                segment = sorted(segment, key=lambda x: x[0], reverse=True)
            else:
                segment = sorted(segment, key=lambda x: x[0])
            if len(segment) >= 2:
                best_waypoints = [[orig_lat, orig_lon]] + [[n[0], n[1]] for n in segment] + [[dest_lat, dest_lon]]

        # Check East Coast Expressway E8
        elif (orig_lon > 102.0 or dest_lon > 102.0) and orig_lon < 104.5 and dest_lon < 104.5 and orig_lat < 6.5:
            segment = [node for node in HIGHWAY_NETWORK["east_coast_e8"] if min(orig_lat, dest_lat) - 0.3 <= node[0] <= max(orig_lat, dest_lat) + 0.3]
            if orig_lat > dest_lat:
                segment = sorted(segment, key=lambda x: x[0], reverse=True)
            else:
                segment = sorted(segment, key=lambda x: x[0])
            if len(segment) >= 2:
                best_waypoints = [[orig_lat, orig_lon]] + [[n[0], n[1]] for n in segment] + [[dest_lat, dest_lon]]

    # Dense smooth terrain-following curve fallback
    if len(best_waypoints) < 2:
        best_waypoints = generate_dense_corridor_curve([orig_lat, orig_lon], [dest_lat, dest_lon], num_steps=25)

    return best_waypoints


def aggregate_destinations_by_state(destinations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical Level-of-Detail (LOD) 16-State Macro Aggregation:
    Groups district records into 16 state macro centroids.

    Computes:
    - State Centroid coordinates (mean lat, mean lon)
    - Summed peak daily demand (daily_demand_peak)
    - Summed sustainable capacity (sustainable_capacity)
    - Mean hotel occupancy rate (latest_aor_pct)
    - Summed hotel inventory (total_rooms)
    - Mean continuous capacity pressure (continuous_pressure)
    - Mean poverty rate (poverty_rate)
    - Scaled 3D elevation: max(25000.0, float(tot_demand) * 0.8)
    - Representative district name (highest demand district) for interactive navigation
    - Pre-formatted clean HTML tooltip string (tooltip_html)
    """
    if destinations_df.empty:
        return pd.DataFrame(columns=[
            "state_name", "district_name", "rep_district", "destination_name", "name",
            "lat", "lon", "daily_demand_peak", "sustainable_capacity",
            "total_rooms", "latest_aor_pct", "continuous_pressure",
            "poverty_rate", "district_count", "elevation", "tooltip_html"
        ])

    state_records = []
    for s_name, s_df in destinations_df.groupby("state_name", sort=True):
        s_lat = float(s_df["lat"].mean())
        s_lon = float(s_df["lon"].mean())
        tot_demand = float(s_df["daily_demand_peak"].sum()) if "daily_demand_peak" in s_df else 0.0
        tot_cap = float(s_df["sustainable_capacity"].sum()) if "sustainable_capacity" in s_df else tot_demand * 1.2
        tot_rooms = float(s_df["total_rooms"].sum()) if "total_rooms" in s_df else 0.0
        avg_aor = float(s_df["latest_aor_pct"].mean()) if "latest_aor_pct" in s_df else 55.0
        avg_pressure = float(s_df["continuous_pressure"].mean()) if "continuous_pressure" in s_df else (tot_demand / max(1.0, tot_cap))
        avg_poverty = float(s_df["poverty_rate"].mean()) if "poverty_rate" in s_df else 5.0
        dist_count = len(s_df)

        # Representative district for clicking state macro column
        if "daily_demand_peak" in s_df and "district_name" in s_df:
            rep_district = str(s_df.sort_values(by="daily_demand_peak", ascending=False).iloc[0]["district_name"])
        elif "district_name" in s_df:
            rep_district = str(s_df.iloc[0]["district_name"])
        else:
            rep_district = str(s_name)

        elevation = max(25000.0, tot_demand * 0.8)

        tooltip_html = (
            f"{s_name}\n"
            f"Macro State ({dist_count} Districts)\n"
            f"────────────────────────────\n"
            f"Total Peak Influx: {tot_demand:,.0f} pax/day\n"
            f"Sustainable Capacity: {tot_cap:,.0f} pax/day\n"
            f"Total Hotel Inventory: {tot_rooms:,.0f} rooms\n"
            f"Mean State Occupancy: {avg_aor:.1f}% • Stress: {avg_pressure*100:.1f}%\n"
            f"Click state column to focus and zoom into districts"
        )

        state_records.append({
            "name": str(s_name),
            "state_name": str(s_name),
            "district_name": rep_district,
            "rep_district": rep_district,
            "destination_name": str(s_name),
            "lat": s_lat,
            "lon": s_lon,
            "daily_demand_peak": tot_demand,
            "sustainable_capacity": tot_cap,
            "total_rooms": tot_rooms,
            "latest_aor_pct": avg_aor,
            "continuous_pressure": avg_pressure,
            "poverty_rate": avg_poverty,
            "district_count": dist_count,
            "elevation": elevation,
            "tooltip_html": tooltip_html
        })

    return pd.DataFrame(state_records)


def render_pydeck_3d_elevation_map(
    hotspot: dict,
    alternatives: list,
    all_destinations_df: pd.DataFrame,
    lod_mode: str = "district",
    google_maps_api_key: str = None,
    visible_layers=None,
    ksas_watch_keys=None,
    map_zoom: Optional[float] = None,
) -> pdk.Deck:
    """
    Renders an interactive 3D WebGL PyDeck cockpit (sole map engine):
    - In 'district' mode: renders all 110 districts as 3D extruded cylinders.
    - In 'state' mode: merges districts into 16 state macro-columns.
    - In 'auto' mode: selects state macro when map_zoom <= AUTO_MACRO_ZOOM_MAX
      (whole-Malaysia view), else district micro. The dashboard LOD radio
      switch selects the mode explicitly (manual switch).
    - Accentuates active Hotspot in glowing crimson red and Top 3 Alternatives in emerald, cyan, and amber.
    - Draws one KSAS alert disk per alerting zone: red on the inside with
      white borders (pickable=False), and nothing else.
    - Draws 3D ground multi-modal transit routes (PathLayer) with live traffic delay colors.
    - Draws high-contrast 3D tensile arcs (ArcLayer) leaping between origin and relief destinations.
    - visible_layers: optional set of overlay layer ids to include (see
      MAP_TOGGLE_LAYERS). District/state cylinders are always rendered.
      Defaults to every layer visible. Legacy "ksas_markers" is accepted
      as an alias of "ksas_buffers".
    - ksas_watch_keys: optional set of KSAS zone keys from get_ksas_watchlist()
      that render alongside the hotspot alert (landing emergence).
    - map_zoom: optional deck zoom driving 'auto' LOD + initial ViewState.
    """
    h_lat = float(hotspot.get("lat", 5.4141))
    h_lon = float(hotspot.get("lon", 100.3288))
    h_name = str(hotspot.get("destination_name", hotspot.get("district_name", "Hotspot")))

    alt_names = [str(alt.get("candidate_name", alt.get("destination_name", alt.get("district_name", "")))) for alt in alternatives]

    # Overlay visibility (Plotly-legend-style toggles from the dashboard).
    # District/state cylinders always render; overlays honour the set.
    _visible = None if visible_layers is None else set(visible_layers)
    if _visible is not None and "ksas_markers" in _visible:
        _visible = set(_visible) | {"ksas_buffers"}

    def _shown(layer_id: str) -> bool:
        if _visible is None:
            return True
        if layer_id in _visible:
            return True
        if layer_id == "ksas_buffers" and (
            "ksas_markers" in _visible or "ksas_alert" in _visible or "ksas_alert_beacon" in _visible
        ):
            return True
        return False

    # Manual LOD: the dashboard switch picks state xor district explicitly.
    # 'auto' resolves via the zoom slider to a single layer (legacy behaviour).
    if str(lod_mode).lower() == "auto":
        try:
            _z = float(map_zoom) if map_zoom is not None else 7.2
        except (TypeError, ValueError):
            _z = 7.2
        lod_mode = "state" if _z <= AUTO_MACRO_ZOOM_MAX else "district"

    layers = []

    if lod_mode == "state":
        # -------------------------------------------------------------
        # STATE MACRO AGGREGATION MODE (16 States / Federal Territories)
        # -------------------------------------------------------------
        state_agg_df = aggregate_destinations_by_state(all_destinations_df)
        state_records = []
        for _, s_row in state_agg_df.iterrows():
            s_name = s_row["state_name"]
            tot_demand = float(s_row["daily_demand_peak"])
            tot_cap = float(s_row["sustainable_capacity"])
            tot_rooms = float(s_row["total_rooms"])
            avg_aor = float(s_row["latest_aor_pct"])
            avg_pressure = float(s_row["continuous_pressure"])
            dist_count = int(s_row["district_count"])

            is_hotspot_state = (s_name == hotspot.get("state_name"))

            if is_hotspot_state:
                color = [239, 68, 68, 255] # Hotspot State Crimson
                elevation = max(28000.0, tot_demand * 0.85)
                status = f"🚨 Hotspot State ({dist_count} Districts)"
            elif any(alt.get("state_name") == s_name for alt in alternatives):
                color = [16, 185, 129, 255] # Relief Corridor State Emerald
                elevation = max(22000.0, tot_demand * 0.75)
                status = f"✅ Relief Corridor State ({dist_count} Districts)"
            elif avg_pressure >= 1.0:
                color = [249, 115, 22, 245] # Partial Highlight: Acute Overcapacity State Tangerine
                elevation = max(15000.0, tot_demand * 0.60)
                status = f"⚠️ Acute Stress State ({dist_count} Districts)"
            elif avg_pressure >= 0.80:
                color = [251, 191, 36, 240] # Partial Highlight: Stressed State Warm Amber
                elevation = max(11000.0, tot_demand * 0.50)
                status = f"⚡ Stressed State ({dist_count} Districts)"
            else:
                color = [240, 249, 255, 240] # Ultra-light luminous ice-platinum for noticeability
                elevation = max(5000.0, tot_demand * 0.35)
                status = f"🟢 State Overview ({dist_count} Districts)"

            rec = dict(s_row)
            rec["name"] = str(s_name)
            rec["color"] = color
            rec["status"] = status
            rec["elevation"] = elevation
            rec["demand_formatted"] = f"{tot_demand:,.0f}"
            rec["cap_formatted"] = f"{tot_cap:,.0f}"
            rec["rooms_formatted"] = f"{tot_rooms:,.0f}"
            rec["aor_formatted"] = f"{avg_aor:.1f}"
            rec["pressure_formatted"] = f"{avg_pressure*100:.1f}"
            rec["pov_formatted"] = f"{float(s_row.get('poverty_rate', 5.0)):.1f}"
            rec["ksas_detail"] = ""
            rec["tooltip_html"] = (
                f"<b>{s_name}</b><br/>"
                f"{status}<br/>"
                f"Total Peak Influx: {tot_demand:,.0f} pax/day<br/>"
                f"Sustainable Capacity: {tot_cap:,.0f} pax/day<br/>"
                f"Total Hotel Inventory: {tot_rooms:,.0f} rooms<br/>"
                f"Mean State Occupancy: {avg_aor:.1f}% | Stress: {avg_pressure*100:.1f}%<br/>"
                f"Click state column to focus and zoom into districts"
            )
            state_records.append(rec)

        state_df = pd.DataFrame(state_records)
        state_col_layer = pdk.Layer(
            "ColumnLayer",
            data=state_df,
            id="state_columns",
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1,
            radius=22000,
            disk_resolution=28,
            get_fill_color="color",
            opacity=0.85,
            wireframe=False,
            pickable=True,
            auto_highlight=True,
            # Hide state columns at street zoom so labels take over.
            maxZoom=CYLINDER_HIDE_ZOOM,
        )
        layers.append(state_col_layer)

    else:
        # -------------------------------------------------------------
        # DISTRICT MICRO COCKPIT MODE (110 Districts Across Malaysia)
        # -------------------------------------------------------------
        col_records = []
        for _, row in all_destinations_df.iterrows():
            d_name = str(row["destination_name"])
            dist_name = str(row["district_name"])
            lat = float(row["lat"])
            lon = float(row["lon"])
            demand = float(row.get("daily_demand_peak", 5000))
            pressure = float(row.get("continuous_pressure", 0.75))
            aor = float(row.get("latest_aor_pct", 50.0))
            pov = float(row.get("poverty_rate", 4.5))

            if dist_name == hotspot.get("district_name") or d_name == h_name:
                color = [239, 68, 68, 255] # Full Highlight: Active Hotspot Glowing Crimson Red
                elevation = max(24000.0, demand * 5.0)
                status = "🚨 Active Hotspot (Overcrowded)"
            elif len(alt_names) > 0 and (dist_name == alt_names[0] or d_name == alt_names[0]):
                color = [16, 185, 129, 255] # Luminous Emerald Green
                elevation = max(19000.0, demand * 4.4)
                status = "✅ #1 Primary Relief Corridor"
            elif len(alt_names) > 1 and (dist_name == alt_names[1] or d_name == alt_names[1]):
                color = [2, 132, 199, 240] # Vivid Electric Sky Cyan
                elevation = max(15000.0, demand * 3.7)
                status = "🔹 #2 Secondary Relief Corridor"
            elif len(alt_names) > 2 and (dist_name == alt_names[2] or d_name == alt_names[2]):
                color = [245, 158, 11, 230] # Bright Amber Gold
                elevation = max(12000.0, demand * 3.1)
                status = "🔸 #3 Tertiary Relief Corridor"
            elif pressure >= 1.0:
                color = [249, 115, 22, 245] # Partial Highlight: Acute Overcapacity Tangerine-Coral
                elevation = max(9500.0, demand * 2.2)
                status = "⚠️ Acute Capacity Pressure (>100% Saturated)"
            elif pressure >= 0.80:
                color = [251, 191, 36, 240] # Partial Highlight: Stressed Capacity Warm Amber
                elevation = max(6500.0, demand * 1.6)
                status = "⚡ Stressed Capacity (80–100% Saturated)"
            else:
                color = [240, 249, 255, 245] # Ultra-light luminous ice-platinum (highly noticeable & clickable)
                elevation = max(3500.0, demand * 1.2)
                status = "🟢 Standard District (Click to activate as Hotspot)"

            cap_val = float(row.get("sustainable_capacity", demand * 1.2))
            district_tooltip_clean = (
                f"<b>{d_name}</b> ({row['state_name']})<br/>"
                f"{status}<br/>"
                f"Peak Demand: {demand:,.0f} pax/day<br/>"
                f"Capacity Pressure: {pressure * 100:.1f}%<br/>"
                f"Hotel AOR: {aor:.1f}% | Poverty: {pov:.1f}%<br/>"
                f"Click cylinder to activate as Hotspot"
            )

            col_records.append({
                "name": d_name,
                "district_name": dist_name,
                "state_name": row["state_name"],
                "lat": lat,
                "lon": lon,
                "demand": demand,
                "daily_demand_peak": demand,
                "demand_formatted": f"{demand:,.0f}",
                "sustainable_capacity": cap_val,
                "cap_formatted": f"{cap_val:,.0f}",
                "rooms_formatted": f"{float(row.get('total_rooms', 0.0)):,.0f}",
                "pressure_formatted": f"{pressure * 100:.1f}",
                "aor_formatted": f"{aor:.1f}",
                "pov_formatted": f"{pov:.1f}",
                "elevation": elevation,
                "color": color,
                "status": status,
                "ksas_detail": "",
                "tooltip_html": district_tooltip_clean
            })

        col_df = pd.DataFrame(col_records)
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=col_df,
            id="district_columns",
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1,
            radius=3500,
            disk_resolution=24,
            get_fill_color="color",
            opacity=0.85,
            wireframe=False,
            pickable=True,
            auto_highlight=True,
            # Hide cylinders at street zoom so they never block
            # street-level context; labels take over.
            maxZoom=CYLINDER_HIDE_ZOOM,
        )
        layers.append(column_layer)
        # Street-level district labels: appear only after cylinders hide.
        # Near-ground elevation (no tower) keeps hotspot/relief identity
        # readable without any 3D occlusion.
        label_close_df = pd.DataFrame([{
            "lon": r["lon"],
            "lat": r["lat"],
            "name": r["name"],
            "label_elev": 800.0,
        } for r in col_records])
        district_close_labels = pdk.Layer(
            "TextLayer",
            data=label_close_df,
            id="district_close_labels",
            get_position=["lon", "lat", "label_elev"],
            get_text="name",
            get_size=15,
            size_min_pixels=11,
            size_max_pixels=18,
            get_color=[248, 250, 252, 255],
            get_angle=0,
            font_family="Arial, sans-serif",
            font_weight="bold",
            pickable=False,
            minZoom=CYLINDER_HIDE_ZOOM,
            parameters={"depthTest": False},
        )
        layers.append(district_close_labels)

    # -----------------------------------------------------------------
    # PLANMALAYSIA RFN KSAS OVERLAY (Official Statutory WMS / Raster Draped on Map)
    # -----------------------------------------------------------------
    # Rendered at Tier 0 (ground plane) BEFORE ground transit routes and tensile
    # arcs so that the route from hotspot to other cities is shown above the KSAS overlay.
    # Opacity tuned to 0.20 so the raster serves as subtle context without overpowering
    # roads, terrain, and high-visibility transit corridors.
    if _shown("ksas_overlay"):
        ksas_overlay_layer = pdk.Layer(
            "BitmapLayer",
            id="ksas_overlay",
            image=get_ksas_overlay_image_uri(),
            bounds=[99.6, 1.2, 104.6, 6.8],
            opacity=0.20,
            pickable=False
        )
        layers.append(ksas_overlay_layer)
        ksas_borneo_layer = pdk.Layer(
            "BitmapLayer",
            id="ksas_overlay_borneo",
            image=get_ksas_borneo_image_uri(),
            bounds=[109.4, 0.8, 119.5, 7.5],
            opacity=0.20,
            pickable=False
        )
        layers.append(ksas_borneo_layer)

    # -----------------------------------------------------------------
    # 3D GROUND TRANSIT ROUTES (PathLayer with Real Traffic Colors)
    # -----------------------------------------------------------------
    path_records = []
    for idx, alt in enumerate(alternatives[:3]):
        a_lat = float(alt.get("lat", 4.8500))
        a_lon = float(alt.get("lon", 100.7333))
        a_name = str(alt.get("candidate_name", alt.get("destination_name", f"Alternative #{idx+1}")))
        is_rail, corridor, nominal_mins, t_mode = _truthful_corridor(
            h_lat, h_lon, a_lat, a_lon, h_name, a_name, alt)

        # Calculate data-grounded traffic profile
        traffic_prof = get_corridor_traffic_profile(
            origin_coords=[h_lat, h_lon],
            dest_coords=[a_lat, a_lon],
            origin_name=h_name,
            dest_name=a_name,
            travel_time_nominal_mins=nominal_mins,
            google_maps_api_key=google_maps_api_key
        )
        if is_rail:
            traffic_prof = dict(traffic_prof)
            traffic_prof["summary"] = f"{t_mode} • ~{int(round(nominal_mins))} mins station-to-station + access"

        poly_coords = None
        if is_rail:
            try:
                poly_coords = traffic_engine.build_rail_path_coords(
                    h_lat, h_lon, a_lat, a_lon,
                    rail_detail=corridor.get("rail_detail") if corridor else None)
            except AttributeError:
                poly_coords = None
        if not is_rail:
            poly_coords = traffic_prof.get("polyline_coords") if traffic_prof else None
        if poly_coords and len(poly_coords) >= 2:
            path_coords = [[pt[1], pt[0]] for pt in poly_coords]
        else:
            waypoints = generate_realistic_transit_waypoints(
                h_lat, h_lon, a_lat, a_lon, is_rail=is_rail,
                orig_name=h_name, dest_name=a_name)
            # Convert waypoints to [lon, lat] format required by PyDeck PathLayer
            path_coords = [[wp[1], wp[0]] for wp in waypoints]

        if is_rail:
            path_color = [56, 189, 248, 255] # Luminous Electric Blue / Rail Tie
            line_width = 8
        else:
            c = traffic_prof["color_rgb"]
            # Ensure full 255 alpha for vivid route contrast against the ground map
            path_color = [c[0], c[1], c[2], 255] if len(c) >= 3 else [16, 185, 129, 255]
            line_width = 7

        path_records.append({
            "path": path_coords,
            "corridor": f"{h_name} → {a_name} ({t_mode})",
            "traffic_status": traffic_prof["summary"],
            "color": path_color,
            "width": line_width
        })

    if path_records and _shown("ground_transit_routes"):
        path_df = pd.DataFrame(path_records)
        ground_routes_layer = pdk.Layer(
            "PathLayer",
            data=path_df,
            id="ground_transit_routes",
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=6,
            joint_rounded=True,
            cap_rounded=True,
            pickable=False
        )
        layers.append(ground_routes_layer)

    # -----------------------------------------------------------------
    # 3D TENSILE ARCS (ArcLayer Leaping Over Terrain)
    # -----------------------------------------------------------------
    arcs_records = []
    for idx, alt in enumerate(alternatives[:3]):
        a_lat = float(alt.get("lat", 4.8500))
        a_lon = float(alt.get("lon", 100.7333))
        a_name = str(alt.get("candidate_name", alt.get("destination_name", f"Alternative #{idx+1}")))
        a_score = float(alt.get("final_wsm_score", 85.0))

        target_color = [16, 185, 129, 255] if idx == 0 else ([2, 132, 199, 240] if idx == 1 else [245, 158, 11, 220])

        arcs_records.append({
            "source_name": h_name,
            "target_name": a_name,
            "source_coords": [h_lon, h_lat],
            "target_coords": [a_lon, a_lat],
            "score": a_score,
            "width": max(4, 8 - idx * 2),
            "source_color": [239, 68, 68, 255],
            "target_color": target_color
        })

    if arcs_records and _shown("redistribution_arcs"):
        arcs_df = pd.DataFrame(arcs_records)
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=arcs_df,
            id="redistribution_arcs",
            get_source_position="source_coords",
            get_target_position="target_coords",
            get_source_color="source_color",
            get_target_color="target_color",
            get_width="width",
            pickable=False
        )
        layers.append(arc_layer)

    # -----------------------------------------------------------------
    # KSAS ALERT DISKS (red inside, white borders — nothing else)
    # -----------------------------------------------------------------
    # Rendered LAST (topmost) so ground routes and tensile arcs never bury
    # the alert. pickable=False so district/state cylinders keep exclusive
    # click-to-select.
    for _lyr in build_ksas_pydeck_layers(h_name, h_lat, h_lon,
                                         watch_keys=ksas_watch_keys):
        if _shown(_lyr.id):
            layers.append(_lyr)

    _default_zoom = 7.2 if lod_mode != "state" else 5.8
    try:
        _z = float(map_zoom) if map_zoom is not None else _default_zoom
    except (TypeError, ValueError):
        _z = _default_zoom
    view_state = pdk.ViewState(
        latitude=h_lat if lod_mode != "state" else 4.2105,
        longitude=h_lon if lod_mode != "state" else 102.5,
        zoom=_z,
        pitch=46 if lod_mode != "state" else 35,
        bearing=10
    )

    # Single unified hover template serving state columns and district
    # cylinders alike.
    tooltip_clean_template = (
        "{name} ({state_name})\n"
        "{status}\n"
        "----------------------------\n"
        "Demand: {demand_formatted} pax/day | Capacity: {cap_formatted}\n"
        "Pressure: {pressure_formatted}% | AOR: {aor_formatted}%\n"
        "{ksas_detail}"
    )

    tooltip_spec = {
        "html": tooltip_clean_template,
        "text": tooltip_clean_template,
        "style": {
            "backgroundColor": "#0f172a",
            "color": "#f8fafc",
            "borderRadius": "8px",
            "boxShadow": "0 8px 20px rgba(0,0,0,0.6)",
            "fontSize": "12px",
            "padding": "8px 12px",
            "lineHeight": "1.5",
            "whiteSpace": "pre-line"
        }
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip_spec,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    )
    return deck

