"""
DESTINASI — KSAS Ecological Buffer Grounding Test Suite
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Why this file exists: the 3D KSAS disks are circular visualization proxies, NOT
surveyed cadastral boundaries (no open KSAS polygon shapefile ships in this repo).
These tests enforce that every zone stays honest:
1. Verified anchor coordinates (town / published study point / summit / gateway).
2. Radius derived from published gazetted/reference areas (circle-equivalent),
   except Taman Negara which is an explicitly scoped gateway node.
3. Full provenance on every zone (legal instrument, source, representation disclaimer).
4. Correct 3D layer contract (ids, non-pickable, data fidelity, serialization).
"""

import math
import unittest

import pandas as pd
import pydeck as pdk

from src.map_components import (
    KSAS_ECOLOGICAL_ZONES,
    aggregate_destinations_by_state,
    build_ksas_pydeck_layers,
    get_3d_deck_legend_html,
    get_ksas_ecological_zones,
    render_pydeck_3d_elevation_map,
)

# Independently verified anchors (town / publication / summit / gateway).
# Tolerance 0.05 deg (~5.5 km) keeps genuine anchor choices green while catching drift.
EXPECTED_ANCHORS = {
    "cameron_highlands": (4.4727, 101.3772),  # Tanah Rata town
    "matang_mangrove": (4.8342, 100.6167),  # PLOS ONE VJR point 4°50'3"N 100°37'0"E
    "kinabalu_geopark": (6.0750, 116.5583),  # Mount Kinabalu summit (4,095 m)
    "taman_negara": (4.3833, 102.4000),  # Kuala Tahan gateway vicinity
}

# Published gazetted/reference areas (km2) from UNESCO WHC / Sabah Parks /
# PLOS ONE / NASA / district statistics. Area-derived disks must agree within 10%.
EXPECTED_AREAS = {
    "cameron_highlands": 712.0,  # district-scale highland zone
    "matang_mangrove": 402.9,  # 40,288 ha MMFR
    "kinabalu_geopark": 753.7,  # 75,370 ha WH property
    "taman_negara": 4343.0,  # whole park (node-scoped; NOT area-derived)
}


class TestKSASGrounding(unittest.TestCase):

    def test_exactly_four_unique_zones(self):
        self.assertEqual(len(KSAS_ECOLOGICAL_ZONES), 4)
        keys = [z["key"] for z in KSAS_ECOLOGICAL_ZONES]
        self.assertEqual(sorted(keys), sorted(EXPECTED_ANCHORS.keys()))
        self.assertEqual(len(set(keys)), 4)

    def test_anchors_match_verified_coordinates(self):
        for z in KSAS_ECOLOGICAL_ZONES:
            exp_lat, exp_lon = EXPECTED_ANCHORS[z["key"]]
            self.assertAlmostEqual(z["lat"], exp_lat, delta=0.05,
                                   msg=f"{z['key']} lat drifted from verified anchor")
            self.assertAlmostEqual(z["lon"], exp_lon, delta=0.05,
                                   msg=f"{z['key']} lon drifted from verified anchor")
            # Anchors must sit inside Malaysia.
            self.assertTrue(0.8 <= z["lat"] <= 7.6, f"{z['key']} lat outside Malaysia")
            self.assertTrue(99.0 <= z["lon"] <= 119.6, f"{z['key']} lon outside Malaysia")

    def test_gazetted_areas_match_publications(self):
        for z in KSAS_ECOLOGICAL_ZONES:
            self.assertAlmostEqual(z["gazetted_area_km2"], EXPECTED_AREAS[z["key"]], delta=1.0,
                                   msg=f"{z['key']} gazetted area changed — re-verify source")

    def test_area_derived_disks_consistent_with_gazette(self):
        for z in KSAS_ECOLOGICAL_ZONES:
            if not z["area_derived"]:
                continue
            disk_km2 = math.pi * (z["radius_m"] / 1000.0) ** 2
            err = abs(disk_km2 - z["gazetted_area_km2"]) / z["gazetted_area_km2"]
            self.assertLessEqual(err, 0.10, f"{z['key']} disk/gazette mismatch {err:.1%}")

    def test_radii_sane_and_coarsely_rounded(self):
        for z in KSAS_ECOLOGICAL_ZONES:
            self.assertGreaterEqual(z["radius_m"], 5000)
            self.assertLessEqual(z["radius_m"], 40000)
            self.assertEqual(z["radius_m"] % 500, 0,
                             f"{z['key']} radius implies fake metre-level precision")

    def test_taman_negara_explicitly_node_scoped(self):
        z = next(x for x in KSAS_ECOLOGICAL_ZONES if x["key"] == "taman_negara")
        self.assertFalse(z["area_derived"])
        whole_park_equiv_m = math.sqrt(z["gazetted_area_km2"] / math.pi) * 1000.0
        self.assertGreater(whole_park_equiv_m, 30000)  # ~37.2 km
        self.assertLess(z["radius_m"], whole_park_equiv_m / 2.0,
                        "Node buffer must stay far below whole-park equivalent")
        self.assertIn("gateway", (z["radius_basis"] + z["representation"]).lower())

    def test_provenance_fields_complete_and_specific(self):
        for z in KSAS_ECOLOGICAL_ZONES:
            for field in ("legal", "source", "radius_basis", "representation"):
                self.assertTrue(isinstance(z[field], str) and len(z[field]) >= 20,
                                f"{z['key']}.{field} too thin to audit")
            self.assertIn("NOT", z["representation"],
                          f"{z['key']} must disclaim it is not the legal boundary")

    def test_zone_sources_name_publishers(self):
        src = {z["key"]: z["source"] for z in KSAS_ECOLOGICAL_ZONES}
        self.assertIn("UNESCO", src["kinabalu_geopark"])
        self.assertIn("Sabah Parks", src["kinabalu_geopark"])
        self.assertIn("PLOS", src["matang_mangrove"])
        self.assertIn("UNESCO", src["taman_negara"])
        self.assertIn("DST_043", src["cameron_highlands"])

    def test_alert_matrix(self):
        cases = [
            ("Cameron Highlands", "ksas02_cameron_highlands"),
            ("Kinabalu Park Resort", "ksas05_kinabalu_park"),
            ("Matang Mangrove Homestay", "ksas01_matang_mangrove"),
            ("Taman Negara Getaway", "ksas06_taman_negara"),
            ("George Town", "ksas05_george_town"),
            ("Gua Tempurung", "ksas05_gua_tempurung"),
        ]
        for hotspot, expected_key in cases:
            zones = get_ksas_ecological_zones(hotspot)
            alerted = [z["key"] for z in zones if z["is_alert"]]
            self.assertEqual(alerted, [expected_key], f"hotspot={hotspot}")
        self.assertEqual([z for z in get_ksas_ecological_zones("Johor Bahru") if z["is_alert"]], [])
        self.assertEqual([z for z in get_ksas_ecological_zones("") if z["is_alert"]], [])
        # Short tokens (<=4 chars) never trigger name matching on their own.
        self.assertEqual([z for z in get_ksas_ecological_zones("Ipoh") if z["is_alert"]], [])

    def test_proximity_alert_with_coordinates(self):
        """Hotspot coordinates inside a zone radius alert even without name overlap."""
        zones = get_ksas_ecological_zones("Somewhere Else", 4.59, 101.07)  # Ipoh city
        alerted = [z["key"] for z in zones if z["is_alert"]]
        self.assertIn("ksas03_kinta_limestone", alerted)
        zones = get_ksas_ecological_zones("Cameron Highlands", 4.4735, 101.3789)
        self.assertIn("ksas02_cameron_highlands",
                      [z["key"] for z in zones if z["is_alert"]])

    def test_tahap_colours_follow_rajah6(self):
        """Non-alert disks use PPP KSAS Rajah 6 Tahap codes (not the legacy teal)."""
        from src.map_components import KSAS_TAHAP_COLORS
        zones = get_ksas_ecological_zones("Johor Bahru")  # no alerts
        for z in zones:
            self.assertEqual(z["color"][:3], KSAS_TAHAP_COLORS[z["tahap"]])
            self.assertIn(z["tahap"], ("Tahap 1", "Tahap 2", "Tahap 3"))

    def test_single_alert_enforced(self):
        """Overlapping triggers collapse to exactly one red zone (nearest wins)."""
        zones = get_ksas_ecological_zones("Taman Negara Kinabalu Expedition")
        alerted = [z["key"] for z in zones if z["is_alert"]]
        self.assertEqual(len(alerted), 1)
        self.assertIn(alerted[0], ("ksas05_kinabalu_park", "ksas06_taman_negara"))
        # With coordinates the nearest containing zone wins.
        zones = get_ksas_ecological_zones("Cameron Highlands", 4.4735, 101.3789)
        alerted = [z["key"] for z in zones if z["is_alert"]]
        self.assertEqual(alerted, ["ksas02_cameron_highlands"])

    def test_layer_contract_and_data_fidelity(self):
        from src.map_components import _haversine_m, load_ksas_reference
        ref = load_ksas_reference()
        self.assertIsNotNone(ref)
        layers = build_ksas_pydeck_layers("Cameron Highlands")
        # Single alert-disk layer: red inside, white borders, nothing else.
        self.assertEqual([lyr.id for lyr in layers], ["ksas_buffers"])
        self.assertEqual(layers[0].type, "GeoJsonLayer")
        for lyr in layers:
            self.assertFalse(lyr.pickable)
        buf = layers[0].data if isinstance(layers[0].data, list) else layers[0].data.to_dict("records")
        self.assertEqual(len(buf), 1)
        self.assertTrue(buf[0]["is_alert"])
        self.assertEqual(buf[0]["key"], "ksas02_cameron_highlands")
        self.assertEqual(buf[0]["fill"][:3], [255, 0, 96])
        self.assertEqual(buf[0]["line"][:3], [255, 255, 255])
        self.assertEqual(buf[0]["lw"], 5)
        # Exact ring: every vertex sits on the circle (true size).
        ring = buf[0]["geometry"]["coordinates"][0]
        self.assertEqual(len(ring), 65)
        self.assertEqual(ring[0], ring[-1])
        cx = sum(p[0] for p in ring[:-1]) / 64
        cy = sum(p[1] for p in ring[:-1]) / 64
        for p in ring[:-1]:
            self.assertAlmostEqual(
                _haversine_m(cy, cx, p[1], p[0]), buf[0]["radius_m"],
                delta=buf[0]["radius_m"] * 0.02)

    def test_legend_documents_ksas(self):
        html = get_3d_deck_legend_html()
        self.assertIn("KSAS", html)

    def test_legend_alert_chip_names_hotspot_zone(self):
        """The legend shows a static red/white chip; no per-zone naming."""
        zones = get_ksas_ecological_zones("Cameron Highlands", 4.4735, 101.3789)
        alerts = [z for z in zones if z["is_alert"]]
        self.assertTrue(alerts, "Cameron Highlands must raise a KSAS alert")
        html = get_3d_deck_legend_html(alert_zones=alerts)
        self.assertIn("KSAS alert zone", html)
        self.assertNotIn("Cameron", html)
        plain = get_3d_deck_legend_html(alert_zones=[])
        self.assertIn("KSAS alert zone", plain)

    def test_legend_omits_hidden_layer_chips(self):
        """Chips for hidden layers disappear so the legend mirrors the map."""
        html = get_3d_deck_legend_html(visible_layers={"redistribution_arcs"})
        self.assertIn("Active Hotspot", html)
        self.assertNotIn("Real-World Route", html)
        self.assertNotIn("KSAS", html)
        html = get_3d_deck_legend_html(
            visible_layers={"redistribution_arcs", "ground_transit_routes",
                            "ksas_overlay", "ksas_buffers"})
        self.assertIn("Real-World Route", html)
        self.assertIn("KSAS Tahap 1", html)
        # Static red/white chip documents the alert disks.
        self.assertIn("KSAS alert zone", html)
        html_new = get_3d_deck_legend_html(
            visible_layers={"redistribution_arcs", "ground_transit_routes",
                            "ksas_overlay", "ksas_markers"})
        # Legacy "ksas_markers" id aliases to the alert-disk chip.
        self.assertIn("KSAS alert zone", html_new)

    def test_alert_disks_emphasised_for_visibility(self):
        """Alert disks carry stronger fill/edge styling than base Tahap disks."""
        zones = get_ksas_ecological_zones("Cameron Highlands", 4.4735, 101.3789)
        cam = next(z for z in zones if z["key"] == "ksas02_cameron_highlands")
        self.assertTrue(cam["is_alert"])
        self.assertEqual(cam["color"], [255, 0, 96, 215])
        self.assertEqual(cam["line_color"], [255, 255, 255, 255])
        self.assertEqual(cam["edge_w"], 5)
        calm = next(z for z in zones if not z["is_alert"])
        self.assertEqual(calm["edge_w"], 1)
        self.assertLess(calm["color"][3], cam["color"][3])

    def test_toggle_registry_matches_deck_layers(self):
        """Every MAP_TOGGLE_LAYERS id must exist in a fully-visible deck."""
        from src.map_components import MAP_TOGGLE_LAYERS, ROOT
        df = pd.read_csv(ROOT / "data" / "processed" / "destinations_master.csv")
        hotspot = df[df["district_name"] == "Cameron Highlands"].iloc[0].to_dict()
        from src.recommender_matcher import find_best_alternatives
        alts = find_best_alternatives(
            df[df["district_name"] == "Cameron Highlands"].iloc[0],
            df, top_n=3).to_dict("records")
        deck = render_pydeck_3d_elevation_map(hotspot, alts, df)
        ids = [lyr.id for lyr in deck.layers]
        for layer_id, _label in MAP_TOGGLE_LAYERS:
            self.assertIn(layer_id, ids)

    def test_routes_rendered_above_ksas_overlay(self):
        """The route from hotspot to other cities must be shown above the KSAS overlay."""
        from src.map_components import ROOT
        df = pd.read_csv(ROOT / "data" / "processed" / "destinations_master.csv")
        hotspot = df[df["district_name"] == "Cameron Highlands"].iloc[0].to_dict()
        from src.recommender_matcher import find_best_alternatives
        alts = find_best_alternatives(
            df[df["district_name"] == "Cameron Highlands"].iloc[0],
            df, top_n=3).to_dict("records")
        deck = render_pydeck_3d_elevation_map(hotspot, alts, df)
        ids = [lyr.id for lyr in deck.layers]
        self.assertIn("ksas_overlay", ids)
        self.assertIn("ground_transit_routes", ids)
        self.assertIn("redistribution_arcs", ids)

        idx_overlay = ids.index("ksas_overlay")
        idx_overlay_borneo = ids.index("ksas_overlay_borneo")
        idx_ground = ids.index("ground_transit_routes")
        idx_arcs = ids.index("redistribution_arcs")

        # Ground routes and redistribution arcs must be placed after (above) KSAS raster overlays in render order
        self.assertGreater(idx_ground, idx_overlay)
        self.assertGreater(idx_ground, idx_overlay_borneo)
        self.assertGreater(idx_arcs, idx_overlay)
        self.assertGreater(idx_arcs, idx_overlay_borneo)

        # KSAS overlay opacity must be subtle (<= 0.25) so routes and base map pop
        ksas_lyr = next(lyr for lyr in deck.layers if lyr.id == "ksas_overlay")
        self.assertLessEqual(ksas_lyr.opacity, 0.25)

        # Route lines must be thick and obvious (width_min_pixels >= 5)
        route_lyr = next(lyr for lyr in deck.layers if lyr.id == "ground_transit_routes")
        self.assertGreaterEqual(route_lyr.width_min_pixels, 5)

        # Columns must be solid (no wireframe strips) yet translucent for zooming
        col_lyr = deck.layers[0]
        self.assertLess(col_lyr.opacity, 1.0)
        self.assertFalse(col_lyr.wireframe)

    def test_visible_layers_filtering(self):
        """Hidden overlays are omitted; cylinders always render."""
        df = pd.DataFrame([
            {"destination_id": "D1", "destination_name": "Tanah Rata", "district_name": "Cameron Highlands",
             "state_name": "Pahang", "lat": 4.4735, "lon": 101.3789, "daily_demand_peak": 9000,
             "total_rooms": 8000, "continuous_pressure": 1.33, "latest_aor_pct": 74.0, "poverty_rate": 5.9,
             "sustainable_capacity": 8600},
            {"destination_id": "D2", "destination_name": "Tapah", "district_name": "Batang Padang",
             "state_name": "Perak", "lat": 4.2003, "lon": 101.2725, "daily_demand_peak": 3600,
             "total_rooms": 4000, "continuous_pressure": 0.60, "latest_aor_pct": 50.0, "poverty_rate": 6.0,
             "sustainable_capacity": 6000},
        ])
        hotspot = df.iloc[0].to_dict()
        alts = [{"candidate_name": "Tapah", "destination_name": "Tapah", "district_name": "Batang Padang",
                 "state_name": "Perak", "lat": 4.2003, "lon": 101.2725, "final_wsm_score": 80.0,
                 "transit_mode": "Road", "distance_km": 57.0, "travel_time_mins": 98.0}]
        deck = render_pydeck_3d_elevation_map(
            hotspot, alts, df, visible_layers={"ksas_buffers"})
        ids = [lyr.id for lyr in deck.layers]
        self.assertEqual(ids[0], "district_columns")
        self.assertIn("ksas_buffers", ids)
        self.assertNotIn("ground_transit_routes", ids)
        self.assertNotIn("redistribution_arcs", ids)
        self.assertTrue(len(deck.to_json()) > 0)

    def test_deck_integration_both_lods_and_serialization(self):
        df = pd.DataFrame([
            {"destination_id": "D1", "destination_name": "Tanah Rata", "district_name": "Cameron Highlands",
             "state_name": "Pahang", "lat": 4.4735, "lon": 101.3789, "daily_demand_peak": 9000,
             "total_rooms": 8000, "continuous_pressure": 1.33, "latest_aor_pct": 74.0, "poverty_rate": 5.9,
             "sustainable_capacity": 8600},
            {"destination_id": "D2", "destination_name": "Tapah", "district_name": "Batang Padang",
             "state_name": "Perak", "lat": 4.2003, "lon": 101.2725, "daily_demand_peak": 3600,
             "total_rooms": 4000, "continuous_pressure": 0.60, "latest_aor_pct": 50.0, "poverty_rate": 6.0,
             "sustainable_capacity": 6000},
        ])
        hotspot = df.iloc[0].to_dict()
        alts = [{"candidate_name": "Tapah", "destination_name": "Tapah", "district_name": "Batang Padang",
                 "state_name": "Perak", "lat": 4.2003, "lon": 101.2725, "final_wsm_score": 80.0,
                 "transit_mode": "Road", "distance_km": 57.0, "travel_time_mins": 98.0}]
        for mode in ("district", "state"):
            deck = render_pydeck_3d_elevation_map(hotspot, alts, df, lod_mode=mode)
            self.assertIsInstance(deck, pdk.Deck)
            ids = [lyr.id for lyr in deck.layers]
            self.assertIn("ksas_buffers", ids)
            self.assertTrue(deck.layers[0].pickable)
            self.assertTrue(all(not lyr.pickable for lyr in deck.layers[1:]))
            self.assertTrue(len(deck.to_json()) > 0)

    def test_ksas_proximity_filter_limits_rendered_disks(self):
        """With hotspot coords, only nearby zones + alerts render (no peninsula mass)."""
        from src.map_components import KSAS_DISPLAY_RADIUS_M, load_ksas_reference
        layers = build_ksas_pydeck_layers("Cameron Highlands", 4.4735, 101.3789)
        buf = layers[0].data if isinstance(layers[0].data, list) else layers[0].data.to_dict("records")
        ref = load_ksas_reference()
        self.assertEqual(len(buf), 1)
        keys = {r["key"] for r in buf}
        self.assertIn("ksas02_cameron_highlands", keys)  # alert always renders
        self.assertNotIn("ksas05_kinabalu_park", keys)  # Non-alert retired
        self.assertNotIn("ksas13_jelapang_padi", keys)  # Non-alert retired
        self.assertEqual(KSAS_DISPLAY_RADIUS_M, 150000)

    def test_ksas_true_scale_pixels(self):
        """Footprints are exact sphere geometry (no renderer unit dependence)."""
        from src.map_components import _haversine_m, load_ksas_reference
        layers = build_ksas_pydeck_layers("Cameron Highlands", 4.4735, 101.3789)
        poly = layers[0]
        self.assertEqual(poly.type, "GeoJsonLayer")
        buf = poly.data if isinstance(poly.data, list) else poly.data.to_dict("records")
        ref = load_ksas_reference()
        ref_radii = {r["key"]: int(r["radius_m"]) for _, r in ref.iterrows()}
        for rec in buf:
            self.assertEqual(rec["radius_m"], max(6000, min(40000, ref_radii[rec["key"]])))
            self.assertGreaterEqual(rec["radius_m"], 6000)
            self.assertLessEqual(rec["radius_m"], 40000)
            ring = rec["geometry"]["coordinates"][0]
            cx = sum(p[0] for p in ring[:-1]) / 64
            cy = sum(p[1] for p in ring[:-1]) / 64
            measured = [_haversine_m(cy, cx, p[1], p[0]) for p in ring[:-1]]
            self.assertLess(max(measured) - min(measured), rec["radius_m"] * 0.01)
            self.assertAlmostEqual(sum(measured) / len(measured), rec["radius_m"],
                                    delta=rec["radius_m"] * 0.02)

    def test_ksas_layers_render_topmost(self):
        """KSAS buffers/labels are last so routes/arcs never bury the alert."""
        from src.map_components import MAP_TOGGLE_LAYERS, ROOT
        df = pd.read_csv(ROOT / "data" / "processed" / "destinations_master.csv")
        hotspot = df[df["district_name"] == "Cameron Highlands"].iloc[0].to_dict()
        from src.recommender_matcher import find_best_alternatives
        alts = find_best_alternatives(
            df[df["district_name"] == "Cameron Highlands"].iloc[0],
            df, top_n=3).to_dict("records")
        deck = render_pydeck_3d_elevation_map(hotspot, alts, df)
        ids = [lyr.id for lyr in deck.layers]
        self.assertEqual(ids[-1], "ksas_buffers")
        buf = next(lyr for lyr in deck.layers if lyr.id == "ksas_buffers")
        recs = buf.data if isinstance(buf.data, list) else buf.data.to_dict("records")
        alerts = [r for r in recs if r["is_alert"]]
        self.assertEqual(len(alerts), 1, "exactly one red KSAS disk must render")
        self.assertEqual(alerts[0]["key"], "ksas02_cameron_highlands")
        self.assertEqual(alerts[0]["fill"][:3], [255, 0, 96])
        self.assertEqual(alerts[0]["line"][:3], [255, 255, 255])
        elevs = [r["elev"] for r in recs]
        self.assertEqual(len(set(elevs)), len(elevs))  # staggered: no coplanar flicker

    def test_watchlist_lights_acute_districts_before_any_click(self):
        """Landing emergence: acute districts inside KSAS turn their zone red."""
        from src.map_components import get_ksas_watchlist
        df = pd.DataFrame([
            {"district_name": "Cameron Highlands", "lat": 4.4735, "lon": 101.3789,
             "continuous_pressure": 1.33},
            {"district_name": "Batang Padang", "lat": 4.2003, "lon": 101.2725,
             "continuous_pressure": 0.60},
        ])
        watch = get_ksas_watchlist(df)
        keys = {z["key"] for z in watch}
        self.assertIn("ksas02_cameron_highlands", keys)
        cam = next(z for z in watch if z["key"] == "ksas02_cameron_highlands")
        self.assertTrue(cam["is_alert"])
        self.assertEqual(cam["color"][:3], [255, 0, 96])
        self.assertIn("Cameron Highlands", cam["watch_districts"])
        # Calm districts trigger nothing.
        calm = get_ksas_watchlist(df[df["district_name"] == "Batang Padang"])
        self.assertNotIn("ksas02_cameron_highlands", {z["key"] for z in calm})
        # Empty / malformed input never crashes the landing paint.
        self.assertEqual(get_ksas_watchlist(pd.DataFrame()), [])
        self.assertEqual(get_ksas_watchlist(None), [])

    def test_watch_keys_render_red_alongside_hotspot_alert(self):
        """Watchlist reds coexist with (never replace) the hotspot's alert."""
        layers = build_ksas_pydeck_layers(
            "Ipoh", 4.59, 101.09,
            watch_keys={"ksas02_cameron_highlands"})
        buf = layers[0].data if isinstance(layers[0].data, list) else layers[0].data.to_dict("records")
        reds = [r for r in buf if r["is_alert"]]
        red_keys = {r["key"] for r in reds}
        self.assertIn("ksas02_cameron_highlands", red_keys)
        for r in reds:
            self.assertEqual(r["fill"][:3], [255, 0, 96])
        elevs = [r["elev"] for r in buf]
        self.assertEqual(len(set(elevs)), len(elevs))

    def test_render_forwards_watch_keys(self):
        """Full deck honours ksas_watch_keys; default stays single-alert."""
        from src.map_components import ROOT
        df = pd.read_csv(ROOT / "data" / "processed" / "destinations_master.csv")
        hotspot = {"district_name": "Ipoh", "destination_name": "Ipoh",
                   "state_name": "Perak", "lat": 4.59, "lon": 101.09,
                   "daily_demand_peak": 5000, "total_rooms": 4000,
                   "continuous_pressure": 1.1, "latest_aor_pct": 60.0,
                   "poverty_rate": 5.0, "sustainable_capacity": 6000}
        deck = render_pydeck_3d_elevation_map(
            hotspot, [], df, ksas_watch_keys={"ksas02_cameron_highlands"})
        buf = next(lyr for lyr in deck.layers if lyr.id == "ksas_buffers")
        recs = buf.data if isinstance(buf.data, list) else buf.data.to_dict("records")
        red_keys = {r["key"] for r in recs if r["is_alert"]}
        self.assertIn("ksas03_kinta_limestone", red_keys)  # hotspot's own alert
        self.assertIn("ksas02_cameron_highlands", red_keys)  # watchlist red

    def test_watch_zones_survive_catchment_filter_on_landing(self):
        """Landing default sits far from KSAS: watch reds must still paint."""
        from src.map_components import ROOT, get_ksas_watchlist
        df = pd.read_csv(ROOT / "data" / "processed" / "destinations_master.csv")
        lang = df[df["district_name"] == "Langkawi"].iloc[0].to_dict()
        watch = get_ksas_watchlist(df)
        self.assertTrue(watch, "master data must yield watch zones")
        layers = build_ksas_pydeck_layers(
            str(lang.get("destination_name") or lang["district_name"]),
            lang["lat"], lang["lon"],
            watch_keys={z["key"] for z in watch})
        buf = layers[0].data if isinstance(layers[0].data, list) else layers[0].data.to_dict("records")
        red_keys = {r["key"] for r in buf if r["is_alert"]}
        for z in watch:
            self.assertIn(z["key"], red_keys)
        for r in buf:
            if r["is_alert"]:
                self.assertEqual(r["fill"][:3], [255, 0, 96])


    def test_legend_contains_no_emoji(self):
        """Legend HUD stays professional: no emoji codepoints anywhere."""
        zones = get_ksas_ecological_zones("Cameron Highlands", 4.4735, 101.3789)
        alerts = [z for z in zones if z["is_alert"]]
        for html in (get_3d_deck_legend_html(),
                     get_3d_deck_legend_html(alert_zones=alerts),
                     get_3d_deck_legend_html(visible_layers={"ksas_buffers"})):
            for emoji in ("🌿", "🛣", "🚆", "🚨", "⚠", "✅", "🔹", "🔸"):
                self.assertNotIn(emoji, html)
        self.assertIn("KSAS alert zone",
                      get_3d_deck_legend_html(alert_zones=alerts))

    def test_alert_disks_red_inside_white_borders(self):
        """One red-inside/white-border disk per alert, non-pickable, nothing else."""
        layers = build_ksas_pydeck_layers("Cameron Highlands", 4.4735, 101.3789)
        self.assertEqual([lyr.id for lyr in layers], ["ksas_buffers"])
        disks = layers[0]
        self.assertFalse(disks.pickable)
        recs = disks.data if isinstance(disks.data, list) else disks.data.to_dict("records")
        n_alerts = sum(1 for r in recs if r["is_alert"])
        self.assertEqual(len(recs), n_alerts)
        self.assertTrue(n_alerts >= 1)
        for r in recs:
            self.assertEqual(r["fill"][:3], [255, 0, 96])
            self.assertEqual(r["line"][:3], [255, 255, 255])
            self.assertEqual(r["lw"], 5)

    def test_manual_lod_single_layer_cylinder_hide(self):
        """Manual switch renders exactly one LOD layer; cylinders hide at 13."""
        from src.map_components import CYLINDER_HIDE_ZOOM, ROOT
        self.assertEqual(CYLINDER_HIDE_ZOOM, 13.0)
        df = pd.read_csv(ROOT / "data" / "processed" / "destinations_master.csv")
        hotspot = df[df["district_name"] == "Cameron Highlands"].iloc[0].to_dict()
        from src.recommender_matcher import find_best_alternatives
        alts = find_best_alternatives(
            df[df["district_name"] == "Cameron Highlands"].iloc[0],
            df, top_n=3).to_dict("records")
        deck_micro = render_pydeck_3d_elevation_map(hotspot, alts, df, lod_mode="district")
        ids_micro = [lyr.id for lyr in deck_micro.layers]
        self.assertIn("district_columns", ids_micro)
        self.assertNotIn("state_columns", ids_micro)
        self.assertIn("district_close_labels", ids_micro)
        self.assertIn("ksas_buffers", ids_micro)
        dist_lyr = next(lyr for lyr in deck_micro.layers if lyr.id == "district_columns")
        close_lyr = next(lyr for lyr in deck_micro.layers if lyr.id == "district_close_labels")
        self.assertEqual(dist_lyr.maxZoom, CYLINDER_HIDE_ZOOM)
        self.assertEqual(close_lyr.minZoom, CYLINDER_HIDE_ZOOM)
        deck_macro = render_pydeck_3d_elevation_map(hotspot, alts, df, lod_mode="state")
        ids_macro = [lyr.id for lyr in deck_macro.layers]
        self.assertIn("state_columns", ids_macro)
        self.assertNotIn("district_columns", ids_macro)
        self.assertIn("ksas_buffers", ids_macro)
        self.assertTrue(len(deck_micro.to_json()) > 0)
        self.assertTrue(len(deck_macro.to_json()) > 0)


if __name__ == "__main__":
    unittest.main()
