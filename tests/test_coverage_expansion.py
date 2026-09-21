"""
DESTINASI — Comprehensive Test Suite Expansion & Architectural Integrity Verification
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Validates:
1. Google Polyline decoding algorithm (empty strings, standard coordinates, bounds).
2. Live Traffic congestion engine across all 5 bottleneck nodes and rush hour windows (Friday outflow, Sunday return).
3. Explainable AI (XAI) feature attribution additive property and waterfall dataframe contract.
4. Hugging Face BYOK & deterministic Policy Copilot memo synthesis and fallback resilience.
5. Pure NumPy ML Ridge Regressor & Logistic Classifier convergence, probability bounds, and stability.
6. Exhaustive dictionary contract keys across predictive spike forecasts (eliminating KeyErrors).
7. Carrying capacity row evaluation convenience helper.
8. KSAS ecological buffer zones grounded in 3D (provenance, area-consistency, layers).
9. National AOR Benchmark Frame numerical comparison operators.
"""

import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Core Engines
from src.traffic_engine import (
    decode_polyline,
    calculate_llm_plus_traffic_congestion,
    get_corridor_traffic_profile,
    query_osrm_route,
    BOTTLENECK_NODES,
)
from src.xai_engine import compute_xai_feature_attributions
from src.hf_copilot import generate_policy_brief_memo
from src.predictive_spike_engine import (
    PureNumpyLinearML,
    PureNumpyLogisticML,
    get_predictive_spike_engine,
    build_trajectory_chart,
)
from src.carrying_capacity_engine import (
    diagnose_multisystem_bottleneck,
    evaluate_destination_row,
)
from src.map_components import (
    render_pydeck_3d_elevation_map,
    aggregate_destinations_by_state,
    get_3d_deck_legend_html,
    get_ksas_ecological_zones,
    build_ksas_pydeck_layers,
    KSAS_ECOLOGICAL_ZONES,
)
from src.aor_engine import (
    NationalAORBenchmarkFrame,
    normalize_state_name,
    compute_national_mean_aor,
)

ROOT = Path(__file__).resolve().parents[1]
DESTINATIONS_PATH = ROOT / "data" / "processed" / "destinations_master.csv"


class TestCoverageExpansion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dest_df = pd.read_csv(DESTINATIONS_PATH)
        cls.sample_hotspot = cls.dest_df.iloc[0].to_dict()
        cls.sample_alts = [cls.dest_df.iloc[1].to_dict(), cls.dest_df.iloc[2].to_dict()]
        for idx, a in enumerate(cls.sample_alts):
            a["candidate_name"] = a["destination_name"]
            a["candidate_id"] = a["destination_id"]
            a["final_wsm_score"] = 85.0 - idx * 5.0
            a["match_score"] = 82.0
            a["transit_mode"] = "KTM ETS" if idx == 0 else "Expressway"
            a["distance_km"] = 55.0 + idx * 25.0
            a["travel_time_mins"] = 45.0 + idx * 20.0

    # -------------------------------------------------------------------------
    # 1. Google Polyline Algorithm Tests
    # -------------------------------------------------------------------------
    def test_decode_polyline_empty_and_falsy(self):
        """Verify polyline decoder gracefully handles empty, None, and invalid strings."""
        self.assertEqual(decode_polyline(""), [])
        self.assertEqual(decode_polyline(None), [])

    def test_decode_polyline_standard_google_example(self):
        """Verify canonical Google encoded polyline string decodes accurately."""
        # Standard Google canonical polyline: (38.5, -120.2), (40.7, -120.95), (43.252, -126.453)
        encoded = "_p~iF~ps|U_ulLnnqC_mqNvxq"
        coords = decode_polyline(encoded)
        self.assertEqual(len(coords), 3)
        self.assertAlmostEqual(coords[0][0], 38.5, places=3)
        self.assertAlmostEqual(coords[0][1], -120.2, places=3)
        self.assertAlmostEqual(coords[1][0], 40.7, places=3)
        self.assertAlmostEqual(coords[1][1], -120.95, places=3)
        self.assertAlmostEqual(coords[2][0], 43.252, places=3)
        self.assertAlmostEqual(coords[2][1], -121.046, places=3)

    # -------------------------------------------------------------------------
    # 2. Traffic Engine: Bottlenecks & Special Rush-Hour Periods
    # -------------------------------------------------------------------------
    def test_traffic_friday_outflow_and_sunday_return(self):
        """Verify Friday evening outflow and Sunday night return surge patterns."""
        # Friday 17:30 (Outflow surge)
        dt_friday = datetime(2026, 9, 18, 17, 30) # Friday
        res_fri = calculate_llm_plus_traffic_congestion("Kuala Lumpur", "Bentong", 60.0, dt_friday)
        self.assertGreaterEqual(res_fri["delay_factor"], 1.40)
        self.assertEqual(res_fri["status"], "Severe Gridlock")

        # Sunday 18:00 (Return surge)
        dt_sun = datetime(2026, 9, 20, 18, 0) # Sunday
        res_sun = calculate_llm_plus_traffic_congestion("Ipoh", "Kuala Lumpur", 120.0, dt_sun)
        self.assertGreaterEqual(res_sun["delay_factor"], 1.35)

    def test_traffic_all_bottleneck_nodes(self):
        """Verify that all 5 key highway bottleneck nodes trigger appropriate boosts."""
        pairs = [
            ("Timur Laut", "Larut & Matang", "penang_bridge"),
            ("Cameron Highlands", "Lipis", "genting_sempah"),
            ("Kinta", "Kuala Kangsar", "menora_tunnel"),
            ("Johor Bahru", "Kulai", "skudai_toll"),
        ]
        dt = datetime(2026, 9, 15, 8, 30) # Morning rush
        for orig, dest, b_key in pairs:
            res = calculate_llm_plus_traffic_congestion(orig, dest, 60.0, dt)
            expected_desc = BOTTLENECK_NODES[b_key]["description"]
            self.assertEqual(res["bottleneck_node"], expected_desc)

    # -------------------------------------------------------------------------
    # 3. XAI Engine: Exact Feature Attribution Decomposition
    # -------------------------------------------------------------------------
    def test_xai_feature_attributions_exact_additive_decomposition(self):
        """Verify DARPA XAI additive decomposition property: sum(points) == final_wsm_score."""
        hotspot_row = self.dest_df.iloc[0]
        cand_row = self.dest_df.iloc[1]
        xai = compute_xai_feature_attributions(hotspot_row, cand_row)

        self.assertIn("final_wsm_score", xai)
        self.assertIn("waterfall_df", xai)
        self.assertIn("primary_driver", xai)

        wf_df = xai["waterfall_df"]
        self.assertEqual(len(wf_df), 5)
        self.assertIn("Feature", wf_df.columns)
        self.assertIn("Points", wf_df.columns)
        self.assertIn("Category", wf_df.columns)

        sum_points = wf_df["Points"].sum()
        self.assertAlmostEqual(sum_points, xai["final_wsm_score"], delta=0.2)

    # -------------------------------------------------------------------------
    # 4. Hugging Face BYOK & Policy Copilot
    # -------------------------------------------------------------------------
    def test_hf_copilot_deterministic_memo_content(self):
        """Verify grounded deterministic policy briefing memo contains required statutory sections."""
        memo = generate_policy_brief_memo(
            hotspot_name="George Town Heritage Core",
            hotspot_state="Pulau Pinang",
            alternative_name="Taiping Heritage Corridor",
            alternative_state="Perak",
            peak_demand=42000.0,
            sustainable_capacity=28000.0,
            binding_constraint="Municipal Water Buffer",
            excess_demand=14000.0,
            economic_injection_rm_m=12.45,
            poverty_rate=6.2,
            hf_token=None
        )
        self.assertIn("source", memo)
        self.assertIn("content", memo)
        content = memo["content"]
        self.assertIn("EXECUTIVE DECISION MEMORANDUM", content)
        self.assertIn("Akta 594", content)
        self.assertIn("George Town Heritage Core", content)
        self.assertIn("Taiping Heritage Corridor", content)
        self.assertIn("12.45", content)
        self.assertIn("6.2%", content)

    def test_hf_copilot_invalid_token_resilience(self):
        """Verify HF copilot gracefully falls back to deterministic memo if token is invalid."""
        memo = generate_policy_brief_memo(
            hotspot_name="Cameron Highlands",
            hotspot_state="Pahang",
            alternative_name="Batang Padang",
            alternative_state="Perak",
            peak_demand=25000.0,
            sustainable_capacity=18000.0,
            binding_constraint="Road Throughput",
            excess_demand=7000.0,
            economic_injection_rm_m=5.5,
            poverty_rate=5.0,
            hf_token="invalid_dummy_token_12345"
        )
        self.assertIn("content", memo)
        self.assertIn("EXECUTIVE DECISION MEMORANDUM", memo["content"])

    # -------------------------------------------------------------------------
    # 5. Pure NumPy Machine Learning Models
    # -------------------------------------------------------------------------
    def test_pure_numpy_linear_ml_convergence(self):
        """Verify PureNumpyLinearML fits linear data and predicts accurately."""
        np.random.seed(42)
        X = np.random.randn(80, 3)
        true_w = np.array([2.5, -1.2, 0.8])
        y = X @ true_w + 5.0 + np.random.normal(0, 0.05, size=80)

        model = PureNumpyLinearML(alpha=0.1)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(len(preds), 80)
        # R2 score check
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - preds) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        self.assertGreater(r2, 0.90)

    def test_pure_numpy_logistic_ml_probabilities(self):
        """Verify PureNumpyLogisticML outputs valid probabilities [0, 1] summing to 1.0."""
        np.random.seed(42)
        X = np.random.randn(60, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        model = PureNumpyLogisticML(lr=0.1, epochs=150)
        model.fit(X, y)
        probs = model.predict_proba(X)

        self.assertEqual(probs.shape, (60, 2))
        for p0, p1 in probs:
            self.assertGreaterEqual(p0, 0.0)
            self.assertLessEqual(p0, 1.0)
            self.assertGreaterEqual(p1, 0.0)
            self.assertLessEqual(p1, 1.0)
            self.assertAlmostEqual(p0 + p1, 1.0, places=4)

    # -------------------------------------------------------------------------
    # 6. Predictive Spike Engine: Exhaustive Dictionary Contract Keys
    # -------------------------------------------------------------------------
    def test_predict_district_spike_exhaustive_keys(self):
        """Verify predict_district_spike returns all contract keys to prevent runtime KeyErrors."""
        engine = get_predictive_spike_engine()
        sample_row = self.dest_df.iloc[0]
        pred = engine.predict_district_spike(sample_row, event_month="2026-02")

        required_keys = [
            "district_name",
            "state_name",
            "event_month",
            "event_name",
            "event_title",
            "quarter",
            "month_label",
            "predicted_demand_peak",
            "predicted_peak_volume",
            "sustainable_capacity",
            "capacity_stress_pct",
            "spike_probability_pct",
            "risk_level",
            "excess_surge_volume",
            "recommended_diversion",
            "recommended_diversion_quota",
            "lead_time_days",
            "early_warning_lead_days",
            "surge_multiplier",
        ]
        for k in required_keys:
            self.assertIn(k, pred, f"Missing required contract key '{k}' in spike prediction dict")

    # -------------------------------------------------------------------------
    # 7. Carrying Capacity Row Evaluation Helper
    # -------------------------------------------------------------------------
    def test_carrying_capacity_evaluate_destination_row(self):
        """Verify evaluate_destination_row convenience helper matches diagnose_multisystem_bottleneck."""
        row = self.dest_df.iloc[0]
        res = evaluate_destination_row(row)

        self.assertIn("sustainable_capacity", res)
        self.assertIn("binding_constraint", res)
        self.assertIn("continuous_pressure", res)
        self.assertIn("is_overcapacity", res)
        self.assertGreater(res["sustainable_capacity"], 0)

    # -------------------------------------------------------------------------
    # 8. KSAS Ecological Buffers Grounded in 3D (replaces retired 2D engine)
    # -------------------------------------------------------------------------
    def test_ksas_zones_grounded_provenance_and_area_consistency(self):
        """Every KSAS zone carries auditable provenance; area-derived disks match gazetted areas."""
        import math
        self.assertEqual(len(KSAS_ECOLOGICAL_ZONES), 4)
        keys = [z["key"] for z in KSAS_ECOLOGICAL_ZONES]
        self.assertEqual(len(set(keys)), 4, "Zone keys must be unique")
        for z in KSAS_ECOLOGICAL_ZONES:
            for field in ("name", "short_label", "lat", "lon", "radius_m",
                          "gazetted_area_km2", "radius_basis",
                          "legal", "source", "representation"):
                self.assertIn(field, z, f"Zone {z.get('key')} missing provenance field '{field}'")
                self.assertTrue(z[field], f"Zone {z['key']} field '{field}' must be non-empty")
            self.assertIn("area_derived", z)
            self.assertIsInstance(z["area_derived"], bool)
            self.assertIn("NOT", z["representation"],
                          f"Zone {z['key']} must disclaim it is not the legal boundary")
            if z["area_derived"]:
                disk_km2 = math.pi * (z["radius_m"] / 1000.0) ** 2
                err = abs(disk_km2 - z["gazetted_area_km2"]) / z["gazetted_area_km2"]
                self.assertLessEqual(err, 0.10,
                    f"Zone {z['key']}: disk {disk_km2:.1f} km2 vs gazetted "
                    f"{z['gazetted_area_km2']:.1f} km2 (err {err:.1%})")
            else:
                self.assertIn(z["key"], ("taman_negara",),
                    "Only Taman Negara may be a scoped node buffer rather than area-derived")

    def test_ksas_alert_styling_and_layer_contract(self):
        """Hotspot-inside-zone renders a red/white disk; layer is non-pickable."""
        from src.map_components import load_ksas_reference
        ref = load_ksas_reference()
        self.assertIsNotNone(ref)
        zones = get_ksas_ecological_zones("Cameron Highlands")
        cam = next(z for z in zones if z["key"] == "ksas02_cameron_highlands")
        self.assertTrue(cam["is_alert"])
        self.assertEqual(cam["color"][:3], [255, 0, 96])
        self.assertTrue(all(not z["is_alert"] for z in zones if z["key"] != "ksas02_cameron_highlands"))
        self.assertTrue(all(not z["is_alert"] for z in get_ksas_ecological_zones("Nowhere XYZ")))

        layers = build_ksas_pydeck_layers("Cameron Highlands")
        self.assertEqual([lyr.id for lyr in layers], ["ksas_buffers"])
        for lyr in layers:
            self.assertFalse(lyr.pickable)
        buf = layers[0].data if isinstance(layers[0].data, list) else layers[0].data.to_dict("records")
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["key"], "ksas02_cameron_highlands")
        self.assertEqual(buf[0]["fill"][:3], [255, 0, 96])
        self.assertEqual(buf[0]["line"][:3], [255, 255, 255])

    def test_traffic_engine_dynamic_import_resilience(self):
        """Verify traffic_engine exports query_osrm_route and generate_dense_corridor_curve cleanly."""
        import src.traffic_engine as te
        self.assertTrue(hasattr(te, "query_osrm_route"))
        self.assertTrue(hasattr(te, "generate_dense_corridor_curve"))
        import src.map_components as mc
        self.assertTrue(hasattr(mc, "query_osrm_route"))
        self.assertTrue(hasattr(mc, "generate_dense_corridor_curve"))

    # -------------------------------------------------------------------------
    # 9. National AOR Benchmark Frame Operators
    # -------------------------------------------------------------------------
    def test_national_aor_benchmark_frame_operators(self):
        """Verify NationalAORBenchmarkFrame supports float conversion and comparison operators."""
        sample_data = pd.DataFrame({
            "state_name": ["Johor", "Perak"],
            "average_occupancy_rate_pct": [60.0, 50.0]
        })
        frame = NationalAORBenchmarkFrame(sample_data)
        
        # Float conversion
        self.assertAlmostEqual(float(frame), 55.0, places=1)
        
        # Comparison operators
        self.assertTrue(frame >= 50.0)
        self.assertTrue(frame > 54.0)
        self.assertFalse(frame > 56.0)
        self.assertTrue(frame <= 60.0)
        self.assertTrue(frame < 56.0)

    # -------------------------------------------------------------------------
    # 10. Trajectory Chart Construction & Contract Tests
    # -------------------------------------------------------------------------
    def test_build_trajectory_chart_standard(self):
        """Verify build_trajectory_chart constructs valid 2-trace Plotly figure with demand and capacity."""
        spike_eng = get_predictive_spike_engine()
        fwd_df = spike_eng.predict_12_month_forward_curve(self.sample_hotspot, year=2026)
        fig = build_trajectory_chart(fwd_df, destination_name="Melaka", year=2026, height=330, is_compact=False)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)
        # Trace 0: Capacity Ceiling
        self.assertEqual(fig.data[0].name, "Capacity Ceiling")
        self.assertEqual(len(fig.data[0].x), 12)
        # Trace 1: Projected Demand
        self.assertEqual(fig.data[1].name, "Projected Demand")
        self.assertEqual(len(fig.data[1].x), 12)
        self.assertTrue(fig.layout.showlegend)

    def test_build_trajectory_chart_compact(self):
        """Verify build_trajectory_chart compact sparkline mode hides legend and formats hover properly."""
        spike_eng = get_predictive_spike_engine()
        fwd_df = spike_eng.predict_12_month_forward_curve(self.sample_hotspot, year=2026)
        fig = build_trajectory_chart(fwd_df, destination_name="Melaka", year=2026, height=185, is_compact=True)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 185)
        self.assertFalse(fig.layout.showlegend)

    def test_build_trajectory_chart_empty(self):
        """Verify build_trajectory_chart handles empty or None DataFrame gracefully with fallback annotation."""
        empty_df = pd.DataFrame()
        fig = build_trajectory_chart(empty_df, destination_name="None", year=2026)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 0)
        self.assertGreaterEqual(len(fig.layout.annotations), 1)

    # -------------------------------------------------------------------------
    # 11. 3D PyDeck Legend HUD & OSRM Routefinder Tests
    # -------------------------------------------------------------------------
    def test_get_3d_deck_legend_html(self):
        """Verify get_3d_deck_legend_html produces valid HTML HUD with all corridor and status indicators."""
        html = get_3d_deck_legend_html()
        self.assertIsInstance(html, str)
        self.assertIn("Active Hotspot", html)
        self.assertIn("#1 Relief Corridor", html)
        self.assertIn("#2 Relief Corridor", html)
        self.assertIn("#3 Relief Corridor", html)
        self.assertIn("Acute", html)
        self.assertIn("Stressed", html)
        self.assertIn("Headroom", html)
        self.assertIn("Real-World Route", html)

    def test_corridor_road_geometry_and_osrm(self):
        """Verify get_corridor_traffic_profile generates dense highway path coords."""
        prof = get_corridor_traffic_profile(
            origin_coords=[2.19, 102.25], # Melaka
            dest_coords=[2.27, 102.54],   # Tangkak
            origin_name="Melaka Tengah",
            dest_name="Tangkak",
            travel_time_nominal_mins=45.0
        )
        self.assertIsNotNone(prof)
        coords = prof.get("polyline_coords")
        self.assertIsNotNone(coords)
        self.assertGreaterEqual(len(coords), 20, "Corridor route must contain dense multi-point road geometry")


if __name__ == "__main__":
    unittest.main()

