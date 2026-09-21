"""
DESTINASI — Milestone 1 (R1: 3D WebGL Cylinder Cockpit & Navigation)
Adversarial Stress-Testing & Empirical Verification Suite
Author: challenger_m1_1 (Empirical Challenger: critic, specialist)
Target: DOSM Datathon 2026
"""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import pydeck as pdk

from src.map_components import (
    aggregate_destinations_by_state,
    render_pydeck_3d_elevation_map,
    generate_realistic_transit_waypoints,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "destinations_master.csv"


class TestAll110DistrictsHotspotIsolation(unittest.TestCase):
    """Exhaustive sweep over all 110 Malaysian districts as active hotspot."""

    @classmethod
    def setUpClass(cls):
        cls.destinations_df = pd.read_csv(DATA_PATH)
        cls.num_districts = len(cls.destinations_df)
        assert cls.num_districts == 110, f"Expected 110 districts, found {cls.num_districts}"

    def test_sweep_all_110_districts_district_lod(self):
        """Test every single district as hotspot in district LOD mode (110 renders)."""
        for i in range(self.num_districts):
            hotspot = self.destinations_df.iloc[i].to_dict()
            h_dist = hotspot["district_name"]
            h_state = hotspot["state_name"]

            # Select 3 alternatives cyclically
            alt1 = self.destinations_df.iloc[(i + 1) % self.num_districts].to_dict()
            alt2 = self.destinations_df.iloc[(i + 2) % self.num_districts].to_dict()
            alt3 = self.destinations_df.iloc[(i + 3) % self.num_districts].to_dict()
            alts = [alt1, alt2, alt3]

            deck = render_pydeck_3d_elevation_map(
                hotspot, alts, self.destinations_df, lod_mode="district"
            )

            # 1. Deck validity & JSON serializability
            self.assertIsInstance(deck, pdk.Deck)
            deck_json_str = deck.to_json()
            self.assertTrue(len(deck_json_str) > 0)
            deck_data = json.loads(deck_json_str)

            # 2. Layer presence and IDs
            layer_ids = [layer["id"] for layer in deck_data.get("layers", [])]
            self.assertIn("district_columns", layer_ids)
            self.assertIn("ground_transit_routes", layer_ids)
            self.assertIn("redistribution_arcs", layer_ids)

            # 3. Column Layer properties
            col_layer = next(l for l in deck.layers if l.id == "district_columns")
            self.assertTrue(col_layer.pickable)
            self.assertEqual(col_layer.radius, 3500)

            col_records = (
                col_layer.data
                if isinstance(col_layer.data, list)
                else col_layer.data.to_dict("records")
            )
            self.assertEqual(len(col_records), 110)

            # 4. Color, elevation, and status assertions
            h_record = next(r for r in col_records if r["district_name"] == h_dist)
            self.assertEqual(h_record["color"], [239, 68, 68, 255], f"Hotspot {h_dist} must be crimson")
            self.assertGreaterEqual(h_record["elevation"], 24000.0)
            self.assertIn("🚨 Active Hotspot", h_record["status"])

            a1_record = next(r for r in col_records if r["district_name"] == alt1["district_name"])
            self.assertEqual(a1_record["color"], [16, 185, 129, 255])
            self.assertGreaterEqual(a1_record["elevation"], 19000.0)

            a2_record = next(r for r in col_records if r["district_name"] == alt2["district_name"])
            self.assertEqual(a2_record["color"], [2, 132, 199, 240])

            a3_record = next(r for r in col_records if r["district_name"] == alt3["district_name"])
            self.assertEqual(a3_record["color"], [245, 158, 11, 230])

            # 5. Tooltip HTML cleanliness (Zero template artifacts)
            for rec in col_records:
                html = rec["tooltip_html"]
                self.assertNotIn("{name}", html)
                self.assertNotIn("{district_name}", html)
                self.assertNotIn("{state_name}", html)
                self.assertNotIn("{status}", html)
                self.assertNotIn("{demand}", html)
                self.assertNotIn("{pressure", html)
                self.assertIn(rec["state_name"], html)

            # 6. Secondary layers non-pickable
            for layer in deck.layers:
                if layer.id != "district_columns":
                    self.assertFalse(layer.pickable)

    def test_sweep_all_110_districts_state_lod(self):
        """Test every single district as hotspot in state LOD mode (110 renders)."""
        for i in range(self.num_districts):
            hotspot = self.destinations_df.iloc[i].to_dict()
            h_state = hotspot["state_name"]

            alt1 = self.destinations_df.iloc[(i + 1) % self.num_districts].to_dict()
            alt2 = self.destinations_df.iloc[(i + 2) % self.num_districts].to_dict()
            alts = [alt1, alt2]

            deck = render_pydeck_3d_elevation_map(
                hotspot, alts, self.destinations_df, lod_mode="state"
            )

            # 1. Deck validity & JSON serializability
            self.assertIsInstance(deck, pdk.Deck)
            deck_json_str = deck.to_json()
            deck_data = json.loads(deck_json_str)

            # 2. Layer presence
            layer_ids = [layer["id"] for layer in deck_data.get("layers", [])]
            self.assertIn("state_columns", layer_ids)

            # 3. State Column Layer properties
            state_layer = next(l for l in deck.layers if l.id == "state_columns")
            self.assertTrue(state_layer.pickable)
            self.assertTrue(20000 <= state_layer.radius <= 25000)

            state_records = (
                state_layer.data
                if isinstance(state_layer.data, list)
                else state_layer.data.to_dict("records")
            )
            self.assertEqual(len(state_records), 16)

            # 4. Hotspot State color and elevation
            hs_record = next(r for r in state_records if r["state_name"] == h_state)
            self.assertEqual(hs_record["color"], [239, 68, 68, 255])
            self.assertGreaterEqual(hs_record["elevation"], 26000.0)
            self.assertIn("🚨 Hotspot State", hs_record["status"])

            # 5. Tooltip cleanliness
            for rec in state_records:
                html = rec["tooltip_html"]
                self.assertNotIn("{name}", html)
                self.assertNotIn("{status}", html)
                self.assertNotIn("{tot_demand", html)
                self.assertIn(rec["state_name"], html)


class TestStateAggregationInvariants(unittest.TestCase):
    """Empirical verification of LOD state aggregation conservation and geographic rules."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)
        cls.agg_df = aggregate_destinations_by_state(cls.df)

    def test_state_count_and_uniqueness(self):
        """Exactly 16 state records must be returned without duplicates."""
        self.assertEqual(len(self.agg_df), 16)
        self.assertEqual(self.agg_df["state_name"].nunique(), 16)

    def test_demand_conservation(self):
        """Sum of aggregated state peak demand must strictly match sum of 110 districts."""
        raw_demand_sum = self.df["daily_demand_peak"].sum()
        agg_demand_sum = self.agg_df["daily_demand_peak"].sum()
        self.assertAlmostEqual(agg_demand_sum, raw_demand_sum, places=1)

    def test_capacity_conservation(self):
        """Sum of sustainable capacity must strictly match sum of 110 districts."""
        raw_cap_sum = self.df["sustainable_capacity"].sum()
        agg_cap_sum = self.agg_df["sustainable_capacity"].sum()
        self.assertAlmostEqual(agg_cap_sum, raw_cap_sum, places=1)

    def test_hotel_rooms_conservation(self):
        """Sum of hotel rooms must strictly match sum of 110 districts."""
        raw_rooms_sum = self.df["total_rooms"].sum()
        agg_rooms_sum = self.agg_df["total_rooms"].sum()
        self.assertAlmostEqual(agg_rooms_sum, raw_rooms_sum, places=1)

    def test_district_count_conservation(self):
        """Sum of district counts across all 16 states must strictly equal 110."""
        self.assertEqual(self.agg_df["district_count"].sum(), 110)

    def test_geographic_centroid_bounds(self):
        """All computed state centroids must lie within Malaysian national boundaries."""
        for _, row in self.agg_df.iterrows():
            # Latitude: 1.0°N to 7.5°N
            self.assertTrue(
                1.0 <= row["lat"] <= 7.5,
                f"State {row['state_name']} lat {row['lat']} outside Malaysia bounds",
            )
            # Longitude: 99.5°E to 119.5°E
            self.assertTrue(
                99.5 <= row["lon"] <= 119.5,
                f"State {row['state_name']} lon {row['lon']} outside Malaysia bounds",
            )

    def test_representative_district_fidelity(self):
        """Each state's rep_district must be the district with maximum daily_demand_peak in that state."""
        for s_name, group in self.df.groupby("state_name"):
            expected_rep = group.sort_values(by="daily_demand_peak", ascending=False).iloc[0]["district_name"]
            agg_row = self.agg_df[self.agg_df["state_name"] == s_name].iloc[0]
            self.assertEqual(
                agg_row["rep_district"],
                expected_rep,
                f"State {s_name} rep_district {agg_row['rep_district']} != expected {expected_rep}",
            )


class TestBoundaryConditionsAndEdgeCases(unittest.TestCase):
    """Adversarial testing on boundary conditions, empty inputs, extreme values, and schema variations."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)
        cls.hotspot = cls.df.iloc[0].to_dict()
        cls.alternatives = [cls.df.iloc[1].to_dict(), cls.df.iloc[2].to_dict()]

    def test_empty_dataframe_aggregation(self):
        """aggregate_destinations_by_state on empty DataFrame returns empty DataFrame with expected columns."""
        res = aggregate_destinations_by_state(pd.DataFrame())
        self.assertTrue(res.empty)
        expected_cols = [
            "state_name", "district_name", "rep_district", "destination_name", "name",
            "lat", "lon", "daily_demand_peak", "sustainable_capacity",
            "total_rooms", "latest_aor_pct", "continuous_pressure",
            "poverty_rate", "district_count", "elevation", "tooltip_html"
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns)

    def test_empty_dataframe_deck_rendering(self):
        """render_pydeck_3d_elevation_map on empty DataFrame produces valid Deck and serializable JSON."""
        empty_df = pd.DataFrame(columns=self.df.columns)

        deck_d = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, empty_df, lod_mode="district"
        )
        self.assertIsInstance(deck_d, pdk.Deck)
        json_d = deck_d.to_json()
        self.assertTrue(isinstance(json_d, str) and len(json_d) > 0)

        deck_s = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, empty_df, lod_mode="state"
        )
        self.assertIsInstance(deck_s, pdk.Deck)
        json_s = deck_s.to_json()
        self.assertTrue(isinstance(json_s, str) and len(json_s) > 0)

    def test_zero_alternatives(self):
        """When alternatives list is empty, PathLayer and ArcLayer are cleanly omitted."""
        deck = render_pydeck_3d_elevation_map(
            self.hotspot, [], self.df, lod_mode="district"
        )
        layer_ids = [lyr.id for lyr in deck.layers]
        self.assertIn("district_columns", layer_ids)
        self.assertIn("ksas_buffers", layer_ids)
        self.assertNotIn("ground_transit_routes", layer_ids)
        self.assertNotIn("redistribution_arcs", layer_ids)
        self.assertEqual(deck.layers[0].id, "district_columns")
        json_str = deck.to_json()
        self.assertTrue(len(json_str) > 0)

    def test_many_alternatives_truncation(self):
        """When >3 alternatives are supplied, paths and arcs are strictly capped at 3."""
        many_alts = [self.df.iloc[i].to_dict() for i in range(1, 10)]
        deck = render_pydeck_3d_elevation_map(
            self.hotspot, many_alts, self.df, lod_mode="district"
        )
        path_layer = next(l for l in deck.layers if l.id == "ground_transit_routes")
        arc_layer = next(l for l in deck.layers if l.id == "redistribution_arcs")
        self.assertEqual(len(path_layer.data), 3)
        self.assertEqual(len(arc_layer.data), 3)

    def test_zero_demand_boundary(self):
        """Zero demand districts still maintain minimum elevation clamp for WebGL visibility."""
        zero_df = self.df.copy()
        zero_df["daily_demand_peak"] = 0.0

        deck_d = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, zero_df, lod_mode="district"
        )
        col_layer = deck_d.layers[0]
        col_records = col_layer.data if isinstance(col_layer.data, list) else col_layer.data.to_dict("records")
        for rec in col_records:
            self.assertGreaterEqual(rec["elevation"], 1500.0)

        deck_s = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, zero_df, lod_mode="state"
        )
        s_layer = deck_s.layers[0]
        s_records = s_layer.data if isinstance(s_layer.data, list) else s_layer.data.to_dict("records")
        for rec in s_records:
            self.assertGreaterEqual(rec["elevation"], 4000.0)

    def test_extreme_demand_boundary(self):
        """Extreme demand values (e.g. 10,000,000 pax/day) do not crash elevation scaling or serialization."""
        extreme_df = self.df.copy()
        extreme_df["daily_demand_peak"] = 10_000_000.0

        deck = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, extreme_df, lod_mode="district"
        )
        col_layer = deck.layers[0]
        col_records = col_layer.data if isinstance(col_layer.data, list) else col_layer.data.to_dict("records")
        h_rec = next(r for r in col_records if r["district_name"] == self.hotspot["district_name"])
        self.assertEqual(h_rec["elevation"], 50_000_000.0)
        self.assertTrue(len(deck.to_json()) > 0)

    def test_negative_demand_clamped(self):
        """Negative demand values are clamped to the minimum elevation threshold."""
        neg_df = self.df.copy()
        neg_df["daily_demand_peak"] = -99999.0

        deck = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, neg_df, lod_mode="district"
        )
        col_layer = deck.layers[0]
        col_records = col_layer.data if isinstance(col_layer.data, list) else col_layer.data.to_dict("records")
        for rec in col_records:
            self.assertGreaterEqual(rec["elevation"], 1500.0)

    def test_empty_hotspot_dictionary(self):
        """Passing empty hotspot dict uses default coordinates and labels without throwing exceptions."""
        deck = render_pydeck_3d_elevation_map(
            {}, self.alternatives, self.df, lod_mode="district"
        )
        self.assertIsInstance(deck, pdk.Deck)
        self.assertTrue(len(deck.to_json()) > 0)

    def test_minimal_columns_dataframe(self):
        """aggregate_destinations_by_state and render_pydeck survive DataFrame with only minimal columns."""
        min_df = self.df[["destination_name", "district_name", "state_name", "lat", "lon"]].copy()
        agg_min = aggregate_destinations_by_state(min_df)
        self.assertEqual(len(agg_min), 16)
        self.assertIn("daily_demand_peak", agg_min.columns)

        deck_d = render_pydeck_3d_elevation_map(self.hotspot, self.alternatives, min_df, lod_mode="district")
        self.assertIsInstance(deck_d, pdk.Deck)
        deck_s = render_pydeck_3d_elevation_map(self.hotspot, self.alternatives, min_df, lod_mode="state")
        self.assertIsInstance(deck_s, pdk.Deck)

    def test_pathlayer_coordinate_order_is_lon_lat(self):
        """PyDeck PathLayer requires [[lon, lat], ...] coordinates order."""
        deck = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, self.df, lod_mode="district"
        )
        path_layer = next(l for l in deck.layers if l.id == "ground_transit_routes")
        path_records = path_layer.data if isinstance(path_layer.data, list) else path_layer.data.to_dict("records")
        for r in path_records:
            coords = r["path"]
            self.assertGreaterEqual(len(coords), 2)
            for pt in coords:
                lon, lat = pt[0], pt[1]
                self.assertTrue(99.0 <= lon <= 120.0, f"Longitude {lon} out of range")
                self.assertTrue(1.0 <= lat <= 8.0, f"Latitude {lat} out of range")

    def test_arclayer_coordinate_order_is_lon_lat(self):
        """PyDeck ArcLayer requires [lon, lat] coordinates for source and target."""
        deck = render_pydeck_3d_elevation_map(
            self.hotspot, self.alternatives, self.df, lod_mode="district"
        )
        arc_layer = next(l for l in deck.layers if l.id == "redistribution_arcs")
        arc_records = arc_layer.data if isinstance(arc_layer.data, list) else arc_layer.data.to_dict("records")
        for r in arc_records:
            s_lon, s_lat = r["source_coords"]
            t_lon, t_lat = r["target_coords"]
            self.assertTrue(99.0 <= s_lon <= 120.0)
            self.assertTrue(1.0 <= s_lat <= 8.0)
            self.assertTrue(99.0 <= t_lon <= 120.0)
            self.assertTrue(1.0 <= t_lat <= 8.0)


if __name__ == "__main__":
    unittest.main()
