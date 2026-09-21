"""
Unit tests for DESTINASI Next-Gen Upgrades:
1. Live Traffic Engine (Google Maps & LLM/PLUS sensors)
2. International Inbound & Blended Economic Models
3. 3D WebGL LOD Cockpit (State & District levels) with Ground Transit Routes
"""

import unittest
from datetime import datetime
from pathlib import Path
import pandas as pd
import pydeck as pdk

from src.traffic_engine import (
    calculate_llm_plus_traffic_congestion,
    get_corridor_traffic_profile,
    BOTTLENECK_NODES
)
from src.economic_impact_model import (
    calculate_redistribution_economic_impact,
    get_top_inbound_markets,
    get_inbound_category_breakdown,
    INBOUND_SPEND_PER_NIGHT,
    INBOUND_ALOS,
    DTS_2024_SPEND_PER_NIGHT
)
from src.map_components import render_pydeck_3d_elevation_map


class TestNextGenEngines(unittest.TestCase):

    def test_llm_plus_traffic_engine_rush_hour(self):
        """Verify LLM/PLUS traffic calculation evaluates rush hours and returns valid attributes."""
        # Test morning rush hour (08:30)
        dt_morning = datetime(2026, 9, 15, 8, 30) # Tuesday 8:30 AM
        res_morning = calculate_llm_plus_traffic_congestion(
            origin_name="Timur Laut",
            dest_name="Larut & Matang",
            travel_time_nominal_mins=52.0,
            current_dt=dt_morning
        )
        self.assertGreaterEqual(res_morning["delay_factor"], 1.15)
        self.assertGreater(res_morning["delay_mins"], 0)
        self.assertIn(res_morning["status"], ["Moderate Congestion", "Severe Gridlock"])
        self.assertEqual(len(res_morning["color_rgb"]), 4)
        for c in res_morning["color_rgb"]:
            self.assertTrue(0 <= c <= 255)

        # Test midday off-peak (13:00)
        dt_offpeak = datetime(2026, 9, 15, 13, 0)
        res_offpeak = calculate_llm_plus_traffic_congestion(
            origin_name="Kuantan",
            dest_name="Pekan",
            travel_time_nominal_mins=40.0,
            current_dt=dt_offpeak
        )
        self.assertLess(res_offpeak["delay_factor"], 1.25)
        self.assertEqual(len(res_offpeak["color_rgb"]), 4)

    def test_get_corridor_traffic_profile_fallback(self):
        """Verify get_corridor_traffic_profile cleanly falls back to LLM/PLUS without API key."""
        prof = get_corridor_traffic_profile(
            origin_coords=[5.4141, 100.3288],
            dest_coords=[4.8500, 100.7333],
            origin_name="Timur Laut",
            dest_name="Larut & Matang",
            travel_time_nominal_mins=52.0,
            google_maps_api_key=None
        )
        self.assertIn("LLM / PLUS", prof["source"])
        self.assertGreaterEqual(prof["delay_factor"], 1.0)
        self.assertIn("summary", prof)

    def test_economic_model_inbound_and_blended(self):
        """Verify international inbound and blended economy simulations."""
        # Test Domestic Mode
        dom = calculate_redistribution_economic_impact(
            redirected_visitors_daily=1000,
            days_period=30,
            tourism_mode="domestic"
        )
        self.assertEqual(dom["tourism_mode"], "domestic")
        self.assertAlmostEqual(dom["alos_used"], 2.45, places=2)
        self.assertAlmostEqual(dom["spend_per_night_used"], DTS_2024_SPEND_PER_NIGHT, places=2)

        # Test Inbound Mode
        inb = calculate_redistribution_economic_impact(
            redirected_visitors_daily=1000,
            days_period=30,
            tourism_mode="inbound"
        )
        self.assertEqual(inb["tourism_mode"], "inbound")
        self.assertAlmostEqual(inb["alos_used"], INBOUND_ALOS, places=2)
        self.assertAlmostEqual(inb["spend_per_night_used"], INBOUND_SPEND_PER_NIGHT, places=2)
        self.assertGreater(inb["total_economic_output_million_rm"], dom["total_economic_output_million_rm"])

        # Test Blended Mode
        blend = calculate_redistribution_economic_impact(
            redirected_visitors_daily=1000,
            days_period=30,
            tourism_mode="blended"
        )
        self.assertEqual(blend["tourism_mode"], "blended")
        self.assertGreater(blend["total_economic_output_million_rm"], dom["total_economic_output_million_rm"])
        self.assertLess(blend["total_economic_output_million_rm"], inb["total_economic_output_million_rm"])

    def test_inbound_datasets_query(self):
        """Verify extraction of top inbound markets and categories from comprehensive_expenditure.csv."""
        top_markets = get_top_inbound_markets(year=2024, top_n=5)
        self.assertFalse(top_markets.empty)
        self.assertIn("market", top_markets.columns)
        self.assertIn("value_rm_million", top_markets.columns)

        cat_breakdown = get_inbound_category_breakdown(year=2024)
        self.assertFalse(cat_breakdown.empty)
        self.assertIn("spending_category", cat_breakdown.columns)

    def test_pydeck_3d_district_and_state_modes(self):
        """Verify PyDeck 3D map generator creates valid Deck objects for both LOD modes."""
        dest_df = pd.DataFrame([
            {
                "destination_id": "D1", "destination_name": "George Town", "district_name": "Timur Laut",
                "state_name": "Pulau Pinang", "lat": 5.4141, "lon": 100.3288, "daily_demand_peak": 8000,
                "total_rooms": 12000, "continuous_pressure": 1.84, "latest_aor_pct": 78.5, "poverty_rate": 2.1
            },
            {
                "destination_id": "D2", "destination_name": "Taiping", "district_name": "Larut & Matang",
                "state_name": "Perak", "lat": 4.8500, "lon": 100.7333, "daily_demand_peak": 3200,
                "total_rooms": 6500, "continuous_pressure": 0.46, "latest_aor_pct": 48.0, "poverty_rate": 8.5
            }
        ])
        hotspot = dest_df.iloc[0].to_dict()
        alternatives = [
            {
                "candidate_name": "Taiping", "destination_name": "Taiping", "district_name": "Larut & Matang",
                "lat": 4.8500, "lon": 100.7333, "final_wsm_score": 88.5, "transit_mode": "KTM ETS", "distance_km": 75
            }
        ]

        # 1. District Mode
        deck_dist = render_pydeck_3d_elevation_map(hotspot, alternatives, dest_df, lod_mode="district")
        self.assertIsInstance(deck_dist, pdk.Deck)
        # ColumnLayer, close labels, ground_transit_routes, redistribution_arcs,
        # PLANMalaysia KSAS overlay (Peninsular + Borneo), KSAS alert disk
        self.assertEqual(len(deck_dist.layers), 7)
        col_layer = deck_dist.layers[0]
        self.assertEqual(col_layer.id, "district_columns")
        self.assertTrue(col_layer.pickable)
        layer_ids = [lyr.id for lyr in deck_dist.layers]
        self.assertIn("ksas_buffers", layer_ids)
        self.assertIn("district_close_labels", layer_ids)
        self.assertIn("ksas_overlay", layer_ids)
        self.assertIn("ksas_overlay_borneo", layer_ids)
        self.assertIn("ground_transit_routes", layer_ids)
        self.assertIn("redistribution_arcs", layer_ids)
        for lyr in deck_dist.layers[1:]:
            self.assertFalse(lyr.pickable) # Only cylinders steal clicks

        # Verify clean tooltip configuration
        self.assertTrue("{name}" in deck_dist._tooltip.get("html") or "{tooltip_html}" in deck_dist._tooltip.get("html"))
        records = col_layer.data if isinstance(col_layer.data, list) else col_layer.data.to_dict("records")
        self.assertIn("<b", records[0]["tooltip_html"])
        # 2. State Mode
        deck_state = render_pydeck_3d_elevation_map(hotspot, alternatives, dest_df, lod_mode="state")
        self.assertIsInstance(deck_state, pdk.Deck)
        self.assertEqual(len(deck_state.layers), 6)
        self.assertTrue(deck_state.layers[0].pickable)
        self.assertTrue("{state_name}" in deck_state._tooltip.get("html") or "{tooltip_html}" in deck_state._tooltip.get("html"))


if __name__ == "__main__":
    unittest.main()
