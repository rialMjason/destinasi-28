"""
Unit Tests for DESTINASI Machine Learning Predictive Spike Forecaster & Corridor Synchronization
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.predictive_spike_engine import (
    PredictiveSpikeEngine,
    get_predictive_spike_engine,
    generate_ai_preemptive_advisory
)
from src.recommender_matcher import find_best_alternatives

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


class TestPredictiveSpikeEngine(unittest.TestCase):

    def setUp(self):
        self.engine = get_predictive_spike_engine()
        self.dest_df = pd.read_csv(DATA_DIR / "destinations_master.csv")

    def test_engine_initialization_and_events(self):
        """Verify predictive engine loads calendar events and trains models successfully."""
        self.assertIsNotNone(self.engine)
        events = self.engine.get_upcoming_events()
        self.assertGreaterEqual(len(events), 12, "Must contain at least 12 months of holiday events")
        
        # Check event fields
        sample = events[0]
        self.assertIn("month", sample)
        self.assertIn("cuti_sekolah_days", sample)
        self.assertIn("demand_multiplier", sample)
        self.assertIn("event_highlights", sample)

    def test_predict_district_spike_accuracy_and_bounds(self):
        """Verify spike prediction outputs valid numbers, probabilities [0, 100], and risk levels."""
        cameron_row = self.dest_df[self.dest_df["district_name"] == "Cameron Highlands"].iloc[0]
        pred = self.engine.predict_district_spike(cameron_row, "2026-02") # CNY peak

        self.assertEqual(pred["district_name"], "Cameron Highlands")
        self.assertGreater(pred["predicted_peak_volume"], 0)
        self.assertGreater(pred["capacity_stress_pct"], 100.0, "CNY in Cameron must show elevated/critical stress")
        self.assertGreaterEqual(pred["spike_probability_pct"], 0.0)
        self.assertLessEqual(pred["spike_probability_pct"], 100.0)
        self.assertIn("🔴", pred["risk_level"])
        self.assertGreater(pred["excess_surge_volume"], 0)
        self.assertGreater(pred["recommended_diversion_quota"], 0)
        self.assertIn(pred["early_warning_lead_days"], [3, 7, 14])

    def test_12_month_forward_curve(self):
        """Verify 12-month forward predictive curve returns 12 complete monthly records."""
        penang_row = self.dest_df[self.dest_df["district_name"] == "Timur Laut"].iloc[0]
        curve_df = self.engine.predict_12_month_forward_curve(penang_row, year=2026)

        self.assertEqual(len(curve_df), 12)
        self.assertIn("predicted_demand", curve_df.columns)
        self.assertIn("sustainable_capacity", curve_df.columns)
        self.assertIn("capacity_stress_pct", curve_df.columns)
        self.assertIn("spike_probability_pct", curve_df.columns)

        # December (Year-End Peak) demand should exceed November (Shoulder)
        nov_dem = curve_df[curve_df["month"] == "2026-11"]["predicted_demand"].iloc[0]
        dec_dem = curve_df[curve_df["month"] == "2026-12"]["predicted_demand"].iloc[0]
        self.assertGreater(dec_dem, nov_dem, "December peak must exceed November trough")

    def test_rank_nationwide_at_risk_districts(self):
        """Verify nationwide ranking returns sorted top N at-risk districts."""
        top_10 = self.engine.rank_nationwide_at_risk_districts("2026-02", self.dest_df, top_n=10)
        self.assertEqual(len(top_10), 10)
        
        # Verify descending order of stress_pct
        stresses = top_10["stress_pct"].tolist()
        self.assertEqual(stresses, sorted(stresses, reverse=True), "Must be sorted by descending stress %")

    def test_tangkak_relief_corridor_regional_proximity(self):
        """Verify Tangkak relief corridors prioritize local regional districts (< 180 km) over distant states."""
        tangkak_row = self.dest_df[self.dest_df["district_name"] == "Tangkak"].iloc[0]
        alts = find_best_alternatives(tangkak_row, self.dest_df, top_n=3)
        candidate_names = alts["candidate_name"].tolist()

        # Jasin, Segamat, or Jempol should be recommended, NOT Gua Musang (297 km away)
        self.assertIn("Jasin", candidate_names, "Adjacent district Jasin must be a top relief corridor for Tangkak")
        self.assertNotIn("Gua Musang", candidate_names, "Distant district Gua Musang must not be in Tangkak top 3")
        
        # All candidate distances must be <= 180 km
        for dist in alts["distance_km"]:
            self.assertLessEqual(dist, 180.0, f"Distance {dist} km must be within regional 180 km travel shed")

    def test_ai_preemptive_directive_synthesis(self):
        """Verify preemptive advisory directive synthesizes statutory agency actions."""
        cameron_row = self.dest_df[self.dest_df["district_name"] == "Cameron Highlands"].iloc[0]
        pred = self.engine.predict_district_spike(cameron_row, "2026-02")
        
        advisory = generate_ai_preemptive_advisory(
            spike_prediction=pred,
            top_relief_destination="Batang Padang",
            top_relief_state="Perak"
        )
        self.assertIn("source", advisory)
        self.assertIn("content", advisory)
        self.assertIn("Cameron Highlands", advisory["content"])
        self.assertIn("Batang Padang", advisory["content"])
        self.assertIn("APAD", advisory["content"])
        self.assertIn("MOTAC", advisory["content"])


if __name__ == "__main__":
    unittest.main()
