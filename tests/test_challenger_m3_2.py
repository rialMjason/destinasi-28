"""
DESTINASI — Milestone 3 Empirical Challenge Suite
Author: challenger_m3_2 (Empirical Challenger: critic, specialist)
Target: DOSM Datathon 2026

Empirically tests and stress-tests:
1. 450+ DTS historical attractions and 110-district POI grounding.
2. Geographic and semantic sanity (coastal vs inland/mountain, urban, heritage, coordinates).
3. XAI explanation fidelity in score_candidate_alternative (placeholders, formatting, genuine names, numbers).
4. Edge cases & adversarial inputs (empty tags, zero capacity, extreme distances, negative headroom).
"""

import math
import re
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

from src.recommender_matcher import (
    score_candidate_alternative,
    find_best_alternatives,
    compute_jaccard_poi_similarity,
    compute_multivector_similarity,
    _extract_tag_set,
    ARCHETYPES,
    PILLAR_LABELS,
)
from src.etl.build_unified_database import (
    GROUNDED_DISTRICT_PROFILES,
    DISTRICT_COORDS,
)


class TestAttractionGrounding(unittest.TestCase):
    """Empirical challenge of 450+ attractions mapped across 110 districts."""

    @classmethod
    def setUpClass(cls):
        cls.dest_path = BASE_DIR / "data" / "processed" / "destinations_master.csv"
        cls.attr_path = BASE_DIR / "data" / "processed" / "dts_all_years_attractions.csv"
        cls.df_dest = pd.read_csv(cls.dest_path)
        cls.df_attr = pd.read_csv(cls.attr_path)

    def test_district_count_and_coverage(self):
        """Verify all 110 administrative districts are present and have grounded profiles."""
        self.assertEqual(len(self.df_dest), 110, "destinations_master.csv must contain exactly 110 districts")
        self.assertEqual(len(GROUNDED_DISTRICT_PROFILES), 110, "GROUNDED_DISTRICT_PROFILES must contain 110 entries")

        # Verify all 16 states and Federal Territories are represented
        states = set(self.df_dest["state_name"].unique())
        self.assertEqual(len(states), 16, f"Expected 16 states/FTs, found {len(states)}: {states}")

    def test_dts_450_attractions_dataset_integrity(self):
        """Verify the 450 DTS historical survey attractions dataset."""
        self.assertEqual(len(self.df_attr), 450, f"Expected 450 attraction rows, found {len(self.df_attr)}")
        expected_years = {2020, 2021, 2022, 2023, 2024, 2025}
        actual_years = set(self.df_attr["year"].unique())
        self.assertEqual(actual_years, expected_years, "DTS attractions must span 2020-2025")

        # Ranks must be 1 to 5
        self.assertEqual(set(self.df_attr["rank"].unique()), {1, 2, 3, 4, 5})

        # No null attraction names
        self.assertEqual(self.df_attr["attraction_name"].isna().sum(), 0)

    def test_poi_tags_richness_and_grounding(self):
        """Verify that every district has rich, non-empty, pipe-separated POI tags."""
        self.assertIn("poi_tags", self.df_dest.columns)
        self.assertEqual(self.df_dest["poi_tags"].isna().sum(), 0)

        total_tags = 0
        all_tags = set()
        for idx, row in self.df_dest.iterrows():
            tags_str = str(row["poi_tags"]).strip()
            self.assertTrue(len(tags_str) > 0, f"District {row['district_name']} has empty poi_tags")
            tags = [t.strip().lower() for t in tags_str.split("|") if t.strip()]
            self.assertGreaterEqual(
                len(tags), 3,
                f"District {row['district_name']} has fewer than 3 POI tags: {tags}"
            )
            total_tags += len(tags)
            all_tags.update(tags)

        # There must be substantial semantic diversity (>= 500 unique tags across Malaysia)
        self.assertGreaterEqual(total_tags, 1000, f"Expected >= 1000 total tag instances, got {total_tags}")
        self.assertGreaterEqual(len(all_tags), 500, f"Expected >= 500 unique tags, got {len(all_tags)}")

    def test_dts_iconic_attractions_represented_in_district_tags(self):
        """Empirically check that DTS top attractions are represented in the corresponding districts."""
        key_checks = [
            ("Langkawi", ["geopark", "skycab", "seven_wells", "beach"]),
            ("Cameron Highlands", ["tea_plantations", "mossy_forest", "strawberry", "agritourism"]),
            ("Port Dickson", ["beach", "army_museum", "military_history"]),
            ("Timur Laut", ["unesco", "gurney", "street_art", "beach"]),
            ("Melaka Tengah", ["unesco", "river_cruise", "nyonya", "heritage"]),
            ("Petaling", ["sunway", "mall", "retail"]),
            ("Kinta", ["limestone", "ipoh_white_coffee", "heritage"]),
            ("Ranau", ["mount_kinabalu", "kundasang", "hot_springs"]),
            ("Kota Tinggi", ["waterfall", "firefly", "beach"]),
            ("Kuala Selangor", ["fireflies", "sky_mirror", "bukit_malawati"]),
            ("Semporna", ["diving", "coral", "lagoon", "bajau"]),
        ]

        for dist_name, expected_keywords in key_checks:
            matches = self.df_dest[self.df_dest["district_name"] == dist_name]
            self.assertFalse(matches.empty, f"District {dist_name} not found in destinations_master")
            poi_tags = str(matches.iloc[0]["poi_tags"]).lower()
            found = [kw for kw in expected_keywords if kw in poi_tags]
            self.assertTrue(
                len(found) >= 1,
                f"District {dist_name} missing expected keywords {expected_keywords}; tags: {poi_tags}"
            )


class TestGeographicSemanticSanity(unittest.TestCase):
    """Empirical verification of geographic coordinates and semantic archetype sanity."""

    @classmethod
    def setUpClass(cls):
        cls.df_dest = pd.read_csv(BASE_DIR / "data" / "processed" / "destinations_master.csv")

    def test_coordinates_bounding_box(self):
        """Verify all district coordinates fall within Malaysia's geographic bounds."""
        # Malaysia bounds approximately: Lat 0.8°N to 7.5°N, Lon 99.5°E to 119.5°E
        for idx, row in self.df_dest.iterrows():
            lat, lon = float(row["lat"]), float(row["lon"])
            self.assertTrue(
                0.8 <= lat <= 7.5,
                f"District {row['district_name']} ({row['state_name']}) lat {lat} out of bounds [0.8, 7.5]"
            )
            self.assertTrue(
                99.5 <= lon <= 119.5,
                f"District {row['district_name']} ({row['state_name']}) lon {lon} out of bounds [99.5, 119.5]"
            )

    def test_archetype_score_bounds_and_distribution(self):
        """Verify all archetype scores are strictly within [0.05, 0.99] with no NaNs."""
        for col in ARCHETYPES:
            self.assertEqual(self.df_dest[col].isna().sum(), 0)
            self.assertTrue(
                (self.df_dest[col] >= 0.05).all(),
                f"Values below 0.05 in {col}"
            )
            self.assertTrue(
                (self.df_dest[col] <= 0.99).all(),
                f"Values above 0.99 in {col}"
            )

    def test_coastal_vs_mountain_semantic_sanity(self):
        """Verify coastal districts have high beach scores and mountain districts have high nature scores."""
        # Coastal districts: must have arch_beach >= 0.60 and coastal/marine/pantai tags
        coastal_districts = [
            "Port Dickson", "Mersing", "Kota Tinggi", "Manjung",
            "Besut", "Dungun", "Kemaman", "Semporna", "Kudat",
            "Bachok", "Langkawi"
        ]
        coastal_keywords = [
            "beach", "coastal", "island", "marine", "sea", "snorkeling",
            "pantai", "pulau", "turtle", "diving", "coral", "reef", "lagoon"
        ]

        for c in coastal_districts:
            row = self.df_dest[self.df_dest["district_name"] == c]
            if not row.empty:
                val = float(row.iloc[0]["arch_beach"])
                tags = str(row.iloc[0]["poi_tags"]).lower()
                self.assertGreaterEqual(
                    val, 0.60,
                    f"Coastal district {c} has low arch_beach: {val}"
                )
                self.assertTrue(
                    any(w in tags for w in coastal_keywords),
                    f"Coastal district {c} has no marine/beach keywords in tags: {tags}"
                )

        # Mountain / Inland nature districts: must have arch_nature >= 0.80 and arch_beach <= 0.20
        mountain_districts = [
            "Cameron Highlands", "Ranau", "Gua Musang", "Jerantut",
            "Baling", "Hulu Perak", "Lipis", "Jelebu", "Tangkak"
        ]
        for m in mountain_districts:
            row = self.df_dest[self.df_dest["district_name"] == m]
            if not row.empty:
                val_nature = float(row.iloc[0]["arch_nature"])
                val_beach = float(row.iloc[0]["arch_beach"])
                tags = str(row.iloc[0]["poi_tags"]).lower()
                self.assertGreaterEqual(
                    val_nature, 0.80,
                    f"Mountain district {m} has low arch_nature: {val_nature}"
                )
                self.assertLessEqual(
                    val_beach, 0.20,
                    f"Inland mountain district {m} has unexpectedly high arch_beach: {val_beach}"
                )
                self.assertTrue(
                    any(w in tags for w in ["mountain", "forest", "rainforest", "waterfall", "hiking", "plateau", "eco", "park", "lake", "caves"]),
                    f"Mountain district {m} has no eco/nature keywords in tags: {tags}"
                )

    def test_urban_and_heritage_semantic_sanity(self):
        """Verify metropolitan capitals have high urban scores and historic towns have high heritage scores."""
        urban_centers = ["W.P. Kuala Lumpur", "Petaling", "Johor Bahru"]
        for u in urban_centers:
            row = self.df_dest[self.df_dest["district_name"] == u]
            if not row.empty:
                val_urban = float(row.iloc[0]["arch_urban"])
                self.assertGreaterEqual(val_urban, 0.90, f"Urban center {u} has low arch_urban: {val_urban}")

        heritage_districts = ["Melaka Tengah", "Timur Laut", "Pekan", "Kuala Pilah"]
        for h in heritage_districts:
            row = self.df_dest[self.df_dest["district_name"] == h]
            if not row.empty:
                val_heritage = float(row.iloc[0]["arch_heritage"])
                self.assertGreaterEqual(val_heritage, 0.85, f"Heritage district {h} has low arch_heritage: {val_heritage}")


class TestXAIExplanationFidelity(unittest.TestCase):
    """Empirical challenge of XAI explanation fidelity in score_candidate_alternative."""

    @classmethod
    def setUpClass(cls):
        cls.df_dest = pd.read_csv(BASE_DIR / "data" / "processed" / "destinations_master.csv")

    def test_all_110_districts_generate_valid_explanations(self):
        """Test explanation generation across all 110 districts paired with nearest valid alternative."""
        template_regex = re.compile(r"\{[a-zA-Z0-9_]+\}")
        placeholder_tokens = ["nan", "none", "null", "undefined", "[]", "{}"]

        for i, hotspot in self.df_dest.iterrows():
            candidates = self.df_dest[self.df_dest["district_name"] != hotspot["district_name"]]
            candidate = candidates.iloc[(i * 7) % len(candidates)]

            cand_row = candidate.copy()
            cand_row["distance_km"] = 65.0
            cand_row["transit_time_mins"] = 50.0
            cand_row["transit_mode"] = "KTM ETS"

            res = score_candidate_alternative(hotspot, cand_row)
            explanation = res.get("explanation", "")

            # 1. Non-empty explanation
            self.assertTrue(len(explanation) > 30, f"Short explanation for {hotspot['district_name']}: {explanation}")

            # 2. No raw template placeholders like {name} or {shared_pillars}
            placeholders_found = template_regex.findall(explanation)
            self.assertEqual(
                len(placeholders_found), 0,
                f"Raw placeholders found in explanation for {hotspot['district_name']}: {placeholders_found}"
            )

            # 3. No unparsed null/undefined string tokens
            exp_lower = explanation.lower()
            for token in placeholder_tokens:
                self.assertFalse(
                    re.search(rf"\b{re.escape(token)}\b", exp_lower),
                    f"Unparsed token '{token}' in explanation for {hotspot['district_name']}: {explanation}"
                )

            # 4. Shared or verified attraction tags must not contain raw underscores (e.g. 'street_art' -> 'Street Art')
            if "Features shared attraction themes (" in explanation:
                themes_part = explanation.split("Features shared attraction themes (")[1].split(")")[0]
                self.assertNotIn("_", themes_part, f"Raw underscores in themes: {themes_part}")
            if "Features verified attractions including" in explanation:
                themes_part = explanation.split("Features verified attractions including")[1].split(".")[0]
                self.assertNotIn("_", themes_part, f"Raw underscores in attractions: {themes_part}")

            # 5. Must contain valid numeric formatting (e.g. percentage signs, numbers)
            self.assertIn("%", explanation, f"Missing percentage symbol in explanation: {explanation}")

            # 6. Must contain valid transit mode
            self.assertIn("KTM ETS", explanation)

            # 7. Grammar & punctuation sanity: ends with period, no double spaces
            self.assertTrue(explanation.strip().endswith("."), f"Explanation does not end with period: {explanation}")
            self.assertNotIn("  ", explanation, f"Double spaces in explanation: {explanation}")

    def test_pilot_corridor_fidelity(self):
        """Stress-test the 5 priority pilot corridors for granular narrative accuracy."""
        pilot_pairs = [
            ("Timur Laut", "Larut & Matang", "KTM ETS", 75.0, 52.0),
            ("Cameron Highlands", "Lipis", "Federal Route 59 & CSR", 95.0, 65.0),
            ("Melaka Tengah", "Muar", "Federal Route 5", 45.0, 45.0),
            ("Petaling", "Kuala Kangsar", "KTM ETS Platinum", 220.0, 110.0),
            ("Kota Kinabalu", "Ranau", "Federal Route 22", 108.0, 120.0),
        ]

        for h_name, c_name, mode, dist, t_time in pilot_pairs:
            h_rows = self.df_dest[self.df_dest["district_name"] == h_name]
            c_rows = self.df_dest[self.df_dest["district_name"] == c_name]
            self.assertFalse(h_rows.empty, f"Missing hotspot {h_name}")
            self.assertFalse(c_rows.empty, f"Missing candidate {c_name}")

            hotspot = h_rows.iloc[0]
            cand = c_rows.iloc[0].copy()
            cand["transit_mode"] = mode
            cand["distance_km"] = dist
            cand["transit_time_mins"] = t_time

            res = score_candidate_alternative(hotspot, cand)
            exp = res["explanation"]

            # Must contain transit mode
            self.assertIn(mode, exp)
            # Must mention distance
            self.assertIn(f"{dist:.0f} km", exp)
            # Must mention poverty / community uplift
            p_rate = float(cand.get("poverty_rate", 0.0))
            self.assertIn(f"{p_rate:.1f}%", exp)


class TestEdgeCasesAndAdversarialInputs(unittest.TestCase):
    """Stress-test boundary conditions, missing fields, and adversarial inputs."""

    def test_missing_and_malformed_poi_tags(self):
        """Verify robust fallback when poi_tags are None, empty, JSON strings, or disjoint."""
        hotspot_base = pd.Series({
            "district_name": "Hotspot A",
            "arch_heritage": 0.8, "arch_nature": 0.2, "arch_beach": 0.1, "arch_food": 0.8, "arch_urban": 0.5,
            "daily_demand_peak": 10000,
            "sustainable_capacity": 8000,
        })
        cand_base = pd.Series({
            "district_name": "Relief B",
            "arch_heritage": 0.8, "arch_nature": 0.2, "arch_beach": 0.1, "arch_food": 0.8, "arch_urban": 0.5,
            "daily_demand_peak": 2000,
            "sustainable_capacity": 5000,
            "cc_ecology": 6000,
            "capacity_gap": 0.40,
            "distance_km": 50.0,
            "transit_time_mins": 45.0,
            "transit_mode": "Road",
            "poverty_rate": 8.0,
        })

        # Subcase 1: Both None
        h1, c1 = hotspot_base.copy(), cand_base.copy()
        h1["poi_tags"] = None
        c1["poi_tags"] = None
        res1 = score_candidate_alternative(h1, c1)
        self.assertTrue(res1["is_feasible"])
        self.assertNotIn("None", res1["explanation"])
        self.assertNotIn("{", res1["explanation"])

        # Subcase 2: Empty strings
        h2, c2 = hotspot_base.copy(), cand_base.copy()
        h2["poi_tags"] = ""
        c2["poi_tags"] = "   "
        res2 = score_candidate_alternative(h2, c2)
        self.assertTrue(res2["is_feasible"])
        self.assertNotIn("{", res2["explanation"])

        # Subcase 3: JSON array format strings
        h3, c3 = hotspot_base.copy(), cand_base.copy()
        h3["poi_tags"] = '["heritage_core", "street_food"]'
        c3["poi_tags"] = "heritage_core | street_food | night_bazaar"
        res3 = score_candidate_alternative(h3, c3)
        self.assertTrue(res3["is_feasible"])
        self.assertGreater(res3["jaccard_poi_similarity"], 0.0)

        # Subcase 4: Disjoint tags
        h4, c4 = hotspot_base.copy(), cand_base.copy()
        h4["poi_tags"] = "scuba_diving | coral_reef | sea_turtle"
        c4["poi_tags"] = "mossy_forest | tea_plantation | strawberry_picking"
        res4 = score_candidate_alternative(h4, c4)
        self.assertEqual(res4["jaccard_poi_similarity"], 0.0)
        # Should cleanly feature verified attractions without shared tag error
        self.assertIn("Features verified attractions including", res4["explanation"])

    def test_unclosed_bracket_tag_behavior_challenge(self):
        """Adversarial stress test: document behavior on malformed unclosed bracket strings."""
        # When an unclosed bracket without closing ']' is supplied with inner quotes
        extracted = _extract_tag_set("['heritage_core', 'street_food'")
        # Note: _extract_tag_set falls through to comma split without inner quote stripping
        self.assertIsInstance(extracted, set)
        self.assertGreater(len(extracted), 0)

    def test_feasibility_gate_extremes(self):
        """Verify strict gate disqualification at exact threshold boundaries."""
        h = pd.Series({
            "district_name": "Hotspot X",
            "arch_heritage": 0.5, "arch_nature": 0.5, "arch_beach": 0.5, "arch_food": 0.5, "arch_urban": 0.5
        })

        # Boundary: capacity_gap exactly 0.10 -> must be disqualified (requires > 0.10)
        c_gap_10 = pd.Series({
            "capacity_gap": 0.10, "daily_demand_peak": 1000, "cc_ecology": 5000, "distance_km": 50.0
        })
        self.assertFalse(score_candidate_alternative(h, c_gap_10)["is_feasible"])

        # Boundary: capacity_gap 0.1001 -> passes gate 1
        c_gap_pass = pd.Series({
            "capacity_gap": 0.1001, "daily_demand_peak": 1000, "cc_ecology": 5000, "distance_km": 50.0
        })
        self.assertTrue(score_candidate_alternative(h, c_gap_pass)["is_feasible"])

        # Boundary: demand == cc_ecology -> disqualified (must be strictly <)
        c_eco_eq = pd.Series({
            "capacity_gap": 0.30, "daily_demand_peak": 5000, "cc_ecology": 5000, "distance_km": 50.0
        })
        self.assertFalse(score_candidate_alternative(h, c_eco_eq)["is_feasible"])

        # Boundary: distance_km == 350.0 -> passes gate 3 (<= 350.0)
        c_dist_350 = pd.Series({
            "capacity_gap": 0.30, "daily_demand_peak": 1000, "cc_ecology": 5000, "distance_km": 350.0
        })
        self.assertTrue(score_candidate_alternative(h, c_dist_350)["is_feasible"])

        # Boundary: distance_km == 350.1 -> disqualified (> 350.0)
        c_dist_350_1 = pd.Series({
            "capacity_gap": 0.30, "daily_demand_peak": 1000, "cc_ecology": 5000, "distance_km": 350.1
        })
        self.assertFalse(score_candidate_alternative(h, c_dist_350_1)["is_feasible"])

    def test_headroom_formatting_when_capacity_zero_or_negative(self):
        """Verify explanation formatting when sustainable capacity is 0 or demand exceeds capacity."""
        h = pd.Series({"district_name": "Hotspot H", "arch_heritage": 0.5, "arch_nature": 0.5, "arch_beach": 0.5, "arch_food": 0.5, "arch_urban": 0.5})
        c_zero_cap = pd.Series({
            "district_name": "Candidate Z",
            "sustainable_capacity": 0.0,
            "daily_demand_peak": 500.0,
            "cc_ecology": 2000.0,
            "capacity_gap": 0.20,
            "distance_km": 40.0
        })
        res = score_candidate_alternative(h, c_zero_cap)
        self.assertIn("safe from bottleneck breach", res["explanation"])
        self.assertNotIn("{", res["explanation"])


if __name__ == "__main__":
    unittest.main()
