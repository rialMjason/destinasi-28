"""Unit tests for DESTINASI Core Computational Engines and Reflex State.
Verifies carrying capacity bottleneck diagnosis, economic impact modelling,
and multi-attribute matchmaking in accordance with statutory rules.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.carrying_capacity_engine import (
    diagnose_multisystem_bottleneck,
    compute_physical_carrying_capacity,
    compute_real_carrying_capacity,
    compute_effective_carrying_capacity,
)
from src.economic_impact_model import (
    calculate_redistribution_economic_impact,
    SPEND_SHARES,
    TOURISM_OUTPUT_MULTIPLIER,
    DTS_2024_SPEND_PER_NIGHT,
    INBOUND_SPEND_PER_NIGHT,
    INBOUND_ALOS,
    INBOUND_OUTPUT_MULTIPLIER,
    BLENDED_OUTPUT_MULTIPLIER,
    get_top_inbound_markets,
    get_market_expenditure_breakdown,
    INBOUND_CATEGORIES,
    PROCESSED_EXP_PATH,
)
from src.recommender_matcher import (
    score_candidate_alternative,
    find_best_alternatives,
    compute_jaccard_poi_similarity,
    compute_multivector_similarity,
    ARCHETYPES,
)
from src.aor_engine import (
    ALL_16_STATES,
    REGIONAL_CLUSTERS,
    STATE_ALIASES,
    normalize_state_name,
    get_cluster_states,
    calculate_national_aor_baseline,
    compute_national_mean_aor,
)
from src.map_components import aggregate_destinations_by_state, render_pydeck_3d_elevation_map
from src.xai_engine import simulate_counterfactual_intervention, simulate_counterfactual_policy



class TestDestinasiEngines(unittest.TestCase):

    def test_carrying_capacity_dynamic_bottleneck(self):
        """Verify that carrying capacity is strictly determined by the minimum sub-system."""
        res = diagnose_multisystem_bottleneck(
            demand=50000,
            cc_accommodation=40000,
            cc_transport=30000,
            cc_attraction=35000,
            cc_water_waste=25000, # binding constraint
            cc_ecology=60000,
            cc_social=45000,
        )
        self.assertEqual(res["sustainable_capacity"], 25000)
        self.assertEqual(res["binding_constraint"], "Municipal Water & Waste Buffer")
        self.assertEqual(res["excess_demand"], 25000)  # 50000 - 25000
        self.assertEqual(res["pressure_percent"], 200.0)  # (50000 / 25000) * 100
        self.assertTrue(res["is_overcapacity"])
        self.assertEqual(res["capacity_gap_percent"], -100.0)

    def test_carrying_capacity_spare_capacity(self):
        """Verify spare capacity gap calculation when demand is below bottleneck."""
        res = diagnose_multisystem_bottleneck(
            demand=10000,
            cc_accommodation=20000,
            cc_transport=25000,
            cc_attraction=30000,
            cc_water_waste=15000,
            cc_ecology=40000,
            cc_social=25000,
        )
        self.assertEqual(res["sustainable_capacity"], 15000)
        self.assertEqual(res["excess_demand"], 0)
        self.assertAlmostEqual(res["pressure_percent"], (10000 / 15000) * 100, places=1)
        self.assertFalse(res["is_overcapacity"])
        self.assertAlmostEqual(res["capacity_gap_percent"], ((15000 - 10000) / 15000) * 100, places=1)

    def test_economic_impact_dts_and_tsa(self):
        """Verify that DTS spending shares sum to 1.0 and TSA composite multiplier is 1.75x."""
        self.assertAlmostEqual(sum(SPEND_SHARES.values()), 1.0, places=2)
        self.assertEqual(TOURISM_OUTPUT_MULTIPLIER, 1.75)

        econ = calculate_redistribution_economic_impact(
            redirected_visitors_daily=2000,
            days_period=30,
            alos=2.45,
            spend_per_night=386.73,
            poverty_rate_candidate=6.2,
            poverty_rate_hotspot=0.8,
        )
        # Total nights = 2000 * 30 * 2.45 = 147,000
        # Direct spend = 147,000 * 386.73 = ~RM 56.85M
        self.assertAlmostEqual(econ["direct_spend_million_rm"], 56.849, places=2)
        # Total economic output = 56.849 * 1.75 = ~RM 99.49M
        self.assertAlmostEqual(econ["total_economic_output_million_rm"], 56.849 * 1.75, places=2)
        # Grassroots B40 injection positive
        self.assertGreater(econ["estimated_b40_income_million_rm"], 0)

    def test_matchmaker_scoring_and_explainability(self):
        """Verify that WSM scoring accounts for archetype fit and generates transparent explanations."""
        hotspot_series = pd.Series({
            "destination_name": "George Town Heritage Core",
            "district_name": "Timur Laut",
            "state_name": "Pulau Pinang",
            "arch_heritage": 0.95,
            "arch_nature": 0.20,
            "arch_beach": 0.40,
            "arch_food": 0.95,
            "arch_urban": 0.85,
            "daily_demand_peak": 42000,
            "sustainable_capacity": 28000,
            "poverty_rate": 0.8,
            "mean_household_income": 6500,
        })

        candidate_series = pd.Series({
            "destination_name": "Taiping Heritage & Lake Gardens",
            "district_name": "Larut & Matang",
            "state_name": "Perak",
            "arch_heritage": 0.90,
            "arch_nature": 0.75,
            "arch_beach": 0.10,
            "arch_food": 0.85,
            "arch_urban": 0.60,
            "daily_demand_peak": 9500,
            "sustainable_capacity": 22000,
            "capacity_gap": 0.568,
            "transit_time_mins": 48,
            "transit_fare_rm": 11.50,
            "transit_mode": "KTM Komuter",
            "poverty_rate": 6.2,
            "mean_household_income": 4100,
            "cc_ecology": 25000,
        })

        res = score_candidate_alternative(hotspot_series, candidate_series)
        self.assertGreater(res["final_wsm_score"], 65)
        self.assertGreater(res["match_score"], 80)
        self.assertIn("explanation", res)
        self.assertGreater(len(res["explanation"]), 20)

    def test_all_corridor_computations(self):
        """Verify that dynamic bottleneck, matching, and policy simulation work across all pilot corridors."""
        pilot_path = BASE_DIR / "data" / "processed" / "pilot_corridors.csv"
        self.assertTrue(pilot_path.exists())
        pilot_df = pd.read_csv(pilot_path)

        for corridor_id in [1, 2, 3, 4]:
            c_data = pilot_df[pilot_df["corridor_id"] == corridor_id]
            self.assertEqual(len(c_data), 2)
            h = c_data[c_data["destination_type"] == "hotspot"].iloc[0]
            a = c_data[c_data["destination_type"] == "alternative"].iloc[0]

            diag_h = diagnose_multisystem_bottleneck(
                demand=h["daily_demand_peak"],
                cc_accommodation=h["cc_accommodation"],
                cc_transport=h["cc_transport"],
                cc_attraction=h["cc_attraction"],
                cc_water_waste=h["cc_water_waste"],
                cc_ecology=h["cc_ecology"],
                cc_social=h["cc_social"]
            )
            self.assertTrue(diag_h["is_overcapacity"])
            self.assertGreater(diag_h["excess_demand"], 0)

            diag_a = diagnose_multisystem_bottleneck(
                demand=a["daily_demand_peak"],
                cc_accommodation=a["cc_accommodation"],
                cc_transport=a["cc_transport"],
                cc_attraction=a["cc_attraction"],
                cc_water_waste=a["cc_water_waste"],
                cc_ecology=a["cc_ecology"],
                cc_social=a["cc_social"]
            )
            self.assertFalse(diag_a["is_overcapacity"])
            self.assertGreater(diag_a["capacity_gap_percent"], 0)

            score = score_candidate_alternative(h, a)
            self.assertGreater(score["final_wsm_score"], 50)
            self.assertIn("explanation", score)

            econ = calculate_redistribution_economic_impact(
                redirected_visitors_daily=diag_h["excess_demand"] * 0.25,
                days_period=30,
                alos=2.45,
                poverty_rate_candidate=a["poverty_rate"],
                poverty_rate_hotspot=h["poverty_rate"]
            )
            self.assertGreater(econ["total_economic_output_million_rm"], 0)
            self.assertGreater(econ["estimated_b40_income_million_rm"], 0)

    # -------------------------------------------------------------------------
    # MILESTONE 3 (R3): DYNAMIC POI-DRIVEN MATCHMAKING & FEASIBILITY GATES
    # -------------------------------------------------------------------------

    def test_compute_jaccard_poi_similarity(self):
        """R3: Verify Jaccard POI feature similarity across sets, lists, strings, and nulls."""
        # 1. Identical sets -> 1.0
        set_a = {"heritage_walk", "colonial_architecture", "street_food"}
        self.assertAlmostEqual(compute_jaccard_poi_similarity(set_a, set_a), 1.0, places=4)

        # 2. Disjoint sets -> 0.0
        set_b = {"scuba_diving", "coral_reef", "island_hopping"}
        self.assertAlmostEqual(compute_jaccard_poi_similarity(set_a, set_b), 0.0, places=4)

        # 3. Partial overlap: 2 shared out of 5 unique -> 2/5 = 0.4
        set_c = {"heritage_walk", "street_food", "night_market", "art_gallery"}
        self.assertAlmostEqual(compute_jaccard_poi_similarity(set_a, set_c), 2.0 / 5.0, places=4)

        # 4. Partial overlap with 2 shared out of 6 unique -> 2/6 = 0.3333
        set_d = {"heritage", "food", "temple", "museum"}
        set_e = {"heritage", "food", "nature", "lake"}
        self.assertAlmostEqual(compute_jaccard_poi_similarity(set_d, set_e), 2.0 / 6.0, places=3)

        # 5. Empty sets -> 0.0
        self.assertEqual(compute_jaccard_poi_similarity(set(), set()), 0.0)
        self.assertEqual(compute_jaccard_poi_similarity(set(), set_a), 0.0)

        # 6. String inputs (comma-separated and pipe-separated)
        self.assertAlmostEqual(
            compute_jaccard_poi_similarity("heritage, food, street_art", "heritage, food, nature"),
            2.0 / 4.0,
            places=3
        )
        self.assertAlmostEqual(
            compute_jaccard_poi_similarity("heritage | food | street_art", "heritage | food | nature"),
            2.0 / 4.0,
            places=3
        )

        # 7. None and missing inputs -> 0.0
        self.assertEqual(compute_jaccard_poi_similarity(None, {"heritage"}), 0.0)
        self.assertEqual(compute_jaccard_poi_similarity({"heritage"}, None), 0.0)
        self.assertEqual(compute_jaccard_poi_similarity(None, None), 0.0)

    def test_compute_multivector_similarity(self):
        """R3: Verify multi-vector similarity hybrid score computation and fallback when tags absent."""
        # Case 1: Identical archetypes and identical tags -> 1.0
        row_h = pd.Series({
            "arch_heritage": 1.0, "arch_nature": 0.0, "arch_beach": 0.0, "arch_food": 0.0, "arch_urban": 0.0,
            "poi_tags": "heritage | colonial"
        })
        row_c_ident = pd.Series({
            "arch_heritage": 1.0, "arch_nature": 0.0, "arch_beach": 0.0, "arch_food": 0.0, "arch_urban": 0.0,
            "poi_tags": "heritage | colonial"
        })
        self.assertAlmostEqual(compute_multivector_similarity(row_h, row_c_ident, alpha=0.60), 1.0, places=4)

        # Case 2: Identical archetypes, but completely disjoint tags
        # Cosine = 1.0, Jaccard = 0.0 -> Hybrid = 0.60 * 1.0 + 0.40 * 0.0 = 0.60
        row_c_disjoint = pd.Series({
            "arch_heritage": 1.0, "arch_nature": 0.0, "arch_beach": 0.0, "arch_food": 0.0, "arch_urban": 0.0,
            "poi_tags": "nature | jungle"
        })
        self.assertAlmostEqual(compute_multivector_similarity(row_h, row_c_disjoint, alpha=0.60), 0.60, places=2)

        # Case 3: Fallback when POI tags are absent on both sides -> pure Cosine similarity
        row_h_notags = pd.Series({
            "arch_heritage": 0.95, "arch_nature": 0.20, "arch_beach": 0.40, "arch_food": 0.95, "arch_urban": 0.85
        })
        row_c_notags = pd.Series({
            "arch_heritage": 0.90, "arch_nature": 0.75, "arch_beach": 0.10, "arch_food": 0.85, "arch_urban": 0.60
        })
        fallback_sim = compute_multivector_similarity(row_h_notags, row_c_notags, alpha=0.60)
        # Cosine is ~0.9111
        self.assertGreater(fallback_sim, 0.90)
        self.assertAlmostEqual(fallback_sim, 0.9111, places=3)

        # Case 4: Fallback when POI tags are absent on candidate side only
        fallback_one_sided = compute_multivector_similarity(row_h, row_c_notags, alpha=0.60)
        self.assertGreater(fallback_one_sided, 0.0)

    def test_grounded_poi_archetypes(self):
        """R3: Verify all 110 districts in destinations_master.csv have grounded scores in [0.05, 0.99] and poi_tags."""
        dest_path = BASE_DIR / "data" / "processed" / "destinations_master.csv"
        self.assertTrue(dest_path.exists(), "destinations_master.csv must exist")
        df = pd.read_csv(dest_path)

        # Must have exactly 110 districts
        self.assertEqual(len(df), 110, "destinations_master.csv must contain all 110 districts")

        # Archetype columns must exist
        for col in ARCHETYPES:
            self.assertIn(col, df.columns, f"Missing archetype column: {col}")
            # All values must be bounded within [0.05, 0.99]
            self.assertTrue(
                (df[col] >= 0.05).all(),
                f"Column {col} has values below 0.05: {df[df[col] < 0.05][[col, 'district_name']]}"
            )
            self.assertTrue(
                (df[col] <= 0.99).all(),
                f"Column {col} has values above 0.99: {df[df[col] > 0.99][[col, 'district_name']]}"
            )
            # Must not contain nulls
            self.assertEqual(df[col].isna().sum(), 0, f"Column {col} contains NaN values")

        # Verify poi_tags column
        self.assertIn("poi_tags", df.columns, "destinations_master.csv must contain poi_tags column")
        self.assertEqual(df["poi_tags"].isna().sum(), 0, "poi_tags must not have null values")
        self.assertTrue(
            (df["poi_tags"].str.strip() != "").all(),
            "All 110 districts must have non-empty poi_tags"
        )

        # Spot-check known iconic districts
        timur_laut = df[df["district_name"] == "Timur Laut"].iloc[0]
        self.assertGreater(timur_laut["arch_heritage"], 0.70)
        self.assertGreater(timur_laut["arch_food"], 0.70)

        cameron = df[df["district_name"] == "Cameron Highlands"].iloc[0]
        self.assertGreater(cameron["arch_nature"], 0.85)

        melaka = df[df["district_name"] == "Melaka Tengah"].iloc[0]
        self.assertGreater(melaka["arch_heritage"], 0.85)

        kl = df[df["district_name"].str.contains("Kuala Lumpur")].iloc[0]
        self.assertGreater(kl["arch_urban"], 0.90)

    def test_recommender_feasibility_gates(self):
        """R3: Verify candidate disqualification when capacity gap <= 10% or distance > 350km."""
        hotspot = pd.Series({
            "destination_id": "HOT_01",
            "destination_name": "Hotspot Central",
            "district_name": "Timur Laut",
            "lat": 5.4141,
            "lon": 100.3288,
            "arch_heritage": 0.90, "arch_nature": 0.20, "arch_beach": 0.30, "arch_food": 0.90, "arch_urban": 0.80
        })

        # Test 1: Capacity Gap <= 10% (0.10) -> Disqualified
        cand_overburdened = pd.Series({
            "destination_name": "Overburdened Town",
            "arch_heritage": 0.85, "arch_nature": 0.30, "arch_beach": 0.20, "arch_food": 0.85, "arch_urban": 0.70,
            "capacity_gap": 0.08,  # <= 0.10
            "daily_demand_peak": 8000,
            "cc_ecology": 15000,
            "distance_km": 60.0,
            "transit_time_mins": 50,
            "poverty_rate": 5.0
        })
        res_disq_cap = score_candidate_alternative(hotspot, cand_overburdened)
        self.assertFalse(res_disq_cap["is_feasible"], "Candidate with capacity gap <= 0.10 must be disqualified")

        # Test 2: Demand >= CC_ecology -> Disqualified
        cand_fragile = pd.Series({
            "destination_name": "Fragile Mountain Reserve",
            "arch_heritage": 0.30, "arch_nature": 0.95, "arch_beach": 0.10, "arch_food": 0.60, "arch_urban": 0.20,
            "capacity_gap": 0.30,
            "daily_demand_peak": 12000,
            "cc_ecology": 10000,  # demand >= cc_ecology
            "distance_km": 45.0,
            "transit_time_mins": 40,
            "poverty_rate": 6.0
        })
        res_disq_eco = score_candidate_alternative(hotspot, cand_fragile)
        self.assertFalse(res_disq_eco["is_feasible"], "Candidate with demand >= cc_ecology must be disqualified")

        # Test 3: Transit Distance > 350 km -> Disqualified
        cand_far = pd.Series({
            "destination_name": "Distant Frontier Town",
            "arch_heritage": 0.85, "arch_nature": 0.30, "arch_beach": 0.20, "arch_food": 0.85, "arch_urban": 0.70,
            "capacity_gap": 0.40,
            "daily_demand_peak": 5000,
            "cc_ecology": 20000,
            "distance_km": 420.0,  # > 350.0 km
            "transit_time_mins": 350,
            "poverty_rate": 3.0
        })
        res_disq_dist = score_candidate_alternative(hotspot, cand_far)
        self.assertFalse(res_disq_dist["is_feasible"], "Candidate with distance > 350km must be disqualified")

        # Test 4: Fully feasible candidate passes
        cand_healthy = pd.Series({
            "destination_name": "Feasible Relief Corridor",
            "arch_heritage": 0.85, "arch_nature": 0.30, "arch_beach": 0.20, "arch_food": 0.85, "arch_urban": 0.70,
            "capacity_gap": 0.35,  # > 0.10
            "daily_demand_peak": 4000,  # < cc_ecology
            "cc_ecology": 15000,
            "distance_km": 75.0,  # <= 350.0 km
            "transit_time_mins": 55,
            "poverty_rate": 6.5
        })
        res_healthy = score_candidate_alternative(hotspot, cand_healthy)
        self.assertTrue(res_healthy["is_feasible"], "Compliant candidate must pass all feasibility gates")

        # Test 5: find_best_alternatives excludes candidates that fail feasibility gates
        cand_df = pd.DataFrame([
            {
                "destination_id": "C_FEASIBLE", "destination_name": "Feasible Town", "district_name": "Larut & Matang",
                "lat": 4.8500, "lon": 100.7333,
                "arch_heritage": 0.85, "arch_nature": 0.70, "arch_beach": 0.10, "arch_food": 0.85, "arch_urban": 0.50,
                "capacity_gap": 0.40, "daily_demand_peak": 3000, "cc_ecology": 8000, "poverty_rate": 7.0
            },
            {
                "destination_id": "C_SATURATED", "destination_name": "Saturated Town", "district_name": "Butterworth",
                "lat": 5.3991, "lon": 100.3638,
                "arch_heritage": 0.90, "arch_nature": 0.20, "arch_beach": 0.30, "arch_food": 0.90, "arch_urban": 0.80,
                "capacity_gap": 0.05, "daily_demand_peak": 12000, "cc_ecology": 20000, "poverty_rate": 4.0
            },
            {
                "destination_id": "C_FAR", "destination_name": "Distant Town", "district_name": "Johor Bahru",
                "lat": 1.4927, "lon": 103.7414,
                "arch_heritage": 0.90, "arch_nature": 0.20, "arch_beach": 0.30, "arch_food": 0.90, "arch_urban": 0.90,
                "capacity_gap": 0.50, "daily_demand_peak": 5000, "cc_ecology": 25000, "poverty_rate": 2.5
            }
        ])
        alts = find_best_alternatives(hotspot, cand_df, top_n=3)
        returned_ids = list(alts["candidate_id"])
        self.assertIn("C_FEASIBLE", returned_ids)
        self.assertNotIn("C_SATURATED", returned_ids, "Saturated candidate (gap <= 10%) must be excluded")
        self.assertNotIn("C_FAR", returned_ids, "Distant candidate (> 350 km) must be excluded")

    def test_r3_xai_explanation_narrative_grounding(self):
        """R3: Verify XAI explanation string transparently cites shared pillars, attractions, and headroom."""
        hotspot = pd.Series({
            "destination_name": "George Town Core",
            "district_name": "Timur Laut",
            "arch_heritage": 0.95, "arch_nature": 0.20, "arch_beach": 0.40, "arch_food": 0.95, "arch_urban": 0.85,
            "poi_tags": "george_town_unesco | penang_laksa | street_art"
        })
        candidate = pd.Series({
            "destination_id": "DST_025",
            "destination_name": "Taiping Heritage Core",
            "district_name": "Larut & Matang",
            "state_name": "Perak",
            "arch_heritage": 0.90, "arch_nature": 0.75, "arch_beach": 0.10, "arch_food": 0.85, "arch_urban": 0.50,
            "sustainable_capacity": 5400,
            "daily_demand_peak": 2900,
            "capacity_gap": 0.463,
            "cc_ecology": 5400,
            "distance_km": 75.0,
            "transit_time_mins": 52,
            "transit_mode": "KTM ETS",
            "poverty_rate": 9.0,
            "poi_tags": "taiping_lake_gardens | zoo_taiping | perak_museum"
        })

        res = score_candidate_alternative(hotspot, candidate)
        explanation = res["explanation"]

        # Verify citation of shared pillars (Heritage and/or Food/Gastronomy)
        self.assertTrue(
            any(p in explanation for p in ["Heritage", "Culture", "Food", "Gastronomy"]),
            f"Explanation missing shared pillar citations: {explanation}"
        )
        # Verify citation of spare capacity headroom
        self.assertTrue(
            any(k in explanation for k in ["spare capacity", "headroom", "gap", "buffer", "46.3%"]),
            f"Explanation missing capacity headroom citation: {explanation}"
        )
        # Verify citation of transit mode and distance
        self.assertIn("KTM ETS", explanation)
        self.assertIn("52", explanation)
        # Verify community poverty citation
        self.assertIn("9.0%", explanation)

    def test_expenditure_market_matrix_structure_and_completeness(self):
        """R6: Verify expenditure_market_matrix.csv covers all 38 origin markets and 12 categories."""
        self.assertTrue(PROCESSED_EXP_PATH.exists(), f"Missing processed matrix at {PROCESSED_EXP_PATH}")
        df = pd.read_csv(PROCESSED_EXP_PATH)
        self.assertFalse(df.empty)

        # Check required columns
        for col in ["year", "market", "spending_category", "value_rm_million", "percentage_share"]:
            self.assertIn(col, df.columns)

        # Verify all 38 markets are present in the dataset
        markets = set(df["market"].dropna().unique())
        self.assertEqual(len(markets), 38, f"Expected 38 origin markets, found {len(markets)}")

        # Verify key markets are present
        for m in ["Singapore", "China", "Indonesia", "Australia", "Overall"]:
            self.assertIn(m, markets)

        # Verify 2024 data has all 12 categories for Overall
        ov_2024 = df[(df["year"] == 2024) & (df["market"] == "Overall")]
        cats_found = set(ov_2024["spending_category"].unique())
        for cat in INBOUND_CATEGORIES:
            self.assertIn(cat, cats_found, f"Missing category {cat} in 2024 Overall")

        # Total 2024 Overall expenditure should match MOTAC RM 106,780M
        tot_row = ov_2024[ov_2024["spending_category"] == "All spending categories"]
        self.assertFalse(tot_row.empty)
        self.assertAlmostEqual(tot_row["value_rm_million"].iloc[0], 106780.0, places=1)

        # Verify sum of 12 category percentage shares is ~100%
        shares_sum = ov_2024[ov_2024["spending_category"] != "All spending categories"]["percentage_share"].sum()
        self.assertAlmostEqual(shares_sum, 100.0, delta=1.0)

    def test_economic_model_macroeconomic_modes(self):
        """R6: Verify Domestic, Inbound, and Blended macroeconomic modes in economic impact model."""
        visitors = 1500
        days = 30

        # 1. Domestic Mode (DTS: RM 84.9B/94.9B, ALOS 2.45, RM 386.73/night)
        dom = calculate_redistribution_economic_impact(
            redirected_visitors_daily=visitors,
            days_period=days,
            poverty_rate_candidate=6.5,
            poverty_rate_hotspot=1.0,
            tourism_mode="domestic"
        )
        self.assertEqual(dom["tourism_mode"], "domestic")
        self.assertAlmostEqual(dom["alos_used"], 2.45, places=2)
        self.assertAlmostEqual(dom["spend_per_night_used"], DTS_2024_SPEND_PER_NIGHT, places=2)
        self.assertEqual(dom["output_multiplier"], TOURISM_OUTPUT_MULTIPLIER)
        self.assertIn(dom["macro_volume_rm_billion"], [84.9, 94.9])
        self.assertIn("shopping_retail_rm", dom["sectoral_breakdown"])
        self.assertGreater(dom["estimated_b40_income_million_rm"], 0)

        # 2. Inbound Mode (MOTAC: RM 106.8B, ALOS 4.80, RM 728.50/night)
        inb = calculate_redistribution_economic_impact(
            redirected_visitors_daily=visitors,
            days_period=days,
            poverty_rate_candidate=6.5,
            poverty_rate_hotspot=1.0,
            tourism_mode="inbound"
        )
        self.assertEqual(inb["tourism_mode"], "inbound")
        self.assertAlmostEqual(inb["alos_used"], INBOUND_ALOS, places=2)
        self.assertAlmostEqual(inb["spend_per_night_used"], INBOUND_SPEND_PER_NIGHT, places=2)
        self.assertEqual(inb["output_multiplier"], INBOUND_OUTPUT_MULTIPLIER)
        self.assertAlmostEqual(inb["macro_volume_rm_billion"], 106.8, places=1)
        # Inbound spend and output must be significantly greater than domestic for same visitor count
        self.assertGreater(inb["direct_spend_million_rm"], dom["direct_spend_million_rm"])
        self.assertGreater(inb["total_economic_output_million_rm"], dom["total_economic_output_million_rm"])
        # Verify 12 categories present
        self.assertEqual(len(inb["sectoral_breakdown_12"]), 12)
        for cat in INBOUND_CATEGORIES:
            self.assertIn(cat, inb["sectoral_breakdown_12"])

        # 3. Blended Mode (Total: RM 191.7B/201.7B, ALOS ~3.63, RM ~557.62/night)
        blend = calculate_redistribution_economic_impact(
            redirected_visitors_daily=visitors,
            days_period=days,
            poverty_rate_candidate=6.5,
            poverty_rate_hotspot=1.0,
            tourism_mode="blended"
        )
        self.assertEqual(blend["tourism_mode"], "blended")
        self.assertEqual(blend["output_multiplier"], BLENDED_OUTPUT_MULTIPLIER)
        self.assertIn(blend["macro_volume_rm_billion"], [191.7, 201.7])
        # Blended output must be strictly intermediate between domestic and inbound
        self.assertGreater(blend["total_economic_output_million_rm"], dom["total_economic_output_million_rm"])
        self.assertLess(blend["total_economic_output_million_rm"], inb["total_economic_output_million_rm"])

    def test_inbound_top_markets_and_market_breakdown(self):
        """R6: Verify helper functions get_top_inbound_markets and get_market_expenditure_breakdown."""
        # Top 10 inbound markets
        top_mkts = get_top_inbound_markets(n=10, year=2024)
        self.assertEqual(len(top_mkts), 10)
        self.assertIn("market", top_mkts.columns)
        self.assertIn("value_rm_million", top_mkts.columns)

        # Verify Singapore is the top source market
        top_1_mkt = top_mkts.iloc[0]["market"]
        self.assertEqual(top_1_mkt, "Singapore")
        self.assertAlmostEqual(top_mkts.iloc[0]["value_rm_million"], 27940.0, delta=100.0)

        # Breakdown for Singapore: Shopping should be the highest spending category
        sg_breakdown = get_market_expenditure_breakdown(market="Singapore", year=2024)
        self.assertEqual(len(sg_breakdown), 12)
        self.assertEqual(sg_breakdown.iloc[0]["spending_category"], "Shopping")
        self.assertGreater(sg_breakdown.iloc[0]["percentage_share"], 50.0)

        # Simulation with origin_market="Singapore" should reflect Singapore's shopping share
        sg_sim = calculate_redistribution_economic_impact(
            redirected_visitors_daily=1000,
            days_period=30,
            tourism_mode="inbound",
            origin_market="Singapore"
        )
        self.assertEqual(sg_sim["origin_market"], "Singapore")
        self.assertGreater(sg_sim["sectoral_breakdown_12"]["Shopping"], sg_sim["sectoral_breakdown_12"]["Accommodation"])

        # Fallback test for unknown market: should not crash and should return 12 categories
        fallback_breakdown = get_market_expenditure_breakdown(market="NonexistentCountryXYZ")
        self.assertEqual(len(fallback_breakdown), 12)

    def test_motac_aor_all_16_states_data_sanitization(self):
        """R4: Verify all 16 states/FTs exist in motac_hotel_occupancy_aor_timeseries.csv
        with valid non-zero AOR values across all years 2017–2026, and 2017 anomalies are fixed."""
        csv_path = BASE_DIR / "data" / "processed" / "motac_hotel_occupancy_aor_timeseries.csv"
        self.assertTrue(csv_path.exists(), f"AOR timeseries CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)
        self.assertFalse(df.empty, "AOR dataset must not be empty.")

        # 1. Verify all 16 states/FTs are present in dataset
        unique_states = sorted(df["state"].dropna().unique())
        self.assertEqual(len(unique_states), 16, f"Expected 16 states, found {len(unique_states)}: {unique_states}")
        for st in ALL_16_STATES:
            self.assertIn(st, unique_states, f"State {st} missing from AOR dataset.")

        # 2. Verify coverage across years 2017 to 2026
        years = sorted(df["year"].unique())
        expected_years = list(range(2017, 2027))
        self.assertEqual(years, expected_years, f"Years mismatch: expected {expected_years}, got {years}")

        # 3. Verify every year has all 16 states represented
        for yr in expected_years:
            yr_states = set(df[df["year"] == yr]["state"])
            self.assertEqual(len(yr_states), 16, f"Year {yr} has {len(yr_states)} states instead of 16.")
            self.assertEqual(yr_states, set(ALL_16_STATES), f"Year {yr} states mismatch: {set(ALL_16_STATES) - yr_states}")

        # 4. Verify all AOR values are valid positive percentages
        self.assertTrue(
            (df["average_occupancy_rate_pct"] > 0.0).all(),
            "All AOR percentage values must be strictly positive (> 0.0%)."
        )
        self.assertTrue(
            (df["average_occupancy_rate_pct"] <= 100.0).all(),
            "All AOR percentage values must be <= 100.0%."
        )

        # 5. Verify 2017 sanitization: no index number / token offset artifacts (< 10%)
        df_2017 = df[df["year"] == 2017]
        self.assertGreater(df_2017["average_occupancy_rate_pct"].min(), 30.0, "2017 minimum AOR must be > 30.0%")
        
        # Spot-check official MOTAC annual summary baseline values
        kl_2017 = df_2017[df_2017["state"] == "W.P. Kuala Lumpur"]["average_occupancy_rate_pct"].iloc[0]
        self.assertAlmostEqual(kl_2017, 66.1, delta=2.0, msg="KL 2017 AOR should be ~66.1%")

        melaka_2017 = df_2017[df_2017["state"] == "Melaka"]["average_occupancy_rate_pct"].iloc[0]
        self.assertGreater(melaka_2017, 50.0, msg="Melaka 2017 AOR must be healthy baseline > 50%")

        kedah_2017 = df_2017[df_2017["state"] == "Kedah"]["average_occupancy_rate_pct"].iloc[0]
        self.assertGreater(kedah_2017, 50.0, msg="Kedah 2017 AOR must be healthy baseline > 50%")

        kelantan_2017 = df_2017[df_2017["state"] == "Kelantan"]["average_occupancy_rate_pct"].iloc[0]
        self.assertGreater(kelantan_2017, 40.0, msg="Kelantan 2017 AOR must be healthy baseline > 40%")

    def test_national_benchmark_mean_aor_calculation(self):
        """R4: Verify national benchmark mean AOR calculation function produces valid values bounded in [30.0, 75.0]."""
        csv_path = BASE_DIR / "data" / "processed" / "motac_hotel_occupancy_aor_timeseries.csv"
        df = pd.read_csv(csv_path)

        # Test calculate_national_aor_baseline function
        nat_baseline_df = calculate_national_aor_baseline(df)
        self.assertFalse(nat_baseline_df.empty, "National baseline should not be empty.")
        self.assertEqual(len(nat_baseline_df), 10, "Should have 10 annual benchmark data points (2017–2026).")
        self.assertIn("state", nat_baseline_df.columns)
        self.assertTrue((nat_baseline_df["state"] == "National Mean Baseline").all())

        # Test scalar mean property and compute_national_mean_aor
        overall_nat_mean = compute_national_mean_aor(df)
        self.assertIsInstance(overall_nat_mean, float)
        self.assertGreaterEqual(overall_nat_mean, 30.0, "National mean AOR must be >= 30.0%")
        self.assertLessEqual(overall_nat_mean, 75.0, "National mean AOR must be <= 75.0%")

        # Test DataFrame comparison operator overload
        self.assertGreaterEqual(float(nat_baseline_df), 30.0)
        self.assertLessEqual(float(nat_baseline_df), 75.0)

        # Test quarterly/period grouping
        nat_quarterly_df = calculate_national_aor_baseline(df, group_by_period=True)
        self.assertGreater(len(nat_quarterly_df), len(nat_baseline_df))

        # Fallback on empty dataframe
        empty_nat = calculate_national_aor_baseline(pd.DataFrame())
        self.assertTrue(empty_nat.empty)
        empty_scalar = compute_national_mean_aor(pd.DataFrame())
        self.assertGreaterEqual(empty_scalar, 30.0)
        self.assertLessEqual(empty_scalar, 75.0)

    def test_regional_cluster_presets_cover_all_16_states(self):
        """R4: Verify regional cluster presets correctly cover all 16 states without omission or duplication."""
        # 1. 5 mutually exclusive planning regions
        expected_clusters = ["Northern", "Central", "Southern", "East Coast", "East Malaysia"]
        self.assertEqual(list(REGIONAL_CLUSTERS.keys()), expected_clusters)

        # 2. Check cluster counts
        self.assertEqual(len(REGIONAL_CLUSTERS["Northern"]), 4)  # Perlis, Kedah, Pulau Pinang, Perak
        self.assertEqual(len(REGIONAL_CLUSTERS["Central"]), 3)   # Selangor, W.P. Kuala Lumpur, W.P. Putrajaya
        self.assertEqual(len(REGIONAL_CLUSTERS["Southern"]), 3)  # Negeri Sembilan, Melaka, Johor
        self.assertEqual(len(REGIONAL_CLUSTERS["East Coast"]), 3) # Kelantan, Terengganu, Pahang
        self.assertEqual(len(REGIONAL_CLUSTERS["East Malaysia"]), 3) # Sabah, Sarawak, W.P. Labuan

        # 3. Ensure no state is in more than one regional cluster (disjoint sets)
        seen_states = set()
        for c_name, states in REGIONAL_CLUSTERS.items():
            for s in states:
                self.assertNotIn(s, seen_states, f"State {s} duplicated in multiple regional clusters!")
                seen_states.add(s)

        # 4. Ensure the union of all 5 clusters is exactly all 16 states
        self.assertEqual(seen_states, set(ALL_16_STATES))
        self.assertEqual(len(seen_states), 16)

        # 5. Test get_cluster_states helper
        self.assertEqual(get_cluster_states("All 16 States"), ALL_16_STATES)
        self.assertEqual(get_cluster_states("Northern"), REGIONAL_CLUSTERS["Northern"])
        self.assertEqual(get_cluster_states("UnknownCluster"), [])

        # 6. Test state normalization helper
        self.assertEqual(normalize_state_name("Kuala Lumpur"), "W.P. Kuala Lumpur")
        self.assertEqual(normalize_state_name("KL"), "W.P. Kuala Lumpur")
        self.assertEqual(normalize_state_name("Putrajaya"), "W.P. Putrajaya")
        self.assertEqual(normalize_state_name("Labuan"), "W.P. Labuan")
        self.assertEqual(normalize_state_name("Penang"), "Pulau Pinang")
        self.assertEqual(normalize_state_name("Pahang"), "Pahang")
        self.assertEqual(normalize_state_name(""), "")

    def test_view4_altair_aor_monitoring_chart_spec(self):
        """R4: Verify Altair multi-state AOR chart compiles with saturation band, state lines, and national baseline."""
        import altair as alt
        csv_path = BASE_DIR / "data" / "processed" / "motac_hotel_occupancy_aor_timeseries.csv"
        aor_df = pd.read_csv(csv_path)

        nat_avg = calculate_national_aor_baseline(aor_df)

        # Test with each regional cluster preset
        for cluster_name, states in REGIONAL_CLUSTERS.items():
            filtered = aor_df[aor_df["state"].isin(states)].sort_values("year")
            state_line = alt.Chart(filtered).mark_line(point=True).encode(
                x=alt.X("year:O"), y=alt.Y("average_occupancy_rate_pct:Q"), color=alt.Color("state:N")
            )
            nat_line = alt.Chart(nat_avg).mark_line(color="#64748b", strokeDash=[4, 4]).encode(
                x=alt.X("year:O"), y=alt.Y("average_occupancy_rate_pct:Q")
            )
            sat_band = alt.Chart(pd.DataFrame([{"threshold": 75.0, "ceiling": 95.0}])).mark_rect().encode(
                y=alt.Y("threshold:Q"), y2=alt.Y2("ceiling:Q")
            )
            chart = (sat_band + state_line + nat_line).properties(height=380)
            spec = chart.to_dict()
            self.assertIn("layer", spec)
            self.assertEqual(len(spec["layer"]), 3)

    def test_aggregate_destinations_by_state(self):
        """R1: Verify aggregate_destinations_by_state produces exactly 16 state macro centroids with exact demand preservation."""
        csv_path = BASE_DIR / "data" / "processed" / "destinations_master.csv"
        destinations_df = pd.read_csv(csv_path)

        agg_df = aggregate_destinations_by_state(destinations_df)

        # 1. 16 unique states / federal territories
        self.assertEqual(len(agg_df), 16, "Must produce exactly 16 state macro records.")
        self.assertEqual(agg_df["state_name"].nunique(), 16)

        # 2. Demand and capacity sum preservation
        orig_demand_sum = destinations_df["daily_demand_peak"].sum()
        agg_demand_sum = agg_df["daily_demand_peak"].sum()
        self.assertAlmostEqual(agg_demand_sum, orig_demand_sum, places=1,
                               msg="Aggregated peak demand must strictly match sum of 110 districts.")

        orig_cap_sum = destinations_df["sustainable_capacity"].sum()
        agg_cap_sum = agg_df["sustainable_capacity"].sum()
        self.assertAlmostEqual(agg_cap_sum, orig_cap_sum, places=1,
                               msg="Aggregated sustainable capacity must strictly match sum of 110 districts.")

        # 3. Sum of district counts must equal 110
        self.assertEqual(agg_df["district_count"].sum(), len(destinations_df))

        # 4. Centroid coordinates must be geographically valid in Malaysia
        for _, row in agg_df.iterrows():
            self.assertTrue(1.0 <= row["lat"] <= 7.5, f"State {row['state_name']} lat {row['lat']} out of bounds")
            self.assertTrue(99.5 <= row["lon"] <= 119.5, f"State {row['state_name']} lon {row['lon']} out of bounds")
            self.assertGreaterEqual(row["elevation"], 25000.0)
            self.assertIn("district_name", row)
            self.assertIn("rep_district", row)
            self.assertTrue(len(str(row["district_name"])) > 0)
            # Ensure tooltip_html is present and clean
            self.assertIn("tooltip_html", row)
            self.assertNotIn("{name}", row["tooltip_html"])
            self.assertNotIn("{status}", row["tooltip_html"])
            self.assertNotIn("{tot_demand", row["tooltip_html"])

        # 5. Empty DataFrame safety
        empty_res = aggregate_destinations_by_state(pd.DataFrame())
        self.assertTrue(empty_res.empty)
        self.assertIn("state_name", empty_res.columns)

    def test_pydeck_3d_elevation_map_modes_and_tooltip_schema(self):
        """R1: Verify render_pydeck_3d_elevation_map under district and state modes with zero-artifact tooltips."""
        import pydeck as pdk
        csv_path = BASE_DIR / "data" / "processed" / "destinations_master.csv"
        dest_df = pd.read_csv(csv_path)

        hotspot = dest_df.iloc[0].to_dict()
        alternatives = [dest_df.iloc[1].to_dict(), dest_df.iloc[2].to_dict()]

        # 1. District Mode
        deck_dist = render_pydeck_3d_elevation_map(hotspot, alternatives, dest_df, lod_mode="district")
        self.assertIsInstance(deck_dist, pdk.Deck)
        self.assertTrue("{name}" in deck_dist._tooltip.get("html") or "{tooltip_html}" in deck_dist._tooltip.get("html"))

        # Verify ColumnLayer
        dist_col_layer = deck_dist.layers[0]
        self.assertEqual(dist_col_layer.id, "district_columns")
        self.assertTrue(dist_col_layer.pickable)

        # Verify all records in district ColumnLayer have clean tooltip_html without artifacts
        dist_col_records = dist_col_layer.data if isinstance(dist_col_layer.data, list) else dist_col_layer.data.to_dict("records")
        self.assertEqual(len(dist_col_records), len(dest_df))
        for r in dist_col_records:
            html = r["tooltip_html"]
            self.assertTrue(isinstance(html, str) and len(html) > 20)
            self.assertNotIn("{name}", html, "Artifact {name} detected in tooltip!")
            self.assertNotIn("{state_name}", html, "Artifact {state_name} detected in tooltip!")
            self.assertNotIn("{status}", html, "Artifact {status} detected in tooltip!")
            self.assertNotIn("{demand}", html, "Artifact {demand} detected in tooltip!")
            self.assertNotIn("{pressure", html, "Artifact {pressure} detected in tooltip!")
            self.assertIn(r["state_name"], html)

        # Verify secondary layers (ground paths and tensile arcs) are non-pickable
        for layer in deck_dist.layers[1:]:
            self.assertFalse(layer.pickable, f"Secondary layer {layer.id} must be pickable=False to avoid tooltip collisions.")

        # 2. State Mode
        deck_state = render_pydeck_3d_elevation_map(hotspot, alternatives, dest_df, lod_mode="state")
        self.assertIsInstance(deck_state, pdk.Deck)
        self.assertTrue("{state_name}" in deck_state._tooltip.get("html") or "{tooltip_html}" in deck_state._tooltip.get("html"))

        state_col_layer = deck_state.layers[0]
        self.assertTrue(state_col_layer.pickable)
        # Macro radius slimmed to avoid overlap (22km nominal)
        self.assertGreaterEqual(state_col_layer.radius, 20000)
        self.assertLessEqual(state_col_layer.radius, 25000)

        state_col_records = state_col_layer.data if isinstance(state_col_layer.data, list) else state_col_layer.data.to_dict("records")
        self.assertEqual(len(state_col_records), 16)
        for r in state_col_records:
            html = r["tooltip_html"]
            self.assertTrue(isinstance(html, str) and len(html) > 20)
            self.assertNotIn("{name}", html)
            self.assertNotIn("{status}", html)
            self.assertIn(r["state_name"], html)

    def test_simulate_counterfactual_intervention_export_and_execution(self):
        """R1/XAI: Verify simulate_counterfactual_intervention clean export and parametric sensitivity."""
        csv_path = BASE_DIR / "data" / "processed" / "destinations_master.csv"
        dest_df = pd.read_csv(csv_path)

        hotspot_row = dest_df.iloc[0]
        candidate_row = dest_df.iloc[1]

        # 1. Standard execution
        res = simulate_counterfactual_intervention(
            hotspot_row=hotspot_row,
            candidate_row=candidate_row,
            travel_time_reduction_mins=20,
            capacity_expansion_pct=25
        )

        expected_keys = ["baseline_score", "simulated_score", "delta_score", "is_feasible", "policy_impact", "explanation"]
        for k in expected_keys:
            self.assertIn(k, res)

        self.assertGreaterEqual(res["simulated_score"], res["baseline_score"])
        self.assertGreater(res["delta_score"], 0.0)
        self.assertIsInstance(res["is_feasible"], bool)
        self.assertIn(res["policy_impact"], ["High Priority", "Incremental"])

        # 2. Backward compatibility with simulate_counterfactual_policy
        policy_res = simulate_counterfactual_policy(candidate_row, time_reduction_mins=15.0, extra_rooms_pct=10.0)
        self.assertIn("score_delta_wsm", policy_res)
        self.assertIn("explanation", policy_res)


if __name__ == "__main__":
    unittest.main()

