"""
DESTINASI — Milestone 4 Adversarial Stress Testing & Empirical Challenge Suite
Target: src/aor_engine.py & Regional Cluster Architecture
Author: challenger_m4_1 (critic, specialist)
Target: DOSM Datathon 2026

Exhaustively tests:
1. Canonical State & Regional Cluster Completeness & Disjoint Partitioning
2. State Name Normalization & Alias Mapping Robustness
3. National Mean AOR Baseline Calculation & Period Grouping
4. Annual Baseline Bounds Analysis (2017-2026) & COVID-19 Impact Verification
5. Missing States & Sub-Cluster Mean Robustness
6. Empty Selections & Missing Column Resilience
7. Boundary Inputs: Extreme Percentages (0%, 100%, negative, >100%, NaN, Inf)
8. NationalAORBenchmarkFrame Class Operators & Type Comparison Edge Cases
9. Altair Visual Spec Generation Under Empty & Extreme Selections
10. Monte Carlo Fuzzing (500 randomized trials)
"""

import unittest
from pathlib import Path
import random
import numpy as np
import pandas as pd
import altair as alt

from src.aor_engine import (
    ALL_16_STATES,
    REGIONAL_CLUSTERS,
    STATE_ALIASES,
    normalize_state_name,
    get_cluster_states,
    NationalAORBenchmarkFrame,
    compute_national_mean_aor,
    calculate_national_aor_baseline,
)

BASE_DIR = Path(__file__).resolve().parents[1]
AOR_CSV_PATH = BASE_DIR / "data" / "processed" / "motac_hotel_occupancy_aor_timeseries.csv"
REGIONS_CSV_PATH = BASE_DIR / "data" / "state_planning_regions.csv"


class TestAOREngineStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aor_df = pd.read_csv(AOR_CSV_PATH)

    # -------------------------------------------------------------------------
    # 1. CANONICAL STATE & REGIONAL CLUSTER INTEGRITY
    # -------------------------------------------------------------------------
    def test_all_16_states_exact_composition(self):
        """Verify ALL_16_STATES contains exactly 16 unique, sorted, non-empty state names."""
        self.assertEqual(len(ALL_16_STATES), 16)
        self.assertEqual(len(set(ALL_16_STATES)), 16)
        for st in ALL_16_STATES:
            self.assertIsInstance(st, str)
            self.assertTrue(len(st.strip()) > 0)
        self.assertEqual(ALL_16_STATES, sorted(ALL_16_STATES))

    def test_regional_clusters_partition_and_alignment(self):
        """Verify REGIONAL_CLUSTERS form a mutually exclusive, collectively exhaustive partition matching state_planning_regions.csv."""
        expected_clusters = {"Northern", "Central", "Southern", "East Coast", "East Malaysia"}
        self.assertEqual(set(REGIONAL_CLUSTERS.keys()), expected_clusters)

        all_clustered_states = []
        for cluster_name, states in REGIONAL_CLUSTERS.items():
            self.assertGreater(len(states), 0, f"Cluster {cluster_name} is empty.")
            all_clustered_states.extend(states)

        # Disjointness check
        self.assertEqual(len(all_clustered_states), 16)
        self.assertEqual(set(all_clustered_states), set(ALL_16_STATES))

        # Check against ground-truth planning regions CSV
        if REGIONS_CSV_PATH.exists():
            df_regions = pd.read_csv(REGIONS_CSV_PATH)
            for _, row in df_regions.iterrows():
                st = row["state"]
                reg = row["planning_region"]
                self.assertIn(reg, REGIONAL_CLUSTERS)
                self.assertIn(st, REGIONAL_CLUSTERS[reg])

    def test_get_cluster_states_edge_cases(self):
        """Test get_cluster_states helper with valid, invalid, empty, and None inputs."""
        # Valid clusters
        self.assertEqual(get_cluster_states("All 16 States"), ALL_16_STATES)
        for c_name, c_states in REGIONAL_CLUSTERS.items():
            self.assertEqual(get_cluster_states(c_name), c_states)

        # Unknown / boundary inputs
        self.assertEqual(get_cluster_states("NonExistentCluster"), [])
        self.assertEqual(get_cluster_states(""), [])
        self.assertEqual(get_cluster_states(None), [])
        self.assertEqual(get_cluster_states(12345), [])

        # Immutability: modifying returned list must not alter internal REGIONAL_CLUSTERS
        northern = get_cluster_states("Northern")
        northern.append("AdversarialStateXYZ")
        self.assertNotIn("AdversarialStateXYZ", REGIONAL_CLUSTERS["Northern"])

    # -------------------------------------------------------------------------
    # 2. STATE NAME NORMALIZATION & ALIAS MAPPING
    # -------------------------------------------------------------------------
    def test_normalize_state_name_aliases_and_whitespace(self):
        """Test normalization of known aliases, whitespace, and dirty inputs."""
        # Known aliases
        self.assertEqual(normalize_state_name("Kuala Lumpur"), "W.P. Kuala Lumpur")
        self.assertEqual(normalize_state_name("KL"), "W.P. Kuala Lumpur")
        self.assertEqual(normalize_state_name("Wilayah Persekutuan Kuala Lumpur"), "W.P. Kuala Lumpur")
        self.assertEqual(normalize_state_name("Putrajaya"), "W.P. Putrajaya")
        self.assertEqual(normalize_state_name("Wilayah Persekutuan Putrajaya"), "W.P. Putrajaya")
        self.assertEqual(normalize_state_name("Labuan"), "W.P. Labuan")
        self.assertEqual(normalize_state_name("Wilayah Persekutuan Labuan"), "W.P. Labuan")
        self.assertEqual(normalize_state_name("Penang"), "Pulau Pinang")
        self.assertEqual(normalize_state_name("Pulau Pinang"), "Pulau Pinang")

        # Whitespace handling
        self.assertEqual(normalize_state_name("  Kuala Lumpur  "), "W.P. Kuala Lumpur")
        self.assertEqual(normalize_state_name("  Melaka\t"), "Melaka")
        self.assertEqual(normalize_state_name("  Johor\n"), "Johor")

        # Malformed / boundary inputs
        self.assertEqual(normalize_state_name(""), "")
        self.assertEqual(normalize_state_name("   "), "")
        self.assertEqual(normalize_state_name(None), "")
        self.assertEqual(normalize_state_name(123), "")
        self.assertEqual(normalize_state_name(45.67), "")
        self.assertEqual(normalize_state_name(["Johor"]), "")
        self.assertEqual(normalize_state_name({"state": "Johor"}), "")

    # -------------------------------------------------------------------------
    # 3. NATIONAL BASELINE BOUNDS & EMPIRICAL ANNUAL SWEEP (2017-2026)
    # -------------------------------------------------------------------------
    def test_empirical_annual_national_baseline_bounds(self):
        """
        Verify national mean baseline calculation for each year from 2017 to 2026.
        Empirical observation:
        - 8 of 10 years (2017-2019, 2022-2026) are strictly bounded within [30.0, 75.0].
        - Years 2020 and 2021 drop below 30.0% (2020: 29.80%, 2021: 28.06%) due to COVID-19 pandemic MCO.
        - Overall scalar benchmark across the entire 10-year span is 46.15% (strictly in [30.0, 75.0]).
        """
        baseline_df = calculate_national_aor_baseline(self.aor_df)
        self.assertEqual(len(baseline_df), 10)
        self.assertEqual(list(baseline_df["year"]), list(range(2017, 2027)))

        annual_means = dict(zip(baseline_df["year"], baseline_df["average_occupancy_rate_pct"]))

        # Check normal non-pandemic years are strictly bounded within [30.0, 75.0]
        normal_years = [2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026]
        for yr in normal_years:
            val = annual_means[yr]
            self.assertGreaterEqual(val, 30.0, f"Year {yr} mean {val:.2f}% unexpectedly < 30.0%")
            self.assertLessEqual(val, 75.0, f"Year {yr} mean {val:.2f}% unexpectedly > 75.0%")

        # Check pandemic anomaly years (reflecting genuine MOTAC historical statistics)
        self.assertAlmostEqual(annual_means[2020], 29.80, delta=0.5, msg="2020 COVID MCO mean drop")
        self.assertAlmostEqual(annual_means[2021], 28.06, delta=0.5, msg="2021 COVID MCO mean drop")
        self.assertLess(annual_means[2020], 30.0, "2020 AOR correctly captures COVID downturn below 30%")
        self.assertLess(annual_means[2021], 30.0, "2021 AOR correctly captures COVID downturn below 30%")

        # Check overall aggregate benchmark satisfies [30.0, 75.0]
        overall_mean = compute_national_mean_aor(self.aor_df)
        self.assertGreaterEqual(overall_mean, 30.0)
        self.assertLessEqual(overall_mean, 75.0)
        self.assertAlmostEqual(overall_mean, 46.15, delta=1.5)

    def test_quarterly_baseline_period_grouping(self):
        """Verify calculate_national_aor_baseline with group_by_period=True."""
        q_baseline = calculate_national_aor_baseline(self.aor_df, group_by_period=True)
        self.assertIn("period", q_baseline.columns)
        self.assertGreater(len(q_baseline), 10)

        # Check that periods include Annual, Q1, Q2, Q3, Q4
        unique_periods = set(q_baseline["period"].dropna().unique())
        self.assertTrue({"Annual", "Q1", "Q2"}.issubset(unique_periods))

        # Check all periods have valid positive mean values
        self.assertTrue((q_baseline["average_occupancy_rate_pct"] > 0.0).all())
        self.assertTrue((q_baseline["average_occupancy_rate_pct"] <= 100.0).all())

    # -------------------------------------------------------------------------
    # 4. MISSING STATES & SUBSET SELECTIONS
    # -------------------------------------------------------------------------
    def test_missing_states_and_single_state_subsets(self):
        """Verify baseline calculation works accurately with incomplete state sets."""
        # 1. Single state dataset
        df_johor = self.aor_df[self.aor_df["state"] == "Johor"]
        baseline_johor = calculate_national_aor_baseline(df_johor)
        self.assertEqual(len(baseline_johor), 10)
        self.assertTrue((baseline_johor["state"] == "National Mean Baseline").all())

        # 2. Subset of 5 states
        subset_states = ["Johor", "Kedah", "Melaka", "Perak", "Sabah"]
        df_sub = self.aor_df[self.aor_df["state"].isin(subset_states)]
        baseline_sub = calculate_national_aor_baseline(df_sub)
        self.assertEqual(len(baseline_sub), 10)

        # Manually compute expected 2017 mean for subset
        expected_2017 = df_sub[df_sub["year"] == 2017]["average_occupancy_rate_pct"].mean()
        actual_2017 = baseline_sub[baseline_sub["year"] == 2017]["average_occupancy_rate_pct"].iloc[0]
        self.assertAlmostEqual(actual_2017, expected_2017, places=5)

    def test_missing_state_column_graceful_handling(self):
        """Verify baseline calculation does not crash if 'state' column is missing."""
        df_no_state = self.aor_df[["year", "average_occupancy_rate_pct"]].copy()
        res = calculate_national_aor_baseline(df_no_state)
        self.assertEqual(len(res), 10)
        self.assertTrue((res["state"] == "National Mean Baseline").all())

    # -------------------------------------------------------------------------
    # 5. EMPTY SELECTIONS & MISSING REQUIRED COLUMNS
    # -------------------------------------------------------------------------
    def test_empty_dataframe_and_missing_metric_column(self):
        """Verify calculate_national_aor_baseline and compute_national_mean_aor on empty/missing inputs."""
        # Empty DataFrame
        empty_res = calculate_national_aor_baseline(pd.DataFrame())
        self.assertIsInstance(empty_res, NationalAORBenchmarkFrame)
        self.assertTrue(empty_res.empty)
        self.assertEqual(empty_res.scalar_mean, 0.0)
        self.assertEqual(float(empty_res), 0.0)

        empty_scalar = compute_national_mean_aor(pd.DataFrame())
        self.assertEqual(empty_scalar, 47.50)

        # Missing 'average_occupancy_rate_pct' column
        df_no_aor = pd.DataFrame({"year": [2024], "state": ["Johor"]})
        res_no_aor = calculate_national_aor_baseline(df_no_aor)
        self.assertTrue(res_no_aor.empty)
        scalar_no_aor = compute_national_mean_aor(df_no_aor)
        self.assertEqual(scalar_no_aor, 47.50)

    # -------------------------------------------------------------------------
    # 6. BOUNDARY & ADVERSARIAL VALUE INPUTS
    # -------------------------------------------------------------------------
    def test_boundary_aor_values_zero_and_hundred(self):
        """Verify calculation under boundary 0% and 100% occupancy values."""
        df_boundary = pd.DataFrame({
            "year": [2024, 2024],
            "state": ["Johor", "Kedah"],
            "average_occupancy_rate_pct": [0.0, 100.0]
        })
        baseline = calculate_national_aor_baseline(df_boundary)
        self.assertEqual(len(baseline), 1)
        self.assertAlmostEqual(baseline["average_occupancy_rate_pct"].iloc[0], 50.0)
        self.assertAlmostEqual(float(baseline), 50.0)

    def test_nan_values_in_aor_column(self):
        """Verify pandas mean skips NaN without crashing."""
        df_nan = pd.DataFrame({
            "year": [2024, 2024, 2024],
            "state": ["Johor", "Kedah", "Melaka"],
            "average_occupancy_rate_pct": [60.0, np.nan, 80.0]
        })
        baseline = calculate_national_aor_baseline(df_nan)
        self.assertEqual(len(baseline), 1)
        # Mean of 60.0 and 80.0 is 70.0 (NaN ignored)
        self.assertAlmostEqual(baseline["average_occupancy_rate_pct"].iloc[0], 70.0)

    def test_numeric_invalid_years(self):
        """Verify numeric invalid years (negative, far future) are grouped cleanly."""
        df_invalid_yr = pd.DataFrame({
            "year": [-50, 9999, 2024],
            "state": ["Johor", "Kedah", "Melaka"],
            "average_occupancy_rate_pct": [40.0, 60.0, 50.0]
        })
        baseline = calculate_national_aor_baseline(df_invalid_yr)
        self.assertEqual(len(baseline), 3)
        self.assertEqual(list(baseline["year"]), [-50, 2024, 9999])

    # -------------------------------------------------------------------------
    # 7. NATIONALAORBENCHMARKFRAME OPERATOR OVERLOADS & TYPE COMPLIANCE
    # -------------------------------------------------------------------------
    def test_benchmark_frame_scalar_comparison_operators(self):
        """Verify NationalAORBenchmarkFrame operator overloads for int, float, np.float64."""
        baseline = calculate_national_aor_baseline(self.aor_df)
        mean_val = baseline.scalar_mean

        # Float comparisons
        self.assertTrue(baseline >= 30.0)
        self.assertTrue(baseline <= 75.0)
        self.assertTrue(baseline > 20.0)
        self.assertTrue(baseline < 80.0)
        self.assertFalse(baseline >= 90.0)
        self.assertFalse(baseline <= 20.0)

        # Int comparisons
        self.assertTrue(baseline >= 30)
        self.assertTrue(baseline <= 75)
        self.assertTrue(baseline > 20)
        self.assertTrue(baseline < 80)

        # NumPy float comparison
        self.assertTrue(baseline >= np.float64(30.0))
        self.assertTrue(baseline <= np.float64(75.0))

        # Float conversion
        self.assertAlmostEqual(float(baseline), mean_val, places=4)

    # -------------------------------------------------------------------------
    # 8. ALTAIR CHART SPECIFICATION COMPILES UNDER ALL SCENARIOS
    # -------------------------------------------------------------------------
    def test_altair_chart_spec_with_empty_and_all_clusters(self):
        """Verify Altair multi-state chart specification compiles without error under empty and all cluster states."""
        nat_avg = calculate_national_aor_baseline(self.aor_df)
        sat_band = alt.Chart(pd.DataFrame([{"threshold": 75.0, "ceiling": 95.0}])).mark_rect().encode(
            y=alt.Y("threshold:Q"), y2=alt.Y2("ceiling:Q")
        )
        nat_line = alt.Chart(nat_avg).mark_line(color="#64748b", strokeDash=[4, 4]).encode(
            x=alt.X("year:O"), y=alt.Y("average_occupancy_rate_pct:Q")
        )

        # 1. Empty state selection
        empty_filtered = self.aor_df[self.aor_df["state"].isin([])]
        state_line_empty = alt.Chart(empty_filtered).mark_line(point=True).encode(
            x=alt.X("year:O"), y=alt.Y("average_occupancy_rate_pct:Q"), color=alt.Color("state:N")
        )
        chart_empty = (sat_band + state_line_empty + nat_line).properties(height=380)
        spec_empty = chart_empty.to_dict()
        self.assertIn("layer", spec_empty)
        self.assertEqual(len(spec_empty["layer"]), 3)

        # 2. All 5 clusters + All 16 States
        test_presets = list(REGIONAL_CLUSTERS.keys()) + ["All 16 States"]
        for preset in test_presets:
            states = get_cluster_states(preset)
            filtered = self.aor_df[self.aor_df["state"].isin(states)]
            st_line = alt.Chart(filtered).mark_line(point=True).encode(
                x=alt.X("year:O"), y=alt.Y("average_occupancy_rate_pct:Q"), color=alt.Color("state:N")
            )
            chart = (sat_band + st_line + nat_line).properties(height=380)
            spec = chart.to_dict()
            self.assertIn("layer", spec)
            self.assertEqual(len(spec["layer"]), 3)

    # -------------------------------------------------------------------------
    # 9. MONTE CARLO RANDOM FUZZING (500 TRIALS)
    # -------------------------------------------------------------------------
    def test_monte_carlo_fuzzing_random_subsets_and_metrics(self):
        """Run 500 randomized trials of calculate_national_aor_baseline and verify numerical invariants."""
        rng = random.Random(42)
        years = list(range(2015, 2030))
        states_pool = ALL_16_STATES + ["Atlantis", "Moon", "UnknownState"]

        for trial in range(500):
            n_rows = rng.randint(1, 50)
            trial_years = [rng.choice(years) for _ in range(n_rows)]
            trial_states = [rng.choice(states_pool) for _ in range(n_rows)]
            trial_aor = [rng.uniform(-20.0, 120.0) for _ in range(n_rows)]

            df_trial = pd.DataFrame({
                "year": trial_years,
                "state": trial_states,
                "average_occupancy_rate_pct": trial_aor
            })

            baseline = calculate_national_aor_baseline(df_trial)
            self.assertIsInstance(baseline, NationalAORBenchmarkFrame)
            self.assertFalse(baseline.empty)
            self.assertIn("year", baseline.columns)
            self.assertIn("average_occupancy_rate_pct", baseline.columns)
            self.assertIn("state", baseline.columns)
            self.assertEqual(len(baseline), len(set(trial_years)))

            # Mathematical invariant: national baseline mean across years must be finite
            s_mean = baseline.scalar_mean
            self.assertTrue(np.isfinite(s_mean), f"Trial {trial}: Non-finite scalar mean {s_mean}")
            self.assertEqual(float(baseline), s_mean)


if __name__ == "__main__":
    unittest.main()
