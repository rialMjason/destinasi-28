"""
DESTINASI — Milestone 6 Empirical Challenge & Stress-Test Suite
Author: challenger_m6_1 (Empirical Challenger: critic, specialist)
Target: DOSM Datathon 2026

Adversarial stress-testing and empirical verification of:
1. Boundary conditions: zero visitors, 1M extreme pax/day, negative inputs, 0% vs 100% poverty rate.
2. Model invariants: linear scaling with days & visitors, non-negativity of all outputs.
3. Multiplier fidelity: 1.75x domestic, 1.82x inbound, 1.78x blended.
4. Helper function robustness: get_top_inbound_markets() and get_market_expenditure_breakdown()
   across valid, case-varying, unknown, and malformed inputs.
"""

import math
import unittest
import pandas as pd
import numpy as np

from src.economic_impact_model import (
    calculate_redistribution_economic_impact,
    get_top_inbound_markets,
    get_market_expenditure_breakdown,
    get_inbound_category_breakdown,
    DTS_2024_SPEND_PER_NIGHT,
    DTS_2024_ALOS,
    INBOUND_SPEND_PER_NIGHT,
    INBOUND_ALOS,
    BLENDED_SPEND_PER_NIGHT,
    BLENDED_ALOS,
    TOURISM_OUTPUT_MULTIPLIER,
    INBOUND_OUTPUT_MULTIPLIER,
    BLENDED_OUTPUT_MULTIPLIER,
    INBOUND_CATEGORIES,
    INBOUND_12_CATEGORIES_OVERALL,
    DOMESTIC_12_CATEGORIES,
    BLENDED_12_CATEGORIES
)


class TestMacroeconomicBoundaryConditions(unittest.TestCase):
    """Stress-test edge cases, boundary conditions, and adversarial inputs."""

    def test_zero_visitors_boundary(self):
        """Verify model behavior under zero visitors across all three tourism modes."""
        for mode in ["domestic", "inbound", "blended"]:
            with self.subTest(mode=mode):
                res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=0,
                    days_period=30,
                    tourism_mode=mode
                )
                self.assertEqual(res["total_tourist_trips"], 0)
                self.assertEqual(res["total_tourist_nights"], 0)
                self.assertEqual(res["direct_spend_total_rm"], 0.0)
                self.assertEqual(res["direct_spend_million_rm"], 0.0)
                self.assertEqual(res["total_economic_output_rm"], 0.0)
                self.assertEqual(res["total_economic_output_million_rm"], 0.0)
                self.assertEqual(res["estimated_b40_income_injected_rm"], 0.0)
                self.assertEqual(res["estimated_b40_income_million_rm"], 0.0)

                # Sector breakdowns should be zero
                for cat, val in res["sectoral_breakdown"].items():
                    self.assertEqual(val, 0.0, f"5-sector {cat} must be 0.0")
                for cat, val in res["sectoral_breakdown_12"].items():
                    self.assertEqual(val, 0.0, f"12-cat {cat} must be 0.0")

    def test_extreme_visitors_one_million_pax_daily(self):
        """Stress-test model with 1,000,000 visitors/day over 30 days (30M trips)."""
        extreme_pax = 1_000_000
        days = 30

        for mode in ["domestic", "inbound", "blended"]:
            with self.subTest(mode=mode):
                res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=extreme_pax,
                    days_period=days,
                    tourism_mode=mode
                )
                self.assertEqual(res["total_tourist_trips"], 30_000_000)
                self.assertTrue(math.isfinite(res["direct_spend_total_rm"]))
                self.assertTrue(math.isfinite(res["total_economic_output_rm"]))
                self.assertTrue(math.isfinite(res["estimated_b40_income_injected_rm"]))

                self.assertGreater(res["direct_spend_total_rm"], 1e10)  # > RM 10 Billion
                self.assertGreater(res["total_economic_output_rm"], 1e10)

                # Check sectoral breakdowns are finite and positive
                for cat, val in res["sectoral_breakdown"].items():
                    self.assertTrue(math.isfinite(val) and val > 0)
                for cat, val in res["sectoral_breakdown_12"].items():
                    self.assertTrue(math.isfinite(val) and val >= 0)
                    if mode == "domestic" and cat == "International Airfares":
                        self.assertEqual(val, 0.0, "Domestic tourists do not spend on International Airfares")
                    else:
                        self.assertGreater(val, 0, f"Expected positive spend for {cat} in mode {mode}")

    def test_negative_visitors_clamping(self):
        """Verify negative visitor inputs are clamped to 0 without throwing or yielding negative spend."""
        for neg_val in [-1, -500, -100000]:
            with self.subTest(neg_val=neg_val):
                res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=neg_val,
                    days_period=30,
                    tourism_mode="domestic"
                )
                self.assertEqual(res["total_tourist_trips"], 0.0)
                self.assertEqual(res["direct_spend_total_rm"], 0.0)
                self.assertEqual(res["total_economic_output_rm"], 0.0)
                self.assertEqual(res["estimated_b40_income_injected_rm"], 0.0)

    def test_poverty_rate_zero_vs_one_hundred_and_bounds(self):
        """Empirically test poverty rate behavior: 0% vs 100%, and mathematical bounds in [0.35, 0.65]."""
        visitors = 1000
        days = 30

        # Baseline: candidate poverty rate = 0.0%
        res_0 = calculate_redistribution_economic_impact(
            redirected_visitors_daily=visitors,
            days_period=days,
            poverty_rate_candidate=0.0,
            poverty_rate_hotspot=1.0
        )

        # High poverty: candidate poverty rate = 100.0%
        res_100 = calculate_redistribution_economic_impact(
            redirected_visitors_daily=visitors,
            days_period=days,
            poverty_rate_candidate=100.0,
            poverty_rate_hotspot=1.0
        )

        # Direct spend must be completely unaffected by poverty rate
        self.assertEqual(res_0["direct_spend_total_rm"], res_100["direct_spend_total_rm"])
        self.assertEqual(res_0["total_economic_output_rm"], res_100["total_economic_output_rm"])

        # B40 retention ratio must strictly increase from 0% to 100%
        ratio_0 = res_0["estimated_b40_income_injected_rm"] / res_0["direct_spend_total_rm"]
        ratio_100 = res_100["estimated_b40_income_injected_rm"] / res_100["direct_spend_total_rm"]

        self.assertAlmostEqual(ratio_0, 0.3708, delta=0.001)
        self.assertAlmostEqual(ratio_100, 0.4885, delta=0.001)
        self.assertGreater(ratio_100, ratio_0)

        # Adversarial sweep across extreme poverty ratios
        test_ratios = [
            (-100.0, 1.0),
            (0.0, 100.0),
            (10.0, 10.0),
            (50.0, 0.0),
            (100.0, 0.01),
            (10000.0, 0.001)  # Extreme theoretical poverty
        ]
        for cand, hot in test_ratios:
            r = calculate_redistribution_economic_impact(
                visitors, days, poverty_rate_candidate=cand, poverty_rate_hotspot=hot
            )
            retention = r["estimated_b40_income_injected_rm"] / r["direct_spend_total_rm"]
            self.assertGreaterEqual(retention, 0.35, f"Retention {retention} below 0.35 floor for ({cand}, {hot})")
            self.assertLessEqual(retention, 0.65, f"Retention {retention} above 0.65 cap for ({cand}, {hot})")

    def test_unknown_origin_markets_in_inbound_simulation(self):
        """Verify inbound mode gracefully falls back to national benchmark for unknown origin markets."""
        unknown_markets = ["Atlantis", "Mars Colony", "NonExistentCountry99", "Unknown", ""]

        for mkt in unknown_markets:
            with self.subTest(market=mkt):
                res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=1000,
                    days_period=30,
                    tourism_mode="inbound",
                    origin_market=mkt
                )
                self.assertEqual(len(res["sectoral_breakdown_12"]), 12)
                # Shopping should match national benchmark ~37.36%
                shopping_share = res["category_shares"].get("Shopping", 0.0)
                self.assertAlmostEqual(shopping_share, 0.3736, delta=0.01)
                self.assertGreater(res["direct_spend_total_rm"], 0)


class TestModelInvariantsAndScaling(unittest.TestCase):
    """Verify linear scaling, non-negativity, and output multipliers."""

    def test_linear_scaling_with_visitors(self):
        """Verify strict linear scaling f(k * V) = k * f(V)."""
        base_visitors = 750
        scale_factor = 3.5
        scaled_visitors = base_visitors * scale_factor
        days = 20

        for mode in ["domestic", "inbound", "blended"]:
            with self.subTest(mode=mode):
                base_res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=base_visitors,
                    days_period=days,
                    tourism_mode=mode
                )
                scaled_res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=scaled_visitors,
                    days_period=days,
                    tourism_mode=mode
                )

                ratio_spend = scaled_res["direct_spend_total_rm"] / base_res["direct_spend_total_rm"]
                ratio_output = scaled_res["total_economic_output_rm"] / base_res["total_economic_output_rm"]
                ratio_b40 = scaled_res["estimated_b40_income_injected_rm"] / base_res["estimated_b40_income_injected_rm"]

                self.assertAlmostEqual(ratio_spend, scale_factor, places=4)
                self.assertAlmostEqual(ratio_output, scale_factor, places=4)
                self.assertAlmostEqual(ratio_b40, scale_factor, places=4)

    def test_linear_scaling_with_days(self):
        """Verify strict linear scaling f(k * D) = k * f(D)."""
        visitors = 1200
        days_base = 14
        days_scaled = 28  # 2.0x

        for mode in ["domestic", "inbound", "blended"]:
            with self.subTest(mode=mode):
                base_res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=visitors,
                    days_period=days_base,
                    tourism_mode=mode
                )
                scaled_res = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=visitors,
                    days_period=days_scaled,
                    tourism_mode=mode
                )

                ratio_spend = scaled_res["direct_spend_total_rm"] / base_res["direct_spend_total_rm"]
                self.assertAlmostEqual(ratio_spend, 2.0, places=4)

    def test_non_negativity_of_all_outputs(self):
        """Verify that all outputs are non-negative across a grid of operational inputs."""
        visitor_grid = [0, 1, 10, 500, 1500, 10000]
        days_grid = [1, 7, 30, 90, 365]
        modes = ["domestic", "inbound", "blended"]

        for v in visitor_grid:
            for d in days_grid:
                for mode in modes:
                    res = calculate_redistribution_economic_impact(v, d, tourism_mode=mode)
                    self.assertGreaterEqual(res["direct_spend_total_rm"], 0.0)
                    self.assertGreaterEqual(res["total_economic_output_rm"], 0.0)
                    self.assertGreaterEqual(res["estimated_b40_income_injected_rm"], 0.0)
                    for k, val in res["sectoral_breakdown"].items():
                        self.assertGreaterEqual(val, 0.0, f"{k} was negative for {v}, {d}, {mode}")
                    for k, val in res["sectoral_breakdown_12"].items():
                        self.assertGreaterEqual(val, 0.0, f"{k} was negative for {v}, {d}, {mode}")

    def test_output_multipliers_exactness(self):
        """Verify empirical multipliers: 1.75x (domestic), 1.82x (inbound), and 1.78x (blended)."""
        visitors = 1000
        days = 30

        dom = calculate_redistribution_economic_impact(visitors, days, tourism_mode="domestic")
        self.assertEqual(dom["output_multiplier"], 1.75)
        self.assertAlmostEqual(dom["total_economic_output_rm"] / dom["direct_spend_total_rm"], 1.75, places=5)

        inb = calculate_redistribution_economic_impact(visitors, days, tourism_mode="inbound")
        self.assertEqual(inb["output_multiplier"], 1.82)
        self.assertAlmostEqual(inb["total_economic_output_rm"] / inb["direct_spend_total_rm"], 1.82, places=5)

        bld = calculate_redistribution_economic_impact(visitors, days, tourism_mode="blended")
        self.assertEqual(bld["output_multiplier"], 1.78)
        self.assertAlmostEqual(bld["total_economic_output_rm"] / bld["direct_spend_total_rm"], 1.78, places=5)

    def test_macroeconomic_mode_spend_hierarchy(self):
        """Verify Inbound spend > Blended spend > Domestic spend for identical visitor counts."""
        visitors = 2000
        days = 30

        dom = calculate_redistribution_economic_impact(visitors, days, tourism_mode="domestic")
        inb = calculate_redistribution_economic_impact(visitors, days, tourism_mode="inbound")
        bld = calculate_redistribution_economic_impact(visitors, days, tourism_mode="blended")

        self.assertGreater(inb["direct_spend_total_rm"], bld["direct_spend_total_rm"])
        self.assertGreater(bld["direct_spend_total_rm"], dom["direct_spend_total_rm"])

        self.assertGreater(inb["total_economic_output_rm"], bld["total_economic_output_rm"])
        self.assertGreater(bld["total_economic_output_rm"], dom["total_economic_output_rm"])


class TestHelperFunctionsEmpiricalValidation(unittest.TestCase):
    """Verify get_top_inbound_markets() and get_market_expenditure_breakdown()."""

    def test_top_inbound_markets_default_and_ranking(self):
        """Verify top inbound markets ranking, structure, and top countries."""
        df = get_top_inbound_markets()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertIn("market", df.columns)
        self.assertIn("value_rm_million", df.columns)
        self.assertIn("share_percent", df.columns)

        # Monotonicity: values must be in descending order
        values = df["value_rm_million"].tolist()
        for i in range(len(values) - 1):
            self.assertGreaterEqual(values[i], values[i + 1], "Markets must be sorted descending by expenditure")

        # Check top 3 empirical markets: Singapore (#1), China (#2), Indonesia (#3)
        self.assertEqual(df.iloc[0]["market"], "Singapore")
        self.assertAlmostEqual(df.iloc[0]["value_rm_million"], 27940.0, delta=50.0)
        self.assertEqual(df.iloc[1]["market"], "China")
        self.assertAlmostEqual(df.iloc[1]["value_rm_million"], 20870.0, delta=50.0)
        self.assertEqual(df.iloc[2]["market"], "Indonesia")
        self.assertAlmostEqual(df.iloc[2]["value_rm_million"], 15320.0, delta=50.0)

    def test_top_inbound_markets_parameter_variations(self):
        """Verify behavior of get_top_inbound_markets with n, top_n, and out-of-range years."""
        # Custom n
        df_5 = get_top_inbound_markets(n=5)
        self.assertEqual(len(df_5), 5)

        # top_n keyword alias
        df_top3 = get_top_inbound_markets(top_n=3)
        self.assertEqual(len(df_top3), 3)

        # n=0
        df_0 = get_top_inbound_markets(n=0)
        self.assertEqual(len(df_0), 0)

        # Out-of-range year
        df_invalid_year = get_top_inbound_markets(year=1900)
        self.assertIsInstance(df_invalid_year, pd.DataFrame)
        self.assertEqual(len(df_invalid_year), 0)

    def test_market_breakdown_valid_countries(self):
        """Verify expenditure breakdown for key international markets."""
        countries = ["Singapore", "China", "Indonesia", "Australia", "United Kingdom", "Japan"]
        for c in countries:
            with self.subTest(country=c):
                df = get_market_expenditure_breakdown(market=c, year=2024)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(len(df), 12, f"Market {c} must have exactly 12 categories")
                self.assertEqual(list(df.columns), ["spending_category", "percentage_share", "value_rm_million"])
                # Percentage share sum should be approximately 100%
                self.assertAlmostEqual(df["percentage_share"].sum(), 100.0, delta=0.5)

    def test_market_breakdown_case_insensitivity(self):
        """Verify case insensitivity for market names (e.g. 'singapore' vs 'SINGAPORE')."""
        df_standard = get_market_expenditure_breakdown("Singapore")
        df_lower = get_market_expenditure_breakdown("singapore")
        df_upper = get_market_expenditure_breakdown("SINGAPORE")

        pd.testing.assert_frame_equal(df_standard, df_lower)
        pd.testing.assert_frame_equal(df_standard, df_upper)

    def test_market_breakdown_invalid_and_fallback(self):
        """Verify graceful fallback for unknown, empty, whitespace, and non-string market names."""
        adversarial_inputs = [
            "InvalidCountryName123",
            "Narnia",
            "",
            "   ",
            None,
            12345,
            "All Markets",
            "Overall",
            "Total",
            "National"
        ]
        for m in adversarial_inputs:
            with self.subTest(market=m):
                df = get_market_expenditure_breakdown(market=m, year=2024)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(len(df), 12)
                self.assertEqual(list(df.columns), ["spending_category", "percentage_share", "value_rm_million"])
                self.assertFalse(df.isnull().values.any(), f"NaNs found in fallback breakdown for {m}")


class TestLegacyCompatibility(unittest.TestCase):
    """Verify backward compatibility for legacy positional callers."""

    def test_legacy_positional_caller_signature(self):
        """Verify legacy positional call (visitors, days, alos, poverty_rate_cand, poverty_rate_hot)."""
        # Legacy caller passed poverty rates in 4th and 5th positional slots
        res = calculate_redistribution_economic_impact(1000, 30, 2.45, 6.5, 1.0)
        self.assertAlmostEqual(res["spend_per_night_used"], DTS_2024_SPEND_PER_NIGHT, places=2)
        self.assertAlmostEqual(res["direct_spend_total_rm"], 1000 * 30 * 2.45 * DTS_2024_SPEND_PER_NIGHT, delta=10.0)
        # B40 retention ratio should reflect cand=6.5, hot=1.0 -> ratio ~0.4104
        b40_ratio = res["estimated_b40_income_injected_rm"] / res["direct_spend_total_rm"]
        self.assertAlmostEqual(b40_ratio, 0.4104, delta=0.005)


class TestMonteCarloAndMarketMatrixExhaustive(unittest.TestCase):
    """Exhaustive validation across all 38 origin markets and Monte Carlo stress testing."""

    def test_all_38_origin_markets_breakdown_fidelity(self):
        """Verify get_market_expenditure_breakdown across every single one of the 38 source markets."""
        top_df = get_top_inbound_markets(n=50, year=2024)
        markets = top_df["market"].tolist()
        self.assertEqual(len(markets), 37, "Expected 37 inbound markets excluding Overall")

        all_markets = ["Overall"] + markets
        for m in all_markets:
            with self.subTest(market=m):
                df = get_market_expenditure_breakdown(market=m, year=2024)
                self.assertEqual(len(df), 12, f"Market {m} must have 12 categories")
                self.assertAlmostEqual(df["percentage_share"].sum(), 100.0, delta=0.5)
                self.assertTrue((df["value_rm_million"] >= 0).all())

                # Test simulation with this origin market
                sim = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=500,
                    days_period=14,
                    tourism_mode="inbound",
                    origin_market=m
                )
                self.assertEqual(len(sim["sectoral_breakdown_12"]), 12)
                self.assertGreater(sim["direct_spend_total_rm"], 0)

    def test_monte_carlo_fuzzing_one_thousand_scenarios(self):
        """Run 1,000 randomized parameter configurations to ensure 0 crashes and total invariant compliance."""
        rng = np.random.default_rng(42)
        modes = ["domestic", "inbound", "blended"]
        known_markets = ["Singapore", "China", "Indonesia", "Australia", "UnknownOrigin", None]

        multipliers = {
            "domestic": TOURISM_OUTPUT_MULTIPLIER,
            "inbound": INBOUND_OUTPUT_MULTIPLIER,
            "blended": BLENDED_OUTPUT_MULTIPLIER
        }

        for i in range(1000):
            visitors = float(rng.uniform(-50, 25000))
            days = int(rng.integers(1, 180))
            cand_pov = float(rng.uniform(-5.0, 95.0))
            hot_pov = float(rng.uniform(-5.0, 50.0))
            mode = rng.choice(modes)
            market = rng.choice(known_markets)

            res = calculate_redistribution_economic_impact(
                redirected_visitors_daily=visitors,
                days_period=days,
                poverty_rate_candidate=cand_pov,
                poverty_rate_hotspot=hot_pov,
                tourism_mode=mode,
                origin_market=market
            )

            # Invariant 1: Non-negativity
            self.assertGreaterEqual(res["direct_spend_total_rm"], 0.0)
            self.assertGreaterEqual(res["total_economic_output_rm"], 0.0)
            self.assertGreaterEqual(res["estimated_b40_income_injected_rm"], 0.0)

            # Invariant 2: Multiplier exactness
            expected_mult = multipliers[mode]
            self.assertEqual(res["output_multiplier"], expected_mult)
            if res["direct_spend_total_rm"] > 0:
                calc_mult = res["total_economic_output_rm"] / res["direct_spend_total_rm"]
                self.assertAlmostEqual(calc_mult, expected_mult, places=5)

                # Invariant 3: B40 retention ratio bounds [0.35, 0.65]
                b40_ratio = res["estimated_b40_income_injected_rm"] / res["direct_spend_total_rm"]
                self.assertGreaterEqual(b40_ratio, 0.35 - 1e-6)
                self.assertLessEqual(b40_ratio, 0.65 + 1e-6)

                # Invariant 4: Sector sums ~ direct spend
                sec_5_sum = sum(res["sectoral_breakdown"].values())
                sec_12_sum = sum(res["sectoral_breakdown_12"].values())
                self.assertAlmostEqual(sec_5_sum, res["direct_spend_total_rm"], delta=res["direct_spend_total_rm"] * 0.005)
                self.assertAlmostEqual(sec_12_sum, res["direct_spend_total_rm"], delta=res["direct_spend_total_rm"] * 0.005)


if __name__ == "__main__":
    unittest.main()
