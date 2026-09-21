"""
DESTINASI — Milestone 6 Empirical Challenge Suite
Author: challenger_m6_2 (Empirical Challenger: critic, specialist)
Target: DOSM Datathon 2026

Empirically tests, verifies, and stress-tests:
1. data/processed/expenditure_market_matrix.csv schema completeness (38 markets, 12 categories).
2. Mathematical consistency: percentage sums (100% +/- 0.5%), RM million consistency, 2024 RM 106,780M headline.
3. Top source market rankings and country-specific spending signatures against official benchmarks.
4. Macroeconomic model behavior across Domestic, Inbound, and Blended tourism economies.
5. Adversarial stress-testing: zero visitors, negative inputs, invalid modes, unknown markets, case-insensitivity.
6. Historical data verification (2018–2024 totals and 2023 audience structure).
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.economic_impact_model import (
    calculate_redistribution_economic_impact,
    get_top_inbound_markets,
    get_market_expenditure_breakdown,
    INBOUND_CATEGORIES,
    INBOUND_12_CATEGORIES_OVERALL,
    DOMESTIC_12_CATEGORIES,
    BLENDED_12_CATEGORIES,
    INBOUND_MACRO_VOLUME_BILLION,
    TOURISM_OUTPUT_MULTIPLIER,
    INBOUND_OUTPUT_MULTIPLIER,
    BLENDED_OUTPUT_MULTIPLIER,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_MATRIX_PATH = ROOT / "data" / "processed" / "expenditure_market_matrix.csv"
RAW_EXP_PATH = ROOT / "data" / "raw" / "comprehensive_expenditure.csv"


class TestMilestone6EmpiricalChallenge(unittest.TestCase):
    """Empirical challenge suite for Milestone 6: Comprehensive International Expenditure Integration."""

    @classmethod
    def setUpClass(cls):
        cls.matrix_df = pd.read_csv(PROCESSED_MATRIX_PATH)
        cls.raw_df = pd.read_csv(RAW_EXP_PATH)
        cls.df_2024 = cls.matrix_df[cls.matrix_df["year"] == 2024].copy()

    # -------------------------------------------------------------------------
    # 1. SCHEMA & MARKET COMPLETENESS
    # -------------------------------------------------------------------------

    def test_schema_completeness_and_columns(self):
        """Verify expenditure_market_matrix.csv has correct columns, data types, and zero NaNs."""
        expected_cols = ["year", "market", "spending_category", "value_rm_million", "percentage_share"]
        for col in expected_cols:
            self.assertIn(col, self.matrix_df.columns, f"Missing expected column: {col}")

        # Zero NaNs allowed in critical fields
        self.assertEqual(self.matrix_df["year"].isna().sum(), 0)
        self.assertEqual(self.matrix_df["market"].isna().sum(), 0)
        self.assertEqual(self.matrix_df["spending_category"].isna().sum(), 0)
        self.assertEqual(self.matrix_df["value_rm_million"].isna().sum(), 0)
        self.assertEqual(self.matrix_df["percentage_share"].isna().sum(), 0)

    def test_all_38_origin_markets_present(self):
        """Verify that exactly 38 distinct origin markets exist in the 2024 matrix."""
        unique_markets = set(self.df_2024["market"].unique())
        self.assertEqual(len(unique_markets), 38, f"Expected exactly 38 origin markets, found {len(unique_markets)}")

        expected_key_markets = [
            "Singapore", "China", "Indonesia", "India", "Australia",
            "Brunei Darussalam", "Thailand", "South Korea", "Japan",
            "United Kingdom", "U.S.A", "Others", "Overall"
        ]
        for m in expected_key_markets:
            self.assertIn(m, unique_markets, f"Key origin market '{m}' missing from 2024 dataset")

    def test_all_12_spending_categories_per_market(self):
        """Verify that all 38 markets have exactly the 12 authoritative spending categories plus Total."""
        expected_categories = set(INBOUND_CATEGORIES)

        for mkt, grp in self.df_2024.groupby("market"):
            # Check Total row exists
            total_rows = grp[grp["spending_category"] == "All spending categories"]
            self.assertEqual(len(total_rows), 1, f"Market '{mkt}' missing 'All spending categories' total row")

            # Check 12 categories
            cat_rows = grp[grp["spending_category"] != "All spending categories"]
            found_categories = set(cat_rows["spending_category"].unique())
            self.assertEqual(
                found_categories,
                expected_categories,
                f"Market '{mkt}' category mismatch. Missing: {expected_categories - found_categories}, Extra: {found_categories - expected_categories}"
            )
            self.assertEqual(len(cat_rows), 12, f"Market '{mkt}' does not have exactly 12 category rows")

    # -------------------------------------------------------------------------
    # 2. MATHEMATICAL CONSISTENCY & BENCHMARK VALIDATION
    # -------------------------------------------------------------------------

    def test_category_percentage_shares_sum_to_100(self):
        """Verify that category percentage shares sum to 100% (+/- 0.5%) for all 38 markets in 2024."""
        cat_rows = self.df_2024[self.df_2024["spending_category"] != "All spending categories"]
        pct_sums = cat_rows.groupby("market")["percentage_share"].sum()

        for mkt, s in pct_sums.items():
            self.assertGreaterEqual(s, 99.5, f"Market '{mkt}' percentage share sum {s:.2f}% is below 99.5%")
            self.assertLessEqual(s, 100.5, f"Market '{mkt}' percentage share sum {s:.2f}% exceeds 100.5%")

        # National aggregate (Overall) percentage sum should be extremely close to 100%
        overall_pct_sum = pct_sums["Overall"]
        self.assertAlmostEqual(overall_pct_sum, 100.0, delta=0.05)

    def test_category_rm_million_sums_match_market_headlines(self):
        """Verify that the sum of 12 category RM Million values matches each market headline within 0.5%."""
        cat_rows = self.df_2024[self.df_2024["spending_category"] != "All spending categories"]
        tot_rows = self.df_2024[self.df_2024["spending_category"] == "All spending categories"].set_index("market")["value_rm_million"]
        rm_sums = cat_rows.groupby("market")["value_rm_million"].sum()

        for mkt in tot_rows.index:
            headline = tot_rows[mkt]
            calculated_sum = rm_sums[mkt]
            pct_diff = abs(calculated_sum - headline) / headline * 100.0
            self.assertLess(
                pct_diff, 0.5,
                f"Market '{mkt}' category RM sum ({calculated_sum:.2f}) differs from headline ({headline:.2f}) by {pct_diff:.3f}%"
            )

    def test_2024_overall_inbound_expenditure_headline(self):
        """Verify that 2024 total inbound expenditure strictly matches official RM 106,780.00M."""
        tot_row = self.df_2024[
            (self.df_2024["market"] == "Overall") &
            (self.df_2024["spending_category"] == "All spending categories")
        ]
        self.assertEqual(len(tot_row), 1)
        reported_value = tot_row["value_rm_million"].iloc[0]
        self.assertEqual(reported_value, 106780.0, f"Expected RM 106,780.00M, got {reported_value}")

    def test_sum_of_37_markets_matches_overall_within_tolerance(self):
        """Verify that sum of the 37 individual source markets approximates Overall within 0.01%."""
        tot_rows = self.df_2024[self.df_2024["spending_category"] == "All spending categories"]
        overall_val = tot_rows[tot_rows["market"] == "Overall"]["value_rm_million"].iloc[0]
        sub_markets_sum = tot_rows[tot_rows["market"] != "Overall"]["value_rm_million"].sum()

        diff = abs(overall_val - sub_markets_sum)
        # Official rounding in individual markets creates a minor 0.25 RM Million delta on RM 106.8B
        self.assertLess(diff, 1.0, f"Sum of 37 markets ({sub_markets_sum}) differs from Overall ({overall_val}) by {diff}M")

    # -------------------------------------------------------------------------
    # 3. TOP SOURCE MARKETS & COUNTRY-SPECIFIC SIGNATURES
    # -------------------------------------------------------------------------

    def test_top_source_markets_rankings_and_values(self):
        """Verify top inbound source markets match official MOTAC benchmarks."""
        top10 = get_top_inbound_markets(n=10, year=2024)
        self.assertEqual(len(top10), 10)

        # Expected top 4: Singapore, China, Indonesia, India
        self.assertEqual(top10.iloc[0]["market"], "Singapore")
        self.assertEqual(top10.iloc[0]["value_rm_million"], 27940.0)

        self.assertEqual(top10.iloc[1]["market"], "China")
        self.assertEqual(top10.iloc[1]["value_rm_million"], 20870.0)

        self.assertEqual(top10.iloc[2]["market"], "Indonesia")
        self.assertEqual(top10.iloc[2]["value_rm_million"], 15320.0)

        self.assertEqual(top10.iloc[3]["market"], "India")
        self.assertEqual(top10.iloc[3]["value_rm_million"], 6110.0)

        # Australia must be present in top 10
        mkt_names = top10["market"].tolist()
        self.assertIn("Australia", mkt_names)
        aus_row = top10[top10["market"] == "Australia"].iloc[0]
        self.assertEqual(aus_row["value_rm_million"], 2488.20)

    def test_empirical_country_specific_spending_signatures(self):
        """Verify unique market spending signatures reflect genuine empirical behavior."""
        # Singapore: Cross-border retail -> Shopping dominates (>50%)
        sg = get_market_expenditure_breakdown("Singapore", year=2024)
        sg_shop = sg[sg["spending_category"] == "Shopping"]["percentage_share"].iloc[0]
        self.assertGreater(sg_shop, 50.0, f"Singapore shopping share {sg_shop}% should exceed 50%")

        # Indonesia: Medical tourism -> Medical share exceeds 20%
        indo = get_market_expenditure_breakdown("Indonesia", year=2024)
        indo_med = indo[indo["spending_category"] == "Medical"]["percentage_share"].iloc[0]
        self.assertGreater(indo_med, 20.0, f"Indonesia medical share {indo_med}% should exceed 20%")

        # China: Long-haul package/cultural -> International Airfare & Organised Tour prominent
        cn = get_market_expenditure_breakdown("China", year=2024)
        cn_air = cn[cn["spending_category"] == "International Airfares"]["percentage_share"].iloc[0]
        cn_tour = cn[cn["spending_category"] == "Organised Tour"]["percentage_share"].iloc[0]
        self.assertGreater(cn_air, 15.0)
        self.assertGreater(cn_tour, 10.0)

        # Australia: Long-haul vacation -> Accommodation is highest spending category
        aus = get_market_expenditure_breakdown("Australia", year=2024)
        aus_top_cat = aus.iloc[0]["spending_category"]
        self.assertEqual(aus_top_cat, "Accommodation")

    # -------------------------------------------------------------------------
    # 4. MACROECONOMIC SIMULATION ENGINE FIDELITY
    # -------------------------------------------------------------------------

    def test_macroeconomic_modes_consistency(self):
        """Verify economic impact simulation behaves consistently across all 3 macro modes."""
        visitors = 2000.0
        days = 30

        dom = calculate_redistribution_economic_impact(visitors, days, tourism_mode="domestic")
        inb = calculate_redistribution_economic_impact(visitors, days, tourism_mode="inbound")
        blended = calculate_redistribution_economic_impact(visitors, days, tourism_mode="blended")

        # Inbound has higher ALOS (4.80) and higher spend (RM 728.50) than Domestic (2.45, RM 386.73)
        self.assertGreater(inb["direct_spend_million_rm"], dom["direct_spend_million_rm"] * 2.0)
        self.assertGreater(inb["total_economic_output_million_rm"], dom["total_economic_output_million_rm"] * 2.0)

        # Blended should be strictly intermediate between Domestic and Inbound
        self.assertGreater(blended["direct_spend_million_rm"], dom["direct_spend_million_rm"])
        self.assertLess(blended["direct_spend_million_rm"], inb["direct_spend_million_rm"])
        self.assertGreater(blended["total_economic_output_million_rm"], dom["total_economic_output_million_rm"])
        self.assertLess(blended["total_economic_output_million_rm"], inb["total_economic_output_million_rm"])

        # Check multipliers
        self.assertEqual(dom["output_multiplier"], TOURISM_OUTPUT_MULTIPLIER)
        self.assertEqual(inb["output_multiplier"], INBOUND_OUTPUT_MULTIPLIER)
        self.assertEqual(blended["output_multiplier"], BLENDED_OUTPUT_MULTIPLIER)

    def test_origin_market_injection_into_economic_sim(self):
        """Verify that passing an origin market dynamically adjusts 12-category injection."""
        sim_sg = calculate_redistribution_economic_impact(
            redirected_visitors_daily=1000,
            days_period=30,
            tourism_mode="inbound",
            origin_market="Singapore"
        )
        sim_indo = calculate_redistribution_economic_impact(
            redirected_visitors_daily=1000,
            days_period=30,
            tourism_mode="inbound",
            origin_market="Indonesia"
        )

        # Singapore injection should have much higher Shopping spend than Indonesia
        self.assertGreater(
            sim_sg["sectoral_breakdown_12"]["Shopping"],
            sim_indo["sectoral_breakdown_12"]["Shopping"]
        )

        # Indonesia injection should have much higher Medical spend than Singapore
        self.assertGreater(
            sim_indo["sectoral_breakdown_12"]["Medical"],
            sim_sg["sectoral_breakdown_12"]["Medical"] * 50.0
        )

    # -------------------------------------------------------------------------
    # 5. ADVERSARIAL STRESS TESTING & ROBUSTNESS
    # -------------------------------------------------------------------------

    def test_adversarial_economic_inputs(self):
        """Stress-test economic calculation with zero, negative, and extreme inputs."""
        # Zero visitors
        res_zero = calculate_redistribution_economic_impact(0, days_period=30, tourism_mode="inbound")
        self.assertEqual(res_zero["direct_spend_total_rm"], 0.0)
        self.assertEqual(res_zero["total_economic_output_rm"], 0.0)
        self.assertEqual(res_zero["estimated_b40_income_injected_rm"], 0.0)

        # Negative visitors (must be clamped to 0)
        res_neg = calculate_redistribution_economic_impact(-500, days_period=30, tourism_mode="inbound")
        self.assertEqual(res_neg["direct_spend_total_rm"], 0.0)
        self.assertEqual(res_neg["total_economic_output_rm"], 0.0)

        # Unknown tourism mode -> fallback to domestic
        res_unknown_mode = calculate_redistribution_economic_impact(1000, days_period=30, tourism_mode="hyperloop")
        self.assertEqual(res_unknown_mode["tourism_mode"], "domestic")

        # Case-insensitivity in mode
        res_case = calculate_redistribution_economic_impact(1000, days_period=30, tourism_mode="InBoUnD")
        self.assertEqual(res_case["tourism_mode"], "inbound")

    def test_adversarial_market_breakdown_queries(self):
        """Stress-test get_market_expenditure_breakdown with unknown and malformed inputs."""
        # Non-existent country -> should gracefully fallback to Overall without crashing
        res_unknown = get_market_expenditure_breakdown("Wakanda")
        self.assertEqual(len(res_unknown), 12)
        self.assertEqual(res_unknown.iloc[0]["spending_category"], "Shopping")

        # None / empty / numeric strings
        self.assertEqual(len(get_market_expenditure_breakdown(None)), 12)
        self.assertEqual(len(get_market_expenditure_breakdown("")), 12)
        self.assertEqual(len(get_market_expenditure_breakdown("All Markets")), 12)

        # Case-insensitivity in market names
        res_lower = get_market_expenditure_breakdown("singapore")
        res_upper = get_market_expenditure_breakdown("SINGAPORE")
        self.assertEqual(res_lower["percentage_share"].tolist(), res_upper["percentage_share"].tolist())

    def test_adversarial_top_markets_queries(self):
        """Stress-test get_top_inbound_markets with out-of-bounds limits."""
        # Zero limit
        t_zero = get_top_inbound_markets(n=0)
        self.assertEqual(len(t_zero), 0)

        # Oversized limit (larger than 37 source markets)
        t_large = get_top_inbound_markets(n=100)
        self.assertEqual(len(t_large), 37)  # 38 minus 'Overall'

    # -------------------------------------------------------------------------
    # 6. HISTORICAL DATA CONSISTENCY (2018–2024)
    # -------------------------------------------------------------------------

    def test_historical_total_expenditure_series(self):
        """Verify historical totals exist for all years 2018 to 2024."""
        years_present = sorted(self.matrix_df["year"].unique())
        expected_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.assertEqual(years_present, expected_years)

        # Verify national totals reflect pandemic collapse and post-pandemic recovery
        overall_totals = self.matrix_df[
            (self.matrix_df["market"] == "Overall") &
            (self.matrix_df["spending_category"] == "All spending categories")
        ].set_index("year")["value_rm_million"]

        self.assertAlmostEqual(overall_totals[2018], 84135.17, delta=100.0)  # 2018 baseline
        self.assertAlmostEqual(overall_totals[2019], 89010.0, delta=100.0)   # Pre-pandemic peak (RM 89.0B)
        self.assertAlmostEqual(overall_totals[2020], 12688.16, delta=100.0)  # Pandemic drop
        self.assertAlmostEqual(overall_totals[2021], 238.73, delta=10.0)     # Border lockdown trough
        self.assertAlmostEqual(overall_totals[2022], 28228.28, delta=100.0)  # Initial border reopening
        self.assertAlmostEqual(overall_totals[2023], 74290.0, delta=100.0)   # Accelerating recovery
        self.assertAlmostEqual(overall_totals[2024], 106780.0, delta=100.0)  # Record inbound milestone RM 106.8B


if __name__ == "__main__":
    unittest.main()
