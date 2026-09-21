"""
Reviewer Adversarial & Conformance Test Suite for Milestone 6 (R6)
Tests View 3 UI logic, Altair chart generation, edge cases, and integrity.
"""

import unittest
from pathlib import Path
import pandas as pd
import altair as alt

from src.carrying_capacity_engine import diagnose_multisystem_bottleneck
from src.recommender_matcher import find_best_alternatives
from src.economic_impact_model import (
    calculate_redistribution_economic_impact,
    get_top_inbound_markets,
    get_market_expenditure_breakdown,
    INBOUND_CATEGORIES,
    DOMESTIC_SPEND_SHARES,
    INBOUND_12_CATEGORIES_OVERALL,
    BLENDED_12_CATEGORIES,
    TOURISM_OUTPUT_MULTIPLIER,
    INBOUND_OUTPUT_MULTIPLIER,
    BLENDED_OUTPUT_MULTIPLIER,
    DTS_2024_SPEND_PER_NIGHT,
    INBOUND_SPEND_PER_NIGHT,
    BLENDED_SPEND_PER_NIGHT
)

BASE_DIR = Path(__file__).resolve().parents[1]
DESTINATIONS_PATH = BASE_DIR / "data" / "processed" / "destinations_master.csv"
EXPENDITURES_PATH = BASE_DIR / "data" / "processed" / "expenditure_market_matrix.csv"
RAW_EXP_PATH = BASE_DIR / "data" / "raw" / "comprehensive_expenditure.csv"


class TestReviewerView3Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dest_df = pd.read_csv(DESTINATIONS_PATH)
        cls.exp_df = pd.read_csv(EXPENDITURES_PATH)

    def test_view3_simulation_all_110_districts(self):
        """Verify that View 3 calculation pipeline executes flawlessly for all 110 districts across all 3 modes."""
        modes = ["domestic", "inbound", "blended"]
        for _, hotspot_row in self.dest_df.iterrows():
            top_alts = find_best_alternatives(hotspot_row, self.dest_df, top_n=3)
            if top_alts.empty:
                top_alts = find_best_alternatives(hotspot_row, self.dest_df, top_n=3, alpha=1.0)
            self.assertFalse(top_alts.empty, f"No alternatives found for {hotspot_row['destination_name']}")

            target_alt_name = top_alts["candidate_name"].iloc[0]
            target_cand_match = self.dest_df[self.dest_df["destination_name"] == target_alt_name]
            self.assertFalse(target_cand_match.empty, f"Target {target_alt_name} not in destinations_df")
            target_cand = target_cand_match.iloc[0]

            h_diag = diagnose_multisystem_bottleneck(
                demand=hotspot_row["daily_demand_peak"],
                cc_accommodation=hotspot_row["cc_accommodation"],
                cc_transport=hotspot_row["cc_transport"],
                cc_attraction=hotspot_row["cc_attraction"],
                cc_water_waste=hotspot_row["cc_water_waste"],
                cc_ecology=hotspot_row["cc_ecology"],
                cc_social=hotspot_row["cc_social"]
            )
            daily_diverted = max(500.0, float(h_diag["excess_demand"])) * 0.35

            for mode in modes:
                econ_sim = calculate_redistribution_economic_impact(
                    redirected_visitors_daily=daily_diverted,
                    days_period=30,
                    poverty_rate_candidate=target_cand["poverty_rate"],
                    poverty_rate_hotspot=hotspot_row["poverty_rate"],
                    tourism_mode=mode
                )
                self.assertGreater(econ_sim["direct_spend_million_rm"], 0)
                self.assertGreater(econ_sim["total_economic_output_million_rm"], 0)
                self.assertGreater(econ_sim["estimated_b40_income_million_rm"], 0)
                self.assertEqual(len(econ_sim["sectoral_breakdown_12"]), 12)

    def test_view3_altair_top_10_markets_chart(self):
        """Verify Altair Top 10 Inbound Source Markets chart spec compilation."""
        top_10_df = get_top_inbound_markets(n=10, year=2024)
        self.assertEqual(len(top_10_df), 10)
        mkt_bar_chart = (
            alt.Chart(top_10_df)
            .mark_bar(cornerRadiusEnd=4, color="#0284c7")
            .encode(
                x=alt.X("value_rm_million:Q", title="Total Expenditure (RM Million)"),
                y=alt.Y("market:N", title=None, sort="-x"),
                tooltip=[
                    alt.Tooltip("market:N", title="Source Market"),
                    alt.Tooltip("value_rm_million:Q", title="Total Spend (RM M)", format=",.1f"),
                    alt.Tooltip("share_percent:Q", title="Share of Inbound (%)", format=".2f")
                ]
            )
            .properties(height=320)
        )
        spec = mkt_bar_chart.to_dict()
        self.assertIn("mark", spec)
        self.assertIn("encoding", spec)
        self.assertEqual(spec["mark"]["type"], "bar")

    def test_view3_altair_12_category_breakdown_charts_all_38_markets(self):
        """Verify Altair 12-category expenditure breakdown charts render for benchmark and all 38 origin markets."""
        top_38 = get_top_inbound_markets(n=38, year=2024)["market"].tolist()
        choices = ["All Markets", None] + top_38

        for mkt in choices:
            cat_12_df = get_market_expenditure_breakdown(market=mkt or "All Markets", year=2024)
            self.assertEqual(len(cat_12_df), 12, f"Failed for market: {mkt}")
            cat_bar_chart = (
                alt.Chart(cat_12_df)
                .mark_bar(cornerRadiusEnd=4, color="#0f766e")
                .encode(
                    x=alt.X("percentage_share:Q", title="Expenditure Share (%)"),
                    y=alt.Y("spending_category:N", title=None, sort="-x"),
                    tooltip=[
                        alt.Tooltip("spending_category:N", title="Category"),
                        alt.Tooltip("percentage_share:Q", title="Share (%)", format=".2f"),
                        alt.Tooltip("value_rm_million:Q", title="Expenditure (RM M)", format=",.1f")
                    ]
                )
                .properties(height=320)
            )
            spec = cat_bar_chart.to_dict()
            self.assertEqual(spec["mark"]["type"], "bar")

    def test_view3_injected_sectoral_spend_altair_chart(self):
        """Verify Injected Sectoral Spend chart generation across modes."""
        for mode in ["domestic", "inbound", "blended"]:
            sim = calculate_redistribution_economic_impact(
                redirected_visitors_daily=1200,
                days_period=30,
                poverty_rate_candidate=12.0,
                poverty_rate_hotspot=2.0,
                tourism_mode=mode
            )
            if mode in ["inbound", "blended"] and "sectoral_breakdown_12" in sim:
                local_inj_items = [
                    {"Category": cat, "RM_Million": val / 1e6}
                    for cat, val in sim["sectoral_breakdown_12"].items()
                    if val > 0
                ]
                local_inj_df = pd.DataFrame(local_inj_items).sort_values(by="RM_Million", ascending=False)
                chart = (
                    alt.Chart(local_inj_df)
                    .mark_bar(cornerRadiusEnd=4, color="#10b981")
                    .encode(
                        x=alt.X("RM_Million:Q", title="Injected Spend (RM Million)"),
                        y=alt.Y("Category:N", title=None, sort="-x"),
                        tooltip=["Category:N", alt.Tooltip("RM_Million:Q", title="Injected (RM M)", format=".3f")]
                    )
                    .properties(height=260)
                )
                spec = chart.to_dict()
                self.assertEqual(spec["mark"]["type"], "bar")
            else:
                breakdown_data = pd.DataFrame([
                    {"Category": "Shopping & Handicrafts", "RM_Million": sim["sectoral_breakdown"]["shopping_retail_rm"] / 1e6},
                    {"Category": "Food & Beverage", "RM_Million": sim["sectoral_breakdown"]["food_beverage_rm"] / 1e6},
                    {"Category": "Accommodation & Lodging", "RM_Million": sim["sectoral_breakdown"]["accommodation_rm"] / 1e6},
                    {"Category": "Local Transport & Fuel", "RM_Million": sim["sectoral_breakdown"]["transport_transit_rm"] / 1e6},
                    {"Category": "Tours, Culture & Recreation", "RM_Million": sim["sectoral_breakdown"]["entertainment_other_rm"] / 1e6},
                ])
                chart = (
                    alt.Chart(breakdown_data)
                    .mark_bar(cornerRadiusEnd=4, color="#0f766e")
                    .encode(
                        x=alt.X("RM_Million:Q", title="Injected Spend (RM Million)"),
                        y=alt.Y("Category:N", title=None, sort="-x"),
                        tooltip=["Category:N", alt.Tooltip("RM_Million:Q", format=".2f")]
                    )
                    .properties(height=260)
                )
                spec = chart.to_dict()
                self.assertEqual(spec["mark"]["type"], "bar")

    def test_view3_longitudinal_inbound_expenditure_chart(self):
        """Verify longitudinal recovery chart (2018-2024)."""
        exp_raw_df = pd.read_csv(EXPENDITURES_PATH)
        ts_sub = exp_raw_df[
            (exp_raw_df["market"] == "Overall") &
            (exp_raw_df["spending_category"] == "All spending categories")
        ].sort_values(by="year")
        self.assertFalse(ts_sub.empty)
        self.assertGreaterEqual(len(ts_sub), 6) # 2018-2024
        ts_chart = (
            alt.Chart(ts_sub)
            .mark_line(point=True, color="#0284c7", strokeWidth=2.5)
            .encode(
                x=alt.X("year:O", title="Survey Year"),
                y=alt.Y("value_rm_million:Q", title="Total Inbound Spend (RM Million)"),
                tooltip=[
                    alt.Tooltip("year:O", title="Year"),
                    alt.Tooltip("value_rm_million:Q", title="Total Spend (RM M)", format=",.1f")
                ]
            )
            .properties(height=260)
        )
        spec = ts_chart.to_dict()
        self.assertEqual(spec["mark"]["type"], "line")

    def test_economic_model_mathematical_consistency(self):
        """Adversarial mathematical stress-test: verify no discrepancies between components."""
        for mode in ["domestic", "inbound", "blended"]:
            sim = calculate_redistribution_economic_impact(
                redirected_visitors_daily=2000.0,
                days_period=45,
                poverty_rate_candidate=15.0,
                poverty_rate_hotspot=2.5,
                tourism_mode=mode
            )
            trips = sim["total_tourist_trips"]
            alos = sim["alos_used"]
            spend_night = sim["spend_per_night_used"]
            mult = sim["output_multiplier"]

            expected_direct = trips * alos * spend_night
            self.assertAlmostEqual(sim["direct_spend_total_rm"], expected_direct, places=1)
            self.assertAlmostEqual(sim["direct_spend_million_rm"], expected_direct / 1e6, places=3)

            expected_output = expected_direct * mult
            self.assertAlmostEqual(sim["total_economic_output_rm"], expected_output, places=1)
            self.assertAlmostEqual(sim["total_economic_output_million_rm"], expected_output / 1e6, places=3)

            # Sum of 12 categories should equal direct spend total
            cat_sum = sum(sim["sectoral_breakdown_12"].values())
            self.assertAlmostEqual(cat_sum, expected_direct, delta=expected_direct * 0.01)

            # Sum of 5 sectors should equal direct spend total
            sec_sum = sum(sim["sectoral_breakdown"].values())
            self.assertAlmostEqual(sec_sum, expected_direct, delta=expected_direct * 0.01)


if __name__ == "__main__":
    unittest.main()
