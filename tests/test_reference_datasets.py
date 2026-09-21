"""
DESTINASI — Reference Datasets Test Suite (PPP KSAS + Travel Trends Top-50)
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Covers the two ETL-backed reference datasets:
  A. data/processed/ksas_reference.csv — transcribed from PPP KSAS (Nov 2025):
     Jadual 7 field schema, all 15 KSAS jenis, Jadual 4 Tahap legality,
     Rajah 6 colour codes, circle-equivalent radius consistency, parser errors.
  B. data/processed/google_travel_trends.csv — top-50 demand proxy:
     row count, deterministic grounding in destinations_master + national
     demand calendar, schema/loader validation, error paths.
"""

import math
import unittest
from pathlib import Path

import pandas as pd

from src.etl.extract_ksas import (
    KSAS_TAHAP_COLORS,
    KSAS_TAHAP_RULES,
    KSAS_TYPE_NAMES,
    build_ksas_reference,
    parse_ksas_reference,
    radius_from_hectares,
)
from src.parse_external_data import (
    TRENDS_TOP_N,
    fetch_destination_search_trends,
    load_travel_trends,
)

ROOT = Path(__file__).resolve().parents[1]
KSAS_CSV = ROOT / "data" / "processed" / "ksas_reference.csv"
TRENDS_CSV = ROOT / "data" / "processed" / "google_travel_trends.csv"
MASTER_CSV = ROOT / "data" / "processed" / "destinations_master.csv"


class TestKSASReferenceDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.report = parse_ksas_reference(KSAS_CSV)

    def test_jadual7_schema_present(self):
        """Jadual 7 (m.s 31) fields must all exist (plus documented mapping extras)."""
        jadual7 = ["jenis_ksas", "tahap", "nama", "luas_h", "akt_1", "akt_syarat",
                   "akt_0", "negeri", "daerah", "bp", "bpk", "tahun_data"]
        for col in jadual7:
            self.assertIn(col, self.df.columns, f"missing Jadual 7 field {col}")
        for col in ["lat", "lon", "radius_m", "key", "label_peta", "sumber",
                    "representation", "keyakinan", "ksas_code"]:
            self.assertIn(col, self.df.columns, f"missing mapping extra {col}")

    def test_all_fifteen_types_covered(self):
        """PPP KSAS defines 15 jenis — the dataset must contain every one."""
        codes = sorted({int(c.split()[1]) for c in self.df["ksas_code"].tolist()})
        self.assertEqual(codes, list(range(1, 16)))
        self.assertEqual(self.report["missing_types"], [])
        self.assertEqual(len(self.df), 22)

    def test_tahap_legal_per_jadual4(self):
        """Every row's Tahap must be legal for its jenis (Jadual 4, m.s 17-18)."""
        for _, r in self.df.iterrows():
            no = int(r["ksas_code"].split()[1])
            tahap = int(r["tahap"].split()[1])
            self.assertIn(tahap, KSAS_TAHAP_RULES[no],
                          f"{r['nama']}: {r['tahap']} illegal for {r['ksas_code']}")
            self.assertIn(no, KSAS_TYPE_NAMES)

    def test_tahap_colours_match_rajah6(self):
        """Rajah 6 (m.s 32) colour codes must be exactly R64/G90/B0, R123/G181/B5, R206/G255/B111."""
        self.assertEqual(KSAS_TAHAP_COLORS[1], (64, 90, 0))
        self.assertEqual(KSAS_TAHAP_COLORS[2], (123, 181, 5))
        self.assertEqual(KSAS_TAHAP_COLORS[3], (206, 255, 111))

    def test_radius_consistency_and_coarseness(self):
        """Area-derived radii must match luas_h within rounding; all radii coarse."""
        for _, r in self.df.iterrows():
            self.assertTrue(1000 <= r["radius_m"] <= 40000)
            self.assertEqual(r["radius_m"] % 500, 0)
            if str(r["area_derived"]).lower() == "true":
                expect = radius_from_hectares(float(r["luas_h"]))
                self.assertLessEqual(abs(r["radius_m"] - expect), 500, r["nama"])

    def test_coordinates_inside_malaysia(self):
        self.assertTrue(((0.8 <= self.df["lat"]) & (self.df["lat"] <= 7.6)).all())
        self.assertTrue(((99.0 <= self.df["lon"]) & (self.df["lon"] <= 119.6)).all())

    def test_keys_unique_and_wakil_flagged(self):
        self.assertEqual(self.df["key"].nunique(), len(self.df))
        wakil = self.df[self.df["keyakinan"] == "wakil"]
        self.assertGreater(len(wakil), 0)
        for _, r in wakil.iterrows():
            text = f"{r['representation']} {r['radius_basis']}".lower()
            self.assertTrue(any(w in text for w in ("proxy", "wakil", "node", "not")),
                            f"{r['nama']} wakil row must disclaim precision")

    def test_build_roundtrip_reproduces_csv(self):
        """build_ksas_reference() must regenerate exactly what ships on disk."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ksas_reference.csv"
            df_built, _ = build_ksas_reference(out)
            pd.testing.assert_frame_equal(
                df_built.reset_index(drop=True),
                self.df.reset_index(drop=True))

    def test_parser_rejects_bad_tahap(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "tahap"] = "Tahap 9"
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            bad.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                parse_ksas_reference(p)

    def test_parser_rejects_bad_jenis(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "ksas_code"] = "KSAS 99"
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            bad.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                parse_ksas_reference(p)

    def test_parser_rejects_illegal_tahap_for_jenis(self):
        """KSAS 5 admits only Tahap 2 (Jadual 4) — a Tahap 1 row must fail."""
        bad = self.df.copy()
        idx = bad[bad["ksas_code"] == "KSAS 5"].index[0]
        bad.loc[idx, "tahap"] = "Tahap 1"
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            bad.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                parse_ksas_reference(p)

    def test_parser_rejects_duplicate_keys(self):
        bad = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            bad.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                parse_ksas_reference(p)

    def test_parser_rejects_out_of_bounds_coords(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "lat"] = 45.0
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            bad.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                parse_ksas_reference(p)


class TestTravelTrendsTop50(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.master = pd.read_csv(MASTER_CSV)
        cls.df = load_travel_trends(TRENDS_CSV)

    def test_top50_row_count_and_uniqueness(self):
        self.assertGreaterEqual(len(self.df), TRENDS_TOP_N)
        self.assertEqual(self.df["destination"].nunique(), len(self.df))

    def test_rows_grounded_in_master_ranking(self):
        """Rows must be the top-50 master districts by daily_demand_peak (stable order)."""
        ranked = self.master.sort_values(
            ["daily_demand_peak", "state_name", "district_name"],
            ascending=[False, True, True]).head(TRENDS_TOP_N)
        self.assertEqual(self.df["destination"].tolist(), ranked["destination_name"].tolist())
        peak = float(ranked["daily_demand_peak"].max())
        for _, r in self.df.iterrows():
            demand = float(ranked.loc[ranked["destination_name"] == r["destination"],
                                      "daily_demand_peak"].iloc[0])
            self.assertAlmostEqual(r["relative_search_index"],
                                   round(100.0 * demand / peak, 1), places=1)
        self.assertEqual(self.df["relative_search_index"].max(), 100.0)

    def test_category_follows_pressure_gate(self):
        """Hotspot iff continuous_pressure >= 1.0 (same gate as the capacity engine)."""
        for _, r in self.df.iterrows():
            p = float(self.master.loc[self.master["destination_name"] == r["destination"],
                                       "continuous_pressure"].iloc[0])
            self.assertEqual(r["category"], "Hotspot" if p >= 1.0 else "Alternative")

    def test_peak_months_from_national_calendar(self):
        """Peak months must be the top-2 mean-multiplier months of the demand calendar."""
        cal = pd.read_csv(ROOT / "data" / "processed" / "calendar_holiday_events.csv")
        cal["month_no"] = cal["month"].astype(str).str.slice(5, 7).astype(int)
        means = cal.groupby("month_no")["demand_multiplier"].mean().sort_values(ascending=False)
        names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        hot_month, alt_month = names[means.index[0]], names[means.index[1]]
        hot_rows = self.df[self.df["category"] == "Hotspot"]
        alt_rows = self.df[self.df["category"] == "Alternative"]
        if len(hot_rows):
            self.assertTrue((hot_rows["peak_month"] == hot_month).all())
        if len(alt_rows):
            self.assertTrue((alt_rows["peak_month"] == alt_month).all())

    def test_queries_come_from_district_poi_tags(self):
        """dominant_query must be built from the district's own DTS poi_tags."""
        for _, r in self.df.iterrows():
            tags = self.master.loc[self.master["destination_name"] == r["destination"],
                                   "poi_tags"].iloc[0]
            if pd.notna(tags) and str(tags).strip():
                first = str(tags).split("|")[0].strip().replace("_", " ")
                self.assertIn(first, r["dominant_query"])

    def test_builder_is_deterministic(self):
        """Two builds must produce byte-identical frames (no randomness, no clocks)."""
        a = fetch_destination_search_trends()
        b = fetch_destination_search_trends()
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_loader_rejects_short_table(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "short.csv"
            self.df.head(10).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_travel_trends(p)

    def test_every_row_marks_proxy_method(self):
        self.assertTrue(self.df["method"].astype(str).str.contains("proxy").all())


if __name__ == "__main__":
    unittest.main()
