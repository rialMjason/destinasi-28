"""
DESTINASI — PPP KSAS Reference Dataset ETL Pipeline
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Transcribes the statutory KSAS classification from:
  Panduan Perancangan dan Pengurusan Kawasan Sensitif Alam Sekitar (PPP KSAS),
  PLANMalaysia / KPKT, Cetakan Pertama November 2025, ISBN 978-629-7679-11-2
  (diluluskan MPFN ke-48, 16 Julai 2025; terpakai Semenanjung + WP).

What this module does:
  1. Holds the canonical 22-row transcription (KSAS_CANONICAL_ROWS) covering all
     15 PPP KSAS jenis (Jadual 1, m.s 11-12) with Tahap per Jadual 4 (m.s 17-18)
     and permitted/conditional/prohibited activities condensed from Jadual 5
     (m.s 19-25).
  2. Builds data/processed/ksas_reference.csv via build_ksas_reference().
  3. Parses + cleans + validates any ksas_reference.csv via parse_ksas_reference()
     (Jadual 7 field schema, m.s 31): normalises Tahap variants, coerces luas_h,
     bounds-checks coordinates, enforces unique keys, verifies circle-equivalent
     radii for area-derived rows, and reports missing KSAS types.

Honesty contract: true KSAS limits are polygons held in RS/RT geodatabases
(PPP KSAS §5). No open polygon ships here, so map disks are circular node
proxies. Rows with keyakinan == "wakil" are representative nodes, never
cadastral claims — enforced by tests/test_reference_datasets.py.
"""

from pathlib import Path
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_ksas")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "data" / "processed" / "ksas_reference.csv"

# ---------------------------------------------------------------------------
# PPP KSAS Jadual 1: the 15 statutory KSAS jenis (m.s 11-12).
# ---------------------------------------------------------------------------
KSAS_TYPE_NAMES: Dict[int, str] = {
    1: "Hutan Simpanan Kekal (HSK)",
    2: "Kawasan Bukit, Tanah Tinggi dan Kawasan Cerun",
    3: "Kawasan Simpanan Mineral",
    4: "Kawasan Risiko Bencana",
    5: "Warisan Kebudayaan Ketara dan Warisan Semula Jadi",
    6: "Kawasan Perlindungan Hidupan Liar",
    7: "Kawasan Empangan, Tadahan Air dan Imbuhan Air Bawah Tanah",
    8: "Dataran Banjir",
    9: "Tanah Bencah dan Tanah Gambut",
    10: "Tasik dan Sungai",
    11: "Kawasan Persisiran Pantai",
    12: "Kawasan Perlindungan Marin",
    13: "Kawasan Keterjaminan Makanan",
    14: "Kawasan Bekas Lombong dan Kuari",
    15: "Bekas dan Tapak Pelupusan Sisa Pepejal",
}

# PPP KSAS Jadual 4 (m.s 17-18): permitted Tahap per jenis.
KSAS_TAHAP_RULES: Dict[int, set] = {
    1: {1, 2},
    2: {1, 2, 3},
    3: {3},
    4: {1, 2, 3},
    5: {2},
    6: {1, 2},
    7: {1, 3},
    8: {2},
    9: {1, 2},
    10: {3},
    11: {1, 2, 3},
    12: {1, 2},
    13: {1, 3},
    14: {2},
    15: {3},
}

# PPP KSAS Rajah 6 (m.s 32): official Tahap colour codes for KSAS plans.
KSAS_TAHAP_COLORS: Dict[int, tuple] = {
    1: (64, 90, 0),     # dark green — Tahap 1 (high sensitivity)
    2: (123, 181, 5),   # mid green — Tahap 2 (moderate)
    3: (206, 255, 111),  # light lime — Tahap 3 (low)
}

# Jadual 7 (m.s 31) schema + mapping extras (lat/lon/radius/provenance).
KSAS_COLUMNS = [
    "ksas_code", "jenis_ksas", "tahap", "nama", "luas_h",
    "akt_1", "akt_syarat", "akt_0",
    "negeri", "daerah", "bp", "bpk", "tahun_data",
    "lat", "lon", "radius_m", "area_derived", "radius_basis",
    "sumber", "representation", "keyakinan", "label_peta", "key",
]

NA_BLOCK = "NA"  # reference rows are not tied to an RT Blok Perancangan.
DEFAULT_NODE_RADIUS_M = 8000


def _slug(nama_core: str) -> str:
    """First-two-word ASCII slug, e.g. 'Sungai Pahang (Temerloh)' -> 'sungai_pahang'."""
    core = nama_core.split("(")[0]
    words = re.findall(r"[a-zA-Z]+", core.lower())[:2]
    return "_".join(words) if words else "tapak"


def radius_from_hectares(luas_h: float) -> int:
    """Circle-equivalent radius in metres, coarsely rounded to 500 m (no fake precision)."""
    return int(round(math.sqrt(float(luas_h) / 100.0 / math.pi) * 1000.0 / 500.0) * 500)


def _row(ksas_no: int, tahap: int, nama: str, negeri: str, daerah: str,
         luas_h, lat: float, lon: float, radius_m: int, area_derived: bool,
         radius_basis: str, akt_1: str, akt_syarat: str, akt_0: str,
         sumber: str, representation: str, keyakinan: str, label_peta: str) -> Dict:
    key = f"ksas{ksas_no:02d}_{_slug(nama)}"
    return {
        "ksas_code": f"KSAS {ksas_no}",
        "jenis_ksas": KSAS_TYPE_NAMES[ksas_no],
        "tahap": f"Tahap {tahap}",
        "nama": nama,
        "luas_h": luas_h,
        "akt_1": akt_1,
        "akt_syarat": akt_syarat,
        "akt_0": akt_0,
        "negeri": negeri,
        "daerah": daerah,
        "bp": NA_BLOCK,
        "bpk": NA_BLOCK,
        "tahun_data": 2025,
        "lat": lat,
        "lon": lon,
        "radius_m": radius_m,
        "area_derived": area_derived,
        "radius_basis": radius_basis,
        "sumber": sumber,
        "representation": representation,
        "keyakinan": keyakinan,
        "label_peta": label_peta,
        "key": key,
    }


# Canonical transcription: 22 representative sites covering all 15 PPP KSAS jenis.
# Activities condensed from PPP KSAS Jadual 5 (m.s 19-25) for the row's Tahap.
KSAS_CANONICAL_ROWS: List[Dict] = [
    _row(2, 1, "Cameron Highlands (Tanah Rata)", "Pahang", "Cameron Highlands",
         71200, 4.4735, 101.3789, 15000, True,
         "circle-equivalent of district-scale highland zone (~712 km2)",
         "Pertanian tanah tinggi",
         "Mematuhi syarat pihak berkuasa + RT; pertanian sedia ada tertakluk SMA",
         "Perbandaran",
         "PPP KSAS Jadual 4 (KSAS 2 Tahap 1: >=1,000 m); RTD Cameron Highlands 2030",
         "district-scale circular proxy — NOT the gazetted KSAS polygon",
         "disahkan", "Cameron KSAS"),
    _row(2, 1, "Fraser's Hill (Bukit Fraser)", "Pahang", "Raub",
         None, 3.7117, 101.7367, DEFAULT_NODE_RADIUS_M, False,
         "town-node buffer (luas tidak diterbitkan di sini)",
         "Pertanian tanah tinggi",
         "Mematuhi syarat pihak berkuasa + RT; pertanian sedia ada tertakluk SMA",
         "Perbandaran",
         "PPP KSAS Jadual 4 (KSAS 2 Tahap 1); koordinat bandar Bukit Fraser",
         "town-node circular proxy — NOT the gazetted KSAS polygon",
         "wakil", "Fraser KSAS"),
    _row(1, 2, "Matang Mangrove Forest Reserve", "Perak", "Larut & Matang",
         40288, 4.8342, 100.6167, 11500, True,
         "circle-equivalent of 40,288 ha reserve",
         "Pengusahasilan hutan",
         "Kebenaran PBN; Pengurusan Hutan Secara Berkekalan; ladang hutan hanya di zon Jabatan Perhutanan",
         "Perbandaran",
         "PLOS ONE (MMFR 40,288 ha); NASA (~40,000 ha); gazetted 1906; Akta Perhutanan Negara 1984",
         "reserve-scale circular proxy of crescent reserve — NOT the reserve boundary",
         "disahkan", "Matang KSAS"),
    _row(1, 1, "Endau-Rompin (Johor)", "Johor", "Mersing",
         48905, 2.45, 103.35, 12500, True,
         "circle-equivalent of 48,905 ha Johor portion",
         "Ekopelancongan dan rekreasi",
         "Berintensiti rendah; kawalan carrying capacity",
         "Perbandaran",
         "Johor National Parks Corporation (48,905 ha Johor portion); Hutan Perlindungan rationale",
         "park-scale circular proxy — NOT the gazetted HSK boundary",
         "disahkan", "Endau KSAS"),
    _row(3, 3, "Kinta Limestone Aggregate Belt", "Perak", "Kinta",
         None, 4.55, 101.05, DEFAULT_NODE_RADIUS_M, False,
         "simpanan-semasa node buffer (luas tidak diterbitkan di sini)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Perlombongan",
         "Perlombongan perlu kebenaran PBN dan mematuhi syarat yang ditetapkan",
         "-",
         "PPP KSAS Jadual 4 (KSAS 3 Tahap 3); JMG mineral belt (wakil node)",
         "resource-belt node proxy — NOT a gazetted simpanan boundary",
         "wakil", "Kinta KSAS"),
    _row(4, 1, "Kelantan Floodplain (Sg Kelantan basin)", "Kelantan", "Kota Bharu",
         None, 6.13, 102.24, DEFAULT_NODE_RADIUS_M, False,
         "bencana-node buffer (sem padan daripada kajian teknikal/PBT berkenaan)",
         "Infrastruktur dan utiliti berkepentingan negara",
         "Patuhi pihak berkuasa + RT; laporan geologi dan mitigasi risiko bencana",
         "Semua pembangunan selain infrastruktur dan utiliti berkepentingan negara",
         "PPP KSAS Jadual 4 (KSAS 4 Tahap 1: risiko tinggi); JPS flood records (wakil node)",
         "hazard-node proxy — sempadan sebenar daripada kajian teknikal berkenaan",
         "wakil", "Kelantan KSAS"),
    _row(5, 2, "George Town UNESCO Core", "Pulau Pinang", "Timur Laut",
         259, 5.4141, 100.3288, 1000, True,
         "circle-equivalent of 259 ha core+buffer",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Intensiti rendah; kawalan carrying capacity; laporan berkaitan warisan",
         "Perbandaran; Pertanian; Perlombongan",
         "UNESCO WHS 2008 (259.42 ha core+buffer); Akta Warisan Kebangsaan 2005",
         "heritage-core circular proxy — NOT the gazetted WHS boundary",
         "disahkan", "George Town KSAS"),
    _row(5, 2, "Gua Tempurung", "Perak", "Kampar",
         None, 4.42, 101.19, DEFAULT_NODE_RADIUS_M, False,
         "tapak-node buffer (luas tidak diterbitkan di sini)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Intensiti rendah; kawalan carrying capacity; laporan berkaitan warisan",
         "Perbandaran; Pertanian; Perlombongan",
         "PPP KSAS Jadual 4 (KSAS 5 Tahap 2: warisan semula jadi geologi); contoh nama Jadual 7",
         "site-node circular proxy — NOT a gazetted boundary",
         "wakil", "Tempurung KSAS"),
    _row(5, 2, "Kinabalu Park WHS", "Sabah", "Ranau",
         75370, 6.0750, 116.5583, 15500, True,
         "circle-equivalent of 75,370 ha World Heritage property",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Intensiti rendah; kawalan carrying capacity; laporan berkaitan warisan",
         "Perbandaran; Pertanian; Perlombongan",
         "UNESCO WHC 1012 (75,370 ha); Sabah Parks (gazetted 1964; Enakmen 1984)",
         "park-scale circular proxy centred on summit — NOT the KUGGp territory",
         "disahkan", "Kinabalu KSAS"),
    _row(6, 1, "Taman Negara (Kuala Tahan)", "Pahang", "Jerantut",
         434351, 4.3833, 102.4000, 16000, False,
         "gateway catchment node buffer; whole-park circle-equiv (~37.2 km) deliberately NOT used",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Intensiti rendah; kawalan carrying capacity",
         "Perbandaran; Pengusahasilan hutan; Pertanian; Perlombongan",
         "UNESCO tentative list 5927 (434,351 ha; Pahang 57%); Enakmen 1938/39; PERHILITAN",
         "gateway-node circular proxy — does NOT depict the tri-state park boundary",
         "disahkan", "Taman Negara KSAS"),
    _row(6, 1, "Royal Belum State Park", "Perak", "Hulu Perak",
         117500, 5.55, 101.35, 19500, True,
         "circle-equivalent of 117,500 ha state park",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Intensiti rendah; kawalan carrying capacity",
         "Perbandaran; Pengusahasilan hutan; Pertanian; Perlombongan",
         "Perak State Parks Corporation (117,500 ha taman negeri)",
         "park-scale circular proxy — NOT the gazetted park boundary",
         "disahkan", "Belum KSAS"),
    _row(7, 1, "Empangan Kenyir", "Terengganu", "Hulu Terengganu",
         26000, 5.00, 102.80, 9000, True,
         "circle-equivalent of ~260 km2 lake",
         "Infrastruktur empangan; Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan",
         "Intensiti rendah; teknologi hijau tidak menjejaskan ekosistem; pencemar berhampiran empangan dilarang",
         "Perbandaran; Pembalakan; Pertanian; Perlombongan",
         "PPP KSAS Jadual 4 (KSAS 7 Tahap 1: empangan sedia ada); tasik ~260 km2 (anggaran)",
         "reservoir-scale circular proxy — NOT the empangan reserve boundary",
         "anggaran", "Kenyir KSAS"),
    _row(8, 2, "Sungai Pahang Meander (Temerloh)", "Pahang", "Temerloh",
         None, 3.45, 102.42, DEFAULT_NODE_RADIUS_M, False,
         "river-reach node buffer (sempadan daripada kajian JPS berkenaan)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti; Kawasan takungan dan landskap",
         "Tidak cemar sumber air; reka bentuk ambil kira risiko banjir; tidak halang laluan air",
         "Semua pembangunan selain yang dibenarkan",
         "PPP KSAS Jadual 4 (KSAS 8 Tahap 2); JPS reach records (wakil node)",
         "reach-node proxy — sempadan sebenar daripada kajian teknikal berkenaan",
         "wakil", "Pahang KSAS"),
    _row(9, 1, "Tasek Bera Ramsar", "Pahang", "Bera",
         38446, 3.08, 102.62, 11000, True,
         "circle-equivalent of 38,446 ha Ramsar catchment",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Berintensiti rendah",
         "Perbandaran; Pembalakan; Pertanian; Perlombongan",
         "Ramsar Site 674 (38,446 ha; listed 1994); tapak tanah bencah diwartakan",
         "catchment-scale circular proxy — NOT the Ramsar boundary",
         "disahkan", "Bera KSAS"),
    _row(9, 2, "Pekan-Nenasi Peatland", "Pahang", "Pekan",
         None, 3.35, 103.35, DEFAULT_NODE_RADIUS_M, False,
         "gambut-node buffer (ketebalan/keluasan daripada kajian tanah berkenaan)",
         "Perikanan; Akuakultur; Infrastruktur dan utiliti",
         "Patuhi pihak berkuasa + RT; kajian kesesuaian tapak sebelum dibangunkan",
         "Perbandaran; Pertanian kelapa sawit; Perlombongan",
         "PPP KSAS Jadual 4/5 (KSAS 9 Tahap 2: tanah gambut); wakil node",
         "peat-node proxy — zon risiko sebenar daripada kajian tanah berkenaan",
         "wakil", "Gambut KSAS"),
    _row(10, 3, "Tasik Chini Biosphere", "Pahang", "Pekan",
         None, 3.43, 102.91, DEFAULT_NODE_RADIUS_M, False,
         "tasik-node buffer (sempadan rizab daripada PBT berkenaan)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Mematuhi syarat pihak berkuasa + RT",
         "-",
         "UNESCO Biosphere Reserve 2009; tasik semula jadi (wakil node)",
         "lake-node circular proxy — NOT the reserve boundary",
         "wakil", "Chini KSAS"),
    _row(10, 3, "Sungai Tembeling", "Pahang", "Jerantut",
         None, 4.20, 102.35, DEFAULT_NODE_RADIUS_M, False,
         "river-reach node buffer (sempadan daripada kajian JPS berkenaan)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Mematuhi syarat pihak berkuasa + RT",
         "Pertanian; Industri; Perumahan",
         "PPP KSAS Jadual 4 (KSAS 10 Tahap 3: sungai nilai sokongan); wakil node",
         "reach-node proxy — sempadan sebenar daripada kajian teknikal berkenaan",
         "wakil", "Tembeling KSAS"),
    _row(11, 1, "Cherating Turtle Beach", "Pahang", "Kuantan",
         None, 4.13, 103.39, DEFAULT_NODE_RADIUS_M, False,
         "pantai-node buffer (penentuan tahap merujuk NCVI/RFZPPN2 berkenaan)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Berintensiti rendah",
         "-",
         "PPP KSAS Jadual 4 (KSAS 11 Tahap 1: sokongan hidup/risiko/warisan tinggi); tapak pendaratan penyu",
         "beach-node proxy — 5 km kedaratan / 3 batu nautika sebenar daripada pelan berkenaan",
         "wakil", "Cherating KSAS"),
    _row(12, 1, "Taman Laut Pulau Tioman", "Pahang", "Rompin",
         None, 2.79, 104.17, DEFAULT_NODE_RADIUS_M, False,
         "marin-node buffer (sempadan taman laut daripada Jabatan Perikanan berkenaan)",
         "Ekopelancongan dan rekreasi; Penyelidikan dan pendidikan; Infrastruktur dan utiliti",
         "Intensiti rendah; kawalan carrying capacity",
         "Perikanan dan akuakultur",
         "PPP KSAS Jadual 4 (KSAS 12 Tahap 1: taman laut diwartakan); Akta Perikanan 1985",
         "marine-node proxy — NOT the gazetted marine park boundary",
         "wakil", "Tioman KSAS"),
    _row(13, 1, "Jelapang Padi MADA", "Kedah", "Kota Setar",
         100000, 6.12, 100.37, 18000, True,
         "circle-equivalent of ~100,000 ha scheme",
         "Kemudahan sokongan aktiviti pertanian; Infrastruktur dan utiliti",
         "Mematuhi syarat pihak berkuasa + RT; tanaman selain padi tidak dibenarkan",
         "Semua pembangunan selain kemudahan sokongan pertanian",
         "MADA scheme area ~100,000 ha (anggaran); PPP KSAS Jadual 4 (KSAS 13 Tahap 1)",
         "scheme-scale circular proxy — NOT the jelapang boundary",
         "anggaran", "MADA KSAS"),
    _row(14, 2, "Mamut Copper Mine (ex-mine)", "Sabah", "Ranau",
         None, 6.02, 116.66, DEFAULT_NODE_RADIUS_M, False,
         "bekas-lombong node buffer (rekod lombong berkenaan)",
         "Boleh dipertimbangkan untuk semua jenis pembangunan dengan syarat pihak berkuasa + RT",
         "Mematuhi syarat pihak berkuasa + RT",
         "Perikanan",
         "PPP KSAS Jadual 4/5 (KSAS 14 Tahap 2: bekas lombong bernilai warisan); rekod lombong Mamut",
         "ex-mine node proxy — NOT a surveyed boundary",
         "wakil", "Mamut KSAS"),
    _row(15, 3, "Seelong Sanitary Landfill", "Johor", "Kulai",
         None, 1.60, 103.65, DEFAULT_NODE_RADIUS_M, False,
         "tapak-node buffer (rekod PBT berkenaan)",
         "Taman awam; Infrastruktur dan utiliti",
         "Rujuk Panduan Pemuliharaan Bekas Tapak Pelupusan (Jabatan Landskap Negara)",
         "-",
         "PPP KSAS Jadual 4 (KSAS 15 Tahap 3); tapak sanitari PBT (wakil node)",
         "site-node proxy — saiz/lokasi sebenar daripada rekod PBT berkenaan",
         "wakil", "Seelong KSAS"),
]


def _parse_tahap(raw) -> int:
    """Accepts 'Tahap 1', 'tahap1', 1, 1.0 → 1; raises on anything else."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError("tahap is missing")
    if isinstance(raw, (int, float)) and not pd.isna(raw):
        val = int(raw)
    else:
        m = re.search(r"[123]", str(raw))
        if not m:
            raise ValueError(f"unparseable tahap: {raw!r}")
        val = int(m.group(0))
    if val not in (1, 2, 3):
        raise ValueError(f"tahap out of range: {raw!r}")
    return val


def _parse_ksas_no(raw) -> int:
    """Accepts 'KSAS 12', 'ksas12', 12 → 12."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError("ksas_code is missing")
    if isinstance(raw, (int, float)) and not pd.isna(raw):
        val = int(raw)
    else:
        m = re.search(r"\d{1,2}", str(raw))
        if not m:
            raise ValueError(f"unparseable ksas_code: {raw!r}")
        val = int(m.group(0))
    if val not in KSAS_TYPE_NAMES:
        raise ValueError(f"ksas_code out of range 1..15: {raw!r}")
    return val


def parse_ksas_reference(csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads, cleans and validates ksas_reference.csv against the PPP KSAS schema.
    Returns (clean DataFrame, validation report dict). Raises ValueError on
    critical faults (bad tahap/jenis, out-of-bounds coordinates, duplicate keys,
    radius/area mismatch beyond tolerance).
    """
    input_path = Path(csv_path) if csv_path else OUTPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"KSAS reference CSV not found: {input_path}")

    logger.info("Parsing KSAS reference from %s", input_path)
    df = pd.read_csv(input_path)
    report: Dict = {"rows_raw": len(df), "warnings": [], "errors": []}

    # --- Jadual 7 required fields -------------------------------------------------
    required = ["ksas_code", "jenis_ksas", "tahap", "nama", "akt_1", "akt_0",
                "negeri", "daerah", "tahun_data", "lat", "lon", "radius_m",
                "sumber", "representation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ksas_reference.csv missing Jadual 7 columns: {missing}")

    df = df.copy()
    for col in ("ksas_code", "jenis_ksas", "tahap", "nama", "negeri", "daerah",
                "sumber", "representation", "keyakinan", "label_peta", "radius_basis"):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].isna(), df[col].astype(str).str.strip())

    # --- clean + validate row by row ----------------------------------------------
    df["_ksas_no"] = df["ksas_code"].apply(_parse_ksas_no)
    df["_tahap_no"] = df["tahap"].apply(_parse_tahap)
    df["tahap"] = df["_tahap_no"].apply(lambda v: f"Tahap {v}")
    df["ksas_code"] = df["_ksas_no"].apply(lambda v: f"KSAS {v}")

    # jenis name must be consistent with the statutory number.
    for idx, row in df.iterrows():
        expected = KSAS_TYPE_NAMES[int(row["_ksas_no"])]
        given = str(row["jenis_ksas"])
        if expected.split("(")[0].strip().lower() not in given.lower() and \
                given.lower() not in expected.lower():
            raise ValueError(f"row {idx} ({row['nama']}): jenis '{given}' "
                             f"inconsistent with {row['ksas_code']} '{expected}'")

    # Tahap must be legal for the jenis (Jadual 4).
    for idx, row in df.iterrows():
        if int(row["_tahap_no"]) not in KSAS_TAHAP_RULES[int(row["_ksas_no"])]:
            raise ValueError(f"row {idx} ({row['nama']}): {row['tahap']} illegal for "
                             f"{row['ksas_code']} per PPP KSAS Jadual 4 "
                             f"(allowed: {sorted(KSAS_TAHAP_RULES[int(row['_ksas_no'])])})")

    # luas_h numeric (NaN allowed only for wakil rows).
    df["luas_h"] = pd.to_numeric(df["luas_h"], errors="coerce")
    for idx, row in df.iterrows():
        if pd.isna(row["luas_h"]) and str(row.get("keyakinan", "")).strip().lower() != "wakil":
            raise ValueError(f"row {idx} ({row['nama']}): missing luas_h on a non-wakil row")
        if pd.notna(row["luas_h"]) and float(row["luas_h"]) <= 0:
            raise ValueError(f"row {idx} ({row['nama']}): luas_h must be positive")

    # Coordinates inside Malaysia; radius sane.
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["radius_m"] = pd.to_numeric(df["radius_m"], errors="coerce")
    if df[["lat", "lon", "radius_m"]].isna().any().any():
        raise ValueError("lat/lon/radius_m contain non-numeric values")
    if not (((0.8 <= df["lat"]) & (df["lat"] <= 7.6)).all()):
        raise ValueError("lat outside Malaysia bounds [0.8, 7.6]")
    if not (((99.0 <= df["lon"]) & (df["lon"] <= 119.6)).all()):
        raise ValueError("lon outside Malaysia bounds [99.0, 119.6]")
    if not (((1000 <= df["radius_m"]) & (df["radius_m"] <= 40000)).all()):
        raise ValueError("radius_m outside sane range [1000, 40000]")
    if not (df["radius_m"] % 500 == 0).all():
        raise ValueError("radius_m must be rounded to 500 m (no fake precision)")

    # Circle-equivalent consistency for area-derived rows (tolerance 500 m rounding).
    for idx, row in df.iterrows():
        if str(row.get("area_derived", "")).strip().lower() == "true":
            expect = radius_from_hectares(float(row["luas_h"]))
            if abs(float(row["radius_m"]) - expect) > 500:
                raise ValueError(f"row {idx} ({row['nama']}): radius_m {row['radius_m']} "
                                 f"inconsistent with luas_h {row['luas_h']} (expect ~{expect})")

    # Unique keys; all 15 jenis covered.
    if df["key"].duplicated().any():
        dupes = df.loc[df["key"].duplicated(), "key"].tolist()
        raise ValueError(f"duplicate KSAS keys: {dupes}")
    covered = set(int(v) for v in df["_ksas_no"].tolist())
    missing = sorted(set(KSAS_TYPE_NAMES) - covered)
    if missing:
        report["warnings"].append(f"KSAS jenis missing from dataset: {missing}")
    report["missing_types"] = missing
    report["types_covered"] = sorted(covered)

    # tahun_data sane.
    df["tahun_data"] = pd.to_numeric(df["tahun_data"], errors="coerce")
    if df["tahun_data"].isna().any() or not ((2000 <= df["tahun_data"]) & (df["tahun_data"] <= 2030)).all():
        raise ValueError("tahun_data outside sane range [2000, 2030]")

    # Drop helper columns; enforce column order.
    df = df.drop(columns=["_ksas_no", "_tahap_no"])
    df = df[[c for c in KSAS_COLUMNS if c in df.columns]]
    report["rows_clean"] = len(df)
    logger.info("KSAS reference clean: %d rows, types %s, warnings=%d",
                len(df), report["types_covered"], len(report["warnings"]))
    return df, report


def build_ksas_reference(output_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """Writes the canonical transcription to CSV, then re-parses it for validation."""
    dest = Path(output_path) if output_path else OUTPUT_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(KSAS_CANONICAL_ROWS)
    raw.to_csv(dest, index=False)
    logger.info("Wrote canonical KSAS reference (%d rows) to %s", len(raw), dest)
    return parse_ksas_reference(dest)


if __name__ == "__main__":
    df_ksas, rep = build_ksas_reference()
    print(f"rows: {rep['rows_clean']} | types covered: {rep['types_covered']} | "
          f"missing: {rep['missing_types']} | warnings: {rep['warnings']}")
    print(df_ksas[["ksas_code", "tahap", "nama", "negeri", "key"]].to_string(index=False))