"""
DESTINASI — Authoritative District -> PBT (Pihak Berkuasa Tempatan) Lookup.

Grounded against (accessed Sep 2026):
- KPKT Jabatan Kerajaan Tempatan (JKT) Senarai PBT + FAQ
  (151-152 PBT: 19-20 DB/MB + 40 MP + 92 MD) — https://jkt.kpkt.gov.my/data-pbt-senarai_pbt/
- KPKT OSC Senarai PBT by state (Neg=01..13) — http://portalosc.kpkt.gov.my/osc/PBT2_index.cfm
- KPKT PBTPay PBT list — https://pbtpay.kpkt.gov.my/pbt
- DBKL — https://www.dbkl.gov.my (sole authority for W.P. Kuala Lumpur)
- MBPP — https://www.mbpp.gov.my (covers Timur Laut + Barat Daya, Penang island)
- MBSP covers all three Seberang Perai districts (Penang mainland)
- Perbadanan Putrajaya Act 536 (PPj) — https://www.ppj.gov.my
- Perbadanan Labuan (PL) — https://www.pl.gov.my
- MP Kangar covers the whole of Perlis — https://www.mpkangar.gov.my
- MBKT Cawangan Kuala Nerus — https://mbkt.terengganu.gov.my
- Portal PBT Kedah short codes (MBAS/MPSP/MPKK/MPLBP/MDB/MPKP/MDPT/MDP/MDS/MDY)
  — https://pbt.kedah.gov.my
- Portal PBT Pahang (MBK/MPT/MPB/MPPBDR/MDLipis/MDMaran/MDBera/MDJerantut/
  MDRaub/MDRompin/MDCH) — https://www.pahang.gov.my
- OSC MySMS codes for N. Sembilan (MBS/MPPD/MPJL/MDJELEBU/MDKP/MDR/MDTAMPIN)
- OSC codes Kelantan (MPKB/MDBACHOK/MDGM/MDMACHANG/MDPM/MDPP/MDTMERAH/
  MDTUMPAT/MDJELI) and Terengganu (MBKT/MPKEMAMAN/MPDUNGUN/MDBESUT/MDHT/
  MDMARANG/MDSETIU)
- Sabah OSC Neg=12 (DBKK + MP Sandakan + MP Tawau + MDs + Lembaga Bandaran Kudat)
- Sarawak OSC Neg=13 + MPHlg (DBKU/MBKS/MPP/SMC+MDLBS/BDA/MBM/MPKS/MDSA/
  MDLimbang/MDDM/MDSarikei+MDMJ/MDKapit)

Conventions:
- Format: "Full Malay name (ACRONYM)", matching pilot_corridors.csv precedent.
- Acronyms are the PBT's own official short codes (e.g. MPLBP for
  Majlis Perbandaran Langkawi Bandaraya Pelancongan — verified via
  mplbp.gov.my; NOT "MBLBP"). Same acronym may legitimately repeat
  across states (e.g. MPT = Taiping and Temerloh; MPKK = Kulim and
  Kuala Kangsar; MDR = Raub/Rompin/Rembau) — the full name disambiguates.
- Districts spanned by several PBTs list ALL covering PBTs joined by " / ".
- Keys are (state_name, district_name) exactly as used in DISTRICT_COORDS /
  dist_to_state in build_unified_database.py.
"""

from typing import Dict, Tuple

PBT_LOOKUP: Dict[Tuple[str, str], str] = {
    # ---------------- Johor (OSC Neg=01, 16 PBT) ----------------
    ("Johor", "Johor Bahru"): (
        "Majlis Bandaraya Johor Bahru (MBJB) / "
        "Majlis Bandaraya Iskandar Puteri (MBIP) / "
        "Majlis Bandaraya Pasir Gudang (MBPG)"
    ),
    ("Johor", "Batu Pahat"): "Majlis Perbandaran Batu Pahat (MPBP)",
    ("Johor", "Muar"): "Majlis Perbandaran Muar (MPM)",
    ("Johor", "Kota Tinggi"): (
        "Majlis Daerah Kota Tinggi (MDKT) / "
        "Majlis Perbandaran Pengerang (MPPengerang)"
    ),
    ("Johor", "Segamat"): (
        "Majlis Perbandaran Segamat (MPSegamat) / "
        "Majlis Daerah Labis (MDLabis)"
    ),
    ("Johor", "Kluang"): (
        "Majlis Perbandaran Kluang (MPKluang) / "
        "Majlis Daerah Simpang Renggam (MDSR)"
    ),
    ("Johor", "Pontian"): "Majlis Perbandaran Pontian (MPPn)",
    ("Johor", "Mersing"): "Majlis Daerah Mersing (MDMersing)",
    ("Johor", "Kulai"): "Majlis Perbandaran Kulai (MPKulai)",
    ("Johor", "Tangkak"): "Majlis Daerah Tangkak (MDTangkak)",
    # ---------------- Kedah (Portal PBT Kedah codes) ----------------
    ("Kedah", "Kota Setar"): "Majlis Bandaraya Alor Setar (MBAS)",
    ("Kedah", "Langkawi"): "Majlis Perbandaran Langkawi Bandaraya Pelancongan (MPLBP)",
    ("Kedah", "Kuala Muda"): "Majlis Perbandaran Sungai Petani (MPSP)",
    ("Kedah", "Baling"): "Majlis Daerah Baling (MDB)",
    ("Kedah", "Kulim"): "Majlis Perbandaran Kulim (MPKK)",
    ("Kedah", "Kubang Pasu"): "Majlis Perbandaran Kubang Pasu (MPKP)",
    ("Kedah", "Yan"): "Majlis Daerah Yan (MDY)",
    ("Kedah", "Pendang"): "Majlis Daerah Pendang (MDP)",
    ("Kedah", "Padang Terap"): "Majlis Daerah Padang Terap (MDPT)",
    ("Kedah", "Sik"): "Majlis Daerah Sik (MDS)",
    # ---------------- Kelantan (OSC Neg=03 codes) ----------------
    ("Kelantan", "Kota Bharu"): "Majlis Perbandaran Kota Bharu Bandaraya Islam (MPKB-BRI)",
    ("Kelantan", "Bachok"): "Majlis Daerah Bachok Bandar Pelancongan Islam (MDBachok)",
    ("Kelantan", "Pasir Mas"): "Majlis Daerah Pasir Mas (MDPM)",
    ("Kelantan", "Pasir Puteh"): "Majlis Daerah Pasir Puteh (MDPP)",
    ("Kelantan", "Tanah Merah"): "Majlis Daerah Tanah Merah (MDTanahMerah)",
    ("Kelantan", "Tumpat"): "Majlis Daerah Tumpat (MDTumpat)",
    ("Kelantan", "Machang"): "Majlis Daerah Machang (MDMachang)",
    ("Kelantan", "Gua Musang"): "Majlis Daerah Gua Musang (MDGM)",
    ("Kelantan", "Jeli"): "Majlis Daerah Jeli (MDJeli)",
    ("Kelantan", "Kuala Krai"): "Majlis Daerah Kuala Krai (MDKualaKrai)",
    # ---------------- Melaka (OSC Neg=04) ----------------
    ("Melaka", "Melaka Tengah"): (
        "Majlis Bandaraya Melaka Bersejarah (MBMB) / "
        "Majlis Perbandaran Hang Tuah Jaya (MPHTJ)"
    ),
    ("Melaka", "Alor Gajah"): (
        "Majlis Perbandaran Alor Gajah (MPAG) / "
        "Majlis Perbandaran Hang Tuah Jaya (MPHTJ)"
    ),
    ("Melaka", "Jasin"): (
        "Majlis Perbandaran Jasin (MPJ) / "
        "Majlis Perbandaran Hang Tuah Jaya (MPHTJ)"
    ),
    # ---------------- Negeri Sembilan (OSC Neg=05 MySMS codes) ----------------
    ("Negeri Sembilan", "Seremban"): "Majlis Bandaraya Seremban (MBS)",
    ("Negeri Sembilan", "Port Dickson"): "Majlis Perbandaran Port Dickson (MPPD)",
    ("Negeri Sembilan", "Jempol"): "Majlis Perbandaran Jempol (MPJL)",
    ("Negeri Sembilan", "Kuala Pilah"): "Majlis Daerah Kuala Pilah (MDKP)",
    ("Negeri Sembilan", "Rembau"): "Majlis Daerah Rembau (MDR)",
    ("Negeri Sembilan", "Tampin"): "Majlis Daerah Tampin (MDTampin)",
    ("Negeri Sembilan", "Jelebu"): "Majlis Daerah Jelebu (MDJelebu)",
    # ---------------- Pahang (OSC Neg=06 + pahang.gov.my) ----------------
    ("Pahang", "Kuantan"): "Majlis Bandaraya Kuantan (MBK)",
    ("Pahang", "Bentong"): "Majlis Perbandaran Bentong (MPB)",
    ("Pahang", "Cameron Highlands"): "Majlis Daerah Cameron Highlands (MDCH)",
    ("Pahang", "Rompin"): "Majlis Daerah Rompin (MDRompin)",
    ("Pahang", "Temerloh"): "Majlis Perbandaran Temerloh (MPT)",
    ("Pahang", "Lipis"): "Majlis Daerah Lipis (MDLipis)",
    ("Pahang", "Raub"): "Majlis Daerah Raub (MDRaub)",
    ("Pahang", "Jerantut"): "Majlis Daerah Jerantut (MDJerantut)",
    ("Pahang", "Pekan"): "Majlis Perbandaran Pekan Bandar Diraja (MPPBDR)",
    ("Pahang", "Maran"): "Majlis Daerah Maran (MDMaran)",
    ("Pahang", "Bera"): "Majlis Daerah Bera (MDBera)",
    # ---------------- Pulau Pinang (2 MB) ----------------
    ("Pulau Pinang", "Timur Laut"): "Majlis Bandaraya Pulau Pinang (MBPP)",
    ("Pulau Pinang", "Barat Daya"): "Majlis Bandaraya Pulau Pinang (MBPP)",
    ("Pulau Pinang", "Seberang Perai Tengah"): "Majlis Bandaraya Seberang Perai (MBSP)",
    ("Pulau Pinang", "Seberang Perai Utara"): "Majlis Bandaraya Seberang Perai (MBSP)",
    ("Pulau Pinang", "Seberang Perai Selatan"): "Majlis Bandaraya Seberang Perai (MBSP)",
    # ---------------- Perak (OSC Neg=08 + perak.gov.my) ----------------
    ("Perak", "Kinta"): "Majlis Bandaraya Ipoh (MBI)",
    ("Perak", "Larut & Matang"): "Majlis Perbandaran Taiping (MPT)",
    ("Perak", "Manjung"): "Majlis Perbandaran Manjung (MPM)",
    ("Perak", "Kuala Kangsar"): "Majlis Perbandaran Kuala Kangsar (MPKK)",
    ("Perak", "Batang Padang"): "Majlis Daerah Tapah (MDTapah)",
    ("Perak", "Hilir Perak"): "Majlis Perbandaran Teluk Intan (MPTI)",
    ("Perak", "Kerian"): "Majlis Daerah Kerian (MDKerian)",
    ("Perak", "Kampar"): "Majlis Daerah Kampar (MDKampar)",
    ("Perak", "Muallim"): "Majlis Daerah Tanjong Malim (MDTM)",
    ("Perak", "Perak Tengah"): "Majlis Daerah Perak Tengah (MDPT)",
    ("Perak", "Hulu Perak"): (
        "Majlis Daerah Gerik (MDGerik) / "
        "Majlis Daerah Pengkalan Hulu (MDPengkalanHulu) / "
        "Majlis Daerah Lenggong (MDLenggong)"
    ),
    # ---------------- Perlis (sole PBT statewide) ----------------
    ("Perlis", "Kangar"): "Majlis Perbandaran Kangar (MPKangar)",
    ("Perlis", "Arau"): "Majlis Perbandaran Kangar (MPKangar)",
    ("Perlis", "Padang Besar"): "Majlis Perbandaran Kangar (MPKangar)",
    # ---------------- Selangor (OSC Neg=10 + PTG Selangor) ----------------
    ("Selangor", "Petaling"): (
        "Majlis Bandaraya Petaling Jaya (MBPJ) / "
        "Majlis Bandaraya Subang Jaya (MBSJ) / "
        "Majlis Bandaraya Shah Alam (MBSA)"
    ),
    ("Selangor", "Sepang"): "Majlis Perbandaran Sepang (MPSp)",
    ("Selangor", "Gombak"): (
        "Majlis Perbandaran Selayang (MPS) / "
        "Majlis Perbandaran Ampang Jaya (MPAJ)"
    ),
    ("Selangor", "Kuala Selangor"): "Majlis Perbandaran Kuala Selangor (MPKS)",
    ("Selangor", "Hulu Langat"): (
        "Majlis Perbandaran Kajang (MPKj) / "
        "Majlis Perbandaran Ampang Jaya (MPAJ)"
    ),
    ("Selangor", "Klang"): "Majlis Bandaraya Diraja Klang (MBDK)",
    ("Selangor", "Kuala Langat"): "Majlis Perbandaran Kuala Langat (MPKL)",
    ("Selangor", "Hulu Selangor"): "Majlis Perbandaran Hulu Selangor (MPHS)",
    ("Selangor", "Sabak Bernam"): "Majlis Daerah Sabak Bernam (MDSB)",
    # ---------------- Terengganu (OSC Neg=11) ----------------
    ("Terengganu", "Kuala Terengganu"): "Majlis Bandaraya Kuala Terengganu (MBKT)",
    ("Terengganu", "Kuala Nerus"): "Majlis Bandaraya Kuala Terengganu (MBKT)",
    ("Terengganu", "Kemaman"): "Majlis Perbandaran Kemaman (MPK)",
    ("Terengganu", "Besut"): "Majlis Daerah Besut (MDBesut)",
    ("Terengganu", "Dungun"): "Majlis Perbandaran Dungun (MPD)",
    ("Terengganu", "Marang"): "Majlis Daerah Marang (MDMarang)",
    ("Terengganu", "Setiu"): "Majlis Daerah Setiu (MDSetiu)",
    ("Terengganu", "Hulu Terengganu"): "Majlis Daerah Hulu Terengganu (MDHT)",
    # ---------------- Sabah (OSC Neg=12: DBKK + 2 MP + MDs + LBK) ----------------
    ("Sabah", "Kota Kinabalu"): "Dewan Bandaraya Kota Kinabalu (DBKK)",
    ("Sabah", "Ranau"): "Majlis Daerah Ranau (MDRanau)",
    ("Sabah", "Tawau"): "Majlis Perbandaran Tawau (MPT)",
    ("Sabah", "Sandakan"): "Majlis Perbandaran Sandakan (MPS)",
    ("Sabah", "Keningau"): "Majlis Daerah Keningau (MDK)",
    ("Sabah", "Penampang"): "Majlis Daerah Penampang (MDPenampang)",
    ("Sabah", "Tuaran"): "Majlis Daerah Tuaran (MDTuaran)",
    ("Sabah", "Kudat"): "Lembaga Bandaran Kudat (LBK)",
    ("Sabah", "Lahad Datu"): "Majlis Daerah Lahad Datu (MDLD)",
    ("Sabah", "Semporna"): "Majlis Daerah Semporna (MDSemporna)",
    # ---------------- Sarawak (OSC Neg=13 + MPHlg) ----------------
    ("Sarawak", "Kuching"): (
        "Dewan Bandaraya Kuching Utara (DBKU) / "
        "Majlis Bandaraya Kuching Selatan (MBKS) / "
        "Majlis Perbandaran Padawan (MPP)"
    ),
    ("Sarawak", "Sibu"): (
        "Majlis Perbandaran Sibu (SMC) / "
        "Majlis Daerah Luar Bandar Sibu (MDLBS)"
    ),
    ("Sarawak", "Bintulu"): "Lembaga Kemajuan Bintulu (BDA)",
    ("Sarawak", "Miri"): "Majlis Bandaraya Miri (MBM)",
    ("Sarawak", "Sri Aman"): "Majlis Daerah Sri Aman (MDSA)",
    ("Sarawak", "Samarahan"): "Majlis Perbandaran Kota Samarahan (MPKS)",
    ("Sarawak", "Limbang"): "Majlis Daerah Limbang (MDLimbang)",
    ("Sarawak", "Mukah"): "Majlis Daerah Dalat dan Mukah (MDDM)",
    ("Sarawak", "Sarikei"): (
        "Majlis Daerah Sarikei (MDSarikei) / "
        "Majlis Daerah Maradong dan Julau (MDMJ)"
    ),
    ("Sarawak", "Kapit"): "Majlis Daerah Kapit (MDKapit)",
    # ---------------- Federal Territories ----------------
    ("W.P. Kuala Lumpur", "W.P. Kuala Lumpur"): "Dewan Bandaraya Kuala Lumpur (DBKL)",
    ("W.P. Putrajaya", "W.P. Putrajaya"): "Perbadanan Putrajaya (PPj)",
    ("W.P. Labuan", "W.P. Labuan"): "Perbadanan Labuan (PL)",
}


def get_pbt_name(state: str, district: str) -> str:
    """Return the authoritative PBT name for a (state, district) pair.

    Raises KeyError with a helpful message if the pair is not mapped,
    so new districts fail loudly instead of silently generating a
    fabricated "Majlis ... {district}" name.
    """
    key = (state, district)
    if key not in PBT_LOOKUP:
        raise KeyError(
            f"No authoritative PBT mapping for state={state!r}, district={district!r}. "
            "Add it to src/etl/pbt_mapping.py PBT_LOOKUP (grounded against KPKT JKT/OSC)."
        )
    return PBT_LOOKUP[key]
