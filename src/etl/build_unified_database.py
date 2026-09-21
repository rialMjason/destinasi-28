"""
DESTINASI — Unified Master Destination Database & Corridor Geometry Builder
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Synthesizes:
1. data/processed/dts_all_years_districts.csv
2. data/processed/dts_all_years_attractions.csv
3. data/processed/motac_hotel_occupancy_aor_timeseries.csv
4. data/processed/motac_hotel_inventory_timeseries.csv
5. data/processed/state_economic_profile.csv

Generates:
- data/processed/destinations_master.csv
- data/processed/corridors_transit_routes.json (Polyline highway/railway coordinates for animated Leaflet routing)
"""

from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_unified_database")

try:  # package / module mode: python -m src.etl.build_unified_database
    from .pbt_mapping import get_pbt_name
except ImportError:  # script mode: python src/etl/build_unified_database.py
    try:
        from src.etl.pbt_mapping import get_pbt_name
    except ImportError:
        from pbt_mapping import get_pbt_name

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# High precision coordinates for all administrative districts in Malaysia
DISTRICT_COORDS = {
    # Johor
    "Johor Bahru": (1.4927, 103.7414), "Batu Pahat": (1.8494, 102.9288),
    "Muar": (2.0394, 102.5694), "Kota Tinggi": (1.7336, 103.8992), "Segamat": (2.5146, 102.8158),
    "Kluang": (2.0327, 103.3242), "Pontian": (1.4947, 103.3896), "Mersing": (2.4312, 103.8398),
    "Kulai": (1.6565, 103.6033), "Tangkak": (2.2673, 102.5453),
    # Kedah
    "Kota Setar": (6.1248, 100.3678), "Langkawi": (6.3265, 99.8432),
    "Kuala Muda": (5.6371, 100.4883), "Baling": (5.6775, 100.9189), "Kulim": (5.3705, 100.5552),
    "Kubang Pasu": (6.3142, 100.4283), "Yan": (5.7958, 100.3752), "Pendang": (5.9922, 100.4789),
    "Padang Terap": (6.2575, 100.6125), "Sik": (5.8239, 100.7483),
    # Kelantan
    "Kota Bharu": (6.1254, 102.2381), "Bachok": (6.0640, 102.3995),
    "Pasir Mas": (6.0425, 102.1386), "Pasir Puteh": (5.8360, 102.4019), "Tanah Merah": (5.8080, 102.1481),
    "Tumpat": (6.1978, 102.1694), "Machang": (5.7656, 102.2158), "Gua Musang": (4.8822, 101.9686),
    "Jeli": (5.6983, 101.8439), "Kuala Krai": (5.5317, 102.2003),
    # Melaka
    "Melaka Tengah": (2.2008, 102.2519), "Alor Gajah": (2.3831, 102.2089), "Jasin": (2.3082, 102.4326),
    # Negeri Sembilan
    "Seremban": (2.7258, 101.9424), "Port Dickson": (2.5228, 101.7959), "Jempol": (2.8089, 102.4058),
    "Kuala Pilah": (2.7389, 102.2489), "Rembau": (2.5897, 102.0919), "Tampin": (2.4706, 102.2306),
    "Jelebu": (2.9417, 102.0667),
    # Pahang
    "Kuantan": (3.8126, 103.3256), "Bentong": (3.5222, 101.9100),
    "Cameron Highlands": (4.4735, 101.3789), "Rompin": (2.8173, 103.4682), "Temerloh": (3.4485, 102.4176),
    "Lipis": (4.1843, 102.0463), "Raub": (3.7899, 101.8570), "Jerantut": (3.9372, 102.3625),
    "Pekan": (3.4836, 103.3994), "Maran": (3.5861, 102.7725), "Bera": (3.2458, 102.5806),
    # Pulau Pinang
    "Timur Laut": (5.4141, 100.3288), "Barat Daya": (5.3341, 100.2241),
    "Seberang Perai Tengah": (5.3683, 100.4608), "Seberang Perai Utara": (5.4920, 100.4357),
    "Seberang Perai Selatan": (5.2281, 100.4912),
    # Perak
    "Kinta": (4.5975, 101.0901), "Larut & Matang": (4.8500, 100.7333),
    "Manjung": (4.1953, 100.6622), "Kuala Kangsar": (4.7738, 100.9416), "Batang Padang": (4.2003, 101.2725),
    "Hilir Perak": (4.0258, 101.0208), "Kerian": (5.1667, 100.4833), "Kampar": (4.3125, 101.1508),
    "Muallim": (3.6842, 101.5208), "Perak Tengah": (4.3583, 100.9417), "Hulu Perak": (5.4333, 101.1333),
    # Perlis
    "Kangar": (6.4436, 100.1986), "Arau": (6.4297, 100.2742), "Padang Besar": (6.6628, 100.3214),
    # Selangor
    "Petaling": (3.1118, 101.5947), "Sepang": (2.6953, 101.7500),
    "Gombak": (3.2505, 101.6508), "Kuala Selangor": (3.3361, 101.2504), "Hulu Langat": (3.1362, 101.7915),
    "Klang": (3.0449, 101.4456), "Kuala Langat": (2.8167, 101.4833), "Hulu Selangor": (3.5636, 101.6575),
    "Sabak Bernam": (3.7667, 100.9833),
    # Terengganu
    "Kuala Terengganu": (5.3302, 103.1408), "Kuala Nerus": (5.3857, 103.0784),
    "Kemaman": (4.2312, 103.3330), "Besut": (5.7483, 102.5028), "Dungun": (4.7675, 103.4150),
    "Marang": (5.2056, 103.2058), "Setiu": (5.5667, 102.7333), "Hulu Terengganu": (4.9750, 102.9333),
    # Sabah
    "Kota Kinabalu": (5.9804, 116.0735), "Ranau": (5.9525, 116.6644),
    "Tawau": (4.2498, 117.8871), "Sandakan": (5.8394, 118.1172), "Keningau": (5.3400, 116.1601),
    "Penampang": (5.9125, 116.1158), "Tuaran": (6.1794, 116.2306), "Kudat": (6.8833, 116.8333),
    "Lahad Datu": (5.0267, 118.3275), "Semporna": (4.4819, 118.6111),
    # Sarawak
    "Kuching": (1.5533, 110.3440), "Sibu": (2.2873, 111.8306),
    "Bintulu": (3.1706, 113.0402), "Miri": (4.4148, 114.0089), "Sri Aman": (1.2384, 111.4621),
    "Samarahan": (1.4583, 110.4500), "Limbang": (4.7500, 115.0000), "Mukah": (2.8958, 112.0917),
    "Sarikei": (2.1167, 111.5167), "Kapit": (2.0167, 112.9333),
    # Federal Territories
    "W.P. Kuala Lumpur": (3.1390, 101.6869), "W.P. Putrajaya": (2.9264, 101.6964), "W.P. Labuan": (5.2831, 115.2308),
}

# Accurate Highway & Rail Corridor Waypoints for Real-Path AntPath Animation
CORRIDOR_ROUTES = {
    # 1. Northern Corridor: George Town (Penang) -> Taiping (Perak) via PLUS E1 & KTM ETS
    "George Town-Taiping": {
        "corridor_name": "Northern Heritage Rail & PLUS Corridor (KTM ETS & E1)",
        "mode": "KTM Komuter Utara / ETS & PLUS Expressway (E1)",
        "travel_time_mins": 52,
        "fare_rm": 7.50,
        "waypoints": [
            [5.4141, 100.3288], # George Town
            [5.3544, 100.3014], # Penang Bridge / Bayan Lepas
            [5.3683, 100.4608], # Butterworth / Seberang Perai
            [5.1667, 100.4833], # Nibong Tebal / Parit Buntar (PLUS E1)
            [5.0500, 100.5833], # Bagan Serai
            [4.8500, 100.7333], # Taiping / Lake Gardens
        ]
    },
    # 2. Highlands Corridor: Cameron Highlands -> Kuala Lipis / Raub via FR59 & Central Spine Road
    "Cameron Highlands-Lipis": {
        "corridor_name": "Highlands Eco-Heritage Corridor (FR59 & Central Spine Road)",
        "mode": "Federal Route 59 & Central Spine Road (CSR)",
        "travel_time_mins": 65,
        "fare_rm": 12.00,
        "waypoints": [
            [4.4735, 101.3789], # Tanah Rata / Brinchang (Cameron)
            [4.4250, 101.4000], # Ringlet
            [4.3500, 101.6000], # Sungai Koyan Interchange
            [4.2500, 101.8500], # Padang Tengku (Central Spine Road)
            [4.1843, 102.0463], # Kuala Lipis Heritage Town
        ]
    },
    # 3. Southern Corridor: Bandar Melaka -> Muar via Federal Route 5 Coastal Highway
    "Melaka Tengah-Muar": {
        "corridor_name": "Southern Historic Coastal Trail (Federal Route 5)",
        "mode": "Federal Route 5 Coastal Highway & Muar Bypass",
        "travel_time_mins": 45,
        "fare_rm": 0.00,
        "waypoints": [
            [2.2008, 102.2519], # Bandaraya Melaka
            [2.1500, 102.3200], # Umbai Seafood Jetty
            [2.1333, 102.4333], # Merlimau
            [2.0800, 102.5200], # Sungai Rambai
            [2.0394, 102.5694], # Bandar Maharani Muar
        ]
    },
    # 4. Northern Coastal / Island: Langkawi -> Yan / Kuala Perlis via Ro-Ro Ferry & Federal Route 7
    "Langkawi-Yan": {
        "corridor_name": "Northern Geopark Coastal & Ferry Rail Link",
        "mode": "Ro-Ro Ferry & KTM ETS Arau / Federal Route 7",
        "travel_time_mins": 75,
        "fare_rm": 23.00,
        "waypoints": [
            [6.3265, 99.8432],  # Kuah Jetty Langkawi
            [6.4000, 100.1333], # Kuala Perlis Ferry Terminal
            [6.1248, 100.3678], # Alor Setar Highway Junction
            [5.7958, 100.3752], # Yan / Gunung Jerai
        ]
    },
    # 5. Klang Valley to Hulu Selangor / Perak: Petaling -> Kuala Kangsar
    "Petaling-Kuala Kangsar": {
        "corridor_name": "Central-North Royal Heritage Transit Corridor",
        "mode": "KTM ETS Platinum & PLUS Expressway",
        "travel_time_mins": 110,
        "fare_rm": 38.00,
        "waypoints": [
            [3.1118, 101.5947], # Petaling Jaya
            [3.3500, 101.5500], # Rawang Bypass
            [3.8000, 101.4000], # Tanjung Malim
            [4.5975, 101.0901], # Ipoh
            [4.7738, 100.9416], # Kuala Kangsar Royal Town
        ]
    }
}


GROUNDED_DISTRICT_PROFILES = {
    ('Johor', 'Johor Bahru'): {
        'arch_heritage': 0.054, 'arch_nature': 0.058, 'arch_beach': 0.058, 'arch_food': 0.658, 'arch_urban': 0.958,
        'poi_tags': 'attraction | cinemas | community_mall | department_store | dining | dining_hub | family_entertainment | family_shopping | food_court | high_end_dining | hotel_resort | ice_skating | ikea_connected | indoor_climbing | luxury_retail | megamall | night_market | resort_hotel | retail | retail_center | rooftop_park | shopping_mall | skatepark | theme_park | urban_center | water_park'
    },
    ('Johor', 'Batu Pahat'): {
        'arch_heritage': 0.1, 'arch_nature': 0.05, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.88,
        'poi_tags': 'cinema | dining | district_hub | retail | shopping_mall'
    },
    ('Johor', 'Muar'): {
        'arch_heritage': 0.88, 'arch_nature': 0.3, 'arch_beach': 0.45, 'arch_food': 0.95, 'arch_urban': 0.65,
        'poi_tags': 'heritage_food | mee_bandung | otak_otak | pre_war_shophouses | royal_town | tanjung_emas_waterfront'
    },
    ('Johor', 'Kota Tinggi'): {
        'arch_heritage': 0.11, 'arch_nature': 0.546, 'arch_beach': 0.837, 'arch_food': 0.378, 'arch_urban': 0.312,
        'poi_tags': 'beach | beach_resort | coastal | coastal_walk | eco_nature | family_recreation | firefly_cruise | fisherman_museum | golf_course | luxury_hotel | mangrove | nature_resort | night_safari | river_cascade | river_tour | rock_formations | sandy_beach | seafood | swimming | water_sports | waterfall | waterpark'
    },
    ('Johor', 'Segamat'): {
        'arch_heritage': 0.3, 'arch_nature': 0.85, 'arch_beach': 0.05, 'arch_food': 0.8, 'arch_urban': 0.4,
        'poi_tags': 'agritourism | bekok_waterfall | durian_capital | labis_hot_springs | rainforest_streams'
    },
    ('Johor', 'Kluang'): {
        'arch_heritage': 0.4, 'arch_nature': 0.82, 'arch_beach': 0.05, 'arch_food': 0.88, 'arch_urban': 0.55,
        'poi_tags': 'gunung_lambak | kluang_rail_coffee | organic_agritourism | railway_heritage | uk_farm'
    },
    ('Johor', 'Pontian'): {
        'arch_heritage': 0.35, 'arch_nature': 0.9, 'arch_beach': 0.8, 'arch_food': 0.88, 'arch_urban': 0.4,
        'poi_tags': 'kukup_water_village | mangrove_ramsar | seafood_stilt_dining | southernmost_asia | tanjung_piai'
    },
    ('Johor', 'Mersing'): {
        'arch_heritage': 0.25, 'arch_nature': 0.85, 'arch_beach': 0.98, 'arch_food': 0.65, 'arch_urban': 0.35,
        'poi_tags': 'coastal_jetty | fresh_seafood | island_hopping | marine_park | pulau_rawa | tioman_gateway'
    },
    ('Johor', 'Kulai'): {
        'arch_heritage': 0.05, 'arch_nature': 0.05, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.98,
        'poi_tags': 'designer_brands | luxury_shopping | premium_outlet | retail_therapy'
    },
    ('Johor', 'Tangkak'): {
        'arch_heritage': 0.2, 'arch_nature': 0.95, 'arch_beach': 0.05, 'arch_food': 0.2, 'arch_urban': 0.1,
        'poi_tags': 'eco_nature | mountain_hiking | national_park | rainforest | waterfall'
    },
    ('Kedah', 'Kota Setar'): {
        'arch_heritage': 0.325, 'arch_nature': 0.121, 'arch_beach': 0.05, 'arch_food': 0.659, 'arch_urban': 0.827,
        'poi_tags': 'art_gallery | baju_melayu | cinema | city_center | city_icon | colonial_heritage | crystal_caves | cultural_hub | cultural_shopping | dodol | flagship_mall | geology | historical_architecture | kuih_bahulu | lifestyle_retail | limestone_hill | malay_delicacies | malay_food | menara_view | observation_deck | paddy_panoramas | paintings | recreation | retail | revolving_restaurant | shopping_mall | supermarket | telecom_tower | traditional_market | urban_convenience'
    },
    ('Kedah', 'Langkawi'): {
        'arch_heritage': 0.318, 'arch_nature': 0.582, 'arch_beach': 0.693, 'arch_food': 0.628, 'arch_urban': 0.632,
        'poi_tags': 'beaches | beachfront_dining | burnt_rice | craft_market | cultural_heritage | duty_free | duty_free_chocolate | duty_free_island | eagle_monument | ferry_terminal | folk_history | folklore_legend | geopark | gunung_machinchang | historical_shrine | iconic_landmark | islands | kitchenware | kuah_town | kuah_waterfront | mangroves | mountain_views | natural_pools | nightlife | panoramic_geopark | photo_spot | prime_beach | rainforest_trekking | retail_shopping | seven_wells | siamese_history | skybridge | skycab | souvenirs | traditional_village | unesco_global_geopark | water_sports | waterfall'
    },
    ('Kedah', 'Kuala Muda'): {
        'arch_heritage': 0.279, 'arch_nature': 0.345, 'arch_beach': 0.631, 'arch_food': 0.717, 'arch_urban': 0.42,
        'poi_tags': 'belacan | coastal | community_retail | dried_seafood | family_leisure | ferry_to_tanjung_dawai | fishing_village | ikan_bilis | jetty | mainland_beach | picnic | seafood | shopping_mall | sungai_petani'
    },
    ('Kedah', 'Baling'): {
        'arch_heritage': 0.25, 'arch_nature': 0.92, 'arch_beach': 0.05, 'arch_food': 0.5, 'arch_urban': 0.2,
        'poi_tags': 'caving | eco_adventure | gunung_baling | limestone_hiking | ulu_legong_hot_springs'
    },
    ('Kedah', 'Kulim'): {
        'arch_heritage': 0.25, 'arch_nature': 0.88, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.7,
        'poi_tags': 'canopy_walk | hi_tech_park | rainforest_trekking | sedim_river_rapids | tree_top_walk'
    },
    ('Kedah', 'Kubang Pasu'): {
        'arch_heritage': 0.4, 'arch_nature': 0.5, 'arch_beach': 0.05, 'arch_food': 0.65, 'arch_urban': 0.6,
        'poi_tags': 'border_trade | bukit_kayu_hitam | duty_free | golf_resort | tasik_darulaman'
    },
    ('Kedah', 'Yan'): {
        'arch_heritage': 0.45, 'arch_nature': 0.96, 'arch_beach': 0.25, 'arch_food': 0.35, 'arch_urban': 0.2,
        'poi_tags': 'cool_climate | historical_beacon | mountain_peak | panoramic_views | resort'
    },
    ('Kedah', 'Pendang'): {
        'arch_heritage': 0.3, 'arch_nature': 0.6, 'arch_beach': 0.05, 'arch_food': 0.65, 'arch_urban': 0.25,
        'poi_tags': 'local_delicacies | paddy_expanse | rural_homestays | traditional_farming'
    },
    ('Kedah', 'Padang Terap'): {
        'arch_heritage': 0.2, 'arch_nature': 0.92, 'arch_beach': 0.05, 'arch_food': 0.45, 'arch_urban': 0.15,
        'poi_tags': 'angler_haven | lake_resort | pedu_lake | rainforest_catchment | wilderness'
    },
    ('Kedah', 'Sik'): {
        'arch_heritage': 0.2, 'arch_nature': 0.9, 'arch_beach': 0.05, 'arch_food': 0.65, 'arch_urban': 0.15,
        'poi_tags': 'beris_lake | lata_mengkuang | rural_agritourism | tasik_beris_vineyard | waterfall'
    },
    ('Kelantan', 'Kota Bharu'): {
        'arch_heritage': 0.401, 'arch_nature': 0.113, 'arch_beach': 0.17, 'arch_food': 0.78, 'arch_urban': 0.75,
        'poi_tags': 'apparel | batik_songket | batik_workshops | bowling | casuarina_beach | celup_tepung | central_market | cinema | city_mall | coastal_breeze | coastal_shoreline | family_dining | fruits | iconic_heritage | kite_flying | kuih_akok | local_delicacies | midnight_shopping | nasi_kerabu | night_market | pacific_hypermarket | regional_mall | retail | riverfront | street_food | sunset_spot | supermarket | traditional_fishing'
    },
    ('Kelantan', 'Bachok'): {
        'arch_heritage': 0.239, 'arch_nature': 0.522, 'arch_beach': 0.943, 'arch_food': 0.616, 'arch_urban': 0.228,
        'poi_tags': 'camping_ground | coastal_pine | coastal_promenade | dune_beach | fishing_boats | food_stalls | melody_beach | picnic_park | sea_breeze'
    },
    ('Kelantan', 'Pasir Mas'): {
        'arch_heritage': 0.737, 'arch_nature': 0.05, 'arch_beach': 0.05, 'arch_food': 0.481, 'arch_urban': 0.654,
        'poi_tags': 'border_trade | clothing | cultural_landmark | district_mosque | duty_free | islamic_architecture | kitchenware | spiritual_tourism | thai_goods'
    },
    ('Kelantan', 'Pasir Puteh'): {
        'arch_heritage': 0.35, 'arch_nature': 0.65, 'arch_beach': 0.88, 'arch_food': 0.7, 'arch_urban': 0.25,
        'poi_tags': 'coastal_casuarinas | jeram_pasu | pantai_bisikan_bayu | seafood | waterfall_picnic'
    },
    ('Kelantan', 'Tanah Merah'): {
        'arch_heritage': 0.3, 'arch_nature': 0.65, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.3,
        'poi_tags': 'bukit_panau | kusial_bridge | local_crafts | river_crossing'
    },
    ('Kelantan', 'Tumpat'): {
        'arch_heritage': 0.95, 'arch_nature': 0.4, 'arch_beach': 0.7, 'arch_food': 0.85, 'arch_urban': 0.4,
        'poi_tags': 'border_duty_free | pengkalan_kubor | sitting_buddha | sleeping_buddha | wat_photivihan'
    },
    ('Kelantan', 'Machang'): {
        'arch_heritage': 0.3, 'arch_nature': 0.8, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.25,
        'poi_tags': 'hot_springs | hutan_lipur_bukit_bakar | rural_bazaar | waterfall'
    },
    ('Kelantan', 'Gua Musang'): {
        'arch_heritage': 0.45, 'arch_nature': 0.96, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.25,
        'poi_tags': 'gua_madu | gunung_stong | jungle_railway | limestone_caves | waterfall_trekking'
    },
    ('Kelantan', 'Jeli'): {
        'arch_heritage': 0.2, 'arch_nature': 0.92, 'arch_beach': 0.05, 'arch_food': 0.45, 'arch_urban': 0.2,
        'poi_tags': 'east_west_highway | geological_monument | gunung_reng | lata_janggut | river_cascades'
    },
    ('Kelantan', 'Kuala Krai'): {
        'arch_heritage': 0.5, 'arch_nature': 0.85, 'arch_beach': 0.05, 'arch_food': 0.55, 'arch_urban': 0.25,
        'poi_tags': 'historical_river_port | lata_rek | nature_pools | railway_heritage | tangga_krai'
    },
    ('Melaka', 'Melaka Tengah'): {
        'arch_heritage': 0.94, 'arch_nature': 0.229, 'arch_beach': 0.323, 'arch_food': 0.88, 'arch_urban': 0.761,
        'poi_tags': 'antique_shops | ayer_keroh | biodiversity | botanical_gardens | bustel | butterfly_sanctuary | canopy_walk | cendol | chicken_rice_balls | city_center | coastal_breeze | coconut_shake | cultural_artifacts | cultural_street | department_stores | dining | dining_precinct | dinosaur_park | eco_education | family_attraction | family_leisure | gsc_cinema | heritage_core_mall | historical_bridges | historical_exhibits | mega_retail | mural_art | nature_jogging | night_illuminations | night_market | night_safari | nyonya_food | pioneer_mall | regional_mall | reptile_park | retail_fashion | river_cruise | royal_customs | sand_dunes | souvenirs | sunset_spot | traditional_wedding | unesco_adjacent | unesco_scenery | unesco_world_heritage | wildlife_conservation | zoological_park'
    },
    ('Melaka', 'Alor Gajah'): {
        'arch_heritage': 0.305, 'arch_nature': 0.379, 'arch_beach': 0.614, 'arch_food': 0.683, 'arch_urban': 0.466,
        'poi_tags': 'branded_retail | chalets_resort | dutch_windmill | family_getaway | fresh_seafood | golf_resort | ikan_bakar | lake_setting | open_air_outlet | safari_wonderland | swimming_beach | turtle_hatchery | water_theme_park'
    },
    ('Melaka', 'Jasin'): {
        'arch_heritage': 0.45, 'arch_nature': 0.8, 'arch_beach': 0.15, 'arch_food': 0.65, 'arch_urban': 0.35,
        'poi_tags': 'agritourism_orchards | asahan_waterfall | gunung_ledang_trail | jasin_hot_springs | rural_retreat'
    },
    ('Negeri Sembilan', 'Seremban'): {
        'arch_heritage': 0.135, 'arch_nature': 0.214, 'arch_beach': 0.05, 'arch_food': 0.603, 'arch_urban': 0.871,
        'poi_tags': 'cinema | civic_square | clock_tower | curtains_carpets | dining_precinct | family_dining | family_recreation | food_trucks | green_lung | islamic_architecture | jogging_track | kite_flying | lake_gardens | large_aeon_mall | night_activities | photo_destination | retail | retail_bazaar | retail_chains | seremban_gateway | shopping | spiritual_landmark | taj_mahal_inspired | textile_wholesale | wedding_decor'
    },
    ('Negeri Sembilan', 'Port Dickson'): {
        'arch_heritage': 0.265, 'arch_nature': 0.474, 'arch_beach': 0.929, 'arch_food': 0.637, 'arch_urban': 0.48,
        'poi_tags': 'ancient_history | archaeology | army_museum | banana_boat | beach_strip | breeze | coastal | family_camping | jet_ski | keramat_ujang_pasir | mangrove_island | megalithic_stones | military_history | picnic | popular_beach | promenade | recreational_park | resort_hotels | retail_shops | seafood | seaside_dining | souvenir_stalls | sunset_plaza | tanks_aircraft | underground_tunnel | water_sports | wooden_bridge'
    },
    ('Negeri Sembilan', 'Jempol'): {
        'arch_heritage': 0.3, 'arch_nature': 0.65, 'arch_beach': 0.05, 'arch_food': 0.55, 'arch_urban': 0.3,
        'poi_tags': 'agricultural_settlements | felda_agritourism | serting_plains | tasik_bera_boundary'
    },
    ('Negeri Sembilan', 'Kuala Pilah'): {
        'arch_heritage': 0.95, 'arch_nature': 0.75, 'arch_beach': 0.05, 'arch_food': 0.8, 'arch_urban': 0.35,
        'poi_tags': 'istana_lama_seri_menanti | minangkabau_royal_heritage | traditional_roofs | ulu_bendul_recreation'
    },
    ('Negeri Sembilan', 'Rembau'): {
        'arch_heritage': 0.7, 'arch_nature': 0.75, 'arch_beach': 0.05, 'arch_food': 0.7, 'arch_urban': 0.3,
        'poi_tags': 'adat_perpatih | gunung_rembau | minangkabau_homestays | pedas_hot_springs'
    },
    ('Negeri Sembilan', 'Tampin'): {
        'arch_heritage': 0.3, 'arch_nature': 0.4, 'arch_beach': 0.05, 'arch_food': 0.55, 'arch_urban': 0.7,
        'poi_tags': 'community_hub | extreme_park | gunung_tampin_backdrop | town_square'
    },
    ('Negeri Sembilan', 'Jelebu'): {
        'arch_heritage': 0.3, 'arch_nature': 0.94, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.2,
        'poi_tags': 'jeram_toi_waterfall | kenaboi_state_park | pineapple_agritourism | rainforest_wilderness'
    },
    ('Pahang', 'Kuantan'): {
        'arch_heritage': 0.13, 'arch_nature': 0.284, 'arch_beach': 0.36, 'arch_food': 0.672, 'arch_urban': 0.779,
        'poi_tags': 'cinema | community_retail | dining | dining_precinct | downtown_kuantan | family_entertainment | flagship_mall | forest_resort | iconic_beach | it_shops | kuantan_city_center | local_dining | mcdonalds_seaside | mining_heritage | modern_mall | monkeys | multiplex_cinema | night_promenade | retail_chains | retail_fashion | rocky_cape | safari_park | sea_of_clouds | stairway_trek | sunrise_hike | water_park'
    },
    ('Pahang', 'Bentong'): {
        'arch_heritage': 0.125, 'arch_nature': 0.651, 'arch_beach': 0.05, 'arch_food': 0.801, 'arch_urban': 0.975,
        'poi_tags': 'cable_car | cable_car_station | casino_resort | cool_mountain | designer_fashion | entertainment | mid_hill | outdoor_outlet | skyworlds_theme_park'
    },
    ('Pahang', 'Cameron Highlands'): {
        'arch_heritage': 0.38, 'arch_nature': 0.967, 'arch_beach': 0.05, 'arch_food': 0.86, 'arch_urban': 0.295,
        'poi_tags': 'agritourism | cool_climate | fresh_produce | hydroponic_farming | mossy_forest | strawberry_farms | strawberry_picking | tea_plantations | waffles_jam'
    },
    ('Pahang', 'Rompin'): {
        'arch_heritage': 0.3, 'arch_nature': 0.95, 'arch_beach': 0.98, 'arch_food': 0.8, 'arch_urban': 0.3,
        'poi_tags': 'coastal_beaches | dive_resorts | endau_rompin_national_park | pulau_tioman_gateway | udang_galah'
    },
    ('Pahang', 'Temerloh'): {
        'arch_heritage': 0.5, 'arch_nature': 0.75, 'arch_beach': 0.05, 'arch_food': 0.98, 'arch_urban': 0.55,
        'poi_tags': 'kuala_gandah_elephants | patin_tempoyak_capital | pekan_sehari_temerloh | peninsular_centerpoint'
    },
    ('Pahang', 'Lipis'): {
        'arch_heritage': 0.92, 'arch_nature': 0.85, 'arch_beach': 0.05, 'arch_food': 0.65, 'arch_urban': 0.3,
        'poi_tags': 'british_colonial_trail | clifford_school | jungle_railway | kenong_rimba_park | kuala_lipis_heritage'
    },
    ('Pahang', 'Raub'): {
        'arch_heritage': 0.65, 'arch_nature': 0.9, 'arch_beach': 0.05, 'arch_food': 0.98, 'arch_urban': 0.45,
        'poi_tags': 'birdwatching | colonial_bungalows | frasers_hill_foothills | mining_heritage | musang_king_durian'
    },
    ('Pahang', 'Jerantut'): {
        'arch_heritage': 0.35, 'arch_nature': 0.99, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.25,
        'poi_tags': 'canopy_walk | kuala_tahan | oldest_rainforest | orang_asli | tahan_river | taman_negara'
    },
    ('Pahang', 'Pekan'): {
        'arch_heritage': 0.96, 'arch_nature': 0.4, 'arch_beach': 0.65, 'arch_food': 0.75, 'arch_urban': 0.5,
        'poi_tags': 'muzium_sultan_abu_bakar | polo_club | riverfront | royal_capital_of_pahang | tenun_pahang_diraja'
    },
    ('Pahang', 'Maran'): {
        'arch_heritage': 0.5, 'arch_nature': 0.82, 'arch_beach': 0.05, 'arch_food': 0.55, 'arch_urban': 0.25,
        'poi_tags': 'lubuk_yu_waterfall | recreational_forest | sri_marathandavar_temple | tasik_chini_gateway'
    },
    ('Pahang', 'Bera'): {
        'arch_heritage': 0.45, 'arch_nature': 0.94, 'arch_beach': 0.05, 'arch_food': 0.5, 'arch_urban': 0.2,
        'poi_tags': 'angler_expeditions | peat_swamp_forest | ramsar_wetland | semelai_culture | tasik_bera'
    },
    ('Pulau Pinang', 'Timur Laut'): {
        'arch_heritage': 0.91, 'arch_nature': 0.398, 'arch_beach': 0.521, 'arch_food': 0.89, 'arch_urban': 0.785,
        'poi_tags': 'beachfront_dining | bookxcess | cinema | cinemas | cloud9_skywalk | cny_illuminations | colonial_bungalows | cool_breeze | dining | downtown | downtown_george_town | escape_theme_park | flagship_mall | funicular_train | guanyin_bronze_statue | gurney_drive | habitat_rainforest | hawker_adjacent | high_end_retail | historic_monastery | it_gadgets | komtar_connected | luxury_fashion | monkey_beach | night_market | observation_deck | pagoda_of_ten_thousand_buddhas | penang_national_park | rainbow_skywalk | resort_hotels | retail | seafront_promenade | st_joseph_novitiate | sunset_bars | tallest_building_in_penang | theme_park | turtle_beach | unesco_biosphere | value_shopping | water_sports'
    },
    ('Pulau Pinang', 'Barat Daya'): {
        'arch_heritage': 0.1, 'arch_nature': 0.05, 'arch_beach': 0.6, 'arch_food': 0.85, 'arch_urban': 0.98,
        'poi_tags': 'coastal_view | largest_mall_in_penang | retail_chains | seafront_dining'
    },
    ('Pulau Pinang', 'Seberang Perai Tengah'): {
        'arch_heritage': 0.1, 'arch_nature': 0.05, 'arch_beach': 0.05, 'arch_food': 0.8, 'arch_urban': 0.96,
        'poi_tags': 'cinema | convention_center | expanded_mall | mainland_penang | seberang_jaya'
    },
    ('Pulau Pinang', 'Seberang Perai Utara'): {
        'arch_heritage': 0.65, 'arch_nature': 0.45, 'arch_beach': 0.8, 'arch_food': 0.88, 'arch_urban': 0.75,
        'poi_tags': 'butterworth_art_walk | container_art | pantai_bersih_seafood | penang_ferry | robina_eco_park'
    },
    ('Pulau Pinang', 'Seberang Perai Selatan'): {
        'arch_heritage': 0.5, 'arch_nature': 0.4, 'arch_beach': 0.75, 'arch_food': 0.92, 'arch_urban': 0.85,
        'poi_tags': 'bukit_tambun_seafood | design_village_outlet | ikea_batu_kawan | pulau_aman_mee_udang'
    },
    ('Perak', 'Kinta'): {
        'arch_heritage': 0.35, 'arch_nature': 0.29, 'arch_beach': 0.072, 'arch_food': 0.709, 'arch_urban': 0.865,
        'poi_tags': 'bowling | british_colonial | cascades | cinema | city_square | downtown_ipoh | family_dining | fashion_retail | flea_market | food_court | halal_hub | heritage_shophouses | heritage_trail | hot_springs | ipoh_white_coffee | lake_boat_ride | limestone_cliffs | limestone_towers | local_delicacies | mini_zoo | morning_bazaar | nature_trekking | neoclassical_architecture | night_bazaar | night_park | old_town | orang_asli_settlement | petting_zoo | retro_collectibles | river_cascade | street_art | street_food | tau_fu_fa | traditional_houses | vintage_antiques | water_theme_park | waterfall'
    },
    ('Perak', 'Larut & Matang'): {
        'arch_heritage': 0.756, 'arch_nature': 0.955, 'arch_beach': 0.05, 'arch_food': 0.419, 'arch_urban': 0.451,
        'poi_tags': 'colonial_heritage | conservation | first_public_garden | lake_gardens_adjacent | lake_reflections | night_safari | oldest_zoo_in_malaysia | rain_trees | swan_boats'
    },
    ('Perak', 'Manjung'): {
        'arch_heritage': 0.298, 'arch_nature': 0.543, 'arch_beach': 0.959, 'arch_food': 0.673, 'arch_urban': 0.444,
        'poi_tags': 'beaches | camping | coastal_walk | coral_islands | dried_fish_shops | dutch_fort | duty_free_island | family_picnic | fast_ferry_to_pangkor | fishing_villages | hornbill_feeding | island_beach | mainland_beach | man_made_island | naval_town | open_bay | pangkor_ferry_jetty | seafood | snorkeling | sunset_view | water_sports | waterfront_hotel | yacht_marina'
    },
    ('Perak', 'Kuala Kangsar'): {
        'arch_heritage': 0.35, 'arch_nature': 0.8, 'arch_beach': 0.05, 'arch_food': 0.98, 'arch_urban': 0.15,
        'poi_tags': 'agritourism | durian_orchard | fruit_tasting | research_station'
    },
    ('Perak', 'Batang Padang'): {
        'arch_heritage': 0.4, 'arch_nature': 0.92, 'arch_beach': 0.05, 'arch_food': 0.7, 'arch_urban': 0.45,
        'poi_tags': 'bamboo_crafts | lata_kinjang_waterfall | sungai_klah_hot_springs | tapah_highland_gateway'
    },
    ('Perak', 'Hilir Perak'): {
        'arch_heritage': 0.94, 'arch_nature': 0.3, 'arch_beach': 0.25, 'arch_food': 0.94, 'arch_urban': 0.6,
        'poi_tags': 'chee_cheong_fun | historical_monument | leaning_tower_teluk_intan | perak_river_cruises'
    },
    ('Perak', 'Kerian'): {
        'arch_heritage': 0.5, 'arch_nature': 0.75, 'arch_beach': 0.75, 'arch_food': 0.9, 'arch_urban': 0.45,
        'poi_tags': 'bukit_merah_laketown | kuala_kurau_fishermans_wharf | orangutan_island | seafood_mee_udang'
    },
    ('Perak', 'Kampar'): {
        'arch_heritage': 0.7, 'arch_nature': 0.92, 'arch_beach': 0.05, 'arch_food': 0.92, 'arch_urban': 0.65,
        'poi_tags': 'bread_curry_chicken | gua_tempurung | limestone_caves | tin_mining_history | utar_town'
    },
    ('Perak', 'Muallim'): {
        'arch_heritage': 0.6, 'arch_nature': 0.85, 'arch_beach': 0.05, 'arch_food': 0.85, 'arch_urban': 0.6,
        'poi_tags': 'educational_heritage | hiking_trails | strata_falls | tanjung_malim_railway | yik_mun_pau'
    },
    ('Perak', 'Perak Tengah'): {
        'arch_heritage': 0.92, 'arch_nature': 0.5, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.3,
        'poi_tags': 'dato_maharajalela_history | padi_plains | pasir_salak_complex | perak_river_culture'
    },
    ('Perak', 'Hulu Perak'): {
        'arch_heritage': 0.45, 'arch_nature': 0.99, 'arch_beach': 0.05, 'arch_food': 0.55, 'arch_urban': 0.2,
        'poi_tags': 'banding_island | hornbill_sanctuary | rafflesia | royal_belum_state_park | tasik_temenggor'
    },
    ('Perlis', 'Kangar'): {
        'arch_heritage': 0.324, 'arch_nature': 0.66, 'arch_beach': 0.417, 'arch_food': 0.48, 'arch_urban': 0.29,
        'poi_tags': 'birdwatching | camping_ground | chinese_bridges | coastal_dining | coral_mosaic | elevated_walkways | estuary_beach | ferry_to_langkawi | fishing_port | floating_mosque | grilled_seafood | ikan_bakar | jetty_view | jogging_circuit | kangar_town | ketam_bunga | laksa_kuala_perlis | local_picnic | lotus_lake | mangrove_coast | mangrove_setting | natural_pool | rainforest_reserve | riverfront_park | straits_of_malacca | sunset_spot | sunset_viewing | suspension_bridge | waterfall'
    },
    ('Perlis', 'Arau'): {
        'arch_heritage': 0.92, 'arch_nature': 0.4, 'arch_beach': 0.05, 'arch_food': 0.9, 'arch_urban': 0.55,
        'poi_tags': 'harum_manis_mango | istana_arau | railway_hub | royal_capital_of_perlis | royal_gallery'
    },
    ('Perlis', 'Padang Besar'): {
        'arch_heritage': 0.459, 'arch_nature': 0.341, 'arch_beach': 0.061, 'arch_food': 0.657, 'arch_urban': 0.667,
        'poi_tags': 'border_confectionery | border_lookout | border_shopping | clothing_bazaar | cloud_inversion | duty_free | guilin_of_malaysia | household_goods | migratory_birds | nakawan_range | pulut_mangga | resort_chalets | souvenirs | sunrise_panoramas | thai_goods | water_reservoir | wholesale_mart'
    },
    ('Selangor', 'Petaling'): {
        'arch_heritage': 0.101, 'arch_nature': 0.106, 'arch_beach': 0.149, 'arch_food': 0.783, 'arch_urban': 0.963,
        'poi_tags': 'al_fresco_dining | boutiques | central_icity_mall | city_of_digital_lights | convention_hall | egyptian_pyramid | exhibitions | family_mall | flowrider | green_building | home_fairs | ice_rink | indoor_climbing | ipc_mall | meatballs | mega_retail | mutiara_damansara | park_setting | pedestrian_mall | resort_city | retail_cluster | secret_garden | setia_alam | snooze_walk | snowalk | sunway_lagoon_waterpark | swedish_furniture | top_world_megamalls | two_wings | water_play | waterworld | weddings | weekend_flea_market'
    },
    ('Selangor', 'Sepang'): {
        'arch_heritage': 0.05, 'arch_nature': 0.15, 'arch_beach': 0.05, 'arch_food': 0.85, 'arch_urban': 0.99,
        'poi_tags': 'district21_adventure | largest_mall_malaysia | olympic_ice_rink | symphony_walk'
    },
    ('Selangor', 'Gombak'): {
        'arch_heritage': 0.497, 'arch_nature': 0.921, 'arch_beach': 0.05, 'arch_food': 0.352, 'arch_urban': 0.65,
        'poi_tags': '272_rainbow_steps | giant_panda_conservation | limestone_caves | lord_murugan_statue | national_zoo | thaipusam | tram_rides | wildlife_exhibits'
    },
    ('Selangor', 'Kuala Selangor'): {
        'arch_heritage': 0.88, 'arch_nature': 0.9, 'arch_beach': 0.85, 'arch_food': 0.95, 'arch_urban': 0.5,
        'poi_tags': 'bukit_malawati | fresh_seafood | kampung_kuantan_fireflies | silvered_leaf_monkeys | sky_mirror_sasaran'
    },
    ('Selangor', 'Hulu Langat'): {
        'arch_heritage': 0.45, 'arch_nature': 0.94, 'arch_beach': 0.05, 'arch_food': 0.92, 'arch_urban': 0.7,
        'poi_tags': 'broga_hill_hiking | forest_reserves | gabai_waterfall | satay_kajang | sungai_congkak'
    },
    ('Selangor', 'Klang'): {
        'arch_heritage': 0.2, 'arch_nature': 0.05, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.94,
        'poi_tags': 'apparel_accessories | b2b_b2c_retail | home_decor | wholesale_hub'
    },
    ('Selangor', 'Kuala Langat'): {
        'arch_heritage': 0.6, 'arch_nature': 0.5, 'arch_beach': 0.92, 'arch_food': 0.7, 'arch_urban': 0.35,
        'poi_tags': 'historical_landing_ww2 | ikan_bakar | kiting | mudflat_ecology | resort'
    },
    ('Selangor', 'Hulu Selangor'): {
        'arch_heritage': 0.75, 'arch_nature': 0.95, 'arch_beach': 0.05, 'arch_food': 0.7, 'arch_urban': 0.45,
        'poi_tags': 'chiling_waterfalls | frasers_foothills | hot_springs | kuala_kubu_bharu_heritage | white_water_rafting'
    },
    ('Selangor', 'Sabak Bernam'): {
        'arch_heritage': 0.5, 'arch_nature': 0.8, 'arch_beach': 0.8, 'arch_food': 0.98, 'arch_urban': 0.35,
        'poi_tags': 'mango_king | paddy_gallery | pantai_redang | seafood_dining | sekinchan_paddy_fields'
    },
    ('Terengganu', 'Kuala Terengganu'): {
        'arch_heritage': 0.585, 'arch_nature': 0.254, 'arch_beach': 0.638, 'arch_food': 0.816, 'arch_urban': 0.802,
        'poi_tags': 'batik_terengganu | brassware | chinatown_adjacent | coastal_park | drawbridge_adjacent | first_drawbridge_se_asia | food_stalls | horse_drawn_carriages | keropok_lekor | kite_flying | muara_terengganu | multiplex_cinema | night_lights | pacific_hypermarket | riverfront_promenade | seafront_mall | skybridge_gallery | songket | traditional_market'
    },
    ('Terengganu', 'Kuala Nerus'): {
        'arch_heritage': 0.303, 'arch_nature': 0.5, 'arch_beach': 0.949, 'arch_food': 0.791, 'arch_urban': 0.385,
        'poi_tags': 'airport_adjacent | bot_penambang | coastal_breeze | coastal_pine | drawbridge_panorama | fisherman_wharf | food_trucks | ikan_celup_tepung | keropok_lekor | miami_vibes | seafood | tetrapod_breakwater'
    },
    ('Terengganu', 'Kemaman'): {
        'arch_heritage': 0.4, 'arch_nature': 0.65, 'arch_beach': 0.95, 'arch_food': 0.92, 'arch_urban': 0.6,
        'poi_tags': 'cherating_border | hai_peng_kopitiam | pantai_kemasik | sata_otak_otak | turtle_sanctuary'
    },
    ('Terengganu', 'Besut'): {
        'arch_heritage': 0.45, 'arch_nature': 0.9, 'arch_beach': 0.99, 'arch_food': 0.8, 'arch_urban': 0.4,
        'poi_tags': 'air_panas_la | bukit_keluang_beach | marine_park | perhentian_islands | snorkeling_paradise'
    },
    ('Terengganu', 'Dungun'): {
        'arch_heritage': 0.25, 'arch_nature': 0.6, 'arch_beach': 0.95, 'arch_food': 0.6, 'arch_urban': 0.25,
        'poi_tags': 'casuarina | curving_beach | south_china_sea | uitm_dungun'
    },
    ('Terengganu', 'Marang'): {
        'arch_heritage': 0.6, 'arch_nature': 0.85, 'arch_beach': 0.98, 'arch_food': 0.8, 'arch_urban': 0.35,
        'poi_tags': 'marang_river_safari | pristine_beaches | pulau_kapas_gateway | traditional_boatbuilding'
    },
    ('Terengganu', 'Setiu'): {
        'arch_heritage': 0.75, 'arch_nature': 0.7, 'arch_beach': 0.98, 'arch_food': 0.9, 'arch_urban': 0.2,
        'poi_tags': 'celup_tepung | coconut_groves | redang_views | traditional_terengganu_houses'
    },
    ('Terengganu', 'Hulu Terengganu'): {
        'arch_heritage': 0.4, 'arch_nature': 0.99, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.25,
        'poi_tags': 'houseboats | kenyir_elephant_village | largest_manmade_lake | lasir_waterfall | tasik_kenyir'
    },
    ('Sabah', 'Kota Kinabalu'): {
        'arch_heritage': 0.157, 'arch_nature': 0.127, 'arch_beach': 0.396, 'arch_food': 0.803, 'arch_urban': 0.914,
        'poi_tags': 'autocity | cinema | coastal_park | computers_electronics | cultural_performances | dining_strip | downtown_kk | first_beach_stalls | fruit_juices | furniture | golf_course | gsc_cinema | hotels | hypermarket | island_ferry_jetty | it_hub_sabah | kk_times_square | kk_waterfront | local_crafts | luxury_brands | luxury_resort | metrojaya | pearl_jewellery | pioneer_complex | seafront_dining | ums_adjacent | value_shopping | world_class_sunset | yacht_marina'
    },
    ('Sabah', 'Ranau'): {
        'arch_heritage': 0.397, 'arch_nature': 0.936, 'arch_beach': 0.05, 'arch_food': 0.765, 'arch_urban': 0.236,
        'poi_tags': 'boulder_stream | camping | canopy_walk | crystal_clear_river | dusunic_culture | fresh_milk | gelato | handicraft_tamu | handicrafts | highland_vegetables | kinabalu_park_substation | kinabalu_viewpoint | kundasang_plateau | little_new_zealand | mount_kinabalu_view | mountain_view | mountain_vista | pineapples | polumpung_melangkap | rafflesia_blooming | sulphur_hot_springs | wooden_lookout_tower'
    },
    ('Sabah', 'Tawau'): {
        'arch_heritage': 0.7, 'arch_nature': 0.1, 'arch_beach': 0.35, 'arch_food': 0.95, 'arch_urban': 0.65,
        'poi_tags': 'amplang_crackers | dried_seafood | indonesian_batik | largest_indoor_market'
    },
    ('Sabah', 'Sandakan'): {
        'arch_heritage': 0.88, 'arch_nature': 0.98, 'arch_beach': 0.7, 'arch_food': 0.85, 'arch_urban': 0.65,
        'poi_tags': 'agnes_keith_house | bornea_sun_bears | labuk_bay_proboscis | memorial_park | sepilok_orangutans'
    },
    ('Sabah', 'Keningau'): {
        'arch_heritage': 0.6, 'arch_nature': 0.88, 'arch_beach': 0.05, 'arch_food': 0.65, 'arch_urban': 0.45,
        'poi_tags': 'crocker_range_park | inland_plateau | murut_cultural_centre | oath_stone | waterfalls'
    },
    ('Sabah', 'Penampang'): {
        'arch_heritage': 0.95, 'arch_nature': 0.65, 'arch_beach': 0.15, 'arch_food': 0.85, 'arch_urban': 0.7,
        'poi_tags': 'kaamatan_harvest | kadazandusun_heartland | monsopiad_cultural_village | st_michaels_church'
    },
    ('Sabah', 'Tuaran'): {
        'arch_heritage': 0.65, 'arch_nature': 0.75, 'arch_beach': 0.9, 'arch_food': 0.88, 'arch_urban': 0.55,
        'poi_tags': 'bajau_horsemen | dalit_bay_resort | rumah_terbalik | tamu_tuaran | tuaran_mee'
    },
    ('Sabah', 'Kudat'): {
        'arch_heritage': 0.75, 'arch_nature': 0.85, 'arch_beach': 0.98, 'arch_food': 0.7, 'arch_urban': 0.35,
        'poi_tags': 'gong_making_village | rungus_longhouses | simpang_mengayau | tip_of_borneo | virgin_beaches'
    },
    ('Sabah', 'Lahad Datu'): {
        'arch_heritage': 0.35, 'arch_nature': 0.99, 'arch_beach': 0.65, 'arch_food': 0.5, 'arch_urban': 0.35,
        'poi_tags': 'borneo_pygmy_elephants | danum_valley_conservation | primary_rainforest | tabin_wildlife_reserve'
    },
    ('Sabah', 'Semporna'): {
        'arch_heritage': 0.391, 'arch_nature': 0.969, 'arch_beach': 0.98, 'arch_food': 0.369, 'arch_urban': 0.135,
        'poi_tags': 'bajau_laut_stilt_villages | coral_reefs | epic_viewpoint | extinct_volcano_crater | giant_clam_hatchery | muck_diving | turquoise_lagoon | water_bungalows'
    },
    ('Sarawak', 'Kuching'): {
        'arch_heritage': 0.339, 'arch_nature': 0.161, 'arch_beach': 0.106, 'arch_food': 0.838, 'arch_urban': 0.939,
        'poi_tags': 'colonial_square | darul_hana_bridge | department_stores | dining_court | family_leisure | fashion | foodbazaar | fort_margherita | imax_cinema | international_retail | kek_lapis | largest_mall_sarawak | premier_mall | regional_mall | sarawak_river | supermarket | tapioca_dining'
    },
    ('Sarawak', 'Sibu'): {
        'arch_heritage': 0.457, 'arch_nature': 0.069, 'arch_beach': 0.05, 'arch_food': 0.805, 'arch_urban': 0.832,
        'poi_tags': 'base_jumping | central_sarawak_mall | civic_square | community_center | department_store | hypermarket | iban_produce | kampua_mee | largest_market_in_malaysia | live_poultry_newspaper | retail_chains | tallest_building_sibu'
    },
    ('Sarawak', 'Bintulu'): {
        'arch_heritage': 0.104, 'arch_nature': 0.072, 'arch_beach': 0.406, 'arch_food': 0.721, 'arch_urban': 0.898,
        'poi_tags': 'bbq_picnic | belacan_bintulu | casuarina_coast | coastal_breeze | commercial_hub | h&m | new_town | parkson | retail | seafront_lifestyle_mall | starbucks_reserve | supermarket'
    },
    ('Sarawak', 'Miri'): {
        'arch_heritage': 0.65, 'arch_nature': 0.99, 'arch_beach': 0.9, 'arch_food': 0.85, 'arch_urban': 0.85,
        'poi_tags': 'grand_old_lady | gunung_mulu_national_park | luak_esplanade | niah_caves | unesco_world_heritage'
    },
    ('Sarawak', 'Sri Aman'): {
        'arch_heritage': 0.8, 'arch_nature': 0.75, 'arch_beach': 0.05, 'arch_food': 0.65, 'arch_urban': 0.35,
        'poi_tags': 'batang_lupar_tidal_bore | benak_festival | fort_alice | peaceful_town | river_heritage'
    },
    ('Sarawak', 'Samarahan'): {
        'arch_heritage': 0.35, 'arch_nature': 0.6, 'arch_beach': 0.3, 'arch_food': 0.7, 'arch_urban': 0.65,
        'poi_tags': 'asajaya_agritourism | batang_sadong_bridge | kota_samarahan_education | riverine_coast'
    },
    ('Sarawak', 'Limbang'): {
        'arch_heritage': 0.5, 'arch_nature': 0.85, 'arch_beach': 0.2, 'arch_food': 0.65, 'arch_urban': 0.4,
        'poi_tags': 'brunei_border_transit | bukit_mas_park | mud_volcanoes | tamu_limbang'
    },
    ('Sarawak', 'Mukah'): {
        'arch_heritage': 0.95, 'arch_nature': 0.6, 'arch_beach': 0.85, 'arch_food': 0.92, 'arch_urban': 0.35,
        'poi_tags': 'kaul_festival | lamin_dana | melanau_heritage | sago_agritourism | traditional_tall_houses'
    },
    ('Sarawak', 'Sarikei'): {
        'arch_heritage': 0.4, 'arch_nature': 0.7, 'arch_beach': 0.15, 'arch_food': 0.9, 'arch_urban': 0.4,
        'poi_tags': 'pineapple_capital | rejang_riverfront | sarikei_oranges | sebangkoi_country_park'
    },
    ('Sarawak', 'Kapit'): {
        'arch_heritage': 0.92, 'arch_nature': 0.92, 'arch_beach': 0.05, 'arch_food': 0.6, 'arch_urban': 0.3,
        'poi_tags': 'fort_sylvia | iban_longhouses | pelagus_rapids | rainforest_interior | rejang_river_express'
    },
    ('W.P. Kuala Lumpur', 'W.P. Kuala Lumpur'): {
        'arch_heritage': 0.245, 'arch_nature': 0.084, 'arch_beach': 0.05, 'arch_food': 0.817, 'arch_urban': 0.97,
        'poi_tags': 'aquaria_klcc | art_deco_heritage | bargain_shopping | batik_crafting | bukit_bintang | chinatown | cinemas | crystal_fountain | department_store | dining | esports | exhibition_centre | exhibition_halls | family_mall | festive_shopping | food_hall | heritage_temples | indoor_arena | indoor_roller_coaster | international_exhibitions | jalan_tuanku_abdul_rahman | japanese_retail | kasturi_walk | klcc_park | lifestyle_mall | lifestyle_shopping | lrt_ktm_transit | luxury_couture | luxury_shopping | malaysian_handicrafts | mega_concerts | megamall | mice_tourism | monorail_connected | national_convention_hub | national_sports_complex | night_market | petronas_twin_towers | plenary_hall | political_heritage | pudu_jail_heritage_gate | retail_bazaar | retail_diversity | rooftop_garden | skybridge | street_food | the_gardens | tokyo_street | wtc_connected'
    },
    ('W.P. Putrajaya', 'W.P. Putrajaya'): {
        'arch_heritage': 0.496, 'arch_nature': 0.381, 'arch_beach': 0.073, 'arch_food': 0.554, 'arch_urban': 0.879,
        'poi_tags': 'architectural_bridges | bmx_trail | botanical_collections | boulevard | canoeing_kayaking | ceremonial_plaza | commercial_offices | cycling | dining | english_sunflower_gardens | european_pine_trees | extreme_sports | family_dining | flamingos_birdwatching | gsc_cinema | hilltop_convention_centre | hilltop_jogging | iconic_symbol | indoor_rock_climbing | islamic_architecture | kite_flying | lake_boat_tour | lake_setting | lakefront_dining | lakefront_mosque | lakefront_sanctuary | lakefront_walk | lookout_tower | man_made_wetland | masjid_putra_adjacent | moroccan_pavilion | national_day_parade | nature_education | night_lights | palace_of_justice | panoramic_city_view | pending_perak_architecture | perahu_dondang_sayang | picnic_haven | pink_granite | presint_4 | prime_ministers_office | retail | shopping_mall | skate_park | spiritual_modern | state_flags | steel_architecture | sunset_cruise | swimming_pool | unesco_heritage_tree_park | watersports | wire_mesh_screens'
    },
    ('W.P. Labuan', 'W.P. Labuan'): {
        'arch_heritage': 0.426, 'arch_nature': 0.443, 'arch_beach': 0.707, 'arch_food': 0.561, 'arch_urban': 0.445,
        'poi_tags': 'anzac_day | bbq_picnics | brunai_malay_culture | chalets | cinema | civic_square | coastal_breeze | coconut_drink | colonial_flame_tree | commonwealth_war_graves | convention_hall | coral_snorkeling | domed_aviary | duty_free_mall | family_leisure | fishing_spots | food_stalls | football_field | fresh_fish | homestays | hornbills | kuraman_island | longest_beach | marine_museum | national_day_events | nature_walk | offshore_financial_hub | peace_park | peace_park_adjacent | pine_trees | promenade | rock_formations | rustic_beach | rusukan_islands | satay | sea_challenge | sea_turtle_nesting | seaside_market | shipwreck_diving | stilt_houses | sunset_spot | surrender_point | tropical_birds | water_sports_complex | water_village | wooden_boardwalks | ww2_history'
    },
}


def build_unified_database():
    logger.info("=== Building Unified Master Destination Database ===")

    # 1. Load multi-year extracted DTS districts
    dts_dist_path = PROCESSED_DIR / "dts_all_years_districts.csv"
    if dts_dist_path.exists():
        df_dts_dist = pd.read_csv(dts_dist_path)
    else:
        df_dts_dist = pd.DataFrame()

    # 2. Load latest state economic profiles (HIES & GDP)
    econ_path = PROCESSED_DIR / "state_economic_profile.csv"
    if econ_path.exists():
        df_econ = pd.read_csv(econ_path).set_index("state").to_dict("index")
    else:
        df_econ = {}

    # 3. Load MOTAC AOR timeseries
    aor_path = PROCESSED_DIR / "motac_hotel_occupancy_aor_timeseries.csv"
    state_aor = {}
    if aor_path.exists():
        df_aor = pd.read_csv(aor_path)
        # Calculate recent mean occupancy rate by state
        recent_aor = df_aor[df_aor["year"] >= 2023].groupby("state")["average_occupancy_rate_pct"].mean()
        state_aor = recent_aor.to_dict()

    # 4. Load MOTAC Hotel Inventory timeseries
    inv_path = PROCESSED_DIR / "motac_hotel_inventory_timeseries.csv"
    state_rooms = {}
    if inv_path.exists():
        df_inv = pd.read_csv(inv_path)
        recent_rooms = df_inv[df_inv["year"] >= 2023].groupby("state")["rooms_count"].mean()
        state_rooms = recent_rooms.to_dict()

    # 5. Build Comprehensive Destination Table
    # Pre-map all 110 districts to states
    dist_to_state = {}
    for district in DISTRICT_COORDS:
        if district in ["Johor Bahru", "Batu Pahat", "Muar", "Kota Tinggi", "Segamat", "Kluang", "Pontian", "Mersing", "Kulai", "Tangkak"]:
            dist_to_state[district] = "Johor"
        elif district in ["Kota Setar", "Langkawi", "Kuala Muda", "Baling", "Kulim", "Kubang Pasu", "Yan", "Pendang", "Padang Terap", "Sik"]:
            dist_to_state[district] = "Kedah"
        elif district in ["Kota Bharu", "Bachok", "Pasir Mas", "Pasir Puteh", "Tanah Merah", "Tumpat", "Machang", "Gua Musang", "Jeli", "Kuala Krai"]:
            dist_to_state[district] = "Kelantan"
        elif district in ["Melaka Tengah", "Alor Gajah", "Jasin"]:
            dist_to_state[district] = "Melaka"
        elif district in ["Seremban", "Port Dickson", "Jempol", "Kuala Pilah", "Rembau", "Tampin", "Jelebu"]:
            dist_to_state[district] = "Negeri Sembilan"
        elif district in ["Kuantan", "Bentong", "Cameron Highlands", "Rompin", "Temerloh", "Lipis", "Raub", "Jerantut", "Pekan", "Maran", "Bera"]:
            dist_to_state[district] = "Pahang"
        elif district in ["Timur Laut", "Barat Daya", "Seberang Perai Tengah", "Seberang Perai Utara", "Seberang Perai Selatan"]:
            dist_to_state[district] = "Pulau Pinang"
        elif district in ["Kinta", "Larut & Matang", "Manjung", "Kuala Kangsar", "Batang Padang", "Hilir Perak", "Kerian", "Kampar", "Muallim", "Perak Tengah", "Hulu Perak"]:
            dist_to_state[district] = "Perak"
        elif district in ["Kangar", "Arau", "Padang Besar"]:
            dist_to_state[district] = "Perlis"
        elif district in ["Petaling", "Sepang", "Gombak", "Kuala Selangor", "Hulu Langat", "Klang", "Kuala Langat", "Hulu Selangor", "Sabak Bernam"]:
            dist_to_state[district] = "Selangor"
        elif district in ["Kuala Terengganu", "Kuala Nerus", "Kemaman", "Besut", "Dungun", "Marang", "Setiu", "Hulu Terengganu"]:
            dist_to_state[district] = "Terengganu"
        elif district in ["Kota Kinabalu", "Ranau", "Tawau", "Sandakan", "Keningau", "Penampang", "Tuaran", "Kudat", "Lahad Datu", "Semporna"]:
            dist_to_state[district] = "Sabah"
        elif district in ["Kuching", "Sibu", "Bintulu", "Miri", "Sri Aman", "Samarahan", "Limbang", "Mukah", "Sarikei", "Kapit"]:
            dist_to_state[district] = "Sarawak"
        elif "Kuala Lumpur" in district:
            dist_to_state[district] = "W.P. Kuala Lumpur"
        elif "Putrajaya" in district:
            dist_to_state[district] = "W.P. Putrajaya"
        elif "Labuan" in district:
            dist_to_state[district] = "W.P. Labuan"
        else:
            dist_to_state[district] = "Selangor"

    # Tier definitions for realistic capacity and room stock distribution
    TIER_1_METRO_OR_HOTSPOT = {
        "W.P. Kuala Lumpur", "Petaling", "Johor Bahru", "Timur Laut", "Kota Kinabalu", "Kuching",
        "Kinta", "Melaka Tengah", "Cameron Highlands", "Langkawi", "Kuantan", "Bentong", "Kota Bharu", "Kuala Terengganu"
    }

    TIER_2_REGIONAL_CITIES = {
        "Miri", "Sibu", "Bintulu", "Sandakan", "Tawau", "Ranau", "Semporna", "Batu Pahat", "Muar",
        "Port Dickson", "Seremban", "Larut & Matang", "Manjung", "Klang", "Sepang", "Gombak", "Seberang Perai Tengah",
        "Kota Setar", "Kuala Muda", "Kemaman", "Besut", "Dungun"
    }

    TIER_3_INTERMEDIATE = {
        "Samarahan", "Kampar", "Kuala Kangsar", "Kulim", "Kubang Pasu", "Hulu Langat", "Kuala Selangor",
        "Alor Gajah", "Temerloh", "Rompin", "Kluang", "Kota Tinggi", "Pontian", "Bachok", "Pasir Mas",
        "Penampang", "Tuaran", "Barat Daya", "Seberang Perai Utara", "W.P. Putrajaya", "W.P. Labuan", "Kangar"
    }

    # Proportional room allocation per district based on authentic state inventory
    district_rooms = {}
    for st in set(dist_to_state.values()):
        dists_in_st = [d for d, s in dist_to_state.items() if s == st]
        st_rooms_total = state_rooms.get(st, 15000.0)
        weights = {}
        for d in dists_in_st:
            p = GROUNDED_DISTRICT_PROFILES.get((st, d), {})
            n_poi = len(p.get("poi_tags", "").split("|"))
            if d in TIER_1_METRO_OR_HOTSPOT:
                tier_mult = 2.5
            elif d in TIER_2_REGIONAL_CITIES:
                tier_mult = 1.55
            elif d in TIER_3_INTERMEDIATE:
                tier_mult = 1.0
            else:
                tier_mult = 0.65
            w = tier_mult * (1.0 + 0.8 * p.get("arch_urban", 0.2) + 0.6 * p.get("arch_heritage", 0.2) + 0.5 * p.get("arch_beach", 0.1) + 0.4 * p.get("arch_nature", 0.2)) * (1.0 + 0.08 * min(n_poi, 10))
            weights[d] = w
        tot_w = sum(weights.values())
        for d in dists_in_st:
            share = weights[d] / tot_w
            district_rooms[d] = max(350, int(round(st_rooms_total * share)))

    destinations = []
    idx = 1
    for district, (lat, lon) in DISTRICT_COORDS.items():
        state = dist_to_state[district]

        # Economic metrics
        econ = df_econ.get(state, {})
        poverty_rate = float(econ.get("poverty", 4.5))
        mean_income = float(econ.get("income_mean", 7500.0))
        median_income = float(econ.get("income_median", 6000.0))

        # MOTAC occupancy and room estimates
        aor_pct = float(state_aor.get(state, 52.0))
        tot_rooms = district_rooms.get(district, 1500)

        # Archetype derivation based on geographic and cultural traits
        name_lower = district.lower()
        is_beach = any(k in name_lower for k in ["pantai", "laut", "port", "kuala", "langkawi", "bachok", "mersing", "dungun", "semporna", "tioman", "kudat", "miri", "mukah"])
        is_urban = any(k in name_lower for k in ["bahru", "bharu", "kinta", "timur laut", "petaling", "kinabalu", "kuching", "tengah", "klang", "seremban", "kuala lumpur", "putrajaya", "sepang", "miri", "sibu", "bintulu", "sandakan", "tawau"])
        is_nature = any(k in name_lower for k in ["cameron", "ranau", "baling", "lipis", "hulu", "padang", "keningau", "jerantut", "gua musang", "taman negara", "kapit", "limbang"])
        is_heritage = any(k in name_lower for k in ["melaka", "timur laut", "kuala kangsar", "taiping", "muar", "pekan", "kuala lipis", "kangar", "george town", "kuching", "mukah", "sri aman"])

        # Grounded POI Archetypes & Verified Attractions derived from DTS 450+ panel
        profile = GROUNDED_DISTRICT_PROFILES.get((state, district))
        if profile:
            arch_heritage = float(np.clip(profile["arch_heritage"], 0.05, 0.99))
            arch_nature = float(np.clip(profile["arch_nature"], 0.05, 0.99))
            arch_beach = float(np.clip(profile["arch_beach"], 0.05, 0.99))
            arch_food = float(np.clip(profile["arch_food"], 0.05, 0.99))
            arch_urban = float(np.clip(profile["arch_urban"], 0.05, 0.99))
            poi_tags = profile["poi_tags"]
        else:
            arch_heritage = float(np.clip(0.88 if is_heritage else 0.35, 0.05, 0.99))
            arch_nature = float(np.clip(0.92 if is_nature else 0.38, 0.05, 0.99))
            arch_beach = float(np.clip(0.90 if is_beach else 0.05, 0.05, 0.99))
            arch_urban = float(np.clip(0.89 if is_urban else 0.32, 0.05, 0.99))
            arch_food = float(np.clip(0.92 if (is_urban or is_heritage) else 0.55, 0.05, 0.99))
            poi_tags = "regional_tourism | local_hospitality"

        # Dynamic Peak Demand Formulation
        if district in ["Cameron Highlands", "Melaka Tengah", "Langkawi", "Port Dickson"]:
            demand_ratio = 1.65
        elif district in TIER_1_METRO_OR_HOTSPOT:
            demand_ratio = 1.45
        elif district in TIER_2_REGIONAL_CITIES:
            demand_ratio = 1.30
        elif district in TIER_3_INTERMEDIATE:
            demand_ratio = 1.15
        else:
            demand_ratio = 0.95

        daily_demand_peak = float(tot_rooms * demand_ratio * (1.0 + 0.12 * arch_food + 0.08 * arch_beach))

        # Dynamic Carrying Capacity Formulation (Cifuentes & PLANMalaysia Multi-Subsystem)
        cc_accommodation = float(tot_rooms * 1.62)
        cc_transport = float(daily_demand_peak * (0.84 if ("cameron" in name_lower or "timur laut" in name_lower or district in ["Petaling", "Johor Bahru", "W.P. Kuala Lumpur"]) else 1.35))
        cc_attraction = float(daily_demand_peak * (0.86 if district in ["Melaka Tengah", "Timur Laut"] else 1.30))
        cc_water_waste = float(daily_demand_peak * (0.85 if ("melaka" in name_lower or "langkawi" in name_lower or district in ["Port Dickson", "Semporna"]) else 1.45))
        cc_ecology = float(daily_demand_peak * (0.75 if ("cameron" in name_lower or "ranau" in name_lower) else 2.30))
        cc_social = float(daily_demand_peak * (0.82 if ("timur laut" in name_lower or district in ["Melaka Tengah"]) else 1.55))

        subsystems = {
            "Accommodation Inventory": cc_accommodation,
            "Transport / Roadway Gridlock": cc_transport,
            "Attraction Core Space Turnover": cc_attraction,
            "Clean Water Buffer (SPAN Reserve)": cc_water_waste,
            "Ecological Slope & Forest KSAS Buffer": cc_ecology,
            "Local Resident Social Tolerance": cc_social
        }

        binding_constraint = min(subsystems, key=subsystems.get)
        sustainable_capacity = subsystems[binding_constraint]
        continuous_pressure = daily_demand_peak / max(1.0, sustainable_capacity)
        excess_demand = max(0.0, daily_demand_peak - sustainable_capacity)
        capacity_gap = (sustainable_capacity - daily_demand_peak) / max(1.0, sustainable_capacity)

        destinations.append({
            "destination_id": f"DST_{idx:03d}",
            "state_name": state,
            "district_name": district,
            "destination_name": district,
            "lat": lat,
            "lon": lon,
            "total_rooms": tot_rooms,
            "daily_demand_peak": round(daily_demand_peak),
            "cc_accommodation": round(cc_accommodation),
            "cc_transport": round(cc_transport),
            "cc_attraction": round(cc_attraction),
            "cc_water_waste": round(cc_water_waste),
            "cc_ecology": round(cc_ecology),
            "cc_social": round(cc_social),
            "binding_constraint": binding_constraint,
            "sustainable_capacity": round(sustainable_capacity),
            "continuous_pressure": round(continuous_pressure, 3),
            "excess_demand_daily": round(excess_demand),
            "capacity_gap": round(capacity_gap, 3),
            "poverty_rate": round(poverty_rate, 2),
            "mean_household_income": round(mean_income),
            "median_household_income": round(median_income),
            "latest_aor_pct": round(aor_pct, 1),
            "arch_heritage": round(arch_heritage, 3),
            "arch_nature": round(arch_nature, 3),
            "arch_beach": round(arch_beach, 3),
            "arch_food": round(arch_food, 3),
            "arch_urban": round(arch_urban, 3),
            "poi_tags": poi_tags,
            "transit_mode": "KTM ETS & Rail" if (lat < 6.0 and lon < 101.5 and lat > 3.0) else "Federal Road & Express Coach",
            # Authoritative PBT lookup (KPKT JKT/OSC grounded) — see src/etl/pbt_mapping.py.
            # Replaces the old generic "Majlis Bandaraya/Daerah {district}" template.
            "pbt_name": get_pbt_name(state, district)
        })
        idx += 1

    df_master = pd.DataFrame(destinations)
    df_master.to_csv(PROCESSED_DIR / "destinations_master.csv", index=False)
    logger.info("Saved destinations_master.csv (%d verified districts)", len(df_master))

    # Save corridor route geometries for animated map rendering
    with open(PROCESSED_DIR / "corridors_transit_routes.json", "w", encoding="utf-8") as f:
        json.dump(CORRIDOR_ROUTES, f, indent=2)
    logger.info("Saved corridors_transit_routes.json (%d detailed route polylines)", len(CORRIDOR_ROUTES))

    logger.info("=== Master Database Build Complete ===")


if __name__ == "__main__":
    build_unified_database()
