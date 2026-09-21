# DESTINASI: Technical & Econometric Appendix
### *Authoritative Mathematical Foundations, Econometric Derivations, and System Specifications*

This technical appendix contains the formal mathematical formulations, econometric derivations, geospatial architecture, and data engineering specifications for **DESTINASI** (Decision-Support System for Tourism Dispersal & Grassroots Economic Balancing).

For the layman-accessible project overview and user guide, refer to the main [README.md](README.md).

---

## Table of Contents
1. [Empirical Macroeconomic Derivations (DOSM I-O 2019–2021)](#1-empirical-macroeconomic-derivations-dosm-i-o-20192021)
   - [Leontief Open Model (Type I Output Multiplier)](#leontief-open-model-type-i-output-multiplier)
   - [Gross Value-Added / Real GDP Multiplier](#gross-value-added--real-gdp-multiplier)
   - [Closed-Household Model (Type II Multiplier)](#closed-household-model-type-ii-multiplier)
   - [Sector-Specific Multipliers vs. Composite Tourism Multiplier](#sector-specific-multipliers-vs-composite-tourism-multiplier)
   - [Capacity-Constrained Multiplier Damping](#capacity-constrained-multiplier-damping)
2. [Multi-Tier Carrying Capacity Framework (Cifuentes Model)](#2-multi-tier-carrying-capacity-framework-cifuentes-model)
   - [PCC, RCC, and ECC Formulations](#pcc-rcc-and-ecc-formulations)
   - [Multi-System Bottleneck Envelope](#multi-system-bottleneck-envelope)
3. [Spatial & Experiential Recommender Mathematics](#3-spatial--experiential-recommender-mathematics)
   - [5-Pillar Cosine Vector Archetype Similarity](#5-pillar-cosine-vector-archetype-similarity)
   - [Discrete Jaccard POI Set Similarity](#discrete-jaccard-poi-set-similarity)
   - [Multi-Criteria Weighted Sum Model (WSM)](#multi-criteria-weighted-sum-model-wsm)
4. [Predictive Machine Learning Formulation (Holiday Spike Radar)](#4-predictive-machine-learning-formulation-holiday-spike-radar)
   - [L2 Regularized Ridge Regression](#l2-regularized-ridge-regression)
   - [Logistic Classification & Risk Thresholds](#logistic-classification--risk-thresholds)
   - [Explainable AI (SHAP Waterfall Decomposition)](#explainable-ai-shap-waterfall-decomposition)
5. [Geospatial & 3D WebGL Visualization Architecture](#5-geospatial--3d-webgl-visualization-architecture)
   - [PyDeck Layer Stack & Coordinate Stratification](#pydeck-layer-stack--coordinate-stratification)
   - [PLANMalaysia WMS & Georeferenced Raster Draping](#planmalaysia-wms--georeferenced-raster-draping)
   - [High-Contrast Hazard Beacon Geometry](#high-contrast-hazard-beacon-geometry)
   - [Multi-Tier Transit Routing Engine](#multi-tier-transit-routing-engine)
6. [Data Lineage, Schema & Regulatory Provenance](#6-data-lineage-schema--regulatory-provenance)
   - [Statutory Data Sources Registry](#statutory-data-sources-registry)
   - [PPP KSAS Classification Schema (Jadual 4 & Jadual 7)](#ppp-ksas-classification-schema-jadual-4--jadual-7)
   - [Inbound Expenditure Matrix Taxonomy](#inbound-expenditure-matrix-taxonomy)

---

## 1. Empirical Macroeconomic Derivations (DOSM I-O 2019–2021)

All macroeconomic calculations in DESTINASI are derived directly from the official **Department of Statistics Malaysia (DOSM) Symmetric Input-Output Tables** for benchmark years 2019, 2020, and 2021 (124 commodity sectors, Table 6: *Jadual Input-Output Simetri Penawaran Tempatan pada Harga Asas*).

### Leontief Open Model (Type I Output Multiplier)

Let $Z$ be the $n \times n$ matrix of domestic intermediate transactions ($n = 124$), where $Z_{ij}$ denotes the value of commodity $i$ purchased as an intermediate input by sector $j$. Let $X$ be the $n \times 1$ vector of total gross output, where $X_j$ is the total production of sector $j$.

The direct input coefficients matrix $A$ is defined element-wise by:
$$A_{ij} = \frac{Z_{ij}}{X_j}, \quad \forall i, j \in \{1, \dots, n\}$$

In matrix notation, the economy-wide balance condition relating gross output $X$ to intermediate demand $A X$ and exogenous final demand $Y$ is:
$$X = A X + Y \implies (I - A) X = Y \implies X = (I - A)^{-1} Y = L Y$$

where:
- $I$ is the $n \times n$ identity matrix.
- $L = (I - A)^{-1}$ is the **Leontief Inverse Matrix** (total requirements matrix). Each element $L_{ij}$ represents the total dollar value of production in sector $i$ required directly and indirectly per unit of final demand delivered by sector $j$.

The **Type I Output Multiplier** for sector $j$, denoted $M_j^I$, is the column sum of the Leontief inverse:
$$M_j^I = \sum_{i=1}^{n} L_{ij} = \mathbf{1}^T L_{\bullet j}$$

### Gross Value-Added / Real GDP Multiplier

Gross output multipliers measure total turnover across all production stages and therefore double-count intermediate goods. To establish the true net contribution to Malaysian Gross Domestic Product (GDP), DESTINASI computes the **Type I Gross Value-Added (GVA) Multiplier**, $M_j^{VA}$.

Let $V_j$ be the primary input (Gross Value Added at basic prices: compensation of employees, gross operating surplus, other taxes less subsidies on production) generated in sector $j$. The direct value-added coefficient vector $v$ has elements:
$$v_j = \frac{V_j}{X_j}, \quad 0 < v_j < 1$$

The total economy-wide GDP contribution per RM 1.00 of exogenous final demand directed to sector $j$ is given by the inner product of the value-added coefficient vector with the $j$-th column of the Leontief inverse:
$$M_j^{VA} = \sum_{i=1}^{n} v_i L_{ij} = (v \cdot L)_j$$

By national accounting identity:
$$\sum_{i=1}^n v_i = 1 - \sum_{i=1}^n \sum_{k=1}^n A_{ki} \implies M_j^{VA} \le 1.00$$
for an open Leontief model without induced household closure.

### Closed-Household Model (Type II Multiplier)

To capture the **induced effect** (where direct and indirect labor compensation earned in tourism supply chains is re-spent on consumer goods and services), the open Leontief matrix is augmented by endogenizing the household sector.

Let:
- $w = [w_1, w_2, \dots, w_n]$ be the row vector of labor income coefficients, where $w_j = \frac{W_j}{X_j}$ and $W_j$ is employee compensation in sector $j$.
- $c = [c_1, c_2, \dots, c_n]^T$ be the column vector of household consumption shares, where $c_i = \frac{C_i}{\sum_{k=1}^n C_k}$ and $C_i$ is private household consumption of commodity $i$.

The augmented $(n+1) \times (n+1)$ coefficient matrix $A^*$ is:
$$A^* = \begin{pmatrix} A & c \\ w & 0 \end{pmatrix}$$

The **Type II Leontief Inverse** is:
$$L^* = (I - A^*)^{-1}$$

The **Type II Output Multiplier** for sector $j$ is:
$$M_j^{II} = \sum_{i=1}^{n} L_{ij}^*$$

### Sector-Specific Multipliers vs. Composite Tourism Multiplier

Tourism is not a single standard industrial classification (SIC) code, but an aggregate of diverse consumer-facing activities. Using expenditure weights $s_k$ mapped from DTS components (`data/processed/dts_expenditure_shares.csv`; implementation in `src/economic_multiplier_engine.py:55-64`):
$$M_T^I = \sum_{k \in \text{Tourism}} s_k M_k^I, \quad M_T^{VA} = \sum_{k \in \text{Tourism}} s_k M_k^{VA}, \quad M_T^{II} = \sum_{k \in \text{Tourism}} s_k M_k^{II}$$

Weight mapping used in code (2024 DTS base, sums to ~100%): retail/wholesale ~45.17% (shopping 37.39% + pre-trip 3.89% + 50% visited-household 3.89%), food & beverage ~20.14% (16.25% + 50% visited-household), accommodation ~11.16%, vehicle fuel (refined petroleum proxy) ~12.65%, land transport ~5.38%, air transport ~1.35%, arts/recreation ~4.16%.

Empirical results are the cached Leontief inversions in `data/processed/dosm_io_multipliers.json` (recompute via `src/economic_multiplier_engine.py:extract_io_matrices_for_year`). Default reporting year is 2021:

| Tourism Sub-Sector (I-O commodity) | DTS weight ($s_k$, code) | Type I Output (2019) | Type I Output (2020) | Type I Output (2021) | Type I GVA 2021 | Type II Output 2021 | Type II GVA 2021 |
|---|---|---|---|---|---|---|---|
| **Wholesale & Retail Trade (93)** | 45.17% | 1.5592 | 1.6022 | 1.5428 | 0.8465 | 2.4985 | 1.2798 |
| **Food & Beverage Services (95)** | 20.14% | 1.9526 | 2.0623 | 1.9133 | 0.7901 | 3.0707 | 1.3148 |
| **Accommodation (94)** | 11.16% | 1.6414 | 1.7181 | 1.6719 | 0.8473 | 3.7527 | 1.7907 |
| **Refined Petroleum / Vehicle Fuel proxy (38)** | 12.65% | 2.1813 | 2.2527 | 2.1817 | 0.7221 | 3.5500 | 1.3424 |
| **Land Transport (96)** | 5.38% | 1.9261 | 1.9560 | 1.9216 | 0.8105 | 2.9962 | 1.2977 |
| **Air Transport (98)** | 1.35% | 2.1188 | 2.3000 | 2.2795 | 0.7907 | 3.2853 | 1.2467 |
| **Arts, Entertainment & Recreation (123)** | 4.16% | 1.7714 | 1.8112 | 1.7741 | 0.8050 | 2.8519 | 1.2936 |
| **DTS Composite Tourism** | **100.0%** | **1.7624** | **1.8272** | **1.7525** | **0.8151** | **2.9388** | **1.3528** |

For completeness, cached composites for other benchmark years: 2019 composite `1.7624× output / 0.8154× GVA / 3.0662× Type II output / 1.4034× Type II GVA`; 2020 composite `1.8272× / 0.8362× / 3.2079× / 1.4567×`. Water transport (97) and supporting transport services (100) are computed in JSON but excluded from the DTS composite weights above; do not substitute them for land/air.

> [!NOTE]
> **Resolution of Historical 1.55× Multiplier**:
> $1.5428\times$ (2021) and $1.5592\times$ (2019) are strictly the **Wholesale & Retail Trade** sector Type I multipliers, not whole-tourism. Whole-tourism Type I is **$1.7525\times$ (2021)**, net GDP **$0.8151\times$**, Type II **$2.9388\times$**. Always report turnover vs GDP separately. Prior appendix rows labelled `Shopping 35.7% / F&B 18.2% / Accommodation 14.8% / Land 12.6% / Air 6.5% / Entertainment 5.4%` and `Inbound 1.8210× / Blended 1.7780×` are superseded — use the code weights and JSON composites above unless a separate inbound/blended derivation is published.

### Capacity-Constrained Multiplier Damping

Standard Leontief models assume perfectly elastic factor supply (infinite spare capacity). When a tourist destination exceeds its sustainable carrying capacity ($C_d = \frac{D_{\text{peak}}}{\text{CC}_{\text{sust}}} > 1.0$), acute congestion creates local price inflation, crowding-out of residents, and supply leakage to non-local corporate imports.

DESTINASI models this via an endogenous congestion leakage penalty:
$$M_{\text{eff}}(d) = M_T \cdot \left[1 - \lambda \cdot \max\left(0, C_d - 1.0\right)\right]$$
where:
- $\lambda = 0.25$ is the empirical congestion leakage elasticity parameter.
- Bounded from below by $M_{\text{eff}}(d) \ge 0.50 \cdot M_T$.

Redirecting surplus visitors from an acute hotspot ($C_d = 1.60 \implies M_{\text{eff}} = 0.85 \cdot M_T$) to an under-utilized relief corridor ($C_d = 0.45 \implies M_{\text{eff}} = 1.00 \cdot M_T$) unlocks **$0.15 \cdot M_T$ of previously dissipated economic potential**.

---

## 2. Multi-Tier Carrying Capacity Framework (Cifuentes Model)

Tourism carrying capacity is computed following the **Cifuentes (1992)** methodological standard adopted by the World Tourism Organization (UNWTO) and IUCN.

### PCC, RCC, and ECC Formulations

1. **Physical Carrying Capacity (PCC)**:
   The theoretical upper bound of physical space:
   $$\text{PCC} = \frac{A}{a} \times \frac{V}{v} \times T$$
   - $A$: Usable public tourist area ($m^2$).
   - $a$: Standard individual visitor space footprint ($a = 25.0 \, m^2/\text{pax}$ for urban/heritage sites; $a = 100.0 \, m^2/\text{pax}$ for eco-sensitive nature reserves).
   - $V/v$: Daily turnover coefficient ($V/v = \frac{\text{Daily Operating Hours}}{\text{Average Visit Duration}}$).
   - $T$: Peak operating season coefficient ($T \in [0.8, 1.0]$).

2. **Real Carrying Capacity (RCC)**:
   Applies site-specific environmental and biophysical correction factors:
   $$\text{RCC} = \text{PCC} \times \prod_{i=1}^{k} (1 - \text{CF}_i)$$
   where correction factors $\text{CF}_i \in [0, 1]$ account for:
   - $\text{CF}_{\text{slope}}$: Topographical erosion and landslide vulnerability (dominant in Cameron Highlands and Kundasang).
   - $\text{CF}_{\text{rain}}$: Monsoon and rainfall disruption factors (East Coast November–February).
   - $\text{CF}_{\text{eco}}$: Ecological sensitivity index based on statutory PLANMalaysia KSAS classification ($\text{CF}_{\text{eco}} = 0.70$ in Tahap 1; $0.40$ in Tahap 2; $0.15$ in Tahap 3).

3. **Effective / Sustainable Carrying Capacity (ECC)**:
   Factors in municipal management capacity ($\text{MC}$):
   $$\text{ECC} = \text{RCC} \times \text{MC}$$
   $$\text{MC} = \frac{1}{3} \left( \text{Staffing Ratio} + \text{Infrastructure Score} + \text{Waste Processing Rate} \right)$$

### Multi-System Bottleneck Envelope

The binding capacity threshold for district $d$ is determined by the lower envelope of 6 independent municipal subsystems:
$$\text{CC}_{\text{sust}}(d) = \min \left( \text{CC}_{\text{accom}}(d), \, \text{CC}_{\text{trans}}(d), \, \text{CC}_{\text{water}}(d), \, \text{CC}_{\text{eco}}(d), \, \text{CC}_{\text{attr}}(d), \, \text{CC}_{\text{soc}}(d) \right)$$

- **Actionable Excess Demand**: $\Delta D(d) = \max\left(0, D_{\text{peak}}(d) - \text{CC}_{\text{sust}}(d)\right)$.
- **Capacity Headroom Gap**: $G_{\text{cap}}(d) = 1.0 - \frac{D_{\text{peak}}(d)}{\text{CC}_{\text{sust}}(d)}$.

---

## 3. Spatial & Experiential Recommender Mathematics

To prevent inappropriate diversions (e.g. routing a family seeking cool mountain tea plantations to a heavy-industry container port), DESTINASI implements a 5-pillar hybrid experiential matchmaker.

### 5-Pillar Cosine Vector Archetype Similarity

Every district $d$ is characterized by a normalized 5-dimensional archetype vector:
$$\vec{v}_d = \langle v_{\text{nature}}, \, v_{\text{heritage}}, \, v_{\text{gastronomy}}, \, v_{\text{adventure}}, \, v_{\text{urban}} \rangle, \quad \|\vec{v}_d\|_2 = 1.0$$

The cosine similarity between hotspot $H$ and candidate relief corridor $C$ is:
$$S_{\text{cosine}}(H, C) = \frac{\vec{v}_H \cdot \vec{v}_C}{\|\vec{v}_H\|_2 \|\vec{v}_C\|_2} = \sum_{k=1}^{5} v_{H, k} \cdot v_{C, k}$$

### Discrete Jaccard POI Set Similarity

To ensure micro-attraction compatibility, DESTINASI cross-references verified attraction inventories ($P_d$):
$$S_{\text{Jaccard}}(H, C) = \frac{|P_H \cap P_C|}{|P_H \cup P_C|}$$

The combined **Hybrid Experiential Similarity** is:
$$S_{\text{hybrid}}(H, C) = \alpha \cdot S_{\text{cosine}}(H, C) + (1 - \alpha) \cdot S_{\text{Jaccard}}(H, C)$$
with calibration parameter $\alpha = 0.60$.

### Multi-Criteria Weighted Sum Model (WSM)

The composite corridor suitability score $\text{WSM}(C)$ is calculated as:
$$\text{WSM}(C) = w_m S_{\text{hybrid}} + w_a A(t, m) + w_c G_{\text{cap}} + w_b B_{40} - w_e R_{\text{eco}} - \Omega(d)$$

Subject to feasibility boundary gates:
$$\text{Feasible}(C) = \begin{cases} 
\text{True}, & \text{if } G_{\text{cap}}(C) \ge 0.10 \land D(C) < \text{CC}_{\text{eco}}(C) \land d(H, C) \le 350\text{ km} \\ 
\text{False}, & \text{otherwise} 
\end{cases}$$

Where:
- $A(t, m) = 100 \cdot e^{-0.0075 \cdot t} + \beta_{\text{rail}}$ is the multimodal accessibility score ($t$ in minutes; $\beta_{\text{rail}} = 15$ if electrified rail connects both nodes).
- $B_{40}$ is the local poverty and grassroots economic development index.
- $R_{\text{eco}}$ is the ecological risk penalty if candidate $C$ is in or near a KSAS zone.
- $\Omega(d) = \max(0, (d - 60) \cdot 0.08)$ is the spatial impedance decay penalty for road distances exceeding $60\text{ km}$.

---

## 4. Predictive Machine Learning Formulation (Holiday Spike Radar)

### L2 Regularized Ridge Regression

Visitor arrival intensity $y_{d, t}$ at district $d$ for forecast horizon $t$ is estimated using regularized Ridge Regression (`src/predictive_spike_engine.py:PureNumpyLinearML`, `alpha=0.5`):
$$\min_{\mathbf{\theta}} \sum_{t=1}^{T} \left( y_t - \mathbf{x}_t^T \mathbf{\theta} \right)^2 + \gamma \|\mathbf{\theta}\|_2^2$$

Implemented 19-feature monthly planning proxy (`_build_feature_vector`, train == inference order): base 10 (month, quarter, `cuti_sekolah_days`, `public_holidays`, `long_weekends`, district `demand_multiplier`, `daily_demand_peak`, `sustainable_capacity`, `demand/capacity` ratio, `demand*multiplier`) + proximity 3 (`min_inv=1/(1+d_next)`, `min_exp=exp(-d_next/30)`, `near_holidays` within ±15d of month-mid vs `data/processed/holiday_dates_2025_2026.csv`, ex-ante known) + calendar 4 (`is_school_peak=cuti>=9`, `is_longwk=weekends>=2`, `sin/cos(2*pi*m/12)`) + AOR lags 2 (state `aor_t1` + trailing `aor_ma3` from Q1–Q4 spread to months, past-only district proxy). Training fits 3,960 calendar-capacity scenarios (110 districts × 12 months × 3 noise levels `0.96-1.04`, `np.seed 42`, 2026 reference year); `y_reg = baseline × multiplier × noise`, `y_clf = 1 if y ≥ 1.05·cap`. Treat outputs as calendar-driven planning scenarios, not validated time-series forecasts.

Still aspirational (require district-daily history + search API): higher-harmonic Fourier terms ($k>1$), holiday Gaussian kernels, district demand AR lags $y_{t-1}, y_{t-12}$, and Google search-interest proxy. State-AOR lags are a coarse proxy (source is state × Annual/Quarter, not district-monthly); `aor_t1/ma3` use past months only.

### Logistic Classification & Risk Thresholds

To classify district saturation probability (`src/predictive_spike_engine.py:PureNumpyLogisticML`, GD `lr=0.1, epochs=250`):
$$P(\text{Acute Saturation}) = \sigma\left(\mathbf{w}^T \mathbf{x} + b\right) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Risk states (modelled, require ground review — not automatic SOP triggers):
- **Low Headroom** ($P < 0.40$, $C_d < 0.80$): Modelled spare capacity.
- **Stressed** ($0.40 \le P < 0.70$, $0.80 \le C_d < 1.00$): Pre-alert for review.
- **Acute / Saturated** ($P \ge 0.70$, $C_d \ge 1.00$): Candidate for redistribution review.

### Explainable AI (SHAP Waterfall Decomposition)

For policy explainability, each recommendation or saturation prediction is decomposed into additive linear Shapley feature contributions:
$$\hat{f}(\mathbf{x}) = \phi_0 + \sum_{j=1}^{M} \phi_j$$
where $\phi_0 = \mathbb{E}[f(\mathbf{x})]$ is the national baseline prediction, and $\phi_j = \theta_j (x_j - \bar{x}_j)$ is the explicit contribution of feature $j$ (e.g. $+22\%$ from school holiday proximity, $-8\%$ from high local carrying capacity).

---

## 5. Geospatial & 3D WebGL Visualization Architecture

### PyDeck Layer Stack & Coordinate Stratification

DESTINASI renders geospatial data strictly via WebGL using **PyDeck** (`@deck.gl/core` and `@deck.gl/layers`). To guarantee that cylinders, rasters, and hazard glyphs never occlude each other, rendering elevations are stratified into non-overlapping vertical tiers:

```text
Elevation
   ▲
   │  [Tier 3: 1,000m - 40,000m] 3D District Demand Cylinders (PolygonLayer / ColumnLayer)
   │                    Manual LOD switch: 16-state macro or 110-district micro;
   │                    cylinders hide past zoom 13 so labels take over
   │
   │  [Tier 2: 350m] KSAS Alert Disks (Electric Crimson #FF0060 fill + White Border)
   │
   │  [Tier 1: 50m] Tensile Redistribution Arcs & Ground Transit Paths (ArcLayer / PathLayer)
   │
   │  [Tier 0: 0m Ground Plane] PLANMalaysia Statutory WMS Raster Overlays (BitmapLayer)
   └────────────────────────────────────────────────────────────────────────────────────────► Lat / Lon
```

### PLANMalaysia WMS & Georeferenced Raster Draping

Statutory land use zoning is acquired from the federal **i-Plan GeoServer** (`/geopro/iplan/wms`) or pre-cached high-resolution rasters:
- **Peninsular Malaysia**: Bounding box `[99.6, 1.2, 104.6, 6.8]`, EPSG:4326 (`planmalaysia_ksas_peninsular.png`).
- **Sabah & Sarawak (Borneo)**: Bounding box `[109.4, 0.8, 119.5, 7.5]`, EPSG:4326 (`planmalaysia_ksas_borneo.png`).

RFN KSAS Tahap Classifications rendered:
- **Tahap 1 (Critical / No Development)**: `#8B4513` (Saddle Brown) — Slopes $>35^\circ$, Catchment Dams, National Parks.
- **Tahap 2 (Buffer / Controlled Eco-Tourism)**: `#F4A460` (Sandy Brown) — Permanent Forest Reserves, Heritage Sites.
- **Tahap 3 (Controlled Development)**: `#FFF8DC` (Cornsilk) — Mineral reserves, retention ponds.

### KSAS Alert Disk Geometry

When a tourist hotspot coincides with an ecologically sensitive zone, DESTINASI renders a single alert disk per alerting zone — red on the inside with white borders, and nothing else:
- **Fill**: High-saturation Electric Crimson (`[255, 0, 96, 215]`).
- **Border**: 5px Pure White Stroke (`[255, 255, 255, 255]`, `line_width_min_pixels=3`).
- **Footprint**: exact 64-vertex circle computed in Python from the zone `radius_m` (clamped to 6,000–40,000 m), lightly staggered at `elev = 350.0m + 25m` per disk to eliminate z-fighting.
- **Interaction**: `pickable=False`, so district/state cylinders keep exclusive click-to-select.
- **Zoom behavior**: the manual LOD switch selects the 16-state macro or the 110-district micro view; all cylinders auto-hide past zoom $13$ while street labels, routes and arcs remain.

### Multi-Tier Transit Routing Engine

Road travel times and geometries are resolved via a priority cascade:
1. **Tier 1 (Realtime Live Ops)**: Google Maps Directions API (`duration_in_traffic` vs `duration`).
2. **Tier 2 (Free Non-Realtime Planning)**: Open Source Routing Machine (OSRM) and Valhalla API — extracts real road network geometry, distance in km, and free-flow duration without API keys.
3. **Tier 3 (Deterministic Heuristic Fallback)**: Terrain-based circuity modeling ($\times 1.28$ mountainous, $\times 1.15$ coastal) calibrated with LLM/PLUS sensor curves for key choke points (Menora Tunnel, Penang Bridge, Genting Sempah, Skudai).

---

## 6. Data Lineage, Schema & Regulatory Provenance

### Statutory Data Sources Registry

| Asset | Frequency | Volume (in repo) | Statutory Source | Legal / Administrative Mandate |
|---|---|---|---|---|
| **MOTAC Hotel AOR series** | 2017–2026 Q1 filings | 288 rows (`motac_hotel_occupancy_aor_timeseries.csv`); 134 raw PDFs in `data/raw/` | MOTAC Statistics Division | Tourism Licensing & Star Rating Standard |
| **DOSM DTS visits** | Annual state 2018–2025; district/attraction ranks | State volumes (`dts_timeseries_2018_2025.csv`); district/attraction files are rank-only, no district volumes | Department of Statistics Malaysia | Statistics Act 1965 (Act 281) |
| **DOSM Symmetric I-O Tables** | Benchmark (2019–2021) | 124 Sectors (`dosm_io_multipliers.json` cache) | DOSM National Accounts | System of National Accounts (SNA 2008) |
| **Inbound Expenditure Matrix** | Annual (2018–2024) | 38 Markets / 1,178 rows (`expenditure_market_matrix.csv`) | MOTAC Analytics & Tourism Malaysia | International Visitor Survey (IVS) |
| **OpenDOSM HIES Poverty Matrix** | 2024 state snapshot in repo | 16 states/FTs (`state_economic_profile.csv`); district poverty in `destinations_master.csv` is downscaled proxy | DOSM Social Statistics | Household Income & Expenditure Survey |
| **PLANMalaysia PPP KSAS** | MPFN ke-48 (Nov 2025) | 22 Canonical Nodes (`ksas_reference.csv: 22×23`) + cached Peninsular/Borneo PNGs | PLANMalaysia / NRECC | Town and Country Planning Act 1976 (Act 172) |

### PPP KSAS Classification Schema (Jadual 4 & Jadual 7)

Canonical records in `data/processed/ksas_reference.csv` follow the statutory PLANMalaysia schema:
- `ksas_code`: Functional code (KSAS 1 to KSAS 15).
- `jenis_ksas`: Category description (e.g. *Kawasan Bukit, Tanah Tinggi dan Kawasan Cerun*).
- `tahap`: Statutory protection rank (Tahap 1: Terpelihara Sepenuhnya; Tahap 2: Pembangunan Terkawal; Tahap 3: Terurus).
- `nama`: Official gazetted or representative geographic name.
- `luas_h`: Gazetted land area in hectares (where formally published).
- `akt_1`: Permitted low-impact activities (e.g. *Ekopelancongan berintensiti rendah*).
- `akt_syarat`: Mandatory compliance prerequisites (e.g. *Kajian Kesan Alam Sekitar (EIA), kawalan carrying capacity*).
- `akt_0`: Prohibited activities (e.g. *Perbandaran, Pertanian Komersial, Perlombongan*).
- `negeri` / `daerah`: State and municipal jurisdiction.

### Inbound Expenditure Matrix Taxonomy

Derived from `data/raw/comprehensive_expenditure.csv`, modeling spending distributions across 12 analytical accounts:
1. Accommodation
2. Food & Beverage
3. Passenger Land Transport
4. Passenger Air Transport
5. Entertainment & Recreation
6. Shopping & Retail
7. Medical & Wellness Services
8. Cultural & Heritage Entry Fees
9. Marine & Dive Activities
10. Tour Agency Fees
11. Telecommunication & Utilities
12. Miscellaneous Local Services

---

*This concludes the DESTINASI Technical & Econometric Appendix. For practical execution instructions, launch procedures, and high-level feature walkthroughs, refer to the primary [README.md](README.md).*
