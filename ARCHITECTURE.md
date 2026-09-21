# DESTINASI Architecture

DESTINASI turns official tourism, mobility, environmental, and economic data into an explainable operational decision: **observe pressure, diagnose the constraint, test a relief corridor, and act early**.

## Architecture at a Glance

```mermaid
flowchart TD
    classDef source fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef data fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef intelligence fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef action fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    S1["DOSM<br/>Tourism and economy"]:::source
    S2["MOTAC<br/>Hotels and visitors"]:::source
    S3["PLANMalaysia + transit<br/>Environment<br/>and mobility"]:::source
    E["1. ETL & QUALITY<br/>Extract<br/>Normalize<br/>Validate"]:::data
    D[("2. CURATED DATA<br/>data/processed/")]:::data
    C["Capacity<br/>and surge risk"]:::intelligence
    R["Relief corridors<br/>and mobility"]:::intelligence
    M["Economic impact<br/>and scenarios"]:::intelligence
    X["Explainability<br/>and map layers"]:::intelligence
    A["3. DECISION DASHBOARD<br/>Maps · alerts<br/>recommendations · actions"]:::action

    S1 --> E
    S2 --> E
    S3 --> E
    E --> D
    D --> C
    D --> R
    D --> M
    D --> X
    C --> A
    R --> A
    M --> A
    X --> A
    A -. "what-if feedback" .-> M
```

### How to read the architecture

The platform moves from evidence to action in three simple stages:

1. **Sources and ETL** provide and prepare official tourism, hotel, transport, environmental, and destination information.
2. **The curated data layer** stores consistent datasets that feed four focused branches: capacity and surge risk; relief corridors and mobility; economic impact and what-if scenarios; and explainability and map layers.
3. **The decision dashboard** brings those branches together as maps, alerts, recommendations, and actions for tourism officers.

The feedback arrow represents scenario testing: officers can change an assumption, such as visitor redistribution or transport support, and immediately review the expected operational and economic effect.

## Layer Responsibilities

| Layer | Purpose | Report output |
| --- | --- | --- |
| **Data sources** | Bring together official tourism, hotel, mobility, environmental, and destination signals. | Traceable evidence base |
| **ETL and data quality** | Parse, normalize, validate, and compile consistent district and state datasets. | Reusable analytical inputs |
| **Decision engines** | Calculate capacity, forecast surges, score corridors, estimate economic impact, and explain results. | Ranked and defensible interventions |
| **Decision dashboard** | Present maps, risk alerts, scenarios, recommendations, and policy actions. | Faster operational coordination |

## Decision Path

```mermaid
flowchart TD
    A[Demand signal] --> B{Capacity pressure?}
    B -- No --> C[Monitor]
    B -- Yes --> D[Diagnose bottleneck]
    D --> E[Find feasible relief corridor]
    E --> F[Run economic and policy scenario]
    F --> G[Explain and act]
```

## Runtime Flow

1. The deployment runner starts `app.py`.
2. `app.py` resolves the repository root and executes `dashboard/app.py`.
3. The dashboard loads curated files from `data/processed/` and optional raw inputs from `data/raw/`.
4. Streamlit navigation opens one of four decision views:
     - **Operations Command Cockpit:** live pressure, 3D district map, relief corridors, and intervention scenarios.
     - **Holiday Surge Forecaster:** forward demand, saturation risk, and pre-emptive advisory signals.
     - **Macro Tourism Intelligence:** hotel trends, visitor spending, multipliers, and redistribution economics.
     - **Governance & Methodology:** provenance, assumptions, benchmarks, and transparent methods.
5. The selected view calls the relevant domain engines in `src/`.
6. Engine outputs become maps, charts, alerts, recommendations, economic scenarios, and policy actions.

## Data Preparation Flow

The preparation pipeline is separate from the dashboard runtime, which keeps the live application fast and the data lineage inspectable.

```mermaid
flowchart LR
        A["data/raw/"] --> B["Extract and parse"]
        B --> C["Normalize geography<br/>and schemas"]
        C --> D["Validate and compile"]
        D --> E[("data/processed/")]
        E --> F["Reusable engine inputs"]
```

The main compilation entrypoint is `src/build_comprehensive_datasets.py`. It writes normalized outputs such as district visitation, attraction inventories, origin-destination matrices, expenditure shares, hotel occupancy series, international profiles, and travel-demand signals to `data/processed/`.

## What the Platform Produces

```mermaid
mindmap
    root((DESTINASI))
        Operational clarity
            District pressure map
            Binding bottleneck
            Agency-ready response
        Visitor redistribution
            Experience similarity
            Travel and transit access
            Spare capacity
        Sustainable growth
            Local economic benefit
            Capacity-aware multiplier
            KSAS ecological safeguards
        Trust and governance
            Official data lineage
            Explainable model outputs
            What-if policy scenarios
```

## Domain Engine Responsibilities

| Engine | Responsibility | Main dashboard use |
| --- | --- | --- |
| `carrying_capacity_engine.py` | Finds the binding accommodation, transport, water, ecological, attraction, or social constraint. | Operations alerts and intervention thresholds |
| `recommender_matcher.py` | Scores feasible nearby destinations by experiential similarity, accessibility, capacity, and economic need. | Relief corridor recommendations |
| `predictive_spike_engine.py` | Estimates holiday demand and district saturation risk. | Holiday Surge Forecaster |
| `economic_impact_model.py` | Estimates local economic effects of redirecting visitors. | Macro Tourism Intelligence and what-if analysis |
| `economic_multiplier_engine.py` | Applies DOSM input-output multiplier data. | Economic impact calculations |
| `aor_engine.py` | Loads and summarizes hotel average occupancy rates. | Hotel and state trend analysis |
| `traffic_engine.py` | Scores road and transit accessibility. | Route and corridor comparisons |
| `xai_engine.py` | Explains model outputs and simulates interventions. | Explainable recommendations and policy scenarios |
| `map_components.py` | Builds the PyDeck map, demand layers, transit paths, and KSAS overlays. | Operations Command Cockpit |

## System Boundaries

- **Inputs:** packaged government datasets, reference data, and optional external data extraction.
- **Processing:** Python ETL scripts and focused domain engines using pandas, NumPy, and project-specific models.
- **Presentation:** Streamlit with Altair charts and PyDeck/WebGL geospatial layers.
- **Optional AI:** Hugging Face-backed policy synthesis through the dashboard's BYOK token flow; built-in templates remain available without a token.
- **Deployment:** local Streamlit, Docker, Streamlit Community Cloud, or another container platform.

## Related Documentation

- [Project overview and quick start](README.md)
- [Technical and econometric appendix](TECHNICAL_APPENDIX.md)
- [Application entrypoint](app.py)
- [Dashboard application](dashboard/app.py)

## 5. Executive Decision Cockpit

The DESTINASI interface turns a difficult tourism-management question into a visible
decision loop: **detect pressure, diagnose the constraint, test a relief corridor,
estimate the consequence, and prepare an accountable action.** The figure below is a
report-ready view of the dashboard experience when the live Streamlit application is
not available to the reader.

```mermaid
flowchart LR
     classDef cockpit fill:#102a43,stroke:#2dd4bf,color:#f8fafc,stroke-width:2px
     classDef signal fill:#fff7ed,stroke:#f59e0b,color:#7c2d12,stroke-width:1.5px
     classDef action fill:#ecfdf5,stroke:#10b981,color:#064e3b,stroke-width:1.5px
     classDef trust fill:#f1f5f9,stroke:#64748b,color:#0f172a,stroke-width:1.5px

     A["Executive Decision Cockpit<br/>DESTINASI"]:::cockpit

     V1["1. Operations Command Cockpit<br/><br/>PyDeck 3D WebGL map<br/>110 district columns<br/>Hotspot and KSAS flags<br/>Relief arcs and transit delay<br/>Synchronized inspector drawer"]:::signal
     V2["2. Holiday Surge Forecaster<br/><br/>12-month scenario curves<br/>Festive risk ranking<br/>Spike probability<br/>What-if assumptions<br/>Draft policy briefing"]:::signal
     V3["3. Macro Tourism Intelligence<br/><br/>16-state MOTAC AOR monitor<br/>2017-2026 quarterly trend<br/>TSA multiplier sandbox<br/>Diversion and GVA scenarios<br/>38 inbound markets"]:::signal
     V4["4. Governance and Provenance<br/><br/>Mathematical specifications<br/>Benchmark context<br/>File-level lineage<br/>Assumptions and limitations<br/>Explainable outputs"]:::trust

     I["Officer interaction<br/><br/>Select district or state<br/>Adjust scenario inputs<br/>Inspect bottleneck<br/>Compare corridors<br/>Review draft action"]:::action
     O["Decision packet<br/><br/>Pressure and spike status<br/>Binding constraint<br/>Recommended relief corridor<br/>Economic scenario range<br/>Evidence and review notes"]:::action

     A --> V1
     A --> V2
     A --> V3
     A --> V4
     V1 --> I
     V2 --> I
     V3 --> I
     V4 --> I
     I --> O
     O -. "new assumption / what-if" .-> V1
     O -. "new calendar scenario" .-> V2
     O -. "economic sensitivity" .-> V3
     O -. "audit trail" .-> V4
```

**Figure 5. Executive Decision Cockpit.** The four views share a common selected
district, scenario context, and evidence base. An officer can move from a modelled
hotspot to its binding bottleneck, test a feasible alternative, inspect the estimated
economic effect, and retain the data lineage behind the recommendation. The cockpit
supports planning and review; it does not measure live crowds, command agencies, issue
legal orders, or guarantee visitor behaviour.

### Interaction narrative for the report

1. **Observe:** the Operations Command Cockpit highlights a modelled hotspot on the
    3D map and exposes demand, sustainable capacity, headroom, and the binding subsystem.
2. **Anticipate:** the Holiday Surge Forecaster tests festive-calendar scenarios and
    ranks districts by modelled saturation and spike risk up to 12 months ahead.
3. **Redistribute:** the officer compares the Top 3 feasible relief corridors using
    experience similarity, accessibility, spare capacity, community opportunity, and
    ecological safeguards.
4. **Quantify:** the Macro Tourism Intelligence view estimates output and GVA ranges
    for a proposed diversion using the selected TSA multiplier and capacity damping.
5. **Account:** Governance & Provenance keeps the equations, assumptions, benchmark
    context, and source lineage attached to the decision packet for review.

> **Evidence boundary:** all pressure, capacity, forecast, corridor, and Ringgit
> values shown by the cockpit are modelled planning estimates unless explicitly marked
> as an official source observation. The interface is designed to make that distinction
> visible rather than hide it.
