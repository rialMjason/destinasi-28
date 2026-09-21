"""
DESTINASI — Baseline Dataset Compiler
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026

Role: Offline Data Ingestion & ETL Pipeline.
Pre-computes baseline state metrics and pilot corridor datasets for data/processed/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

def rank_score(series: pd.Series, high_is_good: bool = True) -> pd.Series:
    rank = series.rank(method="average")
    score = (rank - 1) / (len(series) - 1) * 100
    return score if high_is_good else 100 - score


def build_visitors() -> pd.DataFrame:
    wide = pd.read_csv(RAW / "domestic_visitors_state_2017_2024_extracted.csv")
    return wide.melt(
        id_vars="state",
        var_name="year",
        value_name="domestic_visitors_000",
    ).assign(year=lambda frame: frame["year"].astype(int))


def build_baseline(visitors: pd.DataFrame) -> pd.DataFrame:
    population = pd.read_csv(RAW / "population_state.csv")
    population = population[
        (population["date"] == "2024-01-01")
        & (population["sex"] == "both")
        & (population["age"] == "overall")
        & (population["ethnicity"] == "overall")
    ][["state", "population"]].rename(columns={"population": "population_2024_000"})

    hies = pd.read_csv(RAW / "hies_state.csv")
    hies = hies[hies["date"] == "2024-01-01"][
        ["state", "income_median", "income_mean", "poverty", "gini"]
    ].rename(
        columns={
            "income_median": "median_household_income_2024_rm",
            "income_mean": "mean_household_income_2024_rm",
            "poverty": "poverty_rate_2024_pct",
            "gini": "gini_2024",
        }
    )

    current = visitors[visitors["year"] == 2024][
        ["state", "domestic_visitors_000"]
    ].copy()
    previous = visitors[visitors["year"] == 2023][
        ["state", "domestic_visitors_000"]
    ].rename(columns={"domestic_visitors_000": "domestic_visitors_2023_000"})
    current = current.rename(
        columns={"domestic_visitors_000": "domestic_visitors_2024_000"}
    ).merge(previous, on="state", validate="one_to_one")

    hotel = pd.read_csv(RAW / "hotel_supply_state_2023_extracted.csv")
    tourists = pd.read_csv(
        RAW / "domestic_tourists_destination_state_2024_extracted.csv"
    )

    baseline = (
        current.merge(tourists, on="state", validate="one_to_one")
        .merge(population, on="state", validate="one_to_one")
        .merge(hies, on="state", validate="one_to_one")
        .merge(hotel, on="state", validate="one_to_one")
    )
    baseline["visitor_growth_2024_pct"] = (
        baseline["domestic_visitors_2024_000"]
        / baseline["domestic_visitors_2023_000"]
        - 1
    ) * 100
    baseline["annual_visitors_per_resident"] = (
        baseline["domestic_visitors_2024_000"] / baseline["population_2024_000"]
    )
    baseline["annual_overnight_tourists_per_room"] = (
        baseline["overnight_tourists_2024_000"] * 1000
        / baseline["hotel_rooms_2023"]
    )

    intensity = rank_score(baseline["annual_visitors_per_resident"])
    room_pressure = rank_score(baseline["annual_overnight_tourists_per_room"])
    growth = rank_score(baseline["visitor_growth_2024_pct"])
    baseline["tourism_pressure_proxy_score"] = (
        0.4 * intensity + 0.4 * room_pressure + 0.2 * growth
    )
    baseline["spare_capacity_proxy_score"] = (
        100 - baseline["tourism_pressure_proxy_score"]
    )

    poverty_need = rank_score(baseline["poverty_rate_2024_pct"])
    income_need = rank_score(
        baseline["median_household_income_2024_rm"], high_is_good=False
    )
    baseline["community_opportunity_proxy_score"] = (
        0.6 * poverty_need + 0.4 * income_need
    )
    baseline["redistribution_potential_proxy_score"] = (
        0.55 * baseline["spare_capacity_proxy_score"]
        + 0.45 * baseline["community_opportunity_proxy_score"]
    )

    return baseline.sort_values(
        "redistribution_potential_proxy_score", ascending=False
    )


def create_charts(baseline: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")

    chart = baseline.sort_values("tourism_pressure_proxy_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=chart,
        x="tourism_pressure_proxy_score",
        y="state",
        hue="state",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="Tourism pressure proxy varies widely across Malaysian states",
        xlabel="Tourism pressure proxy score (0–100)",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(PROCESSED / "tourism_pressure_proxy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=baseline,
        x="tourism_pressure_proxy_score",
        y="community_opportunity_proxy_score",
        size="domestic_visitors_2024_000",
        sizes=(60, 500),
        hue="redistribution_potential_proxy_score",
        palette="viridis",
        ax=ax,
    )
    for row in baseline.itertuples():
        ax.annotate(
            row.state.replace("W.P. ", ""),
            (row.tourism_pressure_proxy_score, row.community_opportunity_proxy_score),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        title="Initial screening: tourism pressure versus community opportunity",
        xlabel="Tourism pressure proxy score (0–100)",
        ylabel="Community opportunity proxy score (0–100)",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(PROCESSED / "pressure_vs_opportunity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    visitors = build_visitors()
    baseline = build_baseline(visitors)
    visitors.to_csv(PROCESSED / "domestic_visitors_by_state_2017_2024.csv", index=False)
    baseline.to_csv(PROCESSED / "state_baseline.csv", index=False)
    create_charts(baseline)
    print(
        baseline[
            [
                "state",
                "tourism_pressure_proxy_score",
                "community_opportunity_proxy_score",
                "redistribution_potential_proxy_score",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
