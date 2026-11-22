"""Report test QLIKE by forecast year and realized volatility regime."""

import argparse
import csv
from pathlib import Path
from statistics import mean, median

from baseline import MODELS, annualized_volatility, qlike, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecasts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test_breakdown.csv"))
    args = parser.parse_args()
    if args.forecasts.resolve() == args.output.resolve():
        parser.error("Use distinct input and output paths.")

    with args.forecasts.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
    if not rows:
        parser.error("The forecast CSV contains no test rows.")

    cutoff = median(float(row["realized_variance"]) for row in rows)
    years = {}
    regimes = {"Calm": [], "Volatile": []}
    for row in rows:
        years.setdefault(row["forecast_date"][:4], []).append(row)
        regime = "Calm" if float(row["realized_variance"]) <= cutoff else "Volatile"
        regimes[regime].append(row)

    results = []
    for category, groups in (("Year", sorted(years.items())), ("Regime", regimes.items())):
        for group, weeks in groups:
            common = [row for row in weeks if all(row[model] for model in MODELS)]
            losses = {
                model: mean(qlike(float(row["realized_variance"]), float(row[model]))
                            for row in common) if common else None
                for model in MODELS
            }
            best = min(losses.values()) if common else None
            results.append({
                "category": category,
                "group": group,
                "test_weeks": len(weeks),
                "scored_weeks": len(common),
                **losses,
                "winner": ", ".join(model for model in MODELS if losses[model] == best)
                if common else "",
            })

    write_csv(args.output, results)
    print(f"Median test realized volatility: {annualized_volatility(cutoff):.4f}% annualized")


if __name__ == "__main__":
    main()
