"""Run a walk-forward SPY volatility comparison from adjusted closes."""

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from statistics import mean

import numpy as np

from models import (
    DECAYS,
    HORIZON,
    LOOKBACK,
    MIN_HISTORY,
    VARIANCE_FLOOR,
    ewma_variance,
    garch_variance,
)

MODELS = ("rolling", "ewma", "garch", "switch")
SWITCH_QUANTILES = (0.25, 0.40, 0.50, 0.60, 0.75)


def load_prices(path):
    prices = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"Date", "Adj Close"}.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV must have Date and Adj Close columns.")
        for row in reader:
            try:
                session = date.fromisoformat(row["Date"].strip())
                price = float(row["Adj Close"])
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid date or price on line {reader.line_num}.") from error
            if session in prices:
                raise ValueError(f"Duplicate trading date {session}.")
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid adjusted close on {session}.")
            prices[session] = price
    return sorted(prices.items())


def forecast_rows(prices, validation_start, test_start):
    if validation_start >= test_start:
        raise ValueError("Validation must start before the test period.")

    dates = [session for session, _ in prices[1:]]
    squared = np.diff(np.log([price for _, price in prices])) ** 2
    if sum(session < validation_start for session in dates) < MIN_HISTORY:
        raise ValueError(f"Provide at least {MIN_HISTORY} returns before validation.")

    schedule = []
    for i in range(MIN_HISTORY - 1, len(dates) - HORIZON):
        origin, target_end = dates[i], dates[i + HORIZON]
        if origin < validation_start:
            continue
        if origin.isocalendar()[:2] == dates[i + 1].isocalendar()[:2]:
            continue
        split = "test" if origin >= test_start else "validation"
        if split == "validation" and target_end >= test_start:
            continue
        schedule.append((i, origin, target_end, split))

    if {split for _, _, _, split in schedule} != {"validation", "test"}:
        raise ValueError("Choose split dates with eligible weeks in both periods.")

    ewma = {decay: ewma_variance(squared, decay) for decay in DECAYS}
    validation = [i for i, _, _, split in schedule if split == "validation"]
    decay = min(
        DECAYS,
        key=lambda candidate: mean(
            qlike(squared[i + 1 : i + HORIZON + 1].sum(),
                  ewma[candidate][i - LOOKBACK])
            for i in validation
        ),
    )

    rows = []
    for i, origin, target_end, split in schedule:
        history = squared[: i + 1]
        rows.append({
            "split": split,
            "forecast_date": origin,
            "target_end_date": target_end,
            "realized_variance": float(squared[i + 1 : i + HORIZON + 1].sum()),
            "rolling": max(float(HORIZON * history[-LOOKBACK:].mean()), VARIANCE_FLOOR),
            "ewma": float(ewma[decay][i - LOOKBACK]),
            "garch": garch_variance(history),
        })
    cutoff = apply_switch(rows)
    return rows, decay, cutoff


def apply_switch(rows):
    validation = [row for row in rows if row["split"] == "validation"]
    common = [row for row in validation if row["garch"] is not None]
    if not common:
        raise ValueError("Switch calibration requires successful validation GARCH fits.")

    # Rolling annualized volatility is computed entirely from the last 20 returns.
    cutoffs = np.quantile([row["rolling"] for row in validation], SWITCH_QUANTILES)
    cutoff = min(
        cutoffs,
        key=lambda candidate: mean(
            qlike(row["realized_variance"],
                  row["garch"] if row["rolling"] > candidate else row["ewma"])
            for row in common
        ),
    )
    for row in rows:
        model = "garch" if row["rolling"] > cutoff else "ewma"
        if row[model] is None:
            model = "ewma"
        row["switch"] = row[model]
        row["switch_model"] = model
        row["switch_cutoff_annualized_pct"] = annualized_volatility(cutoff)
    return annualized_volatility(cutoff)


def qlike(realized, forecast):
    return math.log(forecast) + realized / forecast


def annualized_volatility(variance):
    return 100 * math.sqrt(252 / HORIZON * variance)


def score(rows):
    for split in ("validation", "test"):
        eligible = [row for row in rows if row["split"] == split]
        common = [row for row in eligible if all(row[model] is not None for model in MODELS)]
        for model in MODELS:
            yield {
                "split": split,
                "model": model,
                "scheduled": len(eligible),
                "issued": sum(row[model] is not None for row in eligible),
                "scored": len(common),
                "qlike": mean(qlike(row["realized_variance"], row[model])
                              for row in common) if common else None,
                "volatility_mae_pp": mean(
                    abs(annualized_volatility(row["realized_variance"])
                        - annualized_volatility(row[model]))
                    for row in common
                ) if common else None,
            }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prices", type=Path)
    parser.add_argument("--validation-start", type=date.fromisoformat, required=True)
    parser.add_argument("--test-start", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, default=Path("forecasts.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics.csv"))
    args = parser.parse_args()

    try:
        if len({path.resolve() for path in (args.prices, args.output, args.metrics)}) != 3:
            raise ValueError("Use distinct paths for the input, forecasts, and metrics.")
        rows, decay, cutoff = forecast_rows(
            load_prices(args.prices), args.validation_start, args.test_start
        )
        metrics = list(score(rows))
        write_csv(args.output, rows)
        write_csv(args.metrics, metrics)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"EWMA decay selected on validation: {decay:.2f}")
    print(f"Switch threshold selected on validation: {cutoff:.4f}% annualized")
    print(f"Saved {len(rows)} forecasts to {args.output} and metrics to {args.metrics}")
    for row in metrics:
        loss = "unavailable" if row["qlike"] is None else f'{row["qlike"]:.4f}'
        print(f'{row["split"]:10} {row["model"]:7} QLIKE {loss} '
              f'({row["scored"]}/{row["scheduled"]} common weeks)')


if __name__ == "__main__":
    main()
