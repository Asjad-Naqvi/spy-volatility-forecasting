"""Plot realized and forecast SPY volatility on held-out test dates."""

import argparse
import csv
import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from baseline import annualized_volatility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecasts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test_volatility.png"))
    args = parser.parse_args()

    with args.forecasts.open(newline="", encoding="utf-8-sig") as file:
        rows = sorted(
            (row for row in csv.DictReader(file) if row["split"] == "test"),
            key=lambda row: row["forecast_date"],
        )
    if not rows:
        parser.error("The forecast CSV contains no test rows.")

    dates = [date.fromisoformat(row["forecast_date"]) for row in rows]
    series = [
        ("realized_variance", "Realized", "#212936", "-", 1.9),
        ("rolling", "20-session rolling average", "#1766a3", "--", 1.5),
        ("ewma", "EWMA", "#aa5200", "-.", 1.5),
        ("garch", "GARCH(1,1)", "#76419a", ":", 2.0),
        ("switch", "EWMA/GARCH switch", "#008573", (0, (1, 2, 4, 2)), 2.0),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    for key, label, color, style, width in series:
        values = [
            annualized_volatility(float(row[key])) if row.get(key) else math.nan
            for row in rows
        ]
        ax.plot(dates, values, label=label, color=color, linestyle=style, linewidth=width)

    ax.set_title("SPY volatility on held-out test dates", loc="left", weight="bold")
    ax.set_xlabel("Forecast date (realized volatility measured over the next five sessions)")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e6ea")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
