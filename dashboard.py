"""Open a local dashboard for the saved SPY volatility backtest."""

import csv
import math
import tkinter as tk
from collections import Counter
from datetime import date
from pathlib import Path
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from baseline import annualized_volatility

ROOT = Path(__file__).resolve().parent
LABELS = {
    "rolling": "Rolling average",
    "ewma": "EWMA",
    "garch": "GARCH(1,1)",
    "switch": "EWMA/GARCH switch",
}
SERIES = [
    ("realized_variance", "Realized", "#202b40", "-"),
    ("rolling", LABELS["rolling"], "#1766a3", "--"),
    ("ewma", LABELS["ewma"], "#aa5200", "-."),
    ("garch", LABELS["garch"], "#76419a", ":"),
    ("switch", LABELS["switch"], "#008573", (0, (1, 2, 4, 2))),
]


def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def table(parent, columns, rows):
    view = ttk.Treeview(parent, columns=columns, show="headings", height=len(rows))
    for index, column in enumerate(columns):
        view.heading(column, text=column)
        view.column(column, width=180 if index == 0 else 112, anchor="w" if index == 0 else "center")
    for row in rows:
        view.insert("", "end", values=row)
    view.pack(fill="x", pady=(10, 18))


def build_dashboard(window):
    forecasts = [row for row in read_csv("forecasts.csv") if row["split"] == "test"]
    metrics = [row for row in read_csv("metrics.csv") if row["split"] == "test"]
    breakdown = read_csv("test_breakdown.csv")
    if not forecasts or "switch_model" not in forecasts[0] or len(metrics) != len(LABELS):
        raise ValueError("Run the backtest and test_breakdown.py before opening the dashboard.")
    ranked = sorted((row for row in metrics if row["qlike"]), key=lambda row: float(row["qlike"]))
    if not ranked:
        raise ValueError("The test results contain no common scores.")
    best = ranked[0]
    picks = Counter(row["switch_model"] for row in forecasts)
    cutoff = float(forecasts[0]["switch_cutoff_annualized_pct"])

    window.title("SPY Volatility Forecasting")
    window.geometry("1040x720")
    window.minsize(830, 600)
    shell = ttk.Frame(window, padding=20)
    shell.pack(fill="both", expand=True)
    ttk.Label(shell, text="SPY Volatility Forecasting", font=("Segoe UI", 22, "bold")).pack(anchor="w")
    ttk.Label(shell, text=(
        f'{best["scored"]} scored test weeks  •  Lowest QLIKE: '
        f'{LABELS[best["model"]]} ({float(best["qlike"]):.4f})'
    )).pack(anchor="w", pady=(4, 16))

    tabs = ttk.Notebook(shell)
    tabs.pack(fill="both", expand=True)
    chart = ttk.Frame(tabs, padding=12)
    results = ttk.Frame(tabs, padding=12)
    choices = ttk.Frame(tabs, padding=12)
    tabs.add(chart, text="Forecast chart")
    tabs.add(results, text="Test scores")
    tabs.add(choices, text="Switch choices")

    controls = ttk.Frame(chart)
    controls.pack(fill="x")
    ttk.Label(controls, text="Forecast year").pack(side="left", padx=(0, 8))
    year = tk.StringVar(value="All")
    selector = ttk.Combobox(
        controls, textvariable=year, state="readonly", width=8,
        values=("All", *sorted({row["forecast_date"][:4] for row in forecasts})),
    )
    selector.pack(side="left")
    visible = {}
    figure = Figure(figsize=(9, 4), layout="constrained")
    axes = figure.add_subplot()
    canvas = FigureCanvasTkAgg(figure, master=chart)

    def draw(*_):
        subset = [row for row in forecasts if year.get() == "All" or row["forecast_date"].startswith(year.get())]
        dates = [date.fromisoformat(row["forecast_date"]) for row in subset]
        axes.clear()
        for key, label, color, style in SERIES:
            if visible[key].get():
                values = [annualized_volatility(float(row[key])) if row[key] else math.nan for row in subset]
                axes.plot(dates, values, label=label, color=color, linestyle=style, linewidth=1.5)
        axes.set(xlabel="Forecast date", ylabel="Annualized volatility (%)", ylim=(0, None))
        axes.grid(axis="y", alpha=0.25)
        if axes.lines:
            axes.legend(loc="upper left", ncols=2, frameon=False)
        canvas.draw_idle()

    for key, label, _, _ in SERIES:
        visible[key] = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text=label, variable=visible[key], command=draw).pack(side="left", padx=(9, 0))
    selector.bind("<<ComboboxSelected>>", draw)
    NavigationToolbar2Tk(canvas, chart)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ttk.Label(chart, text="Realized volatility uses the five sessions after each forecast date.").pack(anchor="w")
    draw()

    ttk.Label(results, text="Overall held-out scores", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    table(results, ("Model", "QLIKE", "Volatility MAE", "Weeks scored"), [
        (LABELS[row["model"]], f'{float(row["qlike"]):.4f}',
         f'{float(row["volatility_mae_pp"]):.2f} pp', row["scored"])
        for row in ranked
    ])
    ttk.Label(results, text="QLIKE by forecast year and realized regime", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    table(results, ("Group", "Weeks", "Rolling", "EWMA", "GARCH", "Switch"), [
        (row["group"], row["scored_weeks"],
         *(f'{float(row[key]):.4f}' if row.get(key) else "—" for key in LABELS))
        for row in breakdown
    ])
    ttk.Label(results, text=(
        f"Above {cutoff:.2f}% trailing annualized volatility, the switch selects GARCH. "
        f"Test choices: EWMA {picks['ewma']}, GARCH {picks['garch']}."
    ), wraplength=750).pack(anchor="w")
    ttk.Label(results, text="Lower QLIKE is better. Calm and volatile groups use the median realized test volatility.").pack(anchor="w", pady=(6, 0))
    build_choices(choices, forecasts, picks, cutoff)
    tabs.select(choices)


def build_choices(parent, forecasts, picks, cutoff):
    ttk.Label(parent, text=f"EWMA {picks['ewma']} weeks   •   GARCH {picks['garch']} weeks",
              font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Label(parent, text=f"GARCH is selected above {cutoff:.2f}% trailing annualized volatility when its fit succeeds.").pack(anchor="w")

    figure = Figure(figsize=(9, 2.1), layout="constrained")
    axes = figure.add_subplot()
    for level, model, color in ((0, "ewma", "#aa5200"), (1, "garch", "#76419a")):
        dates = [date.fromisoformat(row["forecast_date"]) for row in forecasts
                 if row["switch_model"] == model]
        axes.scatter(dates, [level] * len(dates), color=color, s=13)
    axes.set(yticks=[0, 1], yticklabels=["EWMA", "GARCH"], ylim=(-0.4, 1.4),
             xlabel="Forecast date")
    axes.grid(axis="x", alpha=0.25)
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.get_tk_widget().pack(fill="x", pady=(6, 12))
    canvas.draw()

    pane = ttk.Frame(parent)
    pane.pack(fill="both", expand=True)
    columns = ("Forecast date", "Trailing volatility", "Selected model", "Forecast volatility")
    view = ttk.Treeview(pane, columns=columns, show="headings", height=5)
    scrollbar = ttk.Scrollbar(pane, command=view.yview)
    view.configure(yscrollcommand=scrollbar.set)
    for column in columns:
        view.heading(column, text=column)
        view.column(column, width=160, anchor="center")
    for row in reversed(forecasts):
        trailing = annualized_volatility(float(row["rolling"]))
        predicted = annualized_volatility(float(row["switch"]))
        view.insert("", "end", values=(row["forecast_date"], f"{trailing:.2f}%",
                                        LABELS[row["switch_model"]], f"{predicted:.2f}%"))
    scrollbar.pack(side="right", fill="y")
    view.pack(side="left", fill="both", expand=True)


def main():
    window = tk.Tk()
    try:
        build_dashboard(window)
    except (OSError, ValueError) as error:
        window.destroy()
        raise SystemExit(str(error)) from error
    window.mainloop()


if __name__ == "__main__":
    main()
