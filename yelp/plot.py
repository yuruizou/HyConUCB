from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from data import SEEDS

STYLE = (
    ("hyconucb", "HyConUCB", "FF7F0E", "x", "x"),
    ("var_random", "Var-Random", "8C564B", "o", "o"),
    ("var_am", "Var-AM", "2CA02C", "s", "square*"),
    ("conduel", "ConDuel", "9467BD", "D", "diamond*"),
    ("ofuglb_ucb", "OFUGLB-UCB", "1F77B4", "^", "triangle*"),
)
METRICS = ("link_regret", "utility_regret")


def collect(root, *, users=range(100), seeds=range(10), stop=5000):
    root, users, seeds = Path(root), list(users), list(seeds)
    means = {metric: np.zeros((len(seeds), len(STYLE), stop)) for metric in METRICS}
    for si, seed_index in enumerate(seeds):
        seed = SEEDS[seed_index]
        for user in users:
            for ai, (algorithm, *_rest) in enumerate(STYLE):
                path = root / f"seed_{seed}" / f"user_{user:03d}" / f"{algorithm}.npz"
                with np.load(path, allow_pickle=False) as stored:
                    for metric in METRICS:
                        values = stored[metric].reshape(stop)
                        means[metric][si, ai] += np.cumsum(values) / len(users)
        print(f"Loaded seed {seed}: {len(users)} users, five algorithms", flush=True)
    critical = float(student_t.ppf(.975, len(seeds) - 1)) if len(seeds) > 1 else np.nan
    statistics = {}
    for metric, values in means.items():
        mean = values.mean(axis=0)
        sem = (values.std(axis=0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1
               else np.full_like(mean, np.nan))
        statistics[metric] = (mean, mean - critical * sem, mean + critical * sem)
    selection = dict(users=users, seed_indices=seeds, seed_values=[SEEDS[i] for i in seeds], horizon=stop)
    return selection, means, statistics


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(statistics, output, legend):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "path.simplify": False})
    horizon = next(iter(statistics.values()))[0].shape[-1]
    times = np.arange(horizon + 1)
    marker_step, tick_step = max(1, horizon // 10), max(1, horizon // 5)
    suffix = "" if legend else "_no_legend"
    for metric, (mean, low, high) in statistics.items():
        figure, axis = plt.subplots(figsize=(4.5, 2.8))
        for ai, (_, label, color, marker, _) in enumerate(STYLE):
            values = [np.r_[0., x[ai]] for x in (mean, low, high)]
            if np.isfinite(values[1]).all():
                axis.fill_between(times, values[1], values[2], color="#" + color, alpha=.3, linewidth=0)
            axis.plot(times, values[0], color="#" + color, linewidth=1.2, label=label,
                      marker=marker, markevery=list(range(marker_step, horizon + 1, marker_step)), markersize=3,
                      markerfacecolor="none")
        axis.set(xlabel="Time", ylabel="Regret", xlim=(0, horizon), ylim=(0, None))
        axis.set_xticks(np.arange(0, horizon + 1, tick_step))
        axis.tick_params(direction="out", labelsize=8)
        if legend:
            axis.legend(loc="lower right", fontsize=6, framealpha=.5, edgecolor="black", fancybox=False)
        figure.tight_layout(pad=.5)
        stem = "regret" if metric == "link_regret" else "utility_regret"
        for extension in ("pdf", "png", "svg"):
            figure.savefig(output / f"{stem}{suffix}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)


def export_tikz(statistics, output, legend):
    horizon = next(iter(statistics.values()))[0].shape[-1]
    marker_step, tick_step = max(1, horizon // 10), max(1, horizon // 5)
    ticks = ",".join(map(str, range(0, horizon + 1, tick_step)))
    marks = ",".join(map(str, range(marker_step + 1, horizon + 2, marker_step)))
    target = output / "tikz"
    (target / "data").mkdir(parents=True, exist_ok=True)
    suffix = "" if legend else "_no_legend"
    for metric, values in statistics.items():
        stem = ("regret" if metric == "link_regret" else "utility_regret") + suffix
        lines = [r"\providecommand{\PlotDataDir}{data}"]
        lines += [rf"\definecolor{{curve{ai}}}{{HTML}}{{{row[2]}}}" for ai, row in enumerate(STYLE)]
        lines += [r"\begin{tikzpicture}", r"\begin{axis}[width=8.5cm,height=4cm,",
                  rf"xlabel={{\small Time}},ylabel={{\small Regret}},xmin=0,xmax={horizon},ymin=0,",
                  rf"xtick={{{ticks}}},scaled ticks=false,",
                  r"tick align=outside,tick pos=left,axis line style={black},",
                  r"tick label style={font=\tiny},legend pos=south east,",
                  r"legend image post style={mark indices={2}},",
                  r"legend style={font=\fontsize{4}{5}\selectfont,fill opacity=.5,text opacity=1,",
                  r"draw opacity=1,nodes={inner sep=0pt},inner xsep=2pt,inner ysep=1pt}]"]
        for ai, (algorithm, *_rest) in enumerate(STYLE):
            data = np.column_stack((np.arange(horizon + 1), *(np.r_[0., v[ai]] for v in values)))
            np.savetxt(target / "data" / f"{metric}_{algorithm}.dat", data,
                       fmt=["%d", "%.17g", "%.17g", "%.17g"], header="time mean lower upper", comments="")
            name = metric + "_" + algorithm
            if np.isfinite(values[1][ai]).all():
                lines += [rf"\addplot[name path=u{ai},draw=none,forget plot] table[x=time,y=upper]{{\PlotDataDir/{name}.dat}};",
                          rf"\addplot[name path=l{ai},draw=none,forget plot] table[x=time,y=lower]{{\PlotDataDir/{name}.dat}};",
                          rf"\addplot[curve{ai},fill opacity=.3,draw=none,forget plot] fill between[of=u{ai} and l{ai}];"]
        for ai, (algorithm, label, _color, _marker, marker) in enumerate(STYLE):
            lines += [rf"\addplot[curve{ai},thick,mark={marker},mark size=1.5pt,",
                      rf"mark indices={{{marks}}},",
                      r"mark options={fill=white,fill opacity=0}]",
                      rf"table[x=time,y=mean]{{\PlotDataDir/{metric}_{algorithm}.dat}};"]
            if legend:
                lines.append(rf"\addlegendentry{{{label}}}")
        lines += [r"\end{axis}", r"\end{tikzpicture}"]
        (target / f"{stem}.tikz").write_text("\n".join(lines) + "\n", encoding="utf-8")
        wrapper = (r"\documentclass[tikz,border=2pt]{standalone}" "\n"
                   r"\usepackage{pgfplots}\usepgfplotslibrary{fillbetween}\pgfplotsset{compat=1.18}" "\n"
                   r"\begin{document}\input{" + stem + r".tikz}\end{document}" "\n")
        (target / f"{stem}.tex").write_text(wrapper, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--tikz", action="store_true")
    parser.add_argument("--users", type=int, nargs="+", default=list(range(100)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Seed indices, 0..9")
    parser.add_argument("--stop", type=int, default=5000, help="Required saved trajectory length")
    args = parser.parse_args()
    selection, seeds, statistics = collect(args.result.resolve(), users=sorted(set(args.users)),
                                           seeds=sorted(set(args.seeds)), stop=args.stop)
    horizon, repeats = selection["horizon"], len(selection["seed_indices"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    paired = []
    for metric, (mean, low, high) in statistics.items():
        rows = []
        for t in range(horizon + 1):
            row = dict(time=t)
            for ai, (algorithm, *_rest) in enumerate(STYLE):
                for key, values in zip(("mean", "lower", "upper"), (mean, low, high)):
                    row[algorithm + "_" + key] = float(values[ai, t - 1]) if t else 0.
            rows.append(row)
        write_csv(output / f"{metric}_curves.csv", rows)
        for ai, (algorithm, label, *_rest) in enumerate(STYLE):
            summary.append(dict(metric=metric, algorithm=algorithm, label=label,
                                mean=float(mean[ai, -1]), t95_low=float(low[ai, -1]), t95_high=float(high[ai, -1])))
            if ai:
                difference = seeds[metric][:, ai, -1] - seeds[metric][:, 0, -1]
                radius = (float(student_t.ppf(.975, repeats - 1) * difference.std(ddof=1) / np.sqrt(repeats))
                          if repeats > 1 else float("nan"))
                paired.append(dict(metric=metric, baseline=algorithm, mean_difference=float(difference.mean()),
                                   t95_low=float(difference.mean() - radius), t95_high=float(difference.mean() + radius),
                                   relative_reduction_percent=float(100 * difference.mean() / mean[ai, -1]),
                                   hyconucb_wins=int(np.sum(difference > 0))))
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", paired)
    np.savez_compressed(output / "seed_means.npz", **seeds)
    draw(statistics, output, legend=not args.no_legend)
    if args.tikz:
        export_tikz(statistics, output, legend=not args.no_legend)
    record = dict(selection=selection,
                  curve_definition="mean of selected users per seed, then mean and 95% Student-t CI across paired seeds",
                  smoothing="none", summary=summary, paired_comparisons=paired)
    (output / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), summary=summary), indent=2))


if __name__ == "__main__":
    main()
