from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from run import ALGORITHMS, ROOT, SEEDS, load_benchmark, metric_curves, result_path

CURVES = ("expert_regret", "logged_regret", "task_pseudo_regret", "theta_error")
STYLES = {
    "HyConUCB": ((31, 119, 180), "^", "triangle*"),
    "Var-Random": ((140, 86, 75), "o", "o"),
    "Var-AM": ((255, 127, 14), "x", "x"),
    "ConDuel": ((44, 160, 44), "s", "square*"),
    "OFUGLB-UCB": ((148, 103, 189), "D", "diamond*"),
}


def load_results(directory: Path, data_path: Path, *, algorithms=ALGORITHMS, seeds=SEEDS) -> tuple[dict, list[dict], list[dict]]:
    algorithms, seeds = list(algorithms), list(seeds)
    data = load_benchmark(str(data_path))
    curves = {name: [] for name in CURVES}
    rows, per_seed = [], []
    for algorithm in algorithms:
        grouped, values = {name: [] for name in CURVES}, []
        for seed in seeds:
            path = result_path(directory, algorithm, seed)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files if key != "summary"}
            current = metric_curves(arrays, data)
            for name in CURVES:
                grouped[name].append(current[name])
            values.append({name: float(current[name][-1]) for name in CURVES}
                          | {"accuracy": float(arrays["arm_reward"].mean())})
            per_seed.append({"algorithm": algorithm, "seed": seed, **values[-1]})
        for name in CURVES:
            curves[name].append(np.mean(grouped[name], axis=0))
        for metric in (*CURVES, "accuracy"):
            samples = np.asarray([value[metric] for value in values])
            mean = float(samples.mean())
            sd = float(samples.std(ddof=1)) if len(seeds) > 1 else None
            half = student_t.ppf(0.975, len(seeds) - 1) * sd / np.sqrt(len(seeds)) if sd is not None else None
            rows.append({"algorithm": algorithm, "metric": metric, "n": len(seeds),
                         "mean": mean, "sd": sd,
                         "ci95_low": mean - half if half is not None else None,
                         "ci95_high": mean + half if half is not None else None})
    result = {name: np.asarray(values) for name, values in curves.items()}
    result.update(algorithms=np.asarray(algorithms), seeds=np.asarray(seeds),
                  time=np.arange(result[CURVES[0]].shape[1]))
    return result, rows, per_seed


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(result: dict, output: Path) -> None:
    algorithms, time = list(result["algorithms"]), result["time"]
    colors = "\n".join(f"\\definecolor{{llm{i}}}{{RGB}}{{{','.join(map(str, STYLES[name][0]))}}}"
                       for i, name in enumerate(algorithms))
    (output / "plot_style.tex").write_text(
        "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n" + colors + "\n"
        "\\providecommand{\\LLMPlotRoot}{.}\n", encoding="utf-8")
    for metric in CURVES:
        theta = metric == "theta_error"
        with plt.rc_context({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42}):
            figure, axis = plt.subplots(figsize=(6.3, 3.5))
            for index, algorithm in enumerate(algorithms):
                rgb, marker, _ = STYLES[algorithm]
                axis.plot(time, result[metric][index], color=np.asarray(rgb) / 255,
                          linewidth=1.8, linestyle="-", marker=marker, markersize=4.5,
                          markevery=max(1, int(time[-1]) // 10), markerfacecolor="white",
                          markeredgewidth=0.9, label=algorithm)
            axis.set_xlabel("Time")
            axis.set_ylabel(r"$\|\hat\theta_t-\theta^*\|_2$" if theta else "Regret")
            axis.set_xlim(0, time[-1])
            axis.set_xticks(np.linspace(0, time[-1], 6, dtype=int))
            if not theta:
                axis.set_ylim(bottom=min(0, axis.get_ylim()[0]))
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(loc="upper right" if theta else "upper left", fontsize=7,
                        framealpha=0.75, edgecolor="black", handlelength=2.1)
            figure.tight_layout()
            figure.savefig(output / f"{metric}.pdf", bbox_inches="tight")
            figure.savefig(output / f"{metric}.png", dpi=220, bbox_inches="tight")
            plt.close(figure)
        header = "time " + " ".join(f"a{i}" for i in range(len(algorithms)))
        np.savetxt(output / f"{metric}.dat", np.column_stack((time, result[metric].T)),
                   header=header, comments="", fmt="%.15g")
        label = r"$\|\hat\theta_t-\theta^*\|_2$" if theta else "Regret"
        lines = ["\\begin{tikzpicture}", "\\begin{axis}[width=\\linewidth,height=4.2cm,",
                 f"  xlabel={{Time}},ylabel={{{label}}},xmin=0,xmax={int(time[-1])},",
                 "  scaled ticks=false,tick align=outside,tick pos=left,",
                 "  tick label style={font=\\tiny},label style={font=\\small},",
                 "  legend pos=" + ("north east" if theta else "north west") + ",",
                 "  legend style={font=\\tiny,cells={anchor=west}}]"]
        for index, algorithm in enumerate(algorithms):
            lines.extend([
                f"\\addplot[color=llm{index},thick,solid,mark={STYLES[algorithm][2]},",
                f"  mark repeat={max(1, int(time[-1]) // 10)},mark phase=1,",
                f"  mark options={{solid,fill=white,draw=llm{index}}}]",
                f"  table[x=time,y=a{index}] {{\\LLMPlotRoot/{metric}.dat}};",
                f"\\addlegendentry{{{algorithm}}}",
            ])
        lines.extend(["\\end{axis}", "\\end{tikzpicture}", ""])
        (output / f"{metric}.tikz").write_text("\n".join(lines), encoding="utf-8")
        (output / f"{metric}.tex").write_text(
            "\\documentclass[border=2pt]{standalone}\n\\input{plot_style.tex}\n"
            "\\begin{document}\n\\begin{minipage}{8cm}\n"
            f"\\input{{{metric}.tikz}}\n"
            "\\end{minipage}\n\\end{document}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true", help="plot the supplied final 200-seed means")
    parser.add_argument("--result", type=Path, default=ROOT / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmark.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.reference:
        with np.load(ROOT / "data/reference.npz", allow_pickle=False) as archive:
            result = {key: archive[key] for key in archive.files}
        with (ROOT / "data/reference.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        result, rows, per_seed = load_results(args.result.resolve(), args.data.resolve(),
                                               algorithms=dict.fromkeys(args.algorithms), seeds=dict.fromkeys(args.seeds))
        write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "summary.csv", rows)
    draw(result, output)
    print(f"Figures and summary written to {output}")


if __name__ == "__main__":
    main()
