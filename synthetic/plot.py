import argparse
import csv
from pathlib import Path

import numpy as np

from environment import CONDITIONS, SEEDS

STYLE = (
    ("hyconucb", "HyConUCB", "#FF7F0E", "x"),
    ("var-random", "Var-Random", "#8C564B", "o"),
    ("var-am", "Var-AM", "#2CA02C", "s"),
    ("conduel", "ConDuel", "#9467BD", "D"),
    ("ofuglb-ucb", "OFUGLB-UCB", "#1F77B4", "^"),
)
METRICS = ("regret", "parameter_error", "utility_regret")


def read_condition(folder, *, users=range(100), seeds=range(10), stop=5000):
    folder, users, seeds = Path(folder), list(users), list(seeds)
    means = {metric: np.zeros((len(seeds), len(STYLE), stop)) for metric in METRICS}
    for si, seed in enumerate(seeds):
        for user in users:
            for ai, (algorithm, _, _, _) in enumerate(STYLE):
                path = folder / f"seed{seed:02d}" / f"user{user:03d}" / f"{algorithm}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    for metric in METRICS:
                        values = archive[metric].reshape(stop)
                        if metric != "parameter_error":
                            values = np.cumsum(values)
                        means[metric][si, ai] += values / len(users)
        print(f"Loaded {folder.name}, seed {seed}: {len(users)} users", flush=True)
    return dict(users=users, seed_indices=seeds, T=stop), means


def collect(results, setting, *, users=range(100), seeds=range(10), stop=5000):
    users, seeds = list(users), list(seeds)
    records = []
    for value in CONDITIONS[setting]:
        _, means = read_condition(Path(results) / f"{setting}{value}", users=users, seeds=seeds, stop=stop)
        records.append(means)
    data = dict(conditions=np.array(CONDITIONS[setting]), users=np.array(users),
                seed_indices=np.array(seeds), seed_values=np.array([SEEDS[i] for i in seeds]),
                time=np.arange(1, stop + 1), algorithms=np.array([s[0] for s in STYLE]),
                labels=np.array([s[1] for s in STYLE]))
    for metric in METRICS:
        values = data[metric + "_seed_means"] = np.stack([row[metric] for row in records])
        data[metric + "_mean"] = values.mean(axis=1)
        data[metric + "_sd"] = (values.std(axis=1, ddof=1) if values.shape[1] > 1
                               else np.full_like(data[metric + "_mean"], np.nan))
        data[metric + "_se"] = data[metric + "_sd"] / np.sqrt(values.shape[1])
    return data


def draw(data, folder, setting, legend=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
                         "axes.titlesize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": .7, "pdf.fonttype": 42, "path.simplify": False})
    time = data["time"]
    step = max(1, len(time) // 10)
    markers = np.unique(np.r_[0, np.arange(step - 1, len(time), step), len(time) - 1]).tolist()
    suffix = "" if legend else "_no_legend"
    for metric, label in (("regret", "Regret"), ("parameter_error", r"$\|\hat{\theta}_t-\theta^\star\|_2$")):
        fig, axes = plt.subplots(1, len(data["conditions"]), figsize=(11 if legend else 9.2, 3.1), squeeze=False)
        for ci, (condition, ax) in enumerate(zip(data["conditions"], axes[0])):
            for ai, (_, name, color, marker) in enumerate(STYLE):
                mean, se = data[metric + "_mean"][ci, ai], data[metric + "_se"][ci, ai]
                if np.isfinite(se).all():
                    ax.fill_between(time, mean - se, mean + se, color=color, alpha=.3, linewidth=0)
                ax.plot(time, mean, color=color, linewidth=1.4, label=name, marker=marker,
                        markevery=markers, markersize=3.4, markerfacecolor="none")
            ax.set(title="$" + f"{setting}={condition:g}$", xlabel="Time", xlim=(0, time[-1]), ylim=(0, None))
            if ci == 0:
                ax.set_ylabel(label)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.tick_params(direction="out", width=.7, length=3)
        fig.tight_layout(rect=(0, 0, .84 if legend else 1, 1), w_pad=1.1)
        if legend:
            position = axes[0, -1].get_position()
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="center left", ncol=1, frameon=False,
                       bbox_to_anchor=(position.x1 + .015, (position.y0 + position.y1) / 2))
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"{metric}_{setting}{suffix}.{extension}", dpi=300,
                        bbox_inches="tight", pad_inches=.06)
        plt.close(fig)


def analyze(results, setting, *, output=None, legend=True, users=range(100), seeds=range(10), stop=5000):
    data = collect(results, setting, users=users, seeds=seeds, stop=stop)
    folder = (Path(output) if output is not None else Path(__file__).resolve().parent / "figures") / setting
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "curves.npz", **data)
    rows = []
    for ci, condition in enumerate(data["conditions"]):
        for ai, (algorithm, label, _, _) in enumerate(STYLE):
            for metric in METRICS:
                rows.append(dict(setting=setting, value=int(condition), algorithm=algorithm, label=label,
                                 users=len(data["users"]), seeds=len(data["seed_indices"]), T=len(data["time"]),
                                 metric=metric, **{name: float(data[metric + "_" + name][ci, ai, -1])
                                                 for name in ("mean", "sd", "se")}))
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(data, folder, setting, legend)
    print(f"Saved figures and statistics: {folder}", flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("S", "B", "both"), default="both")
    parser.add_argument("--result", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "figures", help="Figure root directory; S/B subdirectories are added")
    parser.add_argument("--users", type=int, nargs="+", default=list(range(100)), help="User IDs, 0..99")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Seed indices, 0..9")
    parser.add_argument("--stop", type=int, default=5000, help="Required saved trajectory length")
    parser.add_argument("--no-legend", action="store_true")
    args = parser.parse_args()
    for setting in CONDITIONS if args.setting == "both" else (args.setting,):
        analyze(args.result.resolve(), setting, output=args.output, legend=not args.no_legend,
                users=sorted(set(args.users)), seeds=sorted(set(args.seeds)), stop=args.stop)
