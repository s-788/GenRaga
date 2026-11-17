"""Summarize existing results only; never rerun or modify project experiments.

Run from any directory with the project's .venv Python. Outputs stay in this
paper's tables/ and figures/ directories. SD means sample SD across five seeds.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures"


def read_csv(relative):
    with (ROOT / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    runs = read_csv("v2_outputs/v2_runs.csv")
    assert len(runs) == 90
    groups = collections.defaultdict(list)
    for row in runs:
        groups[row["raga"], row["mode"], row["method"]].append(row)
    names = ("C-MODE", "random_walk", "weighted_GA")
    summary, comparisons, latex = [], [], []
    for mode in ("strict", "balanced"):
        for raga in ("mohanam", "kalyani", "abhogi"):
            means, formatted = {}, []
            for method in names:
                rows = groups[raga, mode, method]
                assert len(rows) == 5
                values = [float(row["hv"]) for row in rows]
                mean, sd = stats.mean(values), stats.stdev(values)
                means[method] = mean
                formatted.append(f"${mean:.4f} \\pm {sd:.4f}$")
                summary.append(dict(raga=raga, mode=mode, method=method,
                                    n=5, hv_mean=mean, hv_sample_sd=sd,
                                    valid_min=min(float(row["valid"]) for row in rows),
                                    best_mean=stats.mean(float(row["best_mean"]) for row in rows)))
            by_seed = {name: {int(row["seed"]): float(row["hv"])
                              for row in groups[raga, mode, name]} for name in names}
            wins = [sum(by_seed["C-MODE"][s] > by_seed[name][s] for s in range(5))
                    for name in names[1:]]
            gain = 100 * (means["C-MODE"] / means["random_walk"] - 1)
            comparisons.append(dict(raga=raga, mode=mode,
                                    relative_hv_gain_over_random_percent=gain,
                                    wins_over_random=wins[0], wins_over_ga=wins[1]))
            latex.append(f"{raga.capitalize()} & {mode.capitalize()} & " +
                         " & ".join(formatted) + f" & {wins[0]}/5 & {wins[1]}/5 " + chr(92) * 2)
    for name, rows in (("hypervolume_summary.csv", summary), ("paired_comparisons.csv", comparisons)):
        with (TABLES / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    latex[-1] = latex[-1].removesuffix(chr(92) * 2).rstrip()
    (TABLES / "hypervolume_rows.tex").write_text("\n".join(latex) + "\n")

    pieces = read_csv("v2_outputs/eval_pieces.csv")
    assert len(pieces) == 18
    audio, lines = [], []
    for raga in ("mohanam", "kalyani", "abhogi"):
        rows = [row for row in pieces if row["raga"] == raga]
        avg = lambda key: stats.mean(float(row[key]) for row in rows)
        correct = sum(row["id40_tracked"].lower() == "true" for row in rows)
        record = dict(raga=raga, n=len(rows), correct_full_classifier=correct,
                      mean_piece_median_error_cents=avg("pitch_err_med"),
                      mean_piece_p90_error_cents=avg("pitch_err_p90"),
                      mean_pitch_accuracy_50c=avg("pitch_acc_50c"),
                      mean_new_3cell=avg("new_3cell"),
                      max_verbatim_cells=max(int(row["longest_verbatim_cells"]) for row in rows))
        audio.append(record)
        lines.append(f"{raga.capitalize()} & {correct}/6 & {record['mean_piece_median_error_cents']:.3f} & "
                     f"{100 * record['mean_new_3cell']:.1f} & {record['max_verbatim_cells']} " + chr(92) * 2)
    with (TABLES / "audio_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audio[0]))
        writer.writeheader()
        writer.writerows(audio)
    lines[-1] = lines[-1].removesuffix(chr(92) * 2).rstrip()
    (TABLES / "audio_rows.tex").write_text("\n".join(lines) + "\n")

    sources = ("v2_outputs/v2_runs.csv", "v2_outputs/eval_pieces.csv",
               "v2_outputs/eval_summary.json", "corpus_data/alignment_report.json",
               "corpus_data/gamaka_eval.json", "carnatic_unified/recordings.csv",
               "phase1_outputs/phase1_metrics.csv")
    audit = dict(operation="Descriptive reanalysis of saved records; no model runs",
                 standard_deviation="Sample SD, ddof=1, across five seeds",
                 sources_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                 comparisons=comparisons,
                 audio=audio,
                 overall_pitch_error_mean_of_piece_medians=stats.mean(float(r["pitch_err_med"]) for r in pieces))
    (TABLES / "evidence_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
    colors = ("#174a75", "#a5a5a5", "#b86f2c")
    labels = ("C-MODE", "Random walk", "Weighted GA")
    for ax, mode in zip(axes, ("strict", "balanced")):
        for j, method in enumerate(names):
            records = [next(row for row in summary if row["mode"] == mode and row["raga"] == r and row["method"] == method)
                       for r in ("mohanam", "kalyani", "abhogi")]
            ax.bar(np.arange(3) + (j - 1) * .24, [r["hv_mean"] for r in records], width=.23,
                   yerr=[r["hv_sample_sd"] for r in records], capsize=2,
                   color=colors[j], label=labels[j], linewidth=.4, edgecolor="black")
        ax.set_xticks(range(3), ["Mohanam", "Kalyani", "Abhogi"])
        ax.set_title(mode.capitalize() + " constraints")
        ax.set_ylim(0, .86)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Estimated hypervolume")
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout(pad=.65)
    fig.savefig(FIGURES / "hypervolume.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "hypervolume.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"saved_runs": len(runs), "audio_pieces": len(pieces),
                      "balanced_gains_percent": [r["relative_hv_gain_over_random_percent"]
                                                for r in comparisons if r["mode"] == "balanced"]}, indent=2))


if __name__ == "__main__":
    main()
