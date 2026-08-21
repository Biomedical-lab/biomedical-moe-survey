import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

from config import MODEL_NAMES, MODEL_LABELS

matplotlib.rcParams["font.family"] = "serif"

def parse_args():
    p = argparse.ArgumentParser(description="MoE Gating Survey — Visualization")
    p.add_argument("--results_dir", type=str, default="./results")
    return p.parse_args()

def plot_figure5(results_dir):

    results_dir = Path(results_dir)

    exp_a_path = results_dir / "experiment_a_results.csv"
    if not exp_a_path.exists():
        print(f"ERROR: {exp_a_path} not found. Run train.py first.")
        return

    df_multi = pd.read_csv(exp_a_path)
    summary = df_multi.groupby(["dataset", "model"]).agg(
        Acc_mean=("acc", "mean"), Acc_std=("acc", "std"),
        F1_mean=("f1", "mean"), F1_std=("f1", "std"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = {"vqarad": "#4C72B0", "slake": "#55A868", "derm7pt": "#C44E52"}
    xlabels_short = [
        "A1.0\nBaseline", "A1.1\nGMU", "A1.2\nSparse",
        "A1.3\nSoft", "A1.4\nCross-Attn", "A1.5\nMod-Spec",
    ]

    for ax, metric, std_col, ylabel, panel in zip(
        axes[:2],
        ["Acc_mean", "F1_mean"], ["Acc_std", "F1_std"],
        ["Accuracy", "Macro F1"], ["(a)", "(b)"],
    ):
        x = np.arange(len(MODEL_NAMES))
        w = 0.25
        for i, ds in enumerate(["vqarad", "slake", "derm7pt"]):
            vals, errs = [], []
            for m in MODEL_NAMES:
                row = summary[(summary["dataset"] == ds) & (summary["model"] == m)]
                vals.append(row[metric].values[0] if len(row) else 0)
                errs.append(row[std_col].values[0] if len(row) else 0)
            bars = ax.bar(
                x + (i - 1) * w, vals, w,
                label=ds.upper(), color=colors[ds], alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels_short, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0.2, 1.15)
        ax.set_title(f"{panel}  {ylabel} by Gating Architecture", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.35, -0.02), frameon=False,
    )

    ax = axes[2]
    exp_b_path = results_dir / "experiment_b_results.csv"
    cond_xlabels = ["Full\n(img+txt)", "No\nimage", "No\ntext", "Half\nimage"]
    if exp_b_path.exists():
        df_b = pd.read_csv(exp_b_path)
        all_conditions = df_b["condition"].unique()
        x_pos = np.arange(len(all_conditions))
        for ds in ["vqarad", "slake", "derm7pt"]:
            sub = df_b[df_b["dataset"] == ds]
            if len(sub) == 0:
                continue
            ax.plot(
                x_pos, sub["acc"].values, marker="o", label=ds.upper(),
                color=colors[ds], linewidth=2, markersize=7,
            )
            for j, (_, row) in enumerate(sub.iterrows()):
                ax.annotate(
                    f"{row['delta']:+.3f}", (j, row["acc"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7,
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cond_xlabels, fontsize=9)
        ax.set_title("(c)  Accuracy by Missing Condition", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.grid(alpha=0.3)
    else:
        ax.set_title("(c) Missing Modality (pending Exp B)")
        ax.text(0.5, 0.5, "Run evaluate.py first", transform=ax.transAxes, ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(results_dir / "figure5.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "figure5.png", dpi=150, bbox_inches="tight")
    print(f"Figure 5 saved to {results_dir}")
    plt.show()

def run_statistical_tests(results_dir):

    results_dir = Path(results_dir)

    exp_a_path = results_dir / "experiment_a_results.csv"
    if not exp_a_path.exists():
        print(f"ERROR: {exp_a_path} not found. Run train.py first.")
        return

    df_multi = pd.read_csv(exp_a_path)
    test_rows = []
    for ds in df_multi["dataset"].unique():
        base = df_multi[(df_multi["dataset"] == ds) & (df_multi["model"] == "baseline")]["acc"].values
        for name in MODEL_NAMES:
            if name == "baseline":
                continue
            accs = df_multi[(df_multi["dataset"] == ds) & (df_multi["model"] == name)]["acc"].values
            if len(base) == len(accs) and len(base) >= 5:
                try:
                    stat, p = wilcoxon(accs, base, alternative="two-sided")
                except Exception:
                    stat, p = 0, 1.0
                test_rows.append(dict(
                    Dataset=ds, Model=MODEL_LABELS[name],
                    Mean_Diff=round(accs.mean() - base.mean(), 4),
                    W_stat=round(stat, 3), p_value=round(p, 4),
                    Significant="*" if p < 0.05 else "",
                ))

    df_stat = pd.DataFrame(test_rows)
    df_stat.to_csv(results_dir / "stat_tests.csv", index=False)
    print("=== Wilcoxon Signed-Rank Test (vs Baseline A1.0) ===")
    print(df_stat.to_string(index=False))

def main():
    args = parse_args()
    plot_figure5(args.results_dir)
    run_statistical_tests(args.results_dir)

if __name__ == "__main__":
    main()
