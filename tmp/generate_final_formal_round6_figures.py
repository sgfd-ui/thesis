from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "tmp" / "generate_final_with_supplement_figures.py"
spec = importlib.util.spec_from_file_location("final_with_supplement", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

RESULT_ROOT = Path("H:/\u8bba\u6587\u9879\u76ee/compare/results")
FORMAL_MAIN_CSV = (
    RESULT_ROOT
    / "md_aligned_main_6run"
    / "launch_20260624_100902"
    / "final_complete_replicate_best_6round.csv"
)
LEGACY_REUSE_CSV = (
    RESULT_ROOT
    / "md_aligned_main_6run"
    / "launch_20260624_100902"
    / "final_best_ours_selected_round1_round2_fixed.csv"
)
E4_CSV = RESULT_ROOT / "supplement_final\u6574\u7406" / "e4_best_ours_selected.csv"
OUT_DIR = ROOT / "figures" / "experiments" / "final_formal_round6"
SUMMARY_DIR = ROOT / "output" / "experiment_summaries" / "final_formal_round6"

base.MAIN_CSV = FORMAL_MAIN_CSV
base.E4_CSV = E4_CSV
base.OUT_DIR = OUT_DIR
base.SUMMARY_DIR = SUMMARY_DIR


def plot_fig6_formal(df: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E1_skew_strength") & df["setting"].isin(base.MAIN_METHODS)].copy()
    natural = sub[sub["case_id"].str.match(r"^alpha_", na=False)].copy()
    controlled = sub[sub["case_id"].str.startswith("controlled_alpha_", na=False)].copy()
    natural["alpha"] = natural["case_id"].map(base.parse_alpha)
    controlled["alpha"] = controlled["case_id"].map(base.parse_alpha)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), constrained_layout=True)
    base.line_methods(
        axes[0, 0],
        natural,
        "alpha",
        "end_to_end_time",
        base.MAIN_METHODS,
        "(a) Natural skew completion",
        "Modeled time (s)",
    )
    axes[0, 0].set_xlabel("Zipf alpha")
    base.line_methods(
        axes[0, 1],
        controlled,
        "alpha",
        "end_to_end_time",
        base.MAIN_METHODS,
        "(b) Controlled-work completion",
        "Modeled time (s)",
    )
    axes[0, 1].set_xlabel("Alpha")

    join_natural = natural.groupby("alpha", as_index=False)["join_work"].mean()
    join_controlled = controlled.groupby("alpha", as_index=False)["join_work"].mean()
    axes[1, 0].plot(join_natural["alpha"], join_natural["join_work"], marker="o", color="#555555")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("(c) Natural join work")
    axes[1, 0].set_xlabel("Zipf alpha")
    axes[1, 0].set_ylabel("$W_{join}$")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].plot(join_controlled["alpha"], join_controlled["join_work"], marker="o", color="#2f6fbb")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("(d) Controlled-work join work")
    axes[1, 1].set_xlabel("Alpha")
    axes[1, 1].set_ylabel("$W_{join}$")
    axes[1, 1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    base.write_outputs(fig, "fig6_skew_strength")

    for name, data in [("natural", natural), ("controlled", controlled)]:
        piv = (
            data.pivot_table(index="case_id", columns="setting", values="end_to_end_time", aggfunc="first")
            .dropna(subset=base.MAIN_METHODS)
        )
        for b in base.BASELINES:
            summary.append(
                {
                    "section": f"E1_{name}",
                    "metric": f"speedup_vs_{b}",
                    "value": base.geomean(piv[b] / piv["ours"]),
                    "n": len(piv),
                }
            )


def plot_fig9_formal(summary: list[dict]) -> None:
    # The six-round complete replicate keeps one row per method/case and therefore
    # collapses repeated-query mode evidence. Use the previously consolidated
    # round1/round2 reuse table for the mode-level panels.
    reuse_df = pd.read_csv(LEGACY_REUSE_CSV)
    reuse_df = reuse_df[
        reuse_df["run_status"].eq("success")
        & reuse_df["scenario_id"].eq("E7_versioned_incremental_history_reuse")
    ].copy()
    for col in [
        "end_to_end_time",
        "detection_time",
        "validation_time",
        "routing_time",
        "hotset_drift",
        "residual_partition_ratio",
    ]:
        if col in reuse_df.columns:
            reuse_df[col] = pd.to_numeric(reuse_df[col], errors="coerce")
    e1b_audit = pd.read_csv(RESULT_ROOT / "supplement_final\u6574\u7406" / "e1b_controlled_work_audit.csv")
    base.plot_fig12(reuse_df, e1b_audit, summary)


def save_summary_formal(summary: list[dict], df: pd.DataFrame, e4: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(summary)
    out.to_csv(SUMMARY_DIR / "summary_final_formal_round6.csv", index=False, encoding="utf-8-sig")

    complete = base.complete_pivot(df)
    lines = [
        "# Final Formal Round6 Key Findings",
        "",
        f"- Main input rows after success filter: `{len(df)}`.",
        f"- Complete main-method points: `{len(complete)}`.",
        f"- E4 supplement rows: `{len(e4)}` across `{e4['case_id'].nunique()}` cases and `{e4['setting'].nunique()}` methods.",
        "",
        "## Overall speedups",
    ]
    for b in base.BASELINES:
        ratio = complete[b] / complete["ours"]
        lines.append(
            f"- Baseline/Ours `{b}`: geomean `{base.geomean(ratio):.3f}x`, wins for Ours `{int((ratio > 1).sum())}/{int(ratio.count())}`."
        )
    lines.extend(["", "## E1 controlled-work skew"])
    controlled = df[
        df["scenario_id"].eq("E1_skew_strength")
        & df["case_id"].str.startswith("controlled_alpha_", na=False)
        & df["setting"].isin(base.MAIN_METHODS)
    ]
    piv = controlled.pivot_table(index="case_id", columns="setting", values="end_to_end_time", aggfunc="first").dropna(
        subset=base.MAIN_METHODS
    )
    for b in base.BASELINES:
        lines.append(f"- Controlled speedup vs `{b}`: `{base.geomean(piv[b] / piv['ours']):.3f}x`.")
    lines.extend(["", "## E4 supplement"])
    for method in [
        "ours_kcheck_only",
        "ours_residual_candidate_only",
        "ours_no_boundary_lookup",
        "ours",
        "oracle_full_rebuild_exact",
    ]:
        m = e4[e4["setting"].eq(method)]
        lines.append(
            f"- `{method}`: recall@h `{m['recall_at_h'].mean():.3f}`, precision@h `{m['precision_at_h'].mean():.3f}`, mean time `{m['end_to_end_time'].mean():.3f}s`."
        )
    (SUMMARY_DIR / "key_findings_final_formal_round6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base.setup_style()
    df = base.load_main()
    e4 = pd.read_csv(E4_CSV)
    summary: list[dict] = []
    base.plot_fig5(df, summary)
    plot_fig6_formal(df, summary)
    base.plot_fig7(df, summary)
    base.plot_fig8(df, summary)
    plot_fig9_formal(summary)
    base.plot_fig10(e4, summary)
    base.plot_fig11(df, summary)
    base.plot_fig9(df, summary)
    save_summary_formal(summary, df, e4)
    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
