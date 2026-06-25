from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path("H:/\u8bba\u6587\u9879\u76ee/compare/results")
MAIN_CSV = (
    RESULT_ROOT
    / "md_aligned_main_6run"
    / "launch_20260624_100902"
    / "final_best_ours_selected_round1_round2_fixed.csv"
)
SUPP_DIR = RESULT_ROOT / "supplement_final\u6574\u7406"
E4_CSV = SUPP_DIR / "e4_best_ours_selected.csv"
E1B_AUDIT_CSV = SUPP_DIR / "e1b_controlled_work_audit.csv"
STATUS_MD = SUPP_DIR / "supplement_experiment_status.md"

OUT_DIR = ROOT / "figures" / "experiments" / "final_with_supplement"
SUMMARY_DIR = ROOT / "output" / "experiment_summaries" / "final_with_supplement"

MAIN_METHODS = [
    "full_skew_join",
    "amjoin_style",
    "rdma_onesize",
    "topology_aware_parallel_join",
    "aqe_join_reselection",
    "ours",
]

BASELINES = [m for m in MAIN_METHODS if m != "ours"]

LABELS = {
    "full_skew_join": "Full-SkewJoin",
    "amjoin_style": "AMJoin-style",
    "rdma_onesize": "RDMA-OneSize",
    "topology_aware_parallel_join": "Topology-aware",
    "aqe_join_reselection": "AQE-style",
    "ours": "Ours",
    "ours_no_history_reuse": "No history reuse",
    "ours_no_versioned_incremental_stats": "No versioned stats",
    "ours_kcheck_only": "Kcheck-only",
    "ours_residual_candidate_only": "+Residual candidates",
    "ours_no_boundary_lookup": "No boundary lookup",
    "oracle_full_rebuild_exact": "Oracle rebuild",
    "ours_no_rebalance": "No runtime rebalance",
    "ours_no_bounded_probing": "No bounded probing",
    "reuse_without_validation": "No validation",
}

COLORS = {
    "Full-SkewJoin": "#8b6bb1",
    "AMJoin-style": "#c44e52",
    "RDMA-OneSize": "#55a868",
    "Topology-aware": "#dd8452",
    "AQE-style": "#7f7f7f",
    "Ours": "#2f6fbb",
    "Kcheck-only": "#9ecae1",
    "+Residual candidates": "#6baed6",
    "No boundary lookup": "#3182bd",
    "Oracle rebuild": "#969696",
    "No runtime rebalance": "#e17c7c",
    "No bounded probing": "#bcbd22",
    "No history reuse": "#e5ae38",
    "No versioned stats": "#b07aa1",
    "No validation": "#72b7b2",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def geomean(values) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return float("nan")
    return float(math.exp(np.log(vals).mean()))


def write_outputs(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def method_label(method: str) -> str:
    return LABELS.get(method, method)


def method_color(method: str) -> str:
    return COLORS.get(method_label(method), "#4c4c4c")


def parse_alpha(case_id: str) -> float:
    value = case_id
    for prefix in ["controlled_alpha_", "alpha_", "E1B_controlled_work_alpha_"]:
        if value.startswith(prefix):
            value = value.replace(prefix, "")
            break
    return float(value.replace("_", "."))


def parse_numeric_case(case_id: str, x_name: str) -> float:
    if x_name == "input_scale" and case_id.startswith("scale_"):
        return float(case_id.replace("scale_", "").replace("_", "."))
    if x_name == "parallelism":
        if case_id.startswith("nodes_") and "_total_" not in case_id:
            return float(case_id.replace("nodes_", ""))
        if "_total_" in case_id:
            return float(case_id.split("_")[1])
    if x_name == "record_size" and case_id.startswith("record_width_"):
        return float(case_id.replace("record_width_", ""))
    if x_name == "hot_key_count" and case_id.startswith("hot_key_count_"):
        return float(case_id.replace("hot_key_count_", ""))
    if x_name == "hot_overlap" and case_id.startswith("hot_overlap_"):
        return float(case_id.replace("hot_overlap_", "").replace("_", "."))
    if x_name == "table_ratio":
        if case_id.startswith("ratio_1_to_"):
            return 1.0 / float(case_id.replace("ratio_1_to_", ""))
        if case_id.startswith("ratio_") and "_to_1" in case_id:
            return float(case_id.replace("ratio_", "").replace("_to_1", ""))
    return float("nan")


def load_main() -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    df = df[df["run_status"].eq("success")].copy()
    for col in df.columns:
        if col.endswith("_time") or col in [
            "end_to_end_time",
            "modeled_end_to_end_time",
            "load_cv",
            "max_worker_time",
            "avg_worker_time",
            "join_work",
            "output_size",
            "residual_partition_ratio",
            "hotset_drift",
            "rdma_operation_count",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def complete_pivot(df: pd.DataFrame, subset: pd.DataFrame | None = None) -> pd.DataFrame:
    data = df if subset is None else subset
    data = data[data["setting"].isin(MAIN_METHODS)].copy()
    piv = data.pivot_table(
        index=["scenario_id", "case_id"],
        columns="setting",
        values="end_to_end_time",
        aggfunc="first",
    )
    return piv.dropna(subset=MAIN_METHODS)


def line_methods(ax, data: pd.DataFrame, x_col: str, y_col: str, methods, title: str, ylabel: str | None = None):
    for method in methods:
        m = data[data["setting"].eq(method)].sort_values(x_col)
        if m.empty:
            continue
        ax.plot(
            m[x_col],
            m[y_col],
            marker="o",
            lw=1.5,
            ms=4,
            label=method_label(method),
            color=method_color(method),
        )
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def grouped_case_lines(ax, data: pd.DataFrame, title: str, ylabel: str = "Modeled time (s)") -> None:
    cases = list(data["case_id"].drop_duplicates())
    x = np.arange(len(cases))
    for method in MAIN_METHODS:
        vals = []
        for case in cases:
            row = data[data["case_id"].eq(case) & data["setting"].eq(method)]
            vals.append(float(row["end_to_end_time"].iloc[0]) if not row.empty else np.nan)
        ax.plot(x, vals, marker="o", lw=1.3, ms=3.5, label=method_label(method), color=method_color(method))
    ax.set_xticks(x)
    ax.set_xticklabels([short_case(c) for c in cases], rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def short_case(case_id: str) -> str:
    return (
        case_id.replace("ratio_", "")
        .replace("_to_", ":")
        .replace("record_width_", "")
        .replace("hot_key_count_", "")
        .replace("hot_overlap_", "")
        .replace("quadrant_mix_", "")
        .replace("_skew", "")
        .replace("controlled_", "")
        .replace("hotset_drift_", "")
        .replace("slowdown_", "")
        .replace("_", ".")
    )


def plot_fig5(df: pd.DataFrame, summary: list[dict]) -> None:
    piv = complete_pivot(df)
    rows = []
    for method in MAIN_METHODS:
        rows.append(
            {
                "method": method,
                "mean_time": float(piv[method].mean()),
            }
        )
    load = (
        df[df["setting"].isin(MAIN_METHODS)]
        .pivot_table(index=["scenario_id", "case_id"], columns="setting", values="load_cv", aggfunc="first")
        .reindex(piv.index)
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), constrained_layout=True)
    labels = [method_label(r["method"]) for r in rows]
    colors = [method_color(r["method"]) for r in rows]
    axes[0].bar(labels, [r["mean_time"] for r in rows], color=colors, alpha=0.88)
    axes[0].set_title("(a) Mean completion")
    axes[0].set_ylabel("Modeled time (s)")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, [float(load[m].mean()) for m in MAIN_METHODS], color=colors, alpha=0.88)
    axes[1].set_title("(b) Mean load CV")
    axes[1].set_ylabel("Load CV")
    axes[1].tick_params(axis="x", rotation=25)

    speedup_data = [piv[b] / piv["ours"] for b in BASELINES]
    bp = axes[2].boxplot(speedup_data, patch_artist=True, labels=[method_label(b) for b in BASELINES])
    for patch, b in zip(bp["boxes"], BASELINES):
        patch.set_facecolor(method_color(b))
        patch.set_alpha(0.65)
    axes[2].axhline(1.0, color="#333333", ls="--", lw=1)
    axes[2].set_title("(c) Baseline/Ours speedup")
    axes[2].set_ylabel("Speedup")
    axes[2].tick_params(axis="x", rotation=25)
    for i, b in enumerate(BASELINES, start=1):
        gm = geomean(piv[b] / piv["ours"])
        axes[2].text(i, np.nanmax(speedup_data[i - 1]) * 1.03, f"{gm:.2f}x", ha="center", fontsize=8)
        summary.append(
            {
                "section": "overall",
                "metric": f"speedup_vs_{b}",
                "value": gm,
                "n": int((piv[b] / piv["ours"]).count()),
                "wins": int((piv[b] / piv["ours"] > 1).sum()),
            }
        )
    summary.append({"section": "overall", "metric": "complete_main_points", "value": len(piv), "n": len(piv)})
    write_outputs(fig, "fig5_overall_comparison")


def plot_fig6(df: pd.DataFrame, e1b: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E1_skew_strength") & df["setting"].isin(MAIN_METHODS)].copy()
    natural = sub[sub["case_id"].str.match(r"^alpha_", na=False)].copy()
    natural["alpha"] = natural["case_id"].map(parse_alpha)
    e1b = e1b.copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    line_methods(axes[0, 0], natural, "alpha", "end_to_end_time", MAIN_METHODS, "(a) Natural skew completion", "Modeled time (s)")
    axes[0, 0].set_xlabel("Zipf alpha")

    axes[0, 1].plot(e1b["alpha"], e1b["achieved_join_work"], marker="o", color="#2f6fbb", label="Achieved")
    axes[0, 1].axhline(float(e1b["target_join_work"].iloc[0]), color="#333333", lw=1, ls="--", label="Target")
    axes[0, 1].set_title("(b) Controlled-work audit")
    axes[0, 1].set_xlabel("Alpha")
    axes[0, 1].set_ylabel("Join work")
    axes[0, 1].legend(fontsize=8)

    join_work = natural.groupby("alpha", as_index=False)["join_work"].mean()
    axes[1, 0].plot(join_work["alpha"], join_work["join_work"], marker="o", color="#555555")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("(c) Natural join work")
    axes[1, 0].set_xlabel("Zipf alpha")
    axes[1, 0].set_ylabel("$W_{join}$")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(e1b["alpha"].astype(str), e1b["join_work_error_percent"], color="#55a868", alpha=0.85)
    axes[1, 1].axhline(1.0, color="#333333", lw=1, ls="--")
    axes[1, 1].set_title("(d) Controlled-work error")
    axes[1, 1].set_xlabel("Alpha")
    axes[1, 1].set_ylabel("Error (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    summary.append({"section": "E1B", "metric": "max_join_work_error_percent", "value": float(e1b["join_work_error_percent"].max()), "n": len(e1b)})
    write_outputs(fig, "fig6_skew_strength")


def plot_fig7(df: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E2_workload_shape") & df["setting"].isin(MAIN_METHODS)].copy()
    panels = [
        ("table_ratio", "Table ratio"),
        ("record_size", "Record width"),
        ("hot_key_count", "Hot-key count"),
        ("hot_overlap", "Hot overlap"),
        ("quadrant_mix", "HH/HC/CH/CC mix"),
        ("correlation_profile", "Correlation profile"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0), constrained_layout=True)
    for ax, (x_name, title) in zip(axes.flat, panels):
        data = sub[sub["x_name"].eq(x_name)].copy()
        if x_name in {"table_ratio", "record_size", "hot_key_count", "hot_overlap"}:
            data["x_num"] = data.apply(lambda r: parse_numeric_case(str(r["case_id"]), x_name), axis=1)
            for method in MAIN_METHODS:
                m = data[data["setting"].eq(method)].sort_values("x_num")
                ax.plot(m["x_num"], m["end_to_end_time"], marker="o", lw=1.2, ms=3, label=method_label(method), color=method_color(method))
            if x_name in {"record_size", "hot_key_count"}:
                ax.set_xscale("log")
            ax.set_xlabel(title)
            ax.set_ylabel("Modeled time (s)")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        else:
            grouped_case_lines(ax, data, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    write_outputs(fig, "fig7_workload_shape")

    piv = sub.pivot_table(index=["case_id"], columns="setting", values="end_to_end_time", aggfunc="first").dropna(subset=MAIN_METHODS)
    for b in BASELINES:
        summary.append({"section": "E2", "metric": f"speedup_vs_{b}", "value": geomean(piv[b] / piv["ours"]), "n": len(piv)})


def plot_fig8(df: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E3_scale_and_parallelism") & df["setting"].isin(MAIN_METHODS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), constrained_layout=True)
    groups = [
        ("scale_", "(a) Input scale", lambda c: parse_numeric_case(c, "input_scale")),
        ("nodes_", "(b) Strong scaling", lambda c: parse_numeric_case(c, "parallelism")),
        ("nodes_", "(c) Weak scaling", lambda c: parse_numeric_case(c, "parallelism")),
    ]
    for ax, (prefix, title, parser) in zip(axes, groups):
        if "Weak" in title:
            data = sub[sub["case_id"].str.contains("_total_", na=False)].copy()
        elif "Strong" in title:
            data = sub[sub["case_id"].str.startswith("nodes_", na=False) & ~sub["case_id"].str.contains("_total_", na=False)].copy()
        else:
            data = sub[sub["case_id"].str.startswith(prefix, na=False)].copy()
        data["x_num"] = data["case_id"].map(parser)
        line_methods(ax, data, "x_num", "end_to_end_time", MAIN_METHODS, title, "Modeled time (s)")
        ax.set_xlabel("Scale" if "Input" in title else "Nodes")
        if "Input" in title:
            ax.set_xscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    write_outputs(fig, "fig8_scale_parallelism")

    piv = sub.pivot_table(index=["case_id"], columns="setting", values="end_to_end_time", aggfunc="first").dropna(subset=MAIN_METHODS)
    for b in BASELINES:
        summary.append({"section": "E3", "metric": f"speedup_vs_{b}", "value": geomean(piv[b] / piv["ours"]), "n": len(piv)})


def plot_fig9(df: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E5_ablation_mechanism_analysis")].copy()
    order = [
        "ours",
        "ours_no_history_reuse",
        "ours_no_versioned_incremental_stats",
        "ours_kcheck_only",
        "ours_no_boundary_lookup",
        "ours_no_rebalance",
        "ours_no_bounded_probing",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), constrained_layout=True)
    # Completion time by trigger group.
    cases = list(sub["case_id"].drop_duplicates())
    x = np.arange(len(cases))
    width = 0.12
    for idx, method in enumerate(order):
        vals = []
        for case in cases:
            row = sub[sub["case_id"].eq(case) & sub["setting"].eq(method)]
            vals.append(float(row["end_to_end_time"].iloc[0]) if not row.empty else np.nan)
        axes[0, 0].bar(x + (idx - 3) * width, vals, width, label=method_label(method), color=method_color(method), alpha=0.85)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(["History", "M2", "Runtime"])
    axes[0, 0].set_title("(a) Completion by trigger")
    axes[0, 0].set_ylabel("Modeled time (s)")

    slow_rows = []
    for case in cases:
        ours = sub[sub["case_id"].eq(case) & sub["setting"].eq("ours")]["end_to_end_time"]
        if ours.empty:
            continue
        ours_t = float(ours.iloc[0])
        for method in order:
            row = sub[sub["case_id"].eq(case) & sub["setting"].eq(method)]
            if not row.empty:
                slow_rows.append({"case": case, "method": method, "slowdown": float(row["end_to_end_time"].iloc[0]) / ours_t})
    slow = pd.DataFrame(slow_rows)
    labels = [method_label(m) for m in order if m != "ours"]
    vals = [float(slow[slow["method"].eq(m)]["slowdown"].mean()) for m in order if m != "ours"]
    axes[0, 1].bar(labels, vals, color=[method_color(m) for m in order if m != "ours"], alpha=0.85)
    axes[0, 1].axhline(1.0, color="#333333", ls="--", lw=1)
    axes[0, 1].set_title("(b) Slowdown vs Ours")
    axes[0, 1].set_ylabel("Relative time")
    axes[0, 1].tick_params(axis="x", rotation=25)

    for idx, method in enumerate(order):
        vals = []
        for case in cases:
            row = sub[sub["case_id"].eq(case) & sub["setting"].eq(method)]
            vals.append(float(row["load_cv"].iloc[0]) if not row.empty else np.nan)
        axes[1, 0].bar(x + (idx - 3) * width, vals, width, label=method_label(method), color=method_color(method), alpha=0.85)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(["History", "M2", "Runtime"])
    axes[1, 0].set_title("(c) Load CV")
    axes[1, 0].set_ylabel("Load CV")

    axes[1, 1].axis("off")
    mapping = [
        "History: no history reuse / no versioned stats",
        "M2: Kcheck-only / no boundary lookup",
        "Runtime: no rebalance / no bounded probing",
        "Each ablation is evaluated on its trigger case.",
    ]
    axes[1, 1].text(0.02, 0.88, "(d) Ablation mapping", fontsize=10, fontweight="bold")
    axes[1, 1].text(0.02, 0.72, "\n".join(mapping), va="top", fontsize=9)
    axes[0, 0].legend(ncol=3, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.32))
    write_outputs(fig, "fig9_mechanism_ablation")
    for _, row in slow.iterrows():
        summary.append({"section": "E5", "metric": f"{row['case']}::{row['method']}_slowdown", "value": row["slowdown"], "n": 1})


def plot_fig10(e4: pd.DataFrame, summary: list[dict]) -> None:
    methods = [
        "ours_kcheck_only",
        "ours_residual_candidate_only",
        "ours_no_boundary_lookup",
        "ours",
        "oracle_full_rebuild_exact",
    ]
    cases = list(e4["case_id"].drop_duplicates())
    x = np.arange(len(cases))
    width = 0.15
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5), constrained_layout=True)
    metrics = [
        ("recall_at_h", "(a) Recall@h", "Recall@h"),
        ("precision_at_h", "(b) Precision@h", "Precision@h"),
        ("boundary_lookup_count", "(c) Boundary lookup", "Keys"),
        ("end_to_end_time", "(d) Completion", "Modeled time (s)"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        for idx, method in enumerate(methods):
            vals = []
            for case in cases:
                row = e4[e4["case_id"].eq(case) & e4["setting"].eq(method)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (idx - 2) * width, vals, width, label=method_label(method), color=method_color(method), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cases], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if metric in {"recall_at_h", "precision_at_h"}:
            ax.set_ylim(0, 1.08)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    write_outputs(fig, "fig10_m2_candidate_boundary")

    for method in methods:
        m = e4[e4["setting"].eq(method)]
        summary.append({"section": "E4_supplement", "metric": f"{method}_mean_recall_at_h", "value": float(m["recall_at_h"].mean()), "n": len(m)})
        summary.append({"section": "E4_supplement", "metric": f"{method}_mean_precision_at_h", "value": float(m["precision_at_h"].mean()), "n": len(m)})
        summary.append({"section": "E4_supplement", "metric": f"{method}_mean_time", "value": float(m["end_to_end_time"].mean()), "n": len(m)})


def plot_fig11(df: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E6_runtime_perturbation")].copy()
    sub["slowdown"] = sub["case_id"].str.replace("slowdown_", "", regex=False).str.replace("_", ".").astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), constrained_layout=True)
    line_methods(axes[0, 0], sub[sub["setting"].isin(MAIN_METHODS)], "slowdown", "end_to_end_time", MAIN_METHODS, "(a) Runtime slowdown sweep", "Modeled time (s)")
    axes[0, 0].set_xlabel("Slowdown factor")
    line_methods(axes[0, 1], sub[sub["setting"].isin(MAIN_METHODS)], "slowdown", "load_cv", MAIN_METHODS, "(b) Load CV", "Load CV")
    axes[0, 1].set_xlabel("Slowdown factor")
    line_methods(axes[1, 0], sub[sub["setting"].isin(["ours", "ours_no_rebalance"])], "slowdown", "end_to_end_time", ["ours", "ours_no_rebalance"], "(c) Rebalance ablation", "Modeled time (s)")
    axes[1, 0].set_xlabel("Slowdown factor")
    piv = sub[sub["setting"].isin(["ours", "ours_no_rebalance"])].pivot_table(index="slowdown", columns="setting", values="end_to_end_time", aggfunc="first")
    ratio = piv["ours_no_rebalance"] / piv["ours"]
    axes[1, 1].bar(ratio.index.astype(str), ratio.values, color="#e17c7c", alpha=0.85)
    axes[1, 1].axhline(1.0, color="#333333", ls="--", lw=1)
    axes[1, 1].set_title("(d) No-rebalance/Ours")
    axes[1, 1].set_xlabel("Slowdown factor")
    axes[1, 1].set_ylabel("Relative time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    write_outputs(fig, "fig11_runtime_rebalance")
    summary.append({"section": "E6", "metric": "no_rebalance_mean_slowdown", "value": float(ratio.mean()), "n": int(ratio.count())})


def plot_fig12(df: pd.DataFrame, e1b: pd.DataFrame, summary: list[dict]) -> None:
    sub = df[df["scenario_id"].eq("E7_versioned_incremental_history_reuse")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), constrained_layout=True)
    ours = sub[sub["setting"].eq("ours")].copy()
    order_status = ["FULL_REUSE", "PARTIAL_REUSE", "REBUILD"]
    means = [float(ours[ours["reuse_status"].eq(s)]["end_to_end_time"].mean()) for s in order_status]
    axes[0, 0].bar(order_status, means, color=["#55a868", "#dd8452", "#8b6bb1"], alpha=0.88)
    axes[0, 0].set_title("(a) Reuse mode completion")
    axes[0, 0].set_ylabel("Modeled time (s)")
    axes[0, 0].tick_params(axis="x", rotation=15)

    breakdown_cols = [c for c in ["detection_time", "validation_time", "routing_time"] if c in sub.columns]
    bottom = np.zeros(len(order_status))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for col, color in zip(breakdown_cols, colors):
        vals = [float(ours[ours["reuse_status"].eq(s)][col].mean()) for s in order_status]
        axes[0, 1].bar(order_status, vals, bottom=bottom, label=col.replace("_time", ""), color=color, alpha=0.85)
        bottom += np.nan_to_num(vals)
    axes[0, 1].set_title("(b) Control-time breakdown")
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].legend(fontsize=8)

    drift = sub.copy()
    drift["drift"] = drift["case_id"].str.replace("hotset_drift_", "", regex=False).str.replace("_", ".").astype(float)
    for method in ["ours", "ours_no_history_reuse", "ours_no_versioned_incremental_stats", "reuse_without_validation"]:
        m = drift[drift["setting"].eq(method)].groupby("drift", as_index=False)["end_to_end_time"].mean()
        axes[1, 0].plot(m["drift"], m["end_to_end_time"], marker="o", lw=1.4, label=method_label(method), color=method_color(method))
    axes[1, 0].set_title("(c) Hotset drift")
    axes[1, 0].set_xlabel("Drift")
    axes[1, 0].set_ylabel("Modeled time (s)")
    axes[1, 0].grid(axis="y", alpha=0.25)

    counts = ours.groupby(["case_id", "reuse_status"]).size().unstack(fill_value=0).reindex(columns=order_status, fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["#55a868", "#dd8452", "#8b6bb1"], alpha=0.88, legend=False)
    axes[1, 1].set_title("(d) Selected reuse modes")
    axes[1, 1].set_xlabel("Case")
    axes[1, 1].set_ylabel("Rows")
    axes[1, 1].set_xticklabels([short_case(c) for c in counts.index], rotation=25, ha="right")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    write_outputs(fig, "fig12_history_reuse")
    for status, val in zip(order_status, means):
        summary.append({"section": "E7", "metric": f"{status}_mean_time", "value": val, "n": int(ours["reuse_status"].eq(status).sum())})
    summary.append({"section": "E1B", "metric": "controlled_work_audit_pass_rows", "value": int(e1b["validation"].eq("pass").sum()), "n": len(e1b)})


def save_summary(summary: list[dict], df: pd.DataFrame, e4: pd.DataFrame, e1b: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(summary)
    out.to_csv(SUMMARY_DIR / "summary_final_with_supplement.csv", index=False, encoding="utf-8-sig")
    complete = complete_pivot(df)
    lines = [
        "# Final with Supplement Key Findings",
        "",
        f"- Main input rows after success filter: `{len(df)}`.",
        f"- Complete main-method points without `hash_join`: `{len(complete)}`.",
        f"- E4 supplement rows: `{len(e4)}` across `{e4['case_id'].nunique()}` cases and `{e4['setting'].nunique()}` methods.",
        f"- E1B controlled-work audit rows: `{len(e1b)}`; max error `{e1b['join_work_error_percent'].max():.4f}%`.",
        "",
        "## Overall speedups",
    ]
    for b in BASELINES:
        ratio = complete[b] / complete["ours"]
        lines.append(
            f"- Baseline/Ours `{b}`: geomean `{geomean(ratio):.3f}x`, wins for Ours `{int((ratio > 1).sum())}/{int(ratio.count())}`."
        )
    lines.extend(["", "## E4 supplement"])
    for method in ["ours_kcheck_only", "ours_residual_candidate_only", "ours_no_boundary_lookup", "ours", "oracle_full_rebuild_exact"]:
        m = e4[e4["setting"].eq(method)]
        lines.append(
            f"- `{method}`: recall@h `{m['recall_at_h'].mean():.3f}`, precision@h `{m['precision_at_h'].mean():.3f}`, mean time `{m['end_to_end_time'].mean():.3f}s`."
        )
    lines.extend(["", "## Evidence boundaries"])
    lines.append("- E1B audit validates the controlled-work generator, but the main CSV controlled completion rows are not used as fixed-work performance evidence.")
    lines.append("- E6/E7 internal OSP counters are not used as main evidence because current counter fields are unavailable or all zero.")
    (SUMMARY_DIR / "key_findings_final_with_supplement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    df = load_main()
    e4 = pd.read_csv(E4_CSV)
    e1b = pd.read_csv(E1B_AUDIT_CSV)
    summary: list[dict] = []
    plot_fig5(df, summary)
    plot_fig6(df, e1b, summary)
    plot_fig7(df, summary)
    plot_fig8(df, summary)
    plot_fig9(df, summary)
    plot_fig10(e4, summary)
    plot_fig11(df, summary)
    plot_fig12(df, e1b, summary)
    save_summary(summary, df, e4, e1b)
    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
