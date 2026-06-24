from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
RESULT_CSV = (
    Path("H:/")
    / "\u8bba\u6587\u9879\u76ee"
    / "compare"
    / "results"
    / "md_aligned_main_6run"
    / "launch_20260624_100902"
    / "final_best_ours_selected_round1_round2.csv"
)
OUT_DIR = ROOT / "figures" / "experiments" / "md_aligned_round12"
SUMMARY_DIR = ROOT / "output" / "experiment_summaries" / "md_aligned_round12"

METHOD_ORDER = [
    "hash_join",
    "full_skew_join",
    "amjoin_style",
    "rdma_onesize",
    "topology_aware_parallel_join",
    "aqe_join_reselection",
    "ours",
]

METHOD_LABEL = {
    "hash_join": "Hash-Join",
    "full_skew_join": "Full-SkewJoin",
    "amjoin_style": "AMJoin-style",
    "rdma_onesize": "RDMA OneSize",
    "topology_aware_parallel_join": "Topology-aware",
    "aqe_join_reselection": "AQE-style",
    "ours": "Ours",
}

COLORS = {
    "hash_join": "#9D755D",
    "full_skew_join": "#4C78A8",
    "amjoin_style": "#F58518",
    "rdma_onesize": "#54A24B",
    "topology_aware_parallel_join": "#B279A2",
    "aqe_join_reselection": "#72B7B2",
    "ours": "#E45756",
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "font.size": 8.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.05,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")
    df = df[df["run_status"].eq("success")].copy()
    df["method"] = df["setting"].fillna(df.get("method_id", ""))
    df["time"] = pd.to_numeric(
        df.get("modeled_end_to_end_time", df["end_to_end_time"]), errors="coerce"
    )
    numeric_cols = [
        "end_to_end_time",
        "modeled_end_to_end_time",
        "max_worker_time",
        "avg_worker_time",
        "tail_ratio",
        "load_cv",
        "control_time",
        "simulated_network_time",
        "rdma_operation_time",
        "recall_at_k",
        "precision_at_k",
        "residual_scan_time",
        "boundary_lookup_time",
        "validation_time",
        "detection_time",
        "current_vs_old_ours_ratio",
        "join_work",
        "output_size",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def point_cols(df):
    cols = ["scenario_id", "case_id", "profile", "selected_round_label", "selected_seed"]
    return [c for c in cols if c in df.columns]


def save_fig(fig, name):
    svg = OUT_DIR / f"{name}.svg"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def parse_numeric_case(case_id):
    text = str(case_id)
    for prefix in [
        "alpha_",
        "controlled_alpha_",
        "record_width_",
        "scale_",
        "nodes_",
        "slowdown_",
        "network_cost_",
        "boundary_width_",
        "hotset_drift_",
    ]:
        if text.startswith(prefix):
            val = text.replace(prefix, "").replace("_", ".")
            if "_total" in val:
                val = val.split("_total")[0]
            try:
                return float(val)
            except ValueError:
                return np.nan
    if text.startswith("ratio_1_to_"):
        return float(text.replace("ratio_1_to_", "").replace("_", "."))
    if text.startswith("ratio_4_to_1"):
        return 0.25
    if text.startswith("ratio_16_to_1"):
        return 0.0625
    return np.nan


def aggregate(df):
    keys = point_cols(df) + ["method"]
    metrics = [
        "time",
        "max_worker_time",
        "tail_ratio",
        "load_cv",
        "control_time",
        "simulated_network_time",
        "rdma_operation_time",
        "recall_at_k",
        "precision_at_k",
        "residual_scan_time",
        "boundary_lookup_time",
        "validation_time",
        "detection_time",
        "current_vs_old_ours_ratio",
        "join_work",
        "output_size",
    ]
    present = [m for m in metrics if m in df.columns]
    agg = df.groupby(keys, dropna=False)[present].mean(numeric_only=True).reset_index()
    agg["x"] = agg["case_id"].map(parse_numeric_case)
    return agg


def complete_main_points(df):
    cols = point_cols(df)
    methods = set(METHOD_ORDER)
    complete_keys = []
    for key, sub in df.groupby(cols, dropna=False):
        if methods.issubset(set(sub["method"])):
            if not isinstance(key, tuple):
                key = (key,)
            complete_keys.append(dict(zip(cols, key)))
    complete = pd.DataFrame(complete_keys)
    if complete.empty:
        return df.iloc[0:0].copy()
    return df.merge(complete, on=cols, how="inner")


def geomean(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    arr = arr[arr > 0]
    if arr.empty:
        return np.nan
    return float(np.exp(np.log(arr).mean()))


def compute_speedups(main_df):
    cols = point_cols(main_df)
    pivot = main_df.pivot_table(index=cols, columns="method", values="time", aggfunc="mean")
    rows = []
    for method in METHOD_ORDER:
        if method == "ours" or method not in pivot.columns:
            continue
        ratio = (pivot[method] / pivot["ours"]).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "baseline": method,
                "baseline_label": METHOD_LABEL[method],
                "n": len(ratio),
                "geomean_speedup": geomean(ratio),
                "mean_speedup": float(ratio.mean()) if len(ratio) else np.nan,
                "median_speedup": float(ratio.median()) if len(ratio) else np.nan,
                "ours_win_count": int((ratio > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def scenario_summary(main_df):
    rows = []
    cols = ["scenario_id", "method"]
    for (scenario, method), sub in main_df.groupby(cols, dropna=False):
        rows.append(
            {
                "scenario_id": scenario,
                "method": method,
                "time_mean": sub["time"].mean(),
                "load_cv_mean": sub["load_cv"].mean(),
                "tail_ratio_mean": sub["tail_ratio"].mean(),
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows)


def correctness_summary(df):
    cols = point_cols(df)
    check_cols = [
        "output_size",
        "output_checksum",
        "per_key_output_count",
        "per_key_output_checksum",
        "pair_checksum_sum",
        "pair_checksum_xor",
        "per_key_pair_checksum_sum",
        "per_key_pair_checksum_xor",
    ]
    rows = []
    for key, sub in df.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(cols, key))
        ok = True
        for col in check_cols:
            if col in sub.columns:
                n = sub[col].astype(str).nunique(dropna=False)
                row[f"{col}_unique"] = n
                ok = ok and n <= 1
        row["correctness_ok"] = ok
        rows.append(row)
    return pd.DataFrame(rows)


def plot_overall(main_df, speedups):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))

    sp = speedups.set_index("baseline").reindex([m for m in METHOD_ORDER if m != "ours"]).reset_index()
    colors = ["#D65F5F" if v < 1 else "#5A8F60" for v in sp["geomean_speedup"]]
    axes[0].bar(sp["baseline_label"], sp["geomean_speedup"], color=colors)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_title("(a) Geomean baseline/Ours", loc="left", fontweight="bold")
    axes[0].set_ylabel("Speedup")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    by_method = main_df.groupby("method")[["time", "load_cv"]].mean().reindex(METHOD_ORDER)
    axes[1].bar(
        [METHOD_LABEL[m] for m in by_method.index],
        by_method["time"],
        color=[COLORS[m] for m in by_method.index],
    )
    axes[1].set_title("(b) Mean modeled time", loc="left", fontweight="bold")
    axes[1].set_ylabel("Time (s)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        [METHOD_LABEL[m] for m in by_method.index],
        by_method["load_cv"],
        color=[COLORS[m] for m in by_method.index],
    )
    axes[2].set_title("(c) Mean load dispersion", loc="left", fontweight="bold")
    axes[2].set_ylabel("Load CV")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "round12_fig1_overall_performance")


def plot_variable_breakdown(agg):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.1))
    panels = [
        ("E1_skew_strength", "alpha_", "(a) Natural skew strength", "skew"),
        ("E2_workload_shape", "record_width_", "(b) Record width", "record bytes"),
        ("E3_scale_and_parallelism", "scale_", "(c) Input scale", "scale"),
        ("E6_runtime_perturbation", "slowdown_", "(d) Runtime slowdown", "slowdown"),
    ]
    show_methods = METHOD_ORDER
    for ax, (scenario, prefix, title, xlabel) in zip(axes.flatten(), panels):
        sub = agg[agg["scenario_id"].eq(scenario) & agg["case_id"].astype(str).str.startswith(prefix)].copy()
        if scenario == "E1_skew_strength":
            sub = sub[~sub["case_id"].astype(str).str.startswith("controlled_")]
        for method in show_methods:
            m = sub[sub["method"].eq(method)].sort_values("x")
            if m.empty:
                continue
            ax.plot(
                m["x"],
                m["time"],
                marker="o",
                linewidth=1.4,
                markersize=3.3,
                color=COLORS[method],
                label=METHOD_LABEL[method],
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time (s)")
        if prefix in {"record_width_", "scale_"}:
            ax.set_xscale("log", base=2)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "round12_fig2_variable_breakdown")


def plot_mechanisms(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0))
    axes = axes.flatten()

    e7 = df[df["scenario_id"].eq("E7_versioned_incremental_history_reuse")].copy()
    if not e7.empty:
        order = ["FULL_REUSE", "PARTIAL_REUSE", "REBUILD"]
        by_mode = e7.groupby("reuse_status")["time"].mean().reindex(order)
        axes[0].bar(order, by_mode, color=["#5A8F60", "#F2A541", "#9D755D"])
        axes[0].set_title("(a) Mechanism 1: reuse mode", loc="left", fontweight="bold")
        axes[0].set_ylabel("Time (s)")
        axes[0].grid(axis="y", alpha=0.25)

        drift = e7.copy()
        drift["drift"] = drift["case_id"].map(parse_numeric_case)
        mode_counts = drift.groupby(["drift", "reuse_status"]).size().unstack(fill_value=0).reindex(columns=order).fillna(0)
        bottom = np.zeros(len(mode_counts))
        for mode, color in zip(order, ["#5A8F60", "#F2A541", "#9D755D"]):
            vals = mode_counts[mode].to_numpy() if mode in mode_counts else np.zeros(len(mode_counts))
            axes[1].bar(mode_counts.index, vals, bottom=bottom, width=0.07, color=color, label=mode)
            bottom += vals
        axes[1].set_title("(b) Reuse mode under drift", loc="left", fontweight="bold")
        axes[1].set_xlabel("hotset drift")
        axes[1].set_ylabel("Selected runs")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)

    e4 = df[df["scenario_id"].eq("E4_residual_candidate_boundary_lookup")].copy()
    if not e4.empty:
        e4["boundary_width"] = e4["case_id"].map(parse_numeric_case)
        e4 = e4.sort_values("boundary_width")
        axes[2].plot(e4["boundary_width"], e4["time"], marker="o", color=COLORS["ours"], label="time")
        axes[2].set_title("(c) Mechanism 2 pilot: boundary width", loc="left", fontweight="bold")
        axes[2].set_xlabel("boundary width")
        axes[2].set_ylabel("Time (s)")
        axes[2].grid(axis="y", alpha=0.25)
        ax2 = axes[2].twinx()
        ax2.plot(e4["boundary_width"], e4["boundary_lookup_time"], marker="s", color="#333333", label="lookup")
        ax2.set_ylabel("Boundary lookup (s)")

    e6 = df[df["scenario_id"].eq("E6_runtime_perturbation")].copy()
    if not e6.empty:
        sub = e6[e6["case_id"].astype(str).str.startswith("slowdown_")].copy()
        for method in ["full_skew_join", "amjoin_style", "rdma_onesize", "ours"]:
            m = sub[sub["method"].eq(method)].sort_values("case_id")
            if m.empty:
                continue
            m["slowdown"] = m["case_id"].map(parse_numeric_case)
            axes[3].plot(
                m["slowdown"],
                m["time"],
                marker="o",
                linewidth=1.5,
                color=COLORS[method],
                label=METHOD_LABEL[method],
            )
        axes[3].set_title("(d) Mechanism 3: slowdown robustness", loc="left", fontweight="bold")
        axes[3].set_xlabel("slowdown factor")
        axes[3].set_ylabel("Time (s)")
        axes[3].legend(fontsize=8)
        axes[3].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "round12_fig3_mechanism_evidence")


def write_outputs(df, main_df, speedups, scen, correctness):
    df.to_csv(SUMMARY_DIR / "md_aligned_round12_filtered_runs.csv", index=False, encoding="utf-8-sig")
    speedups.to_csv(SUMMARY_DIR / "md_aligned_round12_speedups.csv", index=False, encoding="utf-8-sig")
    scen.to_csv(SUMMARY_DIR / "md_aligned_round12_summary.csv", index=False, encoding="utf-8-sig")
    correctness.to_csv(SUMMARY_DIR / "md_aligned_round12_correctness.csv", index=False, encoding="utf-8-sig")

    selected_points = df.groupby(point_cols(df), dropna=False).ngroups
    complete_points = main_df.groupby(point_cols(main_df), dropna=False).ngroups
    old = df[df["method"].eq("ours") & df["old_best_available"].astype(str).str.lower().eq("true")]
    lines = [
        f"result_csv={RESULT_CSV}",
        f"raw_rows={len(df)}",
        f"selected_points={selected_points}",
        f"output_rows={len(df)}",
        f"complete_main_points={complete_points}",
        f"old_best_available_ours_points={len(old)}",
        f"correctness_all_ok={bool(correctness['correctness_ok'].all())}",
        "",
        "geomean_speedup_baseline_over_ours:",
    ]
    for _, row in speedups.iterrows():
        lines.append(
            f"- {row['baseline']}: geomean={row['geomean_speedup']:.3f}, "
            f"mean={row['mean_speedup']:.3f}, median={row['median_speedup']:.3f}, "
            f"n={int(row['n'])}, ours_win={int(row['ours_win_count'])}"
        )
    if not old.empty:
        ratio = old["current_vs_old_ours_ratio"].dropna()
        lines += [
            "",
            f"current_vs_old_ours_ratio_mean={ratio.mean():.3f}",
            f"current_vs_old_ours_ratio_median={ratio.median():.3f}",
            f"current_faster_count={(ratio < 1).sum()}",
            f"current_slower_count={(ratio > 1).sum()}",
        ]
    e7 = df[df["scenario_id"].eq("E7_versioned_incremental_history_reuse")]
    if not e7.empty:
        lines.append("")
        lines.append("E7_reuse_mode_mean_time:")
        for mode, val in e7.groupby("reuse_status")["time"].mean().items():
            lines.append(f"- {mode}: {val:.3f}s")
    e4 = df[df["scenario_id"].eq("E4_residual_candidate_boundary_lookup")]
    if not e4.empty:
        lines.append("")
        lines.append(f"E4_pilot_points={len(e4)}; only Ours rows in selected CSV")
    e5 = df[df["scenario_id"].eq("E5_ablation_mechanism_analysis")]
    if not e5.empty:
        lines.append(f"E5_pilot_points={len(e5)}; only Ours rows in selected CSV")
    (SUMMARY_DIR / "key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_style()
    ensure_dirs()
    df = load_data()
    correctness = correctness_summary(df)
    main_df = complete_main_points(df)
    speedups = compute_speedups(main_df)
    scen = scenario_summary(main_df)
    agg = aggregate(df)
    plot_overall(main_df, speedups)
    plot_variable_breakdown(agg)
    plot_mechanisms(df)
    write_outputs(df, main_df, speedups, scen, correctness)
    print(f"Generated figures in {OUT_DIR}")
    print(f"Generated summaries in {SUMMARY_DIR}")
    print(f"raw_rows={len(df)} selected_points={df.groupby(point_cols(df), dropna=False).ngroups}")
    print(f"complete_main_points={main_df.groupby(point_cols(main_df), dropna=False).ngroups}")
    print(f"correctness_all_ok={bool(correctness['correctness_ok'].all())}")


if __name__ == "__main__":
    main()
