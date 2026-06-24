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
OUT_DIR = ROOT / "figures" / "experiments" / "round12_legacy_flow"
SUMMARY_DIR = ROOT / "output" / "experiment_summaries" / "round12_legacy_flow"

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
    "rdma_onesize": "RDMA-OneSize",
    "topology_aware_parallel_join": "Topology-aware",
    "aqe_join_reselection": "AQE-style",
    "ours": "Ours",
}

COLORS = {
    "hash_join": "#8C6D62",
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
            "font.size": 8.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
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
        "stable_state_lookup_time",
        "validation_time",
        "detection_time",
        "current_vs_old_ours_ratio",
        "join_work",
        "output_size",
        "local_join_time",
        "initial_plan_synthesis_time",
        "residual_partition_ratio",
        "state_storage_bytes",
        "metadata_compare_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def point_cols(df):
    cols = ["scenario_id", "case_id", "profile", "selected_round_label", "selected_seed"]
    return [c for c in cols if c in df.columns]


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
        "hot_key_count_",
        "hot_overlap_",
    ]:
        if text.startswith(prefix):
            val = text.replace(prefix, "").replace("_total", "").replace("_", ".")
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


def save_fig(fig, name):
    svg = OUT_DIR / f"{name}.svg"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def geomean(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    arr = arr[arr > 0]
    if arr.empty:
        return np.nan
    return float(np.exp(np.log(arr).mean()))


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


def aggregate(df):
    keys = point_cols(df) + ["method"]
    metrics = [
        "time",
        "load_cv",
        "tail_ratio",
        "max_worker_time",
        "avg_worker_time",
        "control_time",
        "rdma_operation_time",
        "simulated_network_time",
        "local_join_time",
        "join_work",
        "output_size",
        "recall_at_k",
        "precision_at_k",
        "boundary_lookup_time",
        "residual_scan_time",
        "initial_plan_synthesis_time",
        "validation_time",
        "current_vs_old_ours_ratio",
    ]
    present = [m for m in metrics if m in df.columns]
    out = df.groupby(keys, dropna=False)[present].mean(numeric_only=True).reset_index()
    out["x"] = out["case_id"].map(parse_numeric_case)
    return out


def compute_speedups(main_df):
    cols = point_cols(main_df)
    pivot = main_df.pivot_table(index=cols, columns="method", values="time", aggfunc="mean")
    rows = []
    for method in METHOD_ORDER:
        if method == "ours":
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


def plot_lines(ax, sub, methods, title, xlabel, ylabel="Modeled time (s)", xscale=None):
    for method in methods:
        m = sub[sub["method"].eq(method)].sort_values("x")
        if m.empty:
            continue
        ax.plot(
            m["x"],
            m["time"],
            marker="o",
            linewidth=1.45,
            markersize=3.2,
            color=COLORS[method],
            label=METHOD_LABEL[method],
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale:
        ax.set_xscale(xscale, base=2)
    ax.grid(axis="y", alpha=0.25)


def plot_fig5_baseline(main_df, speedups):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    sp = speedups.set_index("baseline").reindex([m for m in METHOD_ORDER if m != "ours"]).reset_index()
    colors = ["#D65F5F" if v < 1 else "#5A8F60" for v in sp["geomean_speedup"]]
    axes[0].bar(sp["baseline_label"], sp["geomean_speedup"], color=colors)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_title("(a) Geomean speedup", loc="left", fontweight="bold")
    axes[0].set_ylabel("Baseline / Ours")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].grid(axis="y", alpha=0.25)

    by_method = main_df.groupby("method")[["time", "load_cv"]].mean().reindex(METHOD_ORDER)
    axes[1].bar([METHOD_LABEL[m] for m in by_method.index], by_method["time"], color=[COLORS[m] for m in by_method.index])
    axes[1].set_title("(b) Mean modeled time", loc="left", fontweight="bold")
    axes[1].set_ylabel("Time (s)")
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar([METHOD_LABEL[m] for m in by_method.index], by_method["load_cv"], color=[COLORS[m] for m in by_method.index])
    axes[2].set_title("(c) Mean load CV", loc="left", fontweight="bold")
    axes[2].set_ylabel("Load CV")
    axes[2].tick_params(axis="x", rotation=28)
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "round12_fig5_baseline_overview")


def plot_fig6_skew(agg):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7))
    natural = agg[
        agg["scenario_id"].eq("E1_skew_strength")
        & agg["case_id"].astype(str).str.startswith("alpha_")
        & ~agg["case_id"].astype(str).str.startswith("controlled_")
    ]
    controlled = agg[
        agg["scenario_id"].eq("E1_skew_strength")
        & agg["case_id"].astype(str).str.startswith("controlled_alpha_")
    ]
    methods = METHOD_ORDER
    plot_lines(axes[0], natural, methods, "(a) Natural Zipf skew", "Zipf alpha")
    plot_lines(axes[1], controlled, methods, "(b) Controlled join work", "Zipf alpha")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "round12_fig6_skew_sweep")


def plot_fig7_workload(agg):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.1))
    methods = METHOD_ORDER
    panels = [
        ("ratio_", "(a) Table-size ratio", "relative small side"),
        ("record_width_", "(b) Record width", "bytes"),
        ("hot_overlap_", "(c) Hot-key overlap", "overlap"),
        ("hot_key_count_", "(d) Hot-key count", "count"),
    ]
    for ax, (prefix, title, xlabel) in zip(axes.flatten(), panels):
        sub = agg[
            agg["scenario_id"].eq("E2_workload_shape")
            & agg["case_id"].astype(str).str.startswith(prefix)
        ].copy()
        plot_lines(ax, sub, methods, title, xlabel, xscale="log" if prefix in {"record_width_", "hot_key_count_"} else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "round12_fig7_workload_shape")


def plot_fig8_scale(agg):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    methods = METHOD_ORDER
    scale = agg[agg["scenario_id"].eq("E3_scale_and_parallelism") & agg["case_id"].astype(str).str.startswith("scale_")]
    strong = agg[
        agg["scenario_id"].eq("E3_scale_and_parallelism")
        & agg["case_id"].astype(str).str.match(r"nodes_(4|8|16|32|64)$")
    ]
    weak = agg[
        agg["scenario_id"].eq("E3_scale_and_parallelism")
        & agg["case_id"].astype(str).str.contains("_total_")
    ]
    plot_lines(axes[0], scale, methods, "(a) Input scale", "scale", xscale="log")
    plot_lines(axes[1], strong, methods, "(b) Strong scaling", "workers", xscale="log")
    plot_lines(axes[2], weak, methods, "(c) Weak scaling", "workers", xscale="log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "round12_fig8_scale_parallelism")


def plot_fig9_mechanism_dashboard(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))

    e5 = df[df["scenario_id"].eq("E5_ablation_mechanism_analysis")].copy()
    if not e5.empty:
        labels = {
            "ablation_candidate_boundary": "M2 diag",
            "ablation_history_reuse": "M1 diag",
            "ablation_runtime_rebalance": "M3 diag",
        }
        e5["label"] = e5["case_id"].map(labels).fillna(e5["case_id"])
        axes[0].bar(e5["label"], e5["time"], color="#E45756")
        axes[0].set_title("(a) Diagnostic mechanism cases", loc="left", fontweight="bold")
        axes[0].set_ylabel("Time (s)")
        axes[0].tick_params(axis="x", rotation=18)
        axes[0].grid(axis="y", alpha=0.25)

    e7 = df[df["scenario_id"].eq("E7_versioned_incremental_history_reuse")].copy()
    order = ["FULL_REUSE", "PARTIAL_REUSE", "REBUILD"]
    if not e7.empty:
        by_mode = e7.groupby("reuse_status")["time"].mean().reindex(order)
        axes[1].bar(order, by_mode, color=["#5A8F60", "#F2A541", "#9D755D"])
        axes[1].set_title("(b) Mechanism 1 modes", loc="left", fontweight="bold")
        axes[1].set_ylabel("Time (s)")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].grid(axis="y", alpha=0.25)

    e6 = df[
        df["scenario_id"].eq("E6_runtime_perturbation")
        & df["case_id"].astype(str).str.startswith("slowdown_")
        & df["method"].isin(["full_skew_join", "amjoin_style", "rdma_onesize", "ours"])
    ].copy()
    if not e6.empty:
        e6["slowdown"] = e6["case_id"].map(parse_numeric_case)
        for method in ["full_skew_join", "amjoin_style", "rdma_onesize", "ours"]:
            m = e6[e6["method"].eq(method)].sort_values("slowdown")
            axes[2].plot(m["slowdown"], m["time"], marker="o", linewidth=1.4, color=COLORS[method], label=METHOD_LABEL[method])
        axes[2].set_title("(c) Runtime slowdown", loc="left", fontweight="bold")
        axes[2].set_xlabel("slowdown")
        axes[2].set_ylabel("Time (s)")
        axes[2].legend(fontsize=8)
        axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "round12_fig9_mechanism_dashboard")


def plot_fig10_bilateral_diag(df):
    e4 = df[df["scenario_id"].eq("E4_residual_candidate_boundary_lookup")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6))
    if not e4.empty:
        e4["boundary_width"] = e4["case_id"].map(parse_numeric_case)
        e4 = e4.sort_values("boundary_width")
        axes[0].plot(e4["boundary_width"], e4["time"], marker="o", color=COLORS["ours"])
        axes[0].set_title("(a) Modeled time", loc="left", fontweight="bold")
        axes[0].set_xlabel("boundary width")
        axes[0].set_ylabel("Time (s)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].plot(e4["boundary_width"], e4["boundary_lookup_time"], marker="s", color="#333333")
        axes[1].set_title("(b) Boundary lookup", loc="left", fontweight="bold")
        axes[1].set_xlabel("boundary width")
        axes[1].set_ylabel("Lookup time (s)")
        axes[1].grid(axis="y", alpha=0.25)
        axes[2].plot(e4["boundary_width"], e4["residual_scan_time"], marker="^", color="#4C78A8", label="residual scan")
        if "initial_plan_synthesis_time" in e4.columns:
            axes[2].plot(e4["boundary_width"], e4["initial_plan_synthesis_time"], marker="o", color="#F58518", label="plan")
        axes[2].set_title("(c) Control subtime", loc="left", fontweight="bold")
        axes[2].set_xlabel("boundary width")
        axes[2].set_ylabel("Time (s)")
        axes[2].legend(fontsize=8)
        axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "round12_fig10_bilateral_load_diagnostic")


def plot_fig11_runtime(agg):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7))
    sub = agg[
        agg["scenario_id"].eq("E6_runtime_perturbation")
        & agg["case_id"].astype(str).str.startswith("slowdown_")
    ].copy()
    plot_lines(axes[0], sub, METHOD_ORDER, "(a) Runtime slowdown", "slowdown")
    for method in METHOD_ORDER:
        m = sub[sub["method"].eq(method)].sort_values("x")
        if m.empty:
            continue
        axes[1].plot(m["x"], m["load_cv"], marker="o", linewidth=1.4, color=COLORS[method], label=METHOD_LABEL[method])
    axes[1].set_title("(b) Load dispersion under slowdown", loc="left", fontweight="bold")
    axes[1].set_xlabel("slowdown")
    axes[1].set_ylabel("Load CV")
    axes[1].grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "round12_fig11_runtime_perturbation")


def plot_fig12_reuse(df):
    e7 = df[df["scenario_id"].eq("E7_versioned_incremental_history_reuse")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    if not e7.empty:
        order = ["FULL_REUSE", "PARTIAL_REUSE", "REBUILD"]
        e7["drift"] = e7["case_id"].map(parse_numeric_case)
        by_mode = e7.groupby("reuse_status")["time"].mean().reindex(order)
        axes[0].bar(order, by_mode, color=["#5A8F60", "#F2A541", "#9D755D"])
        axes[0].set_title("(a) Reuse mode time", loc="left", fontweight="bold")
        axes[0].set_ylabel("Time (s)")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(axis="y", alpha=0.25)

        mode_counts = e7.groupby(["drift", "reuse_status"]).size().unstack(fill_value=0).reindex(columns=order).fillna(0)
        bottom = np.zeros(len(mode_counts))
        for mode, color in zip(order, ["#5A8F60", "#F2A541", "#9D755D"]):
            vals = mode_counts[mode].to_numpy()
            axes[1].bar(mode_counts.index, vals, bottom=bottom, width=0.07, color=color, label=mode)
            bottom += vals
        axes[1].set_title("(b) Selected mode by drift", loc="left", fontweight="bold")
        axes[1].set_xlabel("hotset drift")
        axes[1].set_ylabel("runs")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)

        partial = (
            e7[e7["reuse_status"].eq("PARTIAL_REUSE")]
            .groupby("drift", as_index=False)[["residual_scan_time", "boundary_lookup_time"]]
            .mean()
            .sort_values("drift")
        )
        if not partial.empty:
            axes[2].plot(partial["drift"], partial["residual_scan_time"], marker="o", color="#4C78A8", label="residual scan")
            axes[2].plot(partial["drift"], partial["boundary_lookup_time"], marker="s", color="#F58518", label="boundary lookup")
        axes[2].set_title("(c) Partial reuse cost", loc="left", fontweight="bold")
        axes[2].set_xlabel("hotset drift")
        axes[2].set_ylabel("Time (s)")
        axes[2].legend(fontsize=8)
        axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "round12_fig12_reuse_profile")


def write_summary(df, main_df, speedups, correctness):
    df.to_csv(SUMMARY_DIR / "round12_legacy_flow_filtered_runs.csv", index=False, encoding="utf-8-sig")
    speedups.to_csv(SUMMARY_DIR / "round12_legacy_flow_speedups.csv", index=False, encoding="utf-8-sig")
    correctness.to_csv(SUMMARY_DIR / "round12_legacy_flow_correctness.csv", index=False, encoding="utf-8-sig")

    selected_points = df.groupby(point_cols(df), dropna=False).ngroups
    complete_points = main_df.groupby(point_cols(main_df), dropna=False).ngroups
    old = df[df["method"].eq("ours") & df["old_best_available"].astype(str).str.lower().eq("true")]
    e7 = df[df["scenario_id"].eq("E7_versioned_incremental_history_reuse")]
    lines = [
        f"result_csv={RESULT_CSV}",
        f"raw_rows={len(df)}",
        f"selected_points={selected_points}",
        f"output_rows={len(df)}",
        f"complete_main_points={complete_points}",
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
            f"old_best_available_ours_points={len(old)}",
            f"current_vs_old_ours_ratio_mean={ratio.mean():.3f}",
            f"current_vs_old_ours_ratio_median={ratio.median():.3f}",
            f"current_faster_count={(ratio < 1).sum()}",
            f"current_slower_count={(ratio > 1).sum()}",
        ]
    if not e7.empty:
        lines += ["", "E7_reuse_mode_mean_time:"]
        for mode in ["FULL_REUSE", "PARTIAL_REUSE", "REBUILD"]:
            val = e7[e7["reuse_status"].eq(mode)]["time"].mean()
            lines.append(f"- {mode}: {val:.3f}s")
    lines += [
        "",
        f"E4_diagnostic_points={len(df[df['scenario_id'].eq('E4_residual_candidate_boundary_lookup')])}",
        f"E5_diagnostic_points={len(df[df['scenario_id'].eq('E5_ablation_mechanism_analysis')])}",
    ]
    (SUMMARY_DIR / "key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_style()
    ensure_dirs()
    df = load_data()
    correctness = correctness_summary(df)
    main_df = complete_main_points(df)
    speedups = compute_speedups(main_df)
    agg = aggregate(df)
    plot_fig5_baseline(main_df, speedups)
    plot_fig6_skew(agg)
    plot_fig7_workload(agg)
    plot_fig8_scale(agg)
    plot_fig9_mechanism_dashboard(df)
    plot_fig10_bilateral_diag(df)
    plot_fig11_runtime(agg)
    plot_fig12_reuse(df)
    write_summary(df, main_df, speedups, correctness)
    print(f"Generated figures in {OUT_DIR}")
    print(f"Generated summaries in {SUMMARY_DIR}")
    print(f"raw_rows={len(df)} selected_points={df.groupby(point_cols(df), dropna=False).ngroups}")
    print(f"complete_main_points={main_df.groupby(point_cols(main_df), dropna=False).ngroups}")
    print(f"correctness_all_ok={bool(correctness['correctness_ok'].all())}")


if __name__ == "__main__":
    main()
