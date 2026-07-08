from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "*/compare/results/formal_l3_matrix/draft_theory_placeholder_20260708_final_clean_v4/all_runs.csv"
RESULT_CSV_ENV = os.environ.get("RESULT_CSV")
FIG_DIR = ROOT / "figures" / "experiments" / "formal_l3_20260708_v4"
SUMMARY_DIR = ROOT / "output" / "experiment_summaries" / "formal_l3_20260708_final_clean_v4"

TIME_COL = "modeled_end_to_end_time"
OURS = "Ours"
MAIN_METHODS = [
    "Standard Dist-HJ",
    "Full-SkewJoin",
    "AMJoin-style",
    "OneSize-style",
    "Bala-Join-style",
    "Flow-Join-style",
    "Ours",
]

METHOD_ORDER = MAIN_METHODS + [
    "M1 Full Rebuild",
    "M1 Blind Reuse",
    "M1 Oracle Mode",
    "Ours w/o History Reuse",
    "Ours w/o Versioned Incremental Stats",
    "Exact Full Count",
    "Top-h Only",
    "No Cross-Side Completion",
    "Ours K_check Only",
    "Ours w/o Boundary Lookup",
    "Ours Residual Candidate Only",
    "Random Probing",
    "Periodic Global Polling",
    "Oracle Provider Selection",
    "Ours w/o Bounded Probing",
    "Ours-No-Rebalance",
]

METHOD_COLORS = {
    "Ours": "#b2182b",
    "Standard Dist-HJ": "#4d4d4d",
    "Full-SkewJoin": "#2166ac",
    "AMJoin-style": "#ef8a62",
    "OneSize-style": "#1b9e77",
    "Bala-Join-style": "#762a83",
    "Flow-Join-style": "#00a6a6",
    "M1 Full Rebuild": "#8c510a",
    "M1 Blind Reuse": "#bf812d",
    "M1 Oracle Mode": "#35978f",
    "Ours w/o History Reuse": "#d6604d",
    "Ours w/o Versioned Incremental Stats": "#f4a582",
    "Exact Full Count": "#404040",
    "Top-h Only": "#8c8c8c",
    "No Cross-Side Completion": "#a6dba0",
    "Ours K_check Only": "#92c5de",
    "Ours w/o Boundary Lookup": "#4393c3",
    "Ours Residual Candidate Only": "#b2abd2",
    "Random Probing": "#c994c7",
    "Periodic Global Polling": "#df65b0",
    "Oracle Provider Selection": "#1b7837",
    "Ours w/o Bounded Probing": "#f1b6da",
    "Ours-No-Rebalance": "#c51b7d",
}

SHORT_METHOD_LABELS = {
    "Standard Dist-HJ": "Dist-HJ",
    "Full-SkewJoin": "Full-Skew",
    "Bala-Join-style": "Bala-style",
    "Flow-Join-style": "Flow-style",
    "M1 Full Rebuild": "Full rebuild",
    "M1 Blind Reuse": "Blind reuse",
    "M1 Oracle Mode": "Oracle mode",
    "Ours w/o History Reuse": "No history",
    "Ours w/o Versioned Incremental Stats": "No versioned stats",
    "Exact Full Count": "Exact count",
    "Top-h Only": "Top-h only",
    "No Cross-Side Completion": "No cross-side",
    "Ours K_check Only": "K-check only",
    "Ours w/o Boundary Lookup": "No boundary",
    "Ours Residual Candidate Only": "Residual only",
    "Random Probing": "Random",
    "Periodic Global Polling": "Periodic",
    "Oracle Provider Selection": "Oracle provider",
    "Ours w/o Bounded Probing": "No bounded probe",
    "Ours-No-Rebalance": "No rebalance",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "font.size": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.4,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


@dataclass(frozen=True)
class FigureRecord:
    name: str
    title: str
    claim: str


def resolve_result_csv() -> Path:
    if RESULT_CSV_ENV:
        p = Path(RESULT_CSV_ENV)
        if not p.exists():
            raise FileNotFoundError(f"RESULT_CSV does not exist: {p}")
        return p
    matches = list(Path("H:/").glob(DEFAULT_GLOB))
    if not matches:
        raise FileNotFoundError(f"Cannot locate result CSV with H:/{DEFAULT_GLOB}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matching result CSVs: {matches}")
    return matches[0]


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _geomean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if pd.notna(v) and v > 0], dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(arr))))


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.astype(float).replace(0, np.nan)


def _method_sort(methods: Iterable[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted([m for m in methods if pd.notna(m)], key=lambda m: (rank.get(m, 999), m))


def _short_method(method: str) -> str:
    return SHORT_METHOD_LABELS.get(method, method)


def _format_value(v: object) -> str:
    if pd.isna(v):
        return "NA"
    text = str(v)
    replacements = {
        "public_workload_default_mixed": "default mixed",
        "public_workload_hh_dominant": "HH-dominant",
        "public_workload_profile_scaled_public_proxy_1": "profile proxy 1",
        "public_workload_profile_scaled_public_proxy_2": "profile proxy 2",
        "modeled_profile_scaled_pilot": "profile pilot",
    }
    if text in replacements:
        return replacements[text]
    if text.endswith(".0"):
        return text[:-2]
    return text


def _x_sort(values: Iterable[object]) -> list[object]:
    vals = list(values)
    numeric = pd.to_numeric(pd.Series(vals), errors="coerce")
    if len(vals) > 0 and numeric.notna().all():
        return [v for _, v in sorted(zip(numeric.astype(float), vals), key=lambda x: x[0])]
    parsed = []
    for v in vals:
        text = str(v)
        if "x/" in text:
            try:
                left = float(text.split("x/")[0])
                right = float(text.split("x/")[1])
                parsed.append((0, left, right, text, v))
                continue
            except ValueError:
                pass
        parsed.append((1, math.inf, math.inf, text, v))
    return [v for *_ignore, v in sorted(parsed)]


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def _savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _lineplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    methods: list[str] | None = None,
    ylabel: str = "",
    xlabel: str = "",
    logy: bool = False,
    legend: bool = False,
) -> None:
    if methods is None:
        methods = _method_sort(data["method_label"].dropna().unique())
    xs = _x_sort(data[x_col].dropna().unique())
    x_map = {v: i for i, v in enumerate(xs)}
    for method in methods:
        g = data[data["method_label"] == method].copy()
        if g.empty:
            continue
        g["_pos"] = g[x_col].map(x_map)
        g = g.dropna(subset=["_pos", y_col]).sort_values("_pos")
        if g.empty:
            continue
        ax.plot(
            g["_pos"],
            g[y_col],
            marker="o",
            linewidth=2.2 if method == OURS else 1.45,
            markersize=4.2,
            color=METHOD_COLORS.get(method, "#888888"),
            label=_short_method(method),
            alpha=1.0 if method == OURS else 0.85,
        )
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([_format_value(v) for v in xs], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    if legend:
        ax.legend(ncol=2, fontsize=8, loc="upper left")


def _barplot(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    colors: list[str] | None = None,
    ylabel: str = "",
    horizontal: bool = False,
    baseline: float | None = 1.0,
) -> None:
    if colors is None:
        colors = ["#808080"] * len(labels)
    pos = np.arange(len(labels))
    if horizontal:
        ax.barh(pos, values, color=colors, height=0.68)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        if baseline is not None:
            ax.axvline(baseline, color="#222222", linewidth=1.0, linestyle="--")
    else:
        ax.bar(pos, values, color=colors, width=0.72)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        if baseline is not None:
            ax.axhline(baseline, color="#222222", linewidth=1.0, linestyle="--")
    ax.grid(axis="x" if horizontal else "y", color="#e6e6e6", linewidth=0.8)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(resolve_result_csv(), low_memory=False)
    numeric_cols = {
        TIME_COL,
        "end_to_end_time",
        "observed_end_to_end_time",
        "load_cv",
        "control_path_overhead_ratio",
        "scale_factor",
        "x_value",
        "worker_count",
        "actual_input_rows",
        "input_bytes",
        "max_worker_time",
        "control_path_time",
        "validation_time",
        "reuse_detection_time",
        "full_rebuild_detection_time",
        "stable_state_lookup_time",
        "residual_scan_time",
        "state_reuse_confidence",
        "history_update_background_time",
        "partition_version_change_rate",
        "new_hot_key_risk",
        "boundary_lookup_time",
        "boundary_lookup_key_count",
        "boundary_lookup_bytes",
        "candidate_count",
        "candidate_omit_upper_bound",
        "candidate_boundary_width",
        "oversampling_factor_alpha",
        "sketch_error",
        "precision_at_k",
        "recall_at_k",
        "left_recall",
        "right_recall",
        "avg_recall",
        "snapshot_path_overhead_ratio",
        "snapshot_aggregate_resource_time_normalized",
        "snapshot_build_time_total",
        "snapshot_read_time_total",
        "nonoverlapped_snapshot_path_time",
        "snapshot_publish_count",
        "cas_success_rate",
        "invalid_probe_count",
        "stolen_task_count",
        "rdma_operation_count",
        "cas_retry_per_success",
        "cas_retry_count",
        "join_pair_count",
        "output_bytes",
        "worker_imbalance_ratio",
        "top_1_percent_work_share",
        "record_size_bytes",
        "controlled_alpha",
        "table_ratio_R_over_S",
        "hot_keys_per_non_cc_state",
        "target_rank_correlation",
        "actual_rank_correlation",
        "selection_regret",
        "mode_selection_accuracy",
        "mode_selection_cost_ratio",
        "false_certification",
        "macro_f1_non_cc",
        "certified_boundary_recall",
        "certified_boundary_omit_upper_bound",
        "provider_selection_regret",
    }
    for col in numeric_cols.intersection(df.columns):
        df[col] = _num(df[col])
    if TIME_COL not in df.columns:
        df[TIME_COL] = np.nan
    df["plot_time"] = df[TIME_COL]
    for fallback in ["end_to_end_time", "observed_end_to_end_time"]:
        if fallback in df.columns:
            df["plot_time"] = df["plot_time"].fillna(df[fallback])
    valid = df[(df["run_status"].astype(str) == "success") & _truthy(df["is_valid_for_main"])].copy()
    if "x_value" in valid.columns and "case_id" in valid.columns:
        valid["x_value"] = valid["x_value"].astype(object)
        missing_x = valid["x_value"].isna() | (valid["x_value"].astype(str).str.strip() == "")
        valid.loc[missing_x, "x_value"] = valid.loc[missing_x, "case_id"]
    return df, valid


def aggregate(valid: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "plot_time",
        "load_cv",
        "control_path_overhead_ratio",
        "scale_factor",
        "worker_count",
        "actual_input_rows",
        "input_bytes",
        "max_worker_time",
        "control_path_time",
        "validation_time",
        "reuse_detection_time",
        "full_rebuild_detection_time",
        "stable_state_lookup_time",
        "residual_scan_time",
        "state_reuse_confidence",
        "history_update_background_time",
        "partition_version_change_rate",
        "new_hot_key_risk",
        "boundary_lookup_time",
        "boundary_lookup_key_count",
        "boundary_lookup_bytes",
        "candidate_count",
        "candidate_omit_upper_bound",
        "candidate_boundary_width",
        "oversampling_factor_alpha",
        "sketch_error",
        "precision_at_k",
        "recall_at_k",
        "left_recall",
        "right_recall",
        "avg_recall",
        "snapshot_path_overhead_ratio",
        "snapshot_aggregate_resource_time_normalized",
        "snapshot_build_time_total",
        "snapshot_read_time_total",
        "nonoverlapped_snapshot_path_time",
        "snapshot_publish_count",
        "cas_success_rate",
        "invalid_probe_count",
        "stolen_task_count",
        "rdma_operation_count",
        "cas_retry_per_success",
        "cas_retry_count",
        "join_pair_count",
        "output_bytes",
        "worker_imbalance_ratio",
        "top_1_percent_work_share",
        "record_size_bytes",
        "controlled_alpha",
        "table_ratio_R_over_S",
        "hot_keys_per_non_cc_state",
        "target_rank_correlation",
        "actual_rank_correlation",
        "selection_regret",
        "mode_selection_accuracy",
        "mode_selection_cost_ratio",
        "false_certification",
        "macro_f1_non_cc",
        "certified_boundary_recall",
        "certified_boundary_omit_upper_bound",
        "provider_selection_regret",
    ]
    keep_metrics = [c for c in metric_cols if c in valid.columns]
    group_cols = [
        "scenario_id",
        "rq_id",
        "case_id",
        "x_name",
        "x_value",
        "result_role",
        "result_scope",
        "method_label",
        "setting",
    ]
    optional_cols = [
        "ablation_disabled_mechanism",
        "ablation_label",
        "m1_trigger",
        "m2_trigger",
        "m3_trigger",
        "strict_mechanism_ablation",
        "public_workload_id",
        "native_or_profile_scaled",
    ]
    for col in optional_cols:
        if col in valid.columns:
            group_cols.append(col)
    agg = valid.groupby(group_cols, dropna=False)[keep_metrics].mean(numeric_only=True).reset_index()
    counts = valid.groupby(group_cols, dropna=False).size().reset_index(name="raw_row_count")
    return agg.merge(counts, on=group_cols, how="left")


def relative_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    idx = ["scenario_id", "case_id", "x_name", "x_value", "result_scope"]
    pivot = agg.pivot_table(index=idx, columns="method_label", values="plot_time", aggfunc="mean")
    records = []
    for method in [m for m in METHOD_ORDER if m != OURS]:
        if method not in pivot.columns or OURS not in pivot.columns:
            continue
        ratio = _safe_ratio(pivot[method], pivot[OURS])
        tmp = ratio.dropna().reset_index(name="relative_completion_time_ratio")
        tmp["method_label"] = method
        records.append(tmp)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def geomean_summary(ratios: pd.DataFrame) -> pd.DataFrame:
    if ratios.empty:
        return ratios
    rows = []
    for keys, g in ratios.groupby(["scenario_id", "result_scope", "method_label"]):
        scen, scope, method = keys
        values = g["relative_completion_time_ratio"]
        rows.append(
            {
                "scenario_id": scen,
                "result_scope": scope,
                "method_label": method,
                "n_common": values.notna().sum(),
                "geomean_T_method_over_T_ours": _geomean(values),
                "ours_win_count": int((values > 1.0).sum()),
                "method_win_count": int((values < 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_ratio_to_agg(agg: pd.DataFrame) -> pd.DataFrame:
    ratios = relative_ratios(agg)
    if ratios.empty:
        agg["relative_to_ours"] = np.nan
        return agg
    idx = ["scenario_id", "case_id", "x_name", "x_value", "result_scope", "method_label"]
    out = agg.merge(ratios[idx + ["relative_completion_time_ratio"]], on=idx, how="left")
    out["relative_to_ours"] = out["relative_completion_time_ratio"]
    out.loc[out["method_label"] == OURS, "relative_to_ours"] = 1.0
    return out


def _main_formal(agg: pd.DataFrame, scenario: str) -> pd.DataFrame:
    return agg[
        (agg["scenario_id"] == scenario)
        & (agg["result_scope"] == "formal")
        & (agg["result_role"] == "main")
        & (agg["method_label"].isin(MAIN_METHODS))
    ].copy()


def plot_fig5(agg: pd.DataFrame) -> FigureRecord:
    data = _main_formal(agg, "E1_input_scale")
    methods = [m for m in MAIN_METHODS if m in data["method_label"].unique()]
    by_x = data.groupby(["x_value", "method_label"], dropna=False).agg(
        plot_time=("plot_time", "mean"),
        load_cv=("load_cv", "mean"),
        control=("control_path_overhead_ratio", "mean"),
        rel=("relative_to_ours", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))
    _lineplot(axes[0, 0], by_x, "x_value", "plot_time", methods, "Model-estimated time", "Input scale", logy=True, legend=True)
    _panel_label(axes[0, 0], "a")
    ratio = by_x[by_x["method_label"] != OURS].groupby("method_label")["rel"].apply(_geomean).reset_index()
    ratio = ratio[ratio["method_label"].isin(methods)].sort_values("rel", ascending=True)
    _barplot(
        axes[0, 1],
        [_short_method(m) for m in ratio["method_label"]],
        ratio["rel"].tolist(),
        [METHOD_COLORS.get(m, "#888888") for m in ratio["method_label"]],
        "Relative completion time ratio",
        horizontal=True,
    )
    _panel_label(axes[0, 1], "b")
    _lineplot(axes[1, 0], by_x, "x_value", "load_cv", methods, "Load CV", "Input scale")
    _panel_label(axes[1, 0], "c")
    _lineplot(axes[1, 1], by_x, "x_value", "control", methods, "Control-path overhead ratio", "Input scale")
    _panel_label(axes[1, 1], "d")
    fig.suptitle("Fig. 5. Input-scale behavior under increasing join input", y=1.04, fontsize=12)
    _savefig(fig, "fig5_input_scale_main")
    return FigureRecord("fig5_input_scale_main", "Input-scale behavior", "Ours keeps lower modeled completion time and load dispersion as input scale grows.")


def plot_fig6(agg: pd.DataFrame) -> FigureRecord:
    data = _main_formal(agg, "E2_scaling")
    strong = data[data["x_name"] == "worker_count"].copy()
    weak = data[data["x_name"] == "scale_worker_pair"].copy()
    methods = [m for m in MAIN_METHODS if m in data["method_label"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.8))
    ours = strong[strong["method_label"] == OURS].copy()
    if not ours.empty:
        ours["workers_num"] = _num(ours["x_value"])
        ours = ours.groupby("workers_num")["plot_time"].mean().reset_index().sort_values("workers_num")
        base = ours.iloc[0]
        ours["strong_scaling_speedup"] = float(base["plot_time"]) / ours["plot_time"]
        ours["ideal"] = ours["workers_num"] / float(base["workers_num"])
        ours["parallel_efficiency"] = ours["strong_scaling_speedup"] / ours["ideal"]
        axes[0].plot(ours["workers_num"], ours["strong_scaling_speedup"], marker="o", color=METHOD_COLORS[OURS], label="Ours")
        axes[0].plot(ours["workers_num"], ours["ideal"], linestyle="--", color="#666666", label="Ideal")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("Workers")
        axes[0].set_ylabel("Strong-scaling speedup")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", color="#e6e6e6")
        axes[1].plot(ours["workers_num"], ours["parallel_efficiency"], marker="o", color=METHOD_COLORS[OURS])
        axes[1].axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("Workers")
        axes[1].set_ylabel("Parallel efficiency")
        axes[1].set_ylim(0, max(1.05, float(ours["parallel_efficiency"].max()) * 1.12))
        axes[1].grid(axis="y", color="#e6e6e6")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    weak_line = weak.groupby(["x_value", "method_label"], dropna=False)["plot_time"].mean().reset_index()
    if not weak_line.empty:
        weak_meta = weak[["x_value", "worker_count", "scale_factor"]].drop_duplicates("x_value")
        weak_line = weak_line.merge(weak_meta, on="x_value", how="left")
        weak_line["workers_num"] = _num(weak_line["worker_count"])
        weak_line["scale_num"] = _num(weak_line["scale_factor"])
        weak_line = weak_line.sort_values(["method_label", "workers_num", "x_value"])
        base = weak_line.groupby("method_label")["plot_time"].transform("first")
        weak_line["normalized"] = weak_line["plot_time"] / base
        for method in methods:
            g = weak_line[weak_line["method_label"] == method].dropna(subset=["workers_num", "normalized"])
            if g.empty:
                continue
            axes[2].plot(
                g["workers_num"],
                g["normalized"],
                marker="o",
                linewidth=2.2 if method == OURS else 1.45,
                markersize=4.2,
                color=METHOD_COLORS.get(method, "#888888"),
                label=_short_method(method),
                alpha=1.0 if method == OURS else 0.85,
            )
        ticks = weak_line.dropna(subset=["workers_num"]).drop_duplicates("workers_num").sort_values("workers_num")
        axes[2].set_xticks(ticks["workers_num"].tolist())
        axes[2].set_xticklabels([str(int(v)) for v in ticks["workers_num"]])
        axes[2].set_xlabel("Workers (weak scaling)")
        axes[2].set_ylabel("Normalized time")
        axes[2].grid(axis="y", color="#e6e6e6", linewidth=0.8)
        axes[2].legend(ncol=2, fontsize=8, loc="upper left")
    _panel_label(axes[2], "c")
    fig.suptitle("Fig. 6. Scaling behavior separates resource growth from input growth", y=1.05, fontsize=12)
    _savefig(fig, "fig6_scaling")
    return FigureRecord("fig6_scaling", "Scaling behavior", "Strong and weak scaling test whether the lower modeled time persists when resources and input size change.")


def plot_fig7(agg: pd.DataFrame) -> FigureRecord:
    data = _main_formal(agg, "E3_workload_factors")
    methods = [m for m in MAIN_METHODS if m in data["method_label"].unique()]
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 7.2))
    panels = [
        ("skew_alpha", "Natural skew", "a"),
        ("controlled_alpha", "Controlled concentration", "b"),
        ("top_1_percent_work_share", "Top-1% work share", "c"),
        ("record_size_bytes", "Record size", "d"),
        ("state_mix", "State mix", "e"),
        ("table_ratio", "Table ratio", "f"),
        ("target_rank_correlation", "Rank correlation", "g"),
    ]
    for ax, (x_name, title, lab) in zip(axes.flat, panels):
        g = data[data["x_name"] == x_name].groupby(["x_value", "method_label"], dropna=False)["relative_to_ours"].mean().reset_index()
        _lineplot(ax, g, "x_value", "relative_to_ours", methods, "Relative time ratio", title, legend=(lab == "a"))
        ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        _panel_label(ax, lab)
        if x_name in {"state_mix", "table_ratio", "target_rank_correlation"}:
            ax.tick_params(axis="x", labelrotation=18)
    for ax in axes.flat[len(panels):]:
        ax.set_axis_off()
    fig.subplots_adjust(hspace=0.5, wspace=0.32)
    fig.suptitle("Fig. 7. Workload-shape changes preserve the same ranking pattern", y=1.02, fontsize=12)
    _savefig(fig, "fig7_workload_factors")
    return FigureRecord("fig7_workload_factors", "Workload factors", "Ours remains robust across skew, work concentration, record width, state mix, table ratio, and rank correlation.")


def plot_fig8(agg: pd.DataFrame) -> FigureRecord:
    data = agg[(agg["scenario_id"] == "E4_mechanism1") & (agg["result_scope"] == "formal")]
    methods = _method_sort(data["method_label"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    for ax, x_name, title, lab in [
        (axes[0, 0], "residual_partition_ratio", "Residual change", "a"),
        (axes[0, 1], "partition_version_change", "Partition-version change", "b"),
        (axes[0, 2], "new_hot_ratio", "New-hot ratio", "c"),
    ]:
        g = data[data["x_name"] == x_name].groupby(["x_value", "method_label"], dropna=False)["relative_to_ours"].mean().reset_index()
        _lineplot(ax, g, "x_value", "relative_to_ours", methods, "Relative time ratio", title, legend=(lab == "a"))
        ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        _panel_label(ax, lab)
    ours_res = data[(data["x_name"] == "residual_partition_ratio") & (data["method_label"] == OURS)]
    comp = ours_res.groupby("x_value", dropna=False)[["residual_scan_time", "stable_state_lookup_time", "control_path_time"]].mean().reset_index()
    xs = _x_sort(comp["x_value"].dropna().unique())
    comp = comp.set_index("x_value").reindex(xs).reset_index()
    x_pos = np.arange(len(xs))
    axes[1, 0].plot(x_pos, comp["residual_scan_time"], marker="o", color="#ef8a62", label="Residual scan")
    axes[1, 0].plot(x_pos, comp["stable_state_lookup_time"], marker="o", color="#67a9cf", label="Stable lookup")
    axes[1, 0].plot(x_pos, comp["control_path_time"], marker="o", color="#b2182b", label="Control path")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([_format_value(v) for v in xs])
    axes[1, 0].set_xlabel("Residual change")
    axes[1, 0].set_ylabel("Control time")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", color="#e6e6e6")
    _panel_label(axes[1, 0], "d")
    age = data[(data["x_name"] == "history_age") & (data["method_label"] == OURS)].groupby("x_value", dropna=False)[["history_update_background_time", "state_reuse_confidence"]].mean().reset_index()
    xs = _x_sort(age["x_value"].dropna().unique())
    age = age.set_index("x_value").reindex(xs).reset_index()
    x_pos = np.arange(len(xs))
    axes[1, 1].bar(x_pos, age["history_update_background_time"], color="#b2abd2")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([_format_value(v) for v in xs])
    axes[1, 1].set_xlabel("History age")
    axes[1, 1].set_ylabel("Async maintenance time")
    axes[1, 1].grid(axis="y", color="#e6e6e6")
    _panel_label(axes[1, 1], "e")
    sel = data.groupby("method_label", dropna=False)[["selection_regret", "mode_selection_accuracy"]].mean().reset_index()
    sel = sel[sel["method_label"].isin(methods)].copy()
    sel["rank"] = sel["method_label"].map({m: i for i, m in enumerate(methods)})
    sel = sel.sort_values("rank")
    pos = np.arange(len(sel))
    axes[1, 2].bar(pos, sel["selection_regret"], color=[METHOD_COLORS.get(m, "#888888") for m in sel["method_label"]], width=0.65)
    axes[1, 2].set_xticks(pos)
    axes[1, 2].set_xticklabels([_short_method(m) for m in sel["method_label"]], rotation=22, ha="right")
    axes[1, 2].set_ylabel("Selection regret")
    axes[1, 2].grid(axis="y", color="#e6e6e6")
    ax_acc = axes[1, 2].twinx()
    ax_acc.plot(pos, sel["mode_selection_accuracy"], marker="o", color="#222222", linewidth=1.6, label="Mode accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel("Mode accuracy")
    _panel_label(axes[1, 2], "f")
    fig.subplots_adjust(hspace=0.55, wspace=0.34)
    fig.suptitle("Fig. 8. Historical validation chooses when to reuse, update, or rebuild", y=1.02, fontsize=12)
    _savefig(fig, "fig8_history_reuse")
    return FigureRecord("fig8_history_reuse", "Historical validation", "Mechanism 1 reduces planning cost by validating historical state and selecting among reuse, partial update, and rebuild modes.")


def plot_fig9(agg: pd.DataFrame) -> FigureRecord:
    data = agg[(agg["scenario_id"] == "E5_mechanism2") & (agg["result_scope"] == "formal")]
    methods = _method_sort(data["method_label"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5))
    case_data = data[data["x_name"] == "mechanism2_case"].groupby(["x_value", "method_label"], dropna=False).mean(numeric_only=True).reset_index()
    _lineplot(axes[0, 0], case_data, "x_value", "relative_to_ours", methods, "Relative time ratio", "Recognition case", legend=True)
    axes[0, 0].axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    axes[0, 0].tick_params(axis="x", labelrotation=10)
    _panel_label(axes[0, 0], "a")
    pr = data.groupby(["method_label"], dropna=False)[["precision_at_k", "recall_at_k", "certified_boundary_recall"]].mean().reset_index()
    pos = np.arange(len(pr))
    width = 0.36
    axes[0, 1].bar(pos - width / 2, pr["precision_at_k"], width, color="#92c5de", label="Precision")
    axes[0, 1].bar(pos + width / 2, pr["recall_at_k"], width, color="#f4a582", label="Recall")
    if "certified_boundary_recall" in pr.columns:
        axes[0, 1].plot(pos, pr["certified_boundary_recall"], marker="o", color="#222222", linewidth=1.4, label="Certified-boundary recall")
    axes[0, 1].set_xticks(pos)
    axes[0, 1].set_xticklabels([_short_method(m) for m in pr["method_label"]], rotation=20, ha="right")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Hot-key identification quality")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(axis="y", color="#e6e6e6")
    _panel_label(axes[0, 1], "b")
    overhead = data.groupby("method_label", dropna=False)[["candidate_omit_upper_bound", "certified_boundary_omit_upper_bound", "false_certification", "boundary_lookup_time", "boundary_lookup_bytes", "candidate_count"]].mean().reset_index()
    omit_col = "certified_boundary_omit_upper_bound" if "certified_boundary_omit_upper_bound" in overhead.columns else "candidate_omit_upper_bound"
    _barplot(axes[0, 2], [_short_method(m) for m in overhead["method_label"]], overhead[omit_col].tolist(), [METHOD_COLORS.get(m, "#888888") for m in overhead["method_label"]], "Certified omission upper bound", horizontal=True, baseline=None)
    _panel_label(axes[0, 2], "c")
    false_vals = overhead["false_certification"].fillna(0).tolist() if "false_certification" in overhead else [0] * len(overhead)
    _barplot(axes[1, 0], [_short_method(m) for m in overhead["method_label"]], false_vals, [METHOD_COLORS.get(m, "#888888") for m in overhead["method_label"]], "False certification rate", horizontal=True, baseline=None)
    if false_vals and max(false_vals) == 0:
        axes[1, 0].set_xlim(0, 0.1)
        axes[1, 0].text(0.05, 0.5, "all 0", ha="center", va="center", transform=axes[1, 0].transAxes, color="#666666")
    _panel_label(axes[1, 0], "d")
    for ax, x_name, title, lab in [
        (axes[1, 1], "sketch_error", "Sketch error", "e"),
        (axes[1, 2], "candidate_boundary_width", "Boundary width", "f"),
    ]:
        g = data[data["x_name"] == x_name].groupby(["x_value", "method_label"], dropna=False)["relative_to_ours"].mean().reset_index()
        _lineplot(ax, g, "x_value", "relative_to_ours", methods, "Relative time ratio", title)
        ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
        _panel_label(ax, lab)
    fig.subplots_adjust(hspace=0.65, wspace=0.36)
    fig.suptitle("Fig. 9. Boundary-preserving dual-side identification avoids brittle one-sided candidates", y=1.02, fontsize=12)
    _savefig(fig, "fig9_dual_side_recognition")
    return FigureRecord("fig9_dual_side_recognition", "Dual-side recognition", "Mechanism 2 improves current-round load facts by preserving boundary candidates and completing both-side evidence.")


def plot_fig10(agg: pd.DataFrame) -> FigureRecord:
    data = agg[(agg["scenario_id"] == "E6_mechanism3") & (agg["result_scope"] == "formal")]
    methods = _method_sort(data["method_label"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    slow = data[data["x_name"] == "slowdown_factor"].groupby(["x_value", "method_label"], dropna=False)[["relative_to_ours", "load_cv", "worker_imbalance_ratio", "max_worker_time"]].mean().reset_index()
    _lineplot(axes[0, 0], slow, "x_value", "relative_to_ours", methods, "Relative time ratio", "Slow worker factor", legend=True)
    axes[0, 0].axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    _panel_label(axes[0, 0], "a")
    _lineplot(axes[0, 1], slow, "x_value", "load_cv", methods, "Load CV", "Slow worker factor")
    _panel_label(axes[0, 1], "b")
    _lineplot(axes[0, 2], slow, "x_value", "worker_imbalance_ratio", methods, "Worker imbalance ratio", "Slow worker factor")
    _panel_label(axes[0, 2], "c")
    snap = data[data["x_name"] == "K_pool"].groupby(["x_value", "method_label"], dropna=False)[["snapshot_path_overhead_ratio", "snapshot_build_time_total", "snapshot_read_time_total", "nonoverlapped_snapshot_path_time"]].mean().reset_index()
    _lineplot(axes[1, 0], snap, "x_value", "snapshot_path_overhead_ratio", methods, "Snapshot path overhead", "Candidate pool")
    _panel_label(axes[1, 0], "d")
    probe = data[data["x_name"] == "retry_budget"].groupby(["x_value", "method_label"], dropna=False)[["cas_success_rate", "cas_retry_per_success", "cas_retry_count", "invalid_probe_count"]].mean().reset_index()
    _lineplot(axes[1, 1], probe, "x_value", "cas_success_rate", methods, "CAS success rate", "Retry budget")
    _panel_label(axes[1, 1], "e")
    ours_probe = probe[probe["method_label"] == OURS].copy()
    xs = _x_sort(ours_probe["x_value"].dropna().unique())
    ours_probe = ours_probe.set_index("x_value").reindex(xs).reset_index()
    pos = np.arange(len(xs))
    axes[1, 2].bar(pos - 0.18, ours_probe["cas_retry_per_success"], width=0.36, color="#92c5de", label="CAS retry/success")
    axes[1, 2].bar(pos + 0.18, ours_probe["invalid_probe_count"], width=0.36, color="#f4a582", label="Invalid probes")
    axes[1, 2].set_xticks(pos)
    axes[1, 2].set_xticklabels([_format_value(v) for v in xs])
    axes[1, 2].set_xlabel("Retry budget")
    axes[1, 2].set_ylabel("Probe/CAS count")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(axis="y", color="#e6e6e6")
    _panel_label(axes[1, 2], "f")
    fig.subplots_adjust(hspace=0.55, wspace=0.34)
    fig.suptitle("Fig. 10. Runtime rebalancing is most visible under injected long-tail pressure", y=1.02, fontsize=12)
    _savefig(fig, "fig10_runtime_rebalance")
    return FigureRecord("fig10_runtime_rebalance", "Runtime rebalancing", "Mechanism 3 reduces long-tail impact when execution-time remaining work can be safely taken over.")


def plot_fig11(agg: pd.DataFrame) -> FigureRecord:
    data = _main_formal(agg, "E7_external_and_model_sensitivity")
    methods = [m for m in MAIN_METHODS if m in data["method_label"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))
    for ax, x_name, title, lab in [
        (axes[0], "public_workload_id", "External/profile workload", "a"),
        (axes[1], "network_cost_scale", "Network-cost scale", "b"),
    ]:
        g = data[data["x_name"] == x_name].groupby(["x_value", "method_label"], dropna=False)["relative_to_ours"].mean().reset_index()
        if g.empty:
            ax.text(0.5, 0.5, "not in E7 matrix", ha="center", va="center", transform=ax.transAxes, color="#666666")
            ax.set_axis_off()
        else:
            _lineplot(ax, g, "x_value", "relative_to_ours", methods, "Relative time ratio", title, legend=(lab == "a"))
            ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
            ax.tick_params(axis="x", labelrotation=15)
        ax.set_title(title)
        _panel_label(ax, lab)
    fig.suptitle("Fig. 11. External/profile cases and model perturbations test ranking stability", y=1.05, fontsize=12)
    _savefig(fig, "fig11_external_model_sensitivity")
    return FigureRecord("fig11_external_model_sensitivity", "External and model sensitivity", "External/profile workloads and network-cost scaling test whether the main ranking survives beyond the default synthetic configuration.")


def plot_fig12(agg: pd.DataFrame) -> FigureRecord:
    data = agg[(agg["scenario_id"] == "E8_ablation_summary") & (agg["result_scope"] == "formal")]
    methods = _method_sort(data["method_label"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
    g = data.groupby(["x_value", "method_label"], dropna=False)[["relative_to_ours", "load_cv"]].mean().reset_index()
    _lineplot(axes[0], g, "x_value", "relative_to_ours", methods, "Relative time ratio", "Trigger scenario", legend=True)
    axes[0].axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    _panel_label(axes[0], "a")
    _lineplot(axes[1], g, "x_value", "load_cv", methods, "Load CV", "Trigger scenario")
    _panel_label(axes[1], "b")
    ab = data[data["method_label"] != OURS].groupby(["x_value", "ablation_disabled_mechanism"], dropna=False)["relative_to_ours"].apply(_geomean).reset_index()
    pivot = ab.pivot(index="x_value", columns="ablation_disabled_mechanism", values="relative_to_ours")
    xs = _x_sort(pivot.index)
    pivot = pivot.reindex(xs)
    width = 0.24
    pos = np.arange(len(xs))
    for offset, mech in zip([-width, 0, width], ["M1", "M2", "M3"]):
        vals = pivot[mech] if mech in pivot else pd.Series(index=xs, dtype=float)
        axes[2].bar(pos + offset, vals.values, width=width, label=f"no-{mech}", color={"M1": "#d6604d", "M2": "#4393c3", "M3": "#c51b7d"}[mech])
    axes[2].axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    axes[2].set_xticks(pos)
    axes[2].set_xticklabels([_format_value(v) for v in xs], rotation=12, ha="right")
    axes[2].set_ylabel("T_ablation/T_ours")
    axes[2].set_xlabel("Trigger scenario")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", color="#e6e6e6")
    _panel_label(axes[2], "c")
    fig.suptitle("Fig. 12. Whole-mechanism ablations isolate trigger-specific contribution", y=1.05, fontsize=12)
    _savefig(fig, "fig12_mechanism_ablation")
    return FigureRecord("fig12_mechanism_ablation", "Whole-mechanism ablation", "Each disabled mechanism hurts most in the trigger class that requires its corresponding planning, recognition, or runtime-correction capability.")


def audit(df: pd.DataFrame, valid: pd.DataFrame) -> tuple[dict[str, object], str]:
    formal = valid[valid["result_scope"] == "formal"]
    pilot = valid[valid["result_scope"] == "pilot"]
    main_formal = formal[formal["result_role"] == "main"]
    audit_rows = []
    selection_cols = ["selected_node_label", "selected_round_label", "selected_seed"]
    for (scenario, case_id), g in main_formal.groupby(["scenario_id", "case_id"], dropna=False):
        row = {"scenario_id": scenario, "case_id": case_id, "row_count": len(g), "method_count": g["method_label"].nunique()}
        for col in selection_cols:
            row[f"{col}_unique"] = g[col].nunique(dropna=True) if col in g.columns else np.nan
        row["common_selection"] = all(row.get(f"{col}_unique", 0) <= 1 for col in selection_cols)
        audit_rows.append(row)
    audit_df = pd.DataFrame(audit_rows)
    non_common = int((~audit_df["common_selection"]).sum()) if not audit_df.empty else 0
    status = "paired_source_selection" if non_common == 0 else "common_completion_nonpaired_source"
    summary = {
        "raw_rows": len(df),
        "valid_rows": len(valid),
        "formal_rows": len(formal),
        "pilot_rows": len(pilot),
        "scenario_count": formal["scenario_id"].nunique(),
        "case_count": formal["case_id"].nunique(),
        "method_count": formal["method_label"].nunique(),
        "main_formal_case_count": len(audit_df),
        "non_common_selection_case_count": non_common,
        "selection_status": status,
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(SUMMARY_DIR / "selection_uniformity_audit.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 20260708 final-clean-v4 experiment audit",
        "",
        f"- Raw rows: {summary['raw_rows']}",
        f"- Valid rows: {summary['valid_rows']}",
        f"- Formal rows: {summary['formal_rows']}",
        f"- Pilot rows: {summary['pilot_rows']}",
        f"- Formal scenarios: {summary['scenario_count']}",
        f"- Formal cases: {summary['case_count']}",
        f"- Formal methods: {summary['method_count']}",
        f"- Main formal cases audited: {summary['main_formal_case_count']}",
        f"- Cases with non-common selected node/round/seed: {summary['non_common_selection_case_count']}",
        f"- Selection status: `{status}`",
        "",
        "This audit artifact may record data source and pairing details. The manuscript should use common-correct-completion wording and should not expose pipeline file names or directories.",
        "",
    ]
    return summary, "\n".join(lines)


def write_summaries(df: pd.DataFrame, valid: pd.DataFrame, agg: pd.DataFrame, records: list[FigureRecord]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(SUMMARY_DIR / "case_method_summary.csv", index=False, encoding="utf-8-sig")
    ratios = relative_ratios(agg)
    ratios.to_csv(SUMMARY_DIR / "relative_completion_time_ratios.csv", index=False, encoding="utf-8-sig")
    geo = geomean_summary(ratios)
    geo.to_csv(SUMMARY_DIR / "common_case_geomean.csv", index=False, encoding="utf-8-sig")
    audit_summary, audit_md = audit(df, valid)
    (SUMMARY_DIR / "validity_audit.md").write_text(audit_md, encoding="utf-8")

    required_by_figure = {
        "Fig5": ["modeled_end_to_end_time", "load_cv", "control_path_overhead_ratio"],
        "Fig6": ["modeled_end_to_end_time", "worker_count", "scale_worker_pair"],
        "Fig7": ["skew_alpha", "controlled_alpha", "top_1_percent_work_share", "record_size_bytes", "state_mix", "table_ratio", "target_rank_correlation"],
        "Fig8": ["residual_scan_time", "stable_state_lookup_time", "control_path_time", "history_update_background_time", "selection_regret", "mode_selection_accuracy", "partition_version_change", "new_hot_ratio"],
        "Fig9": ["precision_at_k", "recall_at_k", "certified_boundary_recall", "certified_boundary_omit_upper_bound", "false_certification", "boundary_lookup_time", "boundary_lookup_bytes", "sketch_error", "candidate_boundary_width", "oversampling_factor_alpha"],
        "Fig10": ["load_cv", "worker_imbalance_ratio", "max_worker_time", "snapshot_path_overhead_ratio", "cas_success_rate", "invalid_probe_count", "cas_retry_per_success", "cas_retry_count", "provider_selection_regret"],
        "Fig11": ["public_workload_id", "network_cost_scale"],
        "Fig12": ["strict_mechanism_ablation", "ablation_disabled_mechanism", "ablation_label", "m1_trigger", "m2_trigger", "m3_trigger", "load_cv"],
    }
    x_names_available = set(df["x_name"].dropna().astype(str)) if "x_name" in df.columns else set()
    availability = ["# Figure metric availability", ""]
    for fig, cols in required_by_figure.items():
        present = [c for c in cols if c in df.columns or c in x_names_available]
        missing = [c for c in cols if c not in df.columns and c not in x_names_available]
        availability += [f"## {fig}", f"- Present: {', '.join(present) if present else 'none'}", f"- Missing: {', '.join(missing) if missing else 'none'}", ""]
    (SUMMARY_DIR / "figure_metric_availability.md").write_text("\n".join(availability), encoding="utf-8")

    fig_lines = ["# Figure claims", ""]
    for r in records:
        fig_lines.append(f"- `{r.name}`: {r.claim}")
    (SUMMARY_DIR / "figure_claims.md").write_text("\n".join(fig_lines) + "\n", encoding="utf-8")

    fig_map = {
        "E1_input_scale": "Fig5",
        "E2_scaling": "Fig6",
        "E3_workload_factors": "Fig7",
        "E4_mechanism1": "Fig8",
        "E5_mechanism2": "Fig9",
        "E6_mechanism3": "Fig10",
        "E7_external_and_model_sensitivity": "Fig11",
        "E8_ablation_summary": "Fig12",
    }
    table_lines = [
        "# Result summary table",
        "",
        "| Figure | Scenario | Method compared with Ours | Common cases | Geomean relative completion time | Ours wins |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in geo.sort_values(["scenario_id", "geomean_T_method_over_T_ours"], ascending=[True, False]).iterrows():
        table_lines.append(
            f"| {fig_map.get(row['scenario_id'], '')} | {row['scenario_id']} | {row['method_label']} | {int(row['n_common'])} | {row['geomean_T_method_over_T_ours']:.3f} | {int(row['ours_win_count'])} |"
        )
    (SUMMARY_DIR / "result_summary_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    lines = [
        "# Key findings for 20260708 final-clean-v4 figures",
        "",
        "## Data audit",
        f"- Selection status: `{audit_summary['selection_status']}`.",
        f"- Raw rows: {audit_summary['raw_rows']}; valid rows: {audit_summary['valid_rows']}; formal rows: {audit_summary['formal_rows']}; pilot rows: {audit_summary['pilot_rows']}.",
        f"- Formal scenarios: {audit_summary['scenario_count']}; formal cases: {audit_summary['case_count']}; formal methods: {audit_summary['method_count']}.",
        "",
    ]
    for scenario in ["E1_input_scale", "E2_scaling", "E3_workload_factors", "E7_external_and_model_sensitivity"]:
        s = geo[(geo["scenario_id"] == scenario) & (geo["result_scope"] == "formal") & (geo["method_label"].isin([m for m in MAIN_METHODS if m != OURS]))]
        if s.empty:
            continue
        lines += [f"## {scenario}", ""]
        for _, row in s.sort_values("geomean_T_method_over_T_ours", ascending=False).iterrows():
            lines.append(f"- {row['method_label']}: geomean T_method/T_ours={row['geomean_T_method_over_T_ours']:.3f} over {int(row['n_common'])} common cases; Ours wins {int(row['ours_win_count'])}.")
        lines.append("")
    for scenario in ["E4_mechanism1", "E5_mechanism2", "E6_mechanism3", "E8_ablation_summary"]:
        s = geo[(geo["scenario_id"] == scenario) & (geo["result_scope"] == "formal")]
        if s.empty:
            continue
        lines += [f"## {scenario}", ""]
        for _, row in s.sort_values("geomean_T_method_over_T_ours", ascending=False).iterrows():
            lines.append(f"- {row['method_label']}: geomean T_method/T_ours={row['geomean_T_method_over_T_ours']:.3f} over {int(row['n_common'])} common cases.")
        lines.append("")
    e2 = agg[(agg["scenario_id"] == "E2_scaling") & (agg["x_name"] == "worker_count") & (agg["method_label"] == OURS) & (agg["result_scope"] == "formal")]
    if not e2.empty:
        e2 = e2.copy()
        e2["workers"] = _num(e2["x_value"])
        e2 = e2.groupby("workers")["plot_time"].mean().reset_index().sort_values("workers")
        base = e2.iloc[0]
        top = e2.iloc[-1]
        lines += ["## Scaling anchor", f"- Ours strong-scaling speedup from {int(base['workers'])} to {int(top['workers'])} workers: {base['plot_time']/top['plot_time']:.2f}x.", ""]
    (SUMMARY_DIR / "key_findings_formal_l3_20260708_final_clean_v4.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df, valid = load_data()
    formal_valid = valid[valid["result_scope"] == "formal"].copy()
    agg = aggregate(formal_valid)
    agg = add_ratio_to_agg(agg)
    records = [
        plot_fig5(agg),
        plot_fig6(agg),
        plot_fig7(agg),
        plot_fig8(agg),
        plot_fig9(agg),
        plot_fig10(agg),
        plot_fig11(agg),
        plot_fig12(agg),
    ]
    write_summaries(df, valid, agg, records)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
