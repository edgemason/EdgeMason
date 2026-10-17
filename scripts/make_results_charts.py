"""Charts for the "Experimental Results and Discussion" slide.

Reads the JSON results written by run_full_benchmark.py, quantization_analysis.py
and the validation run, and writes PNGs to reports/img/results/.

Usage:
    uv run --with matplotlib python scripts/make_results_charts.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
DATA = REPORTS / "data"
OUT = REPORTS / "img" / "results"

# Reference palette (dataviz skill), light mode, fixed categorical order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL_BLUE = ["#eaf2fc", "#b9d4f4", "#79acea", "#2a78d6", "#1a4f91"]

MODEL_LABELS = {
    "squeezenet1.1-13.onnx": "SqueezeNet 1.1",
    "shufflenet-v2-13.onnx": "ShuffleNet V2",
    "mobilenetv2.onnx": "MobileNetV2",
    "resnet18-v1-13.onnx": "ResNet-18",
    "resnet50.onnx": "ResNet-50",
    "resnet152.onnx": "ResNet-152",
}
CANDIDATE_LABELS = {
    "onnx_fp32_baseline": "FP32 (ONNX Runtime)",
    "onnx_static_int8": "Static INT8 (ONNX Runtime)",
    "tvm_opt_level_3": "FP32 (TVM, untuned)",
    "tvm_int8_quantized": "INT8 (TVM, untuned)",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _load(name: str):
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _grouped_bars(ax, groups, series_names, values, fmt, colors=SERIES):
    """values[series][group]; None draws nothing. Bars are separated by a 2px gap."""
    count = len(series_names)
    width = 0.8 / count
    x = np.arange(len(groups))
    for index, name in enumerate(series_names):
        heights = [v if v is not None else 0 for v in values[index]]
        positions = x - 0.4 + width * (index + 0.5)
        bars = ax.bar(positions, heights, width * 0.92, color=colors[index], label=name,
                      edgecolor=SURFACE, linewidth=1.0)
        for bar, value in zip(bars, values[index]):
            if value is None or fmt is None:
                continue
            ax.annotate(fmt(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK_2)
    ax.set_xticks(x, groups)
    ax.tick_params(axis="x", length=0)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def chart_accuracy(full_run) -> None:
    models = [m for m in full_run["models"] if m.get("candidates")]
    names = list(CANDIDATE_LABELS)
    values = [[next((c["top1"] * 100 for c in m["candidates"] if c["name"] == n and c["top1"] is not None), None)
               for m in models] for n in names]
    groups = [f"{MODEL_LABELS[m['model']]}\n(test n={m['data_split']['test_samples']})" for m in models]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _grouped_bars(ax, groups, [CANDIDATE_LABELS[n] for n in names], values, lambda v: f"{v:.1f}")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 accuracy on held-out test images (%)")
    ax.set_title("Testing accuracy by model and optimization (x86_64, CPU)")
    ax.legend(ncol=2, loc="upper left", fontsize=8.5)
    _save(fig, "testing_accuracy_top1.png")


def chart_latency(full_run) -> None:
    models = [m for m in full_run["models"] if m.get("candidates")]
    groups = [MODEL_LABELS[m["model"]] for m in models]

    def p50(model, name):
        return next((c["latency_p50_ms"] for c in model["candidates"] if c["name"] == name and c["success"]), None)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    _grouped_bars(left, groups, [CANDIDATE_LABELS["onnx_fp32_baseline"], CANDIDATE_LABELS["onnx_static_int8"]],
                  [[p50(m, "onnx_fp32_baseline") for m in models], [p50(m, "onnx_static_int8") for m in models]],
                  lambda v: f"{v:.1f}", colors=SERIES[:2])
    left.set_title("ONNX Runtime: median latency (p50)")
    left.set_ylabel("Latency per image, 1 CPU thread (ms)")
    left.legend(loc="upper left", fontsize=8.5)
    _grouped_bars(right, groups, [CANDIDATE_LABELS["tvm_opt_level_3"], CANDIDATE_LABELS["tvm_int8_quantized"]],
                  [[p50(m, "tvm_opt_level_3") for m in models], [p50(m, "tvm_int8_quantized") for m in models]],
                  lambda v: f"{v:.0f}", colors=SERIES[2:4])
    right.set_title("TVM, untuned: median latency (p50)")
    right.set_ylabel("Latency per image (ms)")
    right.legend(loc="upper left", fontsize=8.5)
    ratios = [p50(m, "tvm_opt_level_3") / p50(m, "onnx_fp32_baseline") for m in models
              if p50(m, "tvm_opt_level_3") and p50(m, "onnx_fp32_baseline")]
    if ratios:
        fig.suptitle(f"Separate panels: untuned TVM FP32 is {min(ratios):.0f}-{max(ratios):.0f}x slower than "
                     "ONNX Runtime FP32; one shared axis would hide the ONNX bars", fontsize=9, color=INK_2, y=-0.02)
    _save(fig, "latency_p50.png")


def chart_size_memory(full_run) -> None:
    models = [m for m in full_run["models"] if m.get("candidates")]
    groups = [MODEL_LABELS[m["model"]] for m in models]
    names = ["onnx_fp32_baseline", "onnx_static_int8"]

    def field(model, name, key):
        return next((c[key] for c in model["candidates"] if c["name"] == name and c["success"]), None)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title, label in [
        (left, "model_size_mb", "Model file size", "Size (MB)"),
        (right, "memory_delta_mb", "Measured RAM (peak - baseline)", "RAM (MB)"),
    ]:
        _grouped_bars(ax, groups, [CANDIDATE_LABELS[n] for n in names],
                      [[field(m, n, key) for m in models] for n in names], lambda v: f"{v:.0f}", colors=SERIES[:2])
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, "size_and_memory.png")


def chart_agreement(analysis) -> None:
    cols = min(3, len(analysis))
    rows = int(np.ceil(len(analysis) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows),
                             gridspec_kw={"wspace": 0.45, "hspace": 0.55})
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(analysis):]:
        ax.set_visible(False)
    vmax = max(max(r["agreement_fp32_vs_static_int8"][k] for k in
                   ["both_right", "fp32_right_int8_wrong", "fp32_wrong_int8_right", "both_wrong"]) for r in analysis)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQUENTIAL_BLUE)
    for ax, result in zip(axes, analysis):
        a = result["agreement_fp32_vs_static_int8"]
        matrix = np.array([[a["both_right"], a["fp32_right_int8_wrong"]],
                           [a["fp32_wrong_int8_right"], a["both_wrong"]]])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="#ffffff" if value > vmax * 0.55 else INK)
        ax.set_xticks([0, 1], ["INT8 right", "INT8 wrong"])
        ax.set_yticks([0, 1], ["FP32 right", "FP32 wrong"])
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{MODEL_LABELS[result['model']]}\n"
                     f"same top-1 prediction: {a['same_top1_prediction'] * 100:.1f}%", fontsize=10.5)
    fig.suptitle(f"FP32 vs static INT8 on the same {analysis[0]['test_samples']} test images", fontsize=12,
                 fontweight="bold", color=INK, y=1.04)
    _save(fig, "agreement_matrices.png")


def chart_ablations(analysis) -> None:
    groups = [MODEL_LABELS[r["model"]] for r in analysis]
    fig, (left, right) = plt.subplots(2, 1, figsize=(11, 8.4), gridspec_kw={"hspace": 0.35})
    names = [("fp32", "FP32"), ("static_int8_per_tensor", "Static INT8, per-tensor (before)"),
             ("static_int8_per_channel", "Static INT8, per-channel (now)")]
    _grouped_bars(left, groups, [label for _, label in names],
                  [[r["variants"][key]["top1"] * 100 for r in analysis] for key, _ in names],
                  None, colors=SERIES[:3])
    left.set_ylim(0, 100)
    left.set_title("Fix 4: per-channel weight scales")
    left.set_ylabel("Top-1 accuracy (%)")
    left.legend(loc="upper left", fontsize=8.5, ncol=3)
    names = [("fp32", "FP32"), ("dynamic_int8", "Dynamic INT8 (dropped for CNNs)"),
             ("static_int8_per_channel", "Static INT8, per-channel")]
    _grouped_bars(right, groups, [label for _, label in names],
                  [[r["variants"][key]["latency_p50_ms_1thread"] for r in analysis] for key, _ in names],
                  None, colors=SERIES[:3])
    right.set_title("Fix 5: dynamic vs static INT8 on CNNs")
    right.set_ylabel("Latency p50, 1 thread (ms)")
    right.legend(loc="upper left", fontsize=8.5)
    fig.text(0.01, 0.005, "Exact values: tables in EdgeMason_Report.md (179 test images, 1 CPU thread).",
             fontsize=8, color=INK_2)
    _save(fig, "ablations.png")


def _study_series(result, key):
    sizes = sorted({p["calibration_size"] for p in result["points"]})
    values = [[p[key]["top1"] * 100 for p in result["points"] if p["calibration_size"] == s] for s in sizes]
    return (np.array(sizes), np.array([np.mean(v) for v in values]),
            np.array([min(v) for v in values]), np.array([max(v) for v in values]))


def _study_axes(study, title):
    cols = 2
    rows = int(np.ceil(len(study) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(study):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK, y=1.0)
    return fig, axes


def _fp32_line(ax, result, x_end):
    """FP32 reference as a gray line; its value goes in the panel title, not on the plot."""
    ax.axhline(result["fp32_test"]["top1"] * 100, color=INK_2, linewidth=1.2, label="FP32 (reference)")


def _panel_title(result):
    return f"{MODEL_LABELS[result['model']]}  (FP32: {result['fp32_test']['top1'] * 100:.1f}%)"


def chart_calibration_size(study) -> None:
    fig, axes = _study_axes(study, f"Static INT8 test accuracy vs calibration-set size "
                                   f"(fixed test set, n={study[0]['test_samples']})")
    for ax, result in zip(axes, study):
        for key, label, color in [("per_channel_test", "INT8 per-channel (shipped)", SERIES[0]),
                                  ("per_tensor_test", "INT8 per-tensor (before Fix 4)", SERIES[1])]:
            sizes, mean, low, high = _study_series(result, key)
            ax.fill_between(sizes, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(sizes, mean, color=color, linewidth=2, marker="o", markersize=5, label=label)
        _fp32_line(ax, result, sizes[-1])
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.set_ylim(0, 100)
        ax.set_title(_panel_title(result))
        ax.set_ylabel("Top-1 on test set (%)")
        ax.grid(True, axis="both")
    fig.supxlabel("Calibration images (log scale); shaded band = min-max over 3 random draws",
                  fontsize=10, color=INK_2)
    axes[0].legend(loc="lower right", fontsize=8.5)
    _save(fig, "calibration_size_accuracy.png")


def chart_train_vs_validation(study) -> None:
    fig, axes = _study_axes(study, "Calibration (\"training\") vs test (\"validation\") accuracy, "
                                   "static INT8 per-channel")
    for ax, result in zip(axes, study):
        for key, label, color in [("per_channel_calibration", "On the calibration images", SERIES[0]),
                                  ("per_channel_test", "On the held-out test images", SERIES[1])]:
            sizes, mean, low, high = _study_series(result, key)
            ax.fill_between(sizes, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(sizes, mean, color=color, linewidth=2, marker="o", markersize=5, label=label)
        _fp32_line(ax, result, sizes[-1])
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.set_ylim(0, 105)
        ax.set_title(_panel_title(result))
        ax.set_ylabel("Top-1 (%)")
        ax.grid(True, axis="both")
    fig.supxlabel("Calibration images (log scale). With 1-5 images, calibration accuracy is a handful of "
                  "images, so it swings between 0% and 100%.", fontsize=10, color=INK_2)
    axes[0].legend(loc="lower right", fontsize=8.5)
    _save(fig, "calibration_vs_test_accuracy.png")


def chart_targets(validation) -> None:
    runs = [d for d in validation["deploys"] if d["priority"] == "latency"]
    order = ["x86_64", "rpi5", "jetson_nano", "rpi4", "rpi3"]
    runs.sort(key=lambda d: order.index(d["hardware"]))
    labels = {"x86_64": "x86_64", "rpi5": "RPi 5", "jetson_nano": "Jetson Nano\n(CPU)", "rpi4": "RPi 4", "rpi3": "RPi 3"}
    names = ["onnx_fp32_baseline", "onnx_static_int8"]
    values = [[next(r["estimated_latency_p50_ms"] for r in d["results"] if r["name"] == n) for d in runs] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _grouped_bars(ax, [labels[d["hardware"]] for d in runs], [CANDIDATE_LABELS[n] for n in names], values,
                  lambda v: f"{v:.0f}", colors=SERIES[:2])
    ax.set_ylabel("Estimated latency on target (ms)")
    ax.set_title("MobileNetV2: estimated latency per target device")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.text(0.01, -0.04, "Estimate = host p50 x per-device, per-model-type multiplier. INT8 multipliers for "
             "ARM targets are placeholders until measured on a real device.", fontsize=8, color=INK_2)
    _save(fig, "target_latency_estimates.png")


def pipeline_diagram() -> None:
    """Block diagram of the EdgeMason pipeline for the report."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("Upload", "ONNX model\n+ calibration\nimages"),
        ("Analyze", "size, params,\noperators,\nmodel family"),
        ("Plan", "candidates\nby model\nfamily"),
        ("Quantize", "FP32 /\nstatic INT8 /\ndynamic INT8"),
        ("Benchmark", "fresh process,\n1 thread,\np50 + RAM"),
        ("Score", "top-1 / top-5\non held-out\nimages"),
        ("Estimate", "host value\nx device\nmultiplier"),
        ("Rank", "priority\n+ limits"),
        ("Package", "ZIP: model,\ninference.py,\nDockerfile"),
    ]
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.set_xlim(0, len(steps) * 1.55)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.grid(False)
    for index, (title, detail) in enumerate(steps):
        x = 0.1 + index * 1.55
        box = FancyBboxPatch((x, 0.45), 1.3, 1.3, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor="#eaf2fc" if index not in (3, 4, 5, 6) else "#b9d4f4",
                             edgecolor="#1a4f91", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + 0.65, 1.5, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + 0.65, 0.92, detail, ha="center", va="center", fontsize=8, color=INK_2, linespacing=1.3)
        if index < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.32, 1.1), (x + 1.53, 1.1), arrowstyle="-|>",
                                         mutation_scale=12, color="#1a4f91", linewidth=1.2))
    ax.text(0.1, 2.0, "EdgeMason deployment pipeline (one deploy request)", fontsize=12,
            fontweight="bold", color=INK)
    ax.text(0.1, 0.12, "Shaded steps (Quantize to Estimate) run once per candidate. Hardware profile (providers, multipliers, memory factor) "
            "feeds Plan, Benchmark and Estimate.", fontsize=8.5, color=INK_2)
    fig.savefig(REPORTS / "img" / "pipeline_diagram.png")
    plt.close(fig)
    print("wrote", REPORTS / "img" / "pipeline_diagram.png")


def main() -> None:
    full_run = _load("full_run_x86_64_2026-10-05.json")
    analysis = _load("quantization_analysis.json")
    validation = _load("validation_run_2026-10-05.json")
    study = _load("calibration_size_study.json")
    if full_run:
        chart_accuracy(full_run)
        chart_latency(full_run)
        chart_size_memory(full_run)
    if analysis:
        chart_agreement(analysis)
        chart_ablations(analysis)
    if validation:
        chart_targets(validation)
    if study:
        chart_calibration_size(study)
        chart_train_vs_validation(study)
    pipeline_diagram()


if __name__ == "__main__":
    main()
