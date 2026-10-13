"""Chart and table for the calibration-scale study (accuracy vs dataset size).

Reads reports/data/calibration_scale_study.json and writes
reports/img/results/calibration_scale.png, and prints a markdown table.

Usage:
    uv run --with matplotlib python scripts/make_calibration_scale_chart.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "reports" / "data" / "calibration_scale_study.json"
OUT = PROJECT_ROOT / "reports" / "img" / "results" / "calibration_scale.png"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LABELS = {
    "squeezenet1.1-13.onnx": "SqueezeNet 1.1", "shufflenet-v2-13.onnx": "ShuffleNet V2",
    "mobilenetv2.onnx": "MobileNetV2", "resnet18-v1-13.onnx": "ResNet-18",
    "resnet50.onnx": "ResNet-50", "resnet152.onnx": "ResNet-152",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK_2, "ytick.color": INK_2,
    "font.size": 9, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def series(result):
    totals = sorted({p["total_dataset"] for p in result["points"]})
    mean, lo, hi = [], [], []
    for t in totals:
        vals = [p["test"]["top1"] * 100 for p in result["points"] if p["total_dataset"] == t]
        mean.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
    return np.array(totals), np.array(mean), np.array(lo), np.array(hi)


def chart(study, out=OUT):
    cols = 3
    rows = int(np.ceil(len(study) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(study):]:
        ax.set_visible(False)
    for ax, result in zip(axes, study):
        totals, mean, lo, hi = series(result)
        x = np.arange(len(totals))  # evenly spaced, avoids log-tick crowding
        fp32 = result["fp32_test"]["top1"] * 100
        ax.axhline(fp32, color=INK_2, linewidth=1.1, label="FP32 (reference)")
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=BLUE, linewidth=2, marker="o", markersize=5,
                label="Static INT8 (per-channel)")
        ax.set_xticks(x, [f"{t}\n({int(t * 0.3)})" for t in totals])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(max(0, min(fp32, lo.min()) - 12), min(100, max(fp32, hi.max()) + 8))
        ax.set_title(f"{LABELS[result['model']]}  (FP32: {fp32:.1f}%)")
        ax.set_ylabel("Top-1 on test set (%)")
        ax.grid(True)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Static INT8 accuracy vs total dataset size (30/70 split, fixed {study[0]['test_samples']}-image "
                 "test set)", fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.supxlabel("Total dataset size (calibration = 30%); shaded band = min-max over random draws",
                  fontsize=10, color=INK_2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def table(study):
    totals = sorted({p["total_dataset"] for p in study[0]["points"]})
    header = ["Model", "FP32"] + [f"{t} ({int(t * 0.3)} cal)" for t in totals]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for result in study:
        row = [LABELS[result["model"]], f"{result['fp32_test']['top1'] * 100:.1f}%"]
        for t in totals:
            vals = [p["test"]["top1"] * 100 for p in result["points"] if p["total_dataset"] == t]
            row.append(f"{np.mean(vals):.1f}%")
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser(description="Calibration-scale chart and table.")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    study = json.loads(args.data.read_text(encoding="utf-8"))
    chart(study, args.out)
    print()
    table(study)


if __name__ == "__main__":
    main()
