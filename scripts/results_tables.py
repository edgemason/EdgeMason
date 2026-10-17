"""Print the results tables (markdown) and render the summary table as a PNG.

Reads the JSON files in reports/data/ so no number is copied by hand.

Usage:
    uv run --with matplotlib python scripts/results_tables.py > reports/tables.md
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import onnx  # noqa: E402

from api.analyzer.model_analyzer import count_parameters  # noqa: E402

DATA = PROJECT_ROOT / "reports" / "data"
IMG = PROJECT_ROOT / "reports" / "img" / "results"
LABELS = {
    "squeezenet1.1-13.onnx": "SqueezeNet 1.1",
    "shufflenet-v2-13.onnx": "ShuffleNet V2",
    "mobilenetv2.onnx": "MobileNetV2",
    "resnet18-v1-13.onnx": "ResNet-18",
    "resnet50.onnx": "ResNet-50",
    "resnet152.onnx": "ResNet-152",
}
CANDIDATES = {
    "onnx_fp32_baseline": "FP32 (ONNX Runtime)",
    "onnx_static_int8": "Static INT8 (ONNX Runtime)",
    "tvm_opt_level_3": "FP32 (TVM, untuned)",
    "tvm_int8_quantized": "INT8 (TVM, untuned)",
}


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def pct(value):
    return f"{value * 100:.1f}%"


def main_table(analysis):
    header = ["Model", "Params", "FP32 size", "INT8 size", "FP32 p50", "INT8 p50", "Speed-up",
              "FP32 top-1 / top-5", "INT8 top-1 / top-5", "Same prediction"]
    # Two-line headers for the PNG version (markdown keeps one line).
    main_table.png_header = ["Model", "Params", "FP32\nsize", "INT8\nsize", "FP32\np50", "INT8\np50",
                             "Speed-up", "FP32 top-1\n/ top-5", "INT8 top-1\n/ top-5", "Same top-1\nprediction"]
    rows = []
    for result in analysis:
        params = count_parameters(onnx.load(str(PROJECT_ROOT / "models" / result["model"])))
        fp32, int8 = result["variants"]["fp32"], result["variants"]["static_int8_per_channel"]
        rows.append([
            LABELS[result["model"]],
            f"{params / 1e6:.1f} M",
            f"{fp32['size_mb']:.1f} MB",
            f"{int8['size_mb']:.1f} MB",
            f"{fp32['latency_p50_ms_1thread']:.1f} ms",
            f"{int8['latency_p50_ms_1thread']:.1f} ms",
            f"{fp32['latency_p50_ms_1thread'] / int8['latency_p50_ms_1thread']:.2f}x",
            f"{pct(fp32['top1'])} / {pct(fp32['top5'])}",
            f"{pct(int8['top1'])} / {pct(int8['top5'])}",
            pct(result["agreement_fp32_vs_static_int8"]["same_top1_prediction"]),
        ])
    return header, rows


def pipeline_table(full_run):
    header = ["Model", "Candidate", "p50 (1 core)", "Timed runs", "RAM (peak - baseline)", "Size", "Top-1",
              "Test images"]
    rows = []
    for model in full_run["models"]:
        for candidate in model["candidates"]:
            winner = " (winner)" if candidate["name"] == model["winner"] else ""
            rows.append([
                LABELS[model["model"]],
                CANDIDATES[candidate["name"]] + winner,
                f"{candidate['latency_p50_ms']:.1f} ms",
                f"{candidate['warmup_runs']}+{candidate['timed_runs']}",
                f"{candidate['memory_delta_mb']:.1f} MB",
                f"{candidate['model_size_mb']:.1f} MB",
                pct(candidate["top1"]),
                str(candidate["test_samples"]),
            ])
    return header, rows


def ablation_table(analysis):
    header = ["Model", "FP32 top-1", "INT8 per-tensor top-1", "INT8 per-channel top-1",
              "FP32 p50", "Dynamic INT8 p50", "Static INT8 p50"]
    rows = []
    for result in analysis:
        v = result["variants"]
        rows.append([
            LABELS[result["model"]],
            pct(v["fp32"]["top1"]),
            pct(v["static_int8_per_tensor"]["top1"]),
            pct(v["static_int8_per_channel"]["top1"]),
            f"{v['fp32']['latency_p50_ms_1thread']:.1f} ms",
            f"{v['dynamic_int8']['latency_p50_ms_1thread']:.1f} ms",
            f"{v['static_int8_per_channel']['latency_p50_ms_1thread']:.1f} ms",
        ])
    return header, rows


def agreement_table(analysis):
    header = ["Model", "Both right", "FP32 right, INT8 wrong", "FP32 wrong, INT8 right", "Both wrong",
              "Same top-1 prediction"]
    rows = []
    for result in analysis:
        a = result["agreement_fp32_vs_static_int8"]
        rows.append([LABELS[result["model"]], str(a["both_right"]), str(a["fp32_right_int8_wrong"]),
                     str(a["fp32_wrong_int8_right"]), str(a["both_wrong"]), pct(a["same_top1_prediction"])])
    return header, rows


def calibration_table(study):
    sizes = sorted({p["calibration_size"] for p in study[0]["points"]})
    header = ["Model", "FP32"] + [f"{s} img" for s in sizes]
    rows = []
    for key, label in [("per_channel_test", "per-channel"), ("per_tensor_test", "per-tensor")]:
        for result in study:
            means = []
            for size in sizes:
                values = [p[key]["top1"] for p in result["points"] if p["calibration_size"] == size]
                means.append(pct(sum(values) / len(values)))
            rows.append([f"{LABELS[result['model']]} ({label})", pct(result["fp32_test"]["top1"])] + means)
    return header, rows


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_png(header, rows, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 0.45 * (len(rows) + 2)))
    ax.axis("off")
    # Wider columns for the "top-1 / top-5" pairs.
    widths = [0.12, 0.07, 0.08, 0.08, 0.085, 0.085, 0.08, 0.125, 0.125, 0.1]
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(cell.get_height() * 1.5)
        cell.set_edgecolor("#e6e5e1")
        if row == 0:
            cell.set_facecolor("#1a4f91")
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#fcfcfb" if row % 2 else "#f2f1ed")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0b0b0b", pad=12)
    fig.patch.set_facecolor("#fcfcfb")
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    analysis = load("quantization_analysis.json")
    full_run = load("full_run_x86_64_2026-10-05.json")
    study = load("calibration_size_study.json")
    summary = main_table(analysis)
    sections = [
        ("Summary: FP32 vs static INT8 (ONNX Runtime, 179 test images, 1 CPU thread)", summary),
        ("Full pipeline on x86_64 (all four candidates)", pipeline_table(full_run)),
        ("Ablations", ablation_table(analysis)),
        ("FP32 vs INT8 agreement (179 test images)", agreement_table(analysis)),
        ("Top-1 vs calibration-set size (mean of draws, 156 test images)", calibration_table(study)),
    ]
    for title, (header, rows) in sections:
        print(f"### {title}\n\n{markdown(header, rows)}\n")
    render_png(main_table.png_header, summary[1], IMG / "summary_table.png",
               "FP32 vs static INT8 (ONNX Runtime, x86_64, 1 CPU thread, 179 held-out ImageNetV2 images)")


if __name__ == "__main__":
    main()
