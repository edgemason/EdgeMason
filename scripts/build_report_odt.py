"""Fill the institute's ODT report template with the EdgeMason report.

Keeps the cover, certificate, declaration and plagiarism pages as they are,
fills the approval sheet, the paper title/author frame, abstract, impact
statement and index terms, and replaces the template's instruction text with
the report body (headings, tables, figures, references) using the template's
own paragraph styles. Tables are generated from reports/data/*.json.

Usage:
    uv run --with lxml --with matplotlib --with pillow python scripts/build_report_odt.py
"""

import io
import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import results_tables as rt  # noqa: E402

TEMPLATE = PROJECT_ROOT / "Minor project_UG Research work- Report prepartion.odt"
OUTPUT = PROJECT_ROOT / "EdgeMason_Minor_Project_Report.odt"
DATA = PROJECT_ROOT / "reports" / "data"
IMG = PROJECT_ROOT / "reports" / "img"

NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "xlink": "http://www.w3.org/1999/xlink",
    "manifest": "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0",
}


def q(tag: str) -> str:
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


# --------------------------------------------------------------------------
# References (verified before use) and citation numbering by first appearance
# --------------------------------------------------------------------------

REFERENCES = {
    "krish": "R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper,” arXiv:1806.08342, Jun. 2018.",
    "jacob": "B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018.",
    "tvm": "T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, M. Cowan, H. Shen, L. Wang, Y. Hu, L. Ceze, C. Guestrin, and A. Krishnamurthy, “TVM: An automated end-to-end optimizing compiler for deep learning,” in Proc. 13th USENIX Symp. Operating Syst. Design Implement. (OSDI), 2018.",
    "ansor": "L. Zheng, C. Jia, M. Sun, Z. Wu, C. H. Yu, A. Haj-Ali, Y. Wang, J. Yang, D. Zhuo, K. Sen, J. E. Gonzalez, and I. Stoica, “Ansor: Generating high-performance tensor programs for deep learning,” in Proc. 14th USENIX Symp. Operating Syst. Design Implement. (OSDI), 2020, pp. 863–879.",
    "match": "M. A. Hamdi, F. Daghero, G. M. Sarda, J. Van Delm, A. Symons, L. Benini, M. Verhelst, D. Jahier Pagliari, and A. Burrello, “MATCH: Model-aware TVM-based compilation for heterogeneous edge devices,” IEEE Trans. Comput.-Aided Design Integr. Circuits Syst., pp. 3844–3857, Oct. 2025.",
    "ratul": "I. J. Ratul, Y. Zhou, and K. Yang, “Accelerating deep learning inference: A comparative analysis of modern acceleration frameworks,” Electronics, vol. 14, Art. no. 2977, 2025, doi: 10.3390/electronics14152977.",
    "nnmeter": "L. L. Zhang, S. Han, J. Wei, N. Zheng, T. Cao, Y. Yang, and Y. Liu, “nn-Meter: Towards accurate latency prediction of deep-learning model inference on diverse edge devices,” in Proc. 19th Annu. Int. Conf. Mobile Syst., Appl., Services (MobiSys), 2021, pp. 81–93, doi: 10.1145/3458864.3467882.",
    "help": "H. Lee, S. Lee, S. Chong, and S. J. Hwang, “HELP: Hardware-adaptive efficient latency prediction for NAS via meta-learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.",
    "evolp": "S. Huai, H. Kong, S. Li, X. Luo, R. Subramaniam, and C. Makaya, “EvoLP: Self-evolving latency predictor for model compression in real-time edge systems,” IEEE Embedded Syst. Lett., vol. 16, no. 2, 2024, doi: 10.1109/LES.2023.3321599.",
    "ortq": "ONNX Runtime, “Quantize ONNX models,” ONNX Runtime documentation. [Online]. Available: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html",
    "mbv2": "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4510–4520.",
    "resnet": "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.",
    "squeeze": "F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer, “SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,” arXiv:1602.07360, 2016.",
    "shuffle": "N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, “ShuffleNet V2: Practical guidelines for efficient CNN architecture design,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018.",
    "imagenetv2": "B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, “Do ImageNet classifiers generalize to ImageNet?” in Proc. 36th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 97, 2019, pp. 5389–5400.",
    "zoo": "ONNX, “ONNX Model Zoo,” GitHub repository. [Online]. Available: https://github.com/onnx/models",
    "tvmdocs": "Apache TVM, “Apache TVM documentation.” [Online]. Available: https://tvm.apache.org/docs/",
}

CITE_ORDER: list[str] = []


def cite(text: str) -> str:
    """Replace {key} / {key1,key2} with IEEE numbers in order of first use."""
    def repl(match):
        keys = [key.strip() for key in match.group(1).split(",")]
        if not all(key in REFERENCES for key in keys):
            return match.group(0)  # set notation such as {latency, memory}, not a citation
        nums = []
        for key in keys:
            if key not in CITE_ORDER:
                CITE_ORDER.append(key)
            nums.append(CITE_ORDER.index(key) + 1)
        return "".join(f"[{n}]" for n in nums)
    return re.sub(r"\{([a-z0-9]+(?:,\s*[a-z0-9]+)*)\}", repl, text)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


ANALYSIS = load("quantization_analysis.json")
FULL_RUN = load("full_run_x86_64_2026-10-05.json")
STUDY = load("calibration_size_study.json")
VALIDATION = load("validation_run_2026-10-05.json")
SCALE = load("calibration_scale_study.json")


def scale_table():
    import numpy as np

    totals = sorted({p["total_dataset"] for p in SCALE[0]["points"]})
    labels = {"squeezenet1.1-13.onnx": "SqueezeNet 1.1", "shufflenet-v2-13.onnx": "ShuffleNet V2",
              "mobilenetv2.onnx": "MobileNetV2", "resnet18-v1-13.onnx": "ResNet-18",
              "resnet50.onnx": "ResNet-50", "resnet152.onnx": "ResNet-152"}
    header = ["Model", "FP32"] + [f"{t} ({int(t * 0.3)} cal)" for t in totals]
    rows = []
    for result in SCALE:
        row = [labels[result["model"]], f"{result['fp32_test']['top1'] * 100:.1f}%"]
        for t in totals:
            vals = [p["test"]["top1"] * 100 for p in result["points"] if p["total_dataset"] == t]
            row.append(f"{np.mean(vals):.1f}%")
        rows.append(row)
    return header, rows


def target_rows():
    labels = {"x86_64": "x86_64", "rpi5": "Raspberry Pi 5", "jetson_nano": "Jetson Nano (CPU)",
              "rpi4": "Raspberry Pi 4", "rpi3": "Raspberry Pi 3"}
    order = list(labels)
    runs = sorted([d for d in VALIDATION["deploys"] if d["priority"] == "latency"],
                  key=lambda d: order.index(d["hardware"]))
    rows = []
    for d in runs:
        r = {x["name"]: x for x in d["results"]}
        fp, i8 = r["onnx_fp32_baseline"], r["onnx_static_int8"]
        winner = "Static INT8" if d["winner"] == "onnx_static_int8" else "FP32"
        rows.append([labels[d["hardware"]],
                     f"{fp['estimated_latency_p50_ms']:.1f} ms (×{fp['latency_scaling_factor']})",
                     f"{i8['estimated_latency_p50_ms']:.1f} ms (×{i8['latency_scaling_factor']})",
                     f"{fp['estimated_memory_mb']:.1f} / {i8['estimated_memory_mb']:.1f} MB", winner])
    return ["Target", "FP32 est. latency", "Static INT8 est. latency", "Est. memory FP32 / INT8", "Winner"], rows


# --------------------------------------------------------------------------
# Report content
# --------------------------------------------------------------------------

ABSTRACT = (
    "Deploying a trained convolutional neural network (CNN) on an edge device requires choosing a numeric "
    "precision, an inference runtime and an execution provider, and the effect of each choice on latency, memory "
    "and accuracy depends on the target hardware. These decisions are usually made by hand and checked "
    "informally on a laptop. This report presents EdgeMason, a web service that automates "
    "the decision. Given an ONNX model, labelled calibration images, a target device profile and user constraints, "
    "EdgeMason plans a set of candidates (FP32, static INT8 and dynamic INT8 under ONNX Runtime, and optionally "
    "Apache TVM), quantizes them, benchmarks each one in an isolated single-thread process, scores accuracy on images "
    "that were not used for calibration, scales host measurements to the target, ranks the candidates and returns a "
    "self-contained deployment package. An audit of the first implementation revealed defects that changed the "
    "selected model, including accuracy graded on calibration data, memory readings that were always zero and a "
    "server that stopped responding during a run; ten corrections were implemented and verified. On six ImageNet "
    "CNNs evaluated on 179 held-out ImageNetV2 images, per-channel static INT8 reduced model size by 3.5–3.9×, "
    "reduced peak memory by up to 3.6× and stayed within 2.3 points of FP32 top-1 accuracy. The INT8 speed-up grew "
    "with model size from 1.17× (MobileNetV2) to 1.75× (ResNet-152), whereas the two smallest networks became slower, "
    "which shows that the choice has to be measured rather than assumed. Five to ten calibration images were enough "
    "for stable INT8 accuracy."
)

IMPACT = (
    "Engineers who deploy vision models on small boards rarely know in advance whether an 8-bit version of their "
    "network will be faster, smaller or less accurate on that particular device, because measuring every option on "
    "every board is slow. EdgeMason reduces this to a single upload: it measures the options, reports the trade-offs "
    "together with their error margins, and returns a package that runs on the target with two commands. The "
    "experiments show why this matters. The same INT8 technique that makes ResNet-152 1.75 times faster makes "
    "SqueezeNet slower, and a common evaluation shortcut overstates INT8 accuracy by several points. With EdgeMason, "
    "a team can choose a deployment configuration from evidence, shrink models by close to four times where accuracy "
    "allows, and avoid configurations that look fast on a laptop but miss the latency budget on a Raspberry Pi."
)

INSTITUTE = "Dr. Shyama Prasad Mukherjee International Institute of Information Technology"
CITY = "Naya Raipur, India"
# (name, italic affiliation lines, plain lines)
AUTHORS = [
    ("Abhay Yadav", ["Department of Computer Science and Engineering", INSTITUTE], [CITY]),
    ("Mayank Dewangan", ["Department of Data Science and Artificial Intelligence", INSTITUTE], [CITY]),
    ("Dr. Kavita Jaiswal", ["Assistant Professor", "Department of Computer Science and Engineering", INSTITUTE],
     [CITY]),
]

INDEX_TERMS = "Apache TVM, edge computing, inference benchmarking, ONNX Runtime, post-training quantization, Raspberry Pi."


def body_blocks():
    """The report body as a list of blocks, in order."""
    B = []
    h1 = lambda t: B.append(("h1", t))
    h2 = lambda t: B.append(("h2", t))
    p = lambda t: B.append(("p", cite(t)))
    eq = lambda t: B.append(("eq", t))
    items = lambda xs: B.append(("list", [cite(x) for x in xs]))
    fig = lambda path, caption, width=6.3: B.append(("fig", path, cite(caption), width))
    table = lambda caption, header, rows, widths=None: B.append(("table", cite(caption), header,
                                                                 [[cite(c) for c in r] for r in rows], widths))
    code = lambda lines: B.append(("code", lines))

    # ---------------------------------------------------------------- I
    h1("Introduction")
    p("Image classification and detection models are increasingly executed on the device that captures the data "
      "rather than in a data centre, which removes network round trips and keeps data local. Single-board computers "
      "such as the Raspberry Pi offer only a few ARM cores and a few gigabytes of memory, so a model that runs "
      "comfortably on a laptop may be too slow or too large on the board. Post-training quantization, which stores "
      "weights and activations as 8-bit integers instead of 32-bit floating point, is the most common remedy: it "
      "reduces model size by about four times and can speed up inference on CPUs with integer instructions "
      "{krish,jacob}. Deep learning compilers such as Apache TVM pursue the same goal from another direction by "
      "generating code that is specialised for one model and one processor {tvm}.")
    p("In practice the benefit of these techniques is not uniform. It depends on the architecture of the network, "
      "on the quantization scheme, on the runtime that executes the graph and on the instruction set of the target "
      "processor. Counting floating-point operations is a poor predictor of real latency once operator fusion and "
      "kernel selection are taken into account {nnmeter}. As a result, practitioners usually try a few options on "
      "their development machine and keep the one that looks best. This informal process is easy to get wrong: "
      "candidates may be compared on different processors, accuracy may be measured on the same images that were "
      "used to calibrate the quantizer, and memory may be read at the wrong moment.")
    p("This report describes EdgeMason, a service that turns the selection into a repeatable measurement. A user "
      "uploads an ONNX model and a set of labelled calibration images, chooses a target device and states a "
      "priority and optional limits on latency and memory. EdgeMason analyses the graph, plans suitable candidates, "
      "quantizes and compiles them, benchmarks them under controlled conditions, measures their accuracy on held-out "
      "images, estimates their behaviour on the target and returns the winning model as a package that runs on the "
      "device with a single command.")

    h1("Motivation")
    p("The project started from a working prototype that already produced a winner and a download for every "
      "request. A systematic audit of that prototype, carried out by driving its dashboard and calling its API "
      "with different targets, priorities and constraints, showed that the answers could not be trusted. With "
      "the default automatic execution provider, the FP32 model ran on the laptop GPU at about 1.06 ms while the "
      "INT8 models fell back to the CPU at about 6.07 ms, so FP32 was selected even for a Raspberry Pi target that "
      "has no GPU. The ranking used laptop latency directly, and one scaling factor per device was applied to every "
      "candidate, so the chosen model never changed with the target. Memory was sampled after the model had been "
      "loaded, so every candidate reported a memory use of zero and the memory priority picked the largest model. "
      "Accuracy was computed on the calibration images themselves, which overstated INT8 accuracy by about seven "
      "points on MobileNetV2 and ResNet-50. The downloaded inference script failed with a type error on its first "
      "line of inference, and the server stopped answering any request while a deployment was running.")
    p("Each of these defects changes which model a user would ship. They also illustrate a broader point: a "
      "deployment assistant is only useful if its measurements are fair, isolated and reported with their "
      "uncertainty. The work in this report therefore focuses as much on measurement methodology as on "
      "optimisation techniques.")

    h2("Contributions")
    p("The main contributions of this work are the following.")
    items([
        "An end-to-end, target-aware deployment pipeline that plans, quantizes, benchmarks, scores, estimates, "
        "ranks and packages candidate models, implemented as a modular FastAPI service with a web dashboard.",
        "A measurement methodology for fair comparison: one execution provider per request, single-thread "
        "benchmarks in a fresh process per candidate, a time-budgeted run count, peak-memory measurement from a "
        "baseline taken before the model is loaded, and a seeded split between calibration and test images with "
        "explicit error margins.",
        "A planner that inspects the operator mix of the graph and selects quantization methods accordingly, so "
        "that convolutional networks receive per-channel static INT8 and never dynamic INT8, independently of "
        "whether ONNX Runtime or TVM executes the model.",
        "An empirical study on six ImageNet CNNs (1.2–60.1 million parameters) covering size, latency, memory, "
        "accuracy, FP32–INT8 agreement, sensitivity to calibration-set size, two ablations and an untuned TVM "
        "baseline.",
        "A framework-independent deployment package whose inference script, dependency list and container recipe "
        "were verified in a clean environment and in Docker.",
    ])
    p("The remainder of the report is organised as follows. Section III reviews related work and identifies the "
      "research gap. Section IV defines the problem formally. Section V presents the proposed framework. Section VI "
      "reports the experimental results, and Section VII concludes with future directions.")

    # ---------------------------------------------------------------- II
    h1("Literature Work")
    p("Work related to EdgeMason falls into three groups: compilers that generate efficient code for edge "
      "processors, empirical comparisons of inference frameworks, and models that predict latency without running "
      "every candidate on the device. Quantization itself is well established: integer-only inference with jointly "
      "designed training was shown to run efficiently on ARM CPUs with close to a fourfold reduction in memory "
      "{jacob}, and a widely used whitepaper reports that per-channel weight quantization with per-layer activation "
      "quantization keeps post-training accuracy within 2% of floating point for many CNNs, with 2–3× CPU "
      "speed-ups {krish}. ONNX Runtime implements both static and dynamic post-training quantization and recommends "
      "static quantization for CNNs {ortq}.")
    p("On the compiler side, TVM introduced an end-to-end compiler stack for deep learning {tvm}, and Ansor "
      "generates high-performance tensor programs by searching a large space of schedules with hardware in the "
      "loop {ansor}. MATCH extends TVM with a model-aware hardware abstraction that maps parts of the network onto "
      "on-chip accelerators of heterogeneous microcontrollers; on the MLPerf Tiny models it lowers latency by up "
      "to 60.88× relative to plain TVM on the DIANA platform and by 2.15× relative to the DORY compiler on GAP9 "
      "{match}. Ratul et al. compare PyTorch, ONNX Runtime, TensorRT, Apache TVM and JAX on an NVIDIA Jetson AGX "
      "Orin in terms of accuracy, inference time, throughput, memory and power, and find that no single framework "
      "is best on every axis {ratul}.")
    p("Latency prediction avoids running every candidate on the device. nn-Meter decomposes a model into fused "
      "kernels and predicts kernel latencies, reaching 99.0%, 99.1% and 83.4% of predictions within ±10% on a "
      "mobile CPU, a mobile GPU and an Intel VPU over 26,000 models {nnmeter}. HELP treats latency estimation on a "
      "new device as a meta-learning problem and produces accurate estimates from as few as ten measurements "
      "{help}. EvoLP keeps refining its predictor while a network is being compressed and was evaluated on three "
      "edge devices and four model variants {evolp}. Table I summarises these works.")
    table("Summary of the reviewed literature",
          ["Ref.", "Technique used", "Platform / data", "Accuracy / measures", "Remarks (limitations)"],
          [
              ["{jacob}", "Integer-only quantization with co-designed training", "ARM CPUs; ImageNet, COCO",
               "Close to 4× memory reduction; better accuracy–latency trade-off on CPUs",
               "Requires training; does not choose between runtimes or devices"],
              ["{krish}", "Post-training and quantization-aware 8-bit quantization of CNNs", "CPUs, DSPs; ImageNet CNNs",
               "Per-channel 8-bit within 2% of FP32; 4× smaller; 2–3× CPU speed-up",
               "Guidance only; no automated selection or packaging"],
              ["{match}", "TVM-based compilation with model-aware hardware abstraction (BYOC)",
               "Heterogeneous MCUs (GAP9, DIANA); MLPerf Tiny",
               "Up to 60.88× lower latency than plain TVM; 2.15× vs. DORY",
               "Optimises one compiler flow for MCUs; no runtime or precision search"],
              ["{ratul}", "Empirical comparison of PyTorch, ONNX Runtime, TensorRT, TVM and JAX",
               "NVIDIA Jetson AGX Orin", "Accuracy, inference time, throughput, memory, power",
               "Single platform; measures but does not automate selection"],
              ["{nnmeter}", "Kernel-level latency prediction with fusion-aware kernel detection",
               "Mobile CPU, mobile GPUs, Intel VPU; 26,000 models",
               "99.0% / 99.1% / 83.4% of predictions within ±10%",
               "Latency only; no accuracy or memory trade-off"],
              ["{help}", "Meta-learned, hardware-adaptive latency predictor for NAS", "Unseen hardware platforms",
               "Accurate estimates from as few as 10 measurements",
               "Targets architecture search, not post-training quantization"],
              ["{evolp}", "Self-evolving latency predictor during model compression",
               "Three edge devices, four model variants", "Outperforms earlier predictors",
               "Covers compression variants; runtime choice not addressed"],
              ["This work", "Target-aware planning, isolated benchmarking, held-out scoring and packaging",
               "x86-64 host; estimates for RPi 3/4/5 and Jetson Nano; six ImageNet CNNs",
               "3.5–3.9× smaller, up to 1.75× faster, ≤2.3-point top-1 loss",
               "ARM estimates use unmeasured INT8 multipliers; TVM untuned"],
          ], widths=[7, 24, 22, 24, 23])
    h2("Research Gaps")
    p("The reviewed works either optimise a single compilation path, report measurements for a fixed set of "
      "frameworks, or predict latency in isolation. Three gaps remain for a practitioner who simply wants the best "
      "deployable model for a given board. First, no reviewed tool closes the loop from an uploaded model to a "
      "verified package while weighing latency, memory and accuracy together. Second, accuracy is rarely reported "
      "with the held-out protocol and error margins that small calibration sets require, so INT8 losses are easy to "
      "understate. Third, the value of INT8 depends on the instruction set of the target processor, which means the "
      "decision must be made per device rather than once per model.")

    # ---------------------------------------------------------------- III
    h1("Problem definition")
    p("Let M be a trained classification model in ONNX format, D_c a set of labelled calibration images and D_t a "
      "disjoint set of labelled test images. A target device h is described by a hardware profile that lists its "
      "allowed execution providers, a latency multiplier s_h(q) for each quantization method q and a memory factor "
      "μ_h. The user supplies a priority π ∈ {latency, memory, accuracy, balanced} and optional limits L_max and "
      "M_max. A candidate is a pair c = (q, r) of a quantization method q ∈ {FP32, static INT8, dynamic INT8} and a "
      "runtime r ∈ {ONNX Runtime, TVM}. For every candidate we measure, on the host, the median latency L(c) and the "
      "peak memory increase M(c), and on D_t the top-1 accuracy A(c). The estimates for the target are")
    eq("L̂_h(c) = s_h(q) · L(c),    M̂_h(c) = μ_h · M(c).")
    p("The feasible set is F = {c : L̂_h(c) ≤ L_max and M̂_h(c) ≤ M_max}, and the selected candidate is")
    eq("c* = argmin over c ∈ F of J_π(c),")
    p("where J_latency = L̂_h, J_memory = M̂_h, J_accuracy = −A, and the balanced objective normalises both resources "
      "by the best feasible value so that milliseconds and megabytes carry equal weight:")
    eq("J_balanced(c) = L̂_h(c) / min L̂_h + M̂_h(c) / min M̂_h.")
    p("The problem is to solve this selection reliably: every candidate must be measured on the same execution "
      "provider, A must be computed on images outside D_c, M must include the cost of loading the model, and the "
      "returned package must run on h without the user knowing which q and r were chosen. If F is empty, no model is "
      "returned.")

    # ---------------------------------------------------------------- IV
    h1("Proposed framework")
    p("Fig. 1 shows the EdgeMason pipeline for one deployment request. The service is written in Python with "
      "FastAPI. HTTP handling lives in a routes package and the logic in a services package, so that analysis, "
      "planning, benchmarking, ranking and packaging can be tested independently. Each step is described below.")
    fig(IMG / "pipeline_diagram.png", "EdgeMason deployment pipeline for one request. Quantization, benchmarking, "
        "scoring and estimation run once per candidate.")
    h2("Analysis and planning")
    p("The analyser reports model size, parameter count and the number of each operator type. The planner classifies "
      "the model as a CNN when convolution operators outnumber matrix-multiplication operators (MatMul and Gemm), and "
      "as MatMul-heavy in the opposite case. FP32 is always planned as a baseline. Static INT8 is added when "
      "calibration images are available. Dynamic INT8, which computes activation scales at run time before every "
      "layer, is only planned for MatMul-heavy models, because on CNNs the run-time conversion before every "
      "convolution costs more than the integer arithmetic saves. Candidates are labelled by method rather than by "
      "framework, so the same rule applies to TVM candidates. When a CNN arrives without calibration data, only FP32 "
      "is planned and the user is told why.")
    h2("Execution provider selection")
    p("Each hardware profile lists the execution providers the device supports in order of preference: CPU only for "
      "the Raspberry Pi boards, and CUDA followed by CPU for the Jetson Nano and x86-64 hosts. One provider is "
      "selected per request and every candidate runs on it, which rules out the unfair GPU-versus-CPU comparison "
      "observed in the prototype. If a candidate cannot run on the chosen GPU provider because some of its INT8 "
      "operators exist only on the CPU, it is reported as unavailable with a plain-language warning rather than being "
      "silently moved to the CPU. The dashboard defaults to the CPU provider.")
    h2("Quantization")
    p("Static INT8 uses ONNX Runtime's quantize-dequantize format with signed 8-bit weights and activations "
      "{ortq}. The graph is first pre-processed (constant folding, Conv–BatchNorm fusion and shape inference), and "
      "weights are quantized per output channel so that channels with small weights keep their resolution. "
      "Activation ranges are calibrated with the min–max method on the calibration images. Dynamic INT8 uses "
      "unsigned 8-bit weights, which were faster than signed weights on the host. The optional TVM path compiles "
      "the FP32 model, or the static INT8 model, into a Relax virtual-machine module {tvm,tvmdocs}.")
    h2("Benchmarking and memory measurement")
    p("Each candidate is benchmarked in a new Python process so that memory left behind by earlier candidates or by "
      "the server cannot leak into its reading. Inside that process the resident memory is recorded before the model "
      "is loaded, the Linux high-water mark is reset at that moment, and the peak is read after the timed runs; the "
      "difference is the memory the model needs. A process-lifetime counter was rejected after it was found to "
      "inherit the parent's memory, which made every model appear to use about 630 MB. Inference uses one thread. "
      "Each candidate receives up to ten warm-up runs and up to one hundred timed runs, with timing stopped after "
      "about ten seconds but never before ten timed runs, so that large models finish quickly while small models "
      "receive the full count. The median (p50) and the 95th percentile are reported.")
    h2("Accuracy evaluation")
    p("A labelled calibration file is shuffled with a fixed seed and split once: 30% of the images, capped at 300, "
      "are used for calibration, and the remainder form the test set on which every candidate is scored. Top-1 and "
      "top-5 accuracy are reported together with warnings when fewer than 100 calibration images or fewer than 200 "
      "test images are available, including the approximate 95% margin of the accuracy estimate.")
    h2("Target estimation and ranking")
    p("Host measurements are converted into target estimates using the profile multipliers of Section IV. The "
      "multiplier for FP32 reflects the relative speed of one core of the target; the INT8 multipliers additionally "
      "reflect whether the target has fast integer dot-product instructions. Ranking and the latency and memory "
      "limits all operate on the target estimates. The priorities follow the objective functions defined in "
      "Section IV.")
    h2("Packaging and service")
    p("The winner is packaged as a ZIP archive containing the model, a single inference script that detects whether "
      "an ONNX or a TVM model is present and loads the matching runtime, a requirements file pinned to the exact "
      "versions used for benchmarking, a multi-architecture Dockerfile and a README with the input shape and the "
      "expected latency, memory and accuracy on the target. The heavy work of a request runs on a worker thread, and "
      "requests are serialised so that two benchmarks never compete for the CPU, which keeps the dashboard "
      "responsive during a run. Fig. 2 shows the dashboard after a deployment for a Raspberry Pi 4 target.")
    fig(IMG / "dashboard_rpi4_cpu.jpg", "EdgeMason dashboard after a MobileNetV2 deployment for a Raspberry Pi 4 "
        "target: estimated latency and memory, data split and small-data warnings, and the per-candidate table.")
    h2("Algorithm")
    p("Algorithm 1 summarises a deployment request.")
    code([
        "Algorithm 1: EdgeMason deployment request",
        "Input : model M, labelled images D, target h, priority π, limits L_max, M_max",
        "Output: deployment package P, or no winner",
        " 1  (D_c, D_t) ← SeededSplit(D, 30%, cap 300)",
        " 2  family ← Classify(OperatorCounts(M))",
        " 3  r ← SelectProvider(profile(h))           // one provider for all candidates",
        " 4  C ← Plan(M, family, D_c ≠ ∅, π, L_max)    // no dynamic INT8 for CNNs",
        " 5  for each candidate c in C do",
        " 6      m_c ← Quantize(M, c, D_c)",
        " 7      (L, Mem) ← BenchmarkInFreshProcess(m_c, r, 1 thread)",
        " 8      A ← TopK(m_c, D_t)",
        " 9      (L̂, M̂) ← (s_h(c) · L, μ_h · Mem)",
        "10  F ← { c : L̂ ≤ L_max and M̂ ≤ M_max }",
        "11  if F = ∅ then return no winner",
        "12  c* ← argmin over F of J_π(c)",
        "13  return Package(m_c*, h, estimates, accuracy)",
    ])

    # ---------------------------------------------------------------- V
    h1("Experimental result and Discussion")
    p("This section reports the evaluation of EdgeMason on six CNNs. All numbers were produced by the scripts in the "
      "project repository and are stored as JSON files, and every table in this section was generated from those "
      "files.")
    h2("Model setting and protocols")
    source = {"SqueezeNet 1.1": ("ONNX Model Zoo {zoo}, opset 7 → 13", "{squeeze}", "CNN without BatchNorm"),
              "ShuffleNet V2": ("ONNX Model Zoo {zoo}, opset 10 → 13", "{shuffle}", "CNN, channel shuffle"),
              "MobileNetV2": ("Project model, opset 13", "{mbv2}", "CNN, depthwise convolutions"),
              "ResNet-18": ("ONNX Model Zoo {zoo}, opset 8 → 13", "{resnet}", "CNN, residual"),
              "ResNet-50": ("Project model, opset 13", "{resnet}", "CNN, residual bottleneck"),
              "ResNet-152": ("Project model, opset 13", "{resnet}", "CNN, residual bottleneck")}
    _, summary_rows = rt.main_table(ANALYSIS)
    p("Six pretrained ImageNet classifiers spanning 1.2 to 60.1 million parameters were evaluated (Table II). "
      "EdgeMason optimises trained models and does not retrain them, so the roles usually played by training and "
      "validation data are taken by the calibration set, from which the quantizer learns its scales, and the "
      "held-out test set, on which accuracy is measured. The images come from ImageNetV2 {imagenetv2}, one image per "
      "class, resized to 232 pixels, centre-cropped to 224 × 224 and normalised with the ImageNet mean and standard "
      "deviation. A fixed-seed split assigns 77 images to calibration and 179 to testing. The three models taken "
      "from the ONNX Model Zoo used an operator-set version older than 13, for which per-channel quantization "
      "produced an invalid graph; they were upgraded to opset 13 with the ONNX version converter, and the upgraded "
      "models gave identical predictions and logits on all 256 images.")
    table("Models evaluated",
          ["Model", "Parameters", "FP32 size", "Source", "Architecture"],
          [[row[0], row[1], row[2], source[row[0]][0], f"{source[row[0]][2]} {source[row[0]][1]}"]
           for row in summary_rows], widths=[16, 13, 13, 30, 28])
    h2("Benchmark setting")
    table("Benchmark setting",
          ["Item", "Setting"],
          [
              ["Host", "Intel Core i7-13620H laptop (AVX2, AVX-VNNI), 16 logical cores"],
              ["Software", "Python 3.11, ONNX Runtime 1.27.0, NumPy 1.26.4, Apache TVM 0.26.0"],
              ["Execution provider", "CPU (the GPU present on the host was not used)"],
              ["Threads", "1 intra-op and 1 inter-op thread"],
              ["Runs", "Up to 10 warm-up and 100 timed runs; timing stops after about 10 s, minimum 10 timed runs"],
              ["Isolation", "Fresh process per candidate; candidates run one at a time"],
              ["Memory", "Peak resident memory minus a baseline taken before loading the model"],
              ["Input", "Batch size 1, 3 × 224 × 224, float32"],
              ["TVM", "Relax, generic LLVM target, no auto-tuning"],
          ], widths=[25, 75])
    p("Table III lists the measurement settings. A single thread is used deliberately. The device latency is estimated by multiplying the host latency by a "
      "device factor, which is only meaningful when like is compared with like, one host core against one device "
      "core. With all sixteen host cores the factor would also have to capture the difference in core count, and "
      "multi-core speed-up differs from model to model, so no single factor could be correct. A single thread also "
      "makes repeated measurements more stable on a processor that mixes performance and efficiency cores, and it "
      "makes the comparison with untuned TVM, which uses one core, fair. The resulting estimates are conservative "
      "for a four-core board.")
    h2("Evaluation measures")
    p("Accuracy is reported as top-1 and top-5 on the held-out test set. With n test images the 95% margin of an "
      "accuracy near 50% is approximately")
    eq("e ≈ 1.96 · √(0.25 / n),")
    p("which gives about ±7.3 points for n = 179, ±7.8 points for n = 156 and ±16.6 points for n = 35. Latency is "
      "the median (p50) per-image time; the speed-up is the FP32 p50 divided by the INT8 p50. Model size is the file "
      "size, and memory is the peak resident memory above the pre-load baseline. Because a 1000-class confusion matrix "
      "built from 179 images would be almost empty, the confusion analysis compares the two models directly: for each "
      "test image it records whether FP32 and INT8 are right or wrong, and it reports the share of images on which "
      "both produce the same top-1 class.")
    h2("Performance analysis")
    header, rows = rt.main_table(ANALYSIS)
    table("FP32 versus per-channel static INT8 with ONNX Runtime (1 CPU thread, 179 held-out images)", header, rows,
          widths=[12, 8, 9, 9, 9, 9, 8, 13, 13, 10])
    p("Table IV shows that static INT8 reduces model size by 3.5–3.9× for every network. The speed-up grows with "
      "model size, from 1.17× for MobileNetV2 and 1.23× for ResNet-18 to 1.46× for ResNet-50 and 1.75× for "
      "ResNet-152, because large, regular convolutions benefit most from the integer instructions of the host. The "
      "two smallest networks behave differently: SqueezeNet runs at 0.75× and ShuffleNet V2 at 0.87× of their FP32 "
      "speed. In these models each layer does so little work that converting values into and out of 8-bit form costs "
      "more than the integer arithmetic saves. This result is the main justification for measuring every candidate "
      "instead of assuming that INT8 always wins.")
    header, rows = rt.pipeline_table(FULL_RUN)
    table("Full pipeline on the x86-64 host with all four candidates (ResNet-152 used a 50-image file: 15 calibration / "
          "35 test images)", header, rows, widths=[12, 26, 11, 9, 13, 10, 9, 10])
    p("Table V lists the complete pipeline output for three models with all four candidates, including TVM. The "
      "measured memory confirms the size reduction at run time: INT8 needs 23.5 MB instead of 43.0 MB for "
      "MobileNetV2, 76.6 MB instead of 255.0 MB for ResNet-50 and 144.2 MB instead of 517.6 MB for ResNet-152, a "
      "reduction of 1.8–3.6×. Untuned TVM produces the same FP32 accuracy as ONNX Runtime, which shows that "
      "compilation is correct, but it is 21–57× slower (Fig. 3). Fig. 4 shows file size and measured memory side by side. ONNX Runtime ships hand-optimised kernels, whereas "
      "TVM generates code that becomes competitive only after auto-tuning on the target {ansor,tvmdocs}; tuning was "
      "outside the scope of this work. Every winning package passed its own self-test, with archive sizes of 3.1, "
      "20.1 and 47.4 MB.")
    fig(IMG / "results" / "latency_p50.png", "Median latency per image on one host core. Left: ONNX Runtime FP32 and "
        "static INT8. Right: untuned TVM, drawn on a separate axis because it is 21–57× slower.")
    fig(IMG / "results" / "size_and_memory.png", "Model file size (left) and measured peak memory above baseline "
        "(right) for FP32 and static INT8.")
    header, rows = target_rows()
    table("Estimated latency and memory of MobileNetV2 on each target (host p50 × device multiplier)", header, rows,
          widths=[20, 22, 24, 22, 12])
    p("Table VI and Fig. 5 show that the target changes the decision. On the x86-64 host and on the Raspberry Pi 5, "
      "whose cores provide fast integer dot products, static INT8 is estimated to be faster. On the Raspberry Pi 3 "
      "and 4 and on the CPU of the Jetson Nano the INT8 multiplier is larger and FP32 wins. The INT8 multipliers for "
      "the ARM targets are placeholders until they are measured on the boards, so these particular winners must be "
      "confirmed on hardware; the mechanism, however, is in place.")
    fig(IMG / "results" / "target_latency_estimates.png", "Estimated MobileNetV2 latency on each target for FP32 "
        "and static INT8.")
    h2("Testing accuracy")
    fig(IMG / "results" / "testing_accuracy_top1.png", "Top-1 accuracy on held-out test images for the four "
        "candidates of the full pipeline.")
    p("Across all six models static INT8 stays within 2.3 points of FP32 top-1 accuracy on 179 test images. The "
      "largest losses occur for ShuffleNet V2 (−2.3) and SqueezeNet (−2.2); ResNet-18 gains 1.1 points and ResNet-152 "
      "is unchanged at 68.7%. All of these differences amount to at most four images and lie inside the ±7.3-point "
      "margin. The full-pipeline run of ResNet-152 in Table V shows a 5.7-point drop, but that run used only 35 test "
      "images (two images of difference) and 15 calibration images; on the 179-image set the two models are equal. "
      "Untuned TVM matches the FP32 accuracy of ONNX Runtime on all three models (Fig. 6).")
    h2("Calibration (training) and validation accuracy")
    header, rows = rt.calibration_table(STUDY)
    table("Static INT8 top-1 accuracy versus calibration-set size (mean of three random draws, fixed 156-image test "
          "set)", header, rows, widths=[26, 9, 9, 9, 9, 9, 9, 10, 10])
    p("To measure how much calibration data is needed, each of four fast models was quantized with 1, 2, 5, 10, 25, "
      "50 and 100 calibration images drawn from a pool of 100, three times per size, and scored on a fixed set of "
      "156 test images (Table VII, Fig. 7). With one or two images, ShuffleNet V2 and ResNet-18 lose 6–8 points and "
      "vary strongly between draws; for ResNet-18 a single image gives anywhere between 28.2% and 55.8%. From five "
      "images onward both models match FP32, and beyond 10–25 images there is no consistent gain. Two further "
      "observations stand out. MobileNetV2 with per-tensor scales stays at about 48–50% for every calibration size, "
      "while per-channel scales match FP32 even with one image, so the choice of weight scales matters far more than "
      "the amount of data. SqueezeNet loses accuracy at 100 images (32.7%), because a larger set brings more extreme "
      "activations that widen the min–max ranges.")
    fig(IMG / "results" / "calibration_size_accuracy.png", "Static INT8 test accuracy versus calibration-set size "
        "for per-channel and per-tensor weight scales. The shaded band spans the three random draws.")
    p("Fig. 8 compares accuracy on the calibration images themselves, which play the role of training accuracy, with "
      "accuracy on the held-out images, which plays the role of validation accuracy. Calibration accuracy is "
      "consistently higher: at 100 images it is 63.0% versus 56.4% for MobileNetV2, 56.0% versus 51.3% for "
      "ShuffleNet V2 and 40.0% versus 32.7% for SqueezeNet. This gap is the optimistic bias that the prototype "
      "reported as accuracy, and it is removed by scoring only on images that were not used for calibration. With "
      "one to five calibration images the calibration accuracy rests on a handful of images and swings between 0% "
      "and 100%.")
    fig(IMG / "results" / "calibration_vs_test_accuracy.png", "Accuracy of per-channel static INT8 on the "
        "calibration images (training role) and on held-out images (validation role).")
    h2("Effect of total dataset size")
    p("The study above varied only the calibration set within the 256-image file. To test whether a larger "
      "overall dataset changes the outcome, a second subset of 1000 ImageNetV2 images, one per class, was "
      "downloaded and shuffled once into a fixed 700-image test set and a 300-image calibration pool. "
      "Calibration subsets of 30, 77, 150, 225 and 300 images were drawn, up to three random draws each, and "
      "every quantized model was scored on the same 700 images. These sizes equal 30% of total datasets of 100, "
      "257, 500, 750 and 1000 images, so each point corresponds to a 30/70 split of a total dataset of that "
      "size, while the fixed 700-image test set keeps every accuracy comparable at a tighter margin of about "
      "±3.7 points.")
    header, rows = scale_table()
    table("Static INT8 top-1 accuracy versus total dataset size (mean of random draws, fixed 700-image test set)",
          header, rows, widths=[18, 10, 14, 14, 15, 15, 16])
    fig(IMG / "results" / "calibration_scale.png", "Static INT8 test accuracy versus total dataset size on the "
        "fixed 700-image test set. The shaded band spans the random draws.")
    p("The curves are flat for every model. Growing the total dataset from 100 to 1000 images, and the "
      "calibration set from 30 to 300 images, changes static INT8 top-1 accuracy by at most about one point, "
      "which is inside the ±3.7-point margin, and INT8 stays within one to two points of FP32 throughout. This "
      "confirms the earlier finding on a larger and more reliable test set: for these CNNs a few tens of "
      "calibration images already saturate the quantizer, and adding more data neither helps nor hurts. The "
      "current warning threshold of 100 calibration images is therefore conservative; about 30 would suffice "
      "for these models.")
    h2("Confusion matrices")
    header, rows = rt.agreement_table(ANALYSIS)
    table("FP32 versus static INT8 agreement on 179 test images", header, rows, widths=[16, 13, 18, 18, 13, 22])
    p("Table VIII and Fig. 9 give the 2 × 2 confusion matrices between FP32 and INT8 predictions. For MobileNetV2 and "
      "the three ResNets the off-diagonal cells contain only two to four images each, and the two models produce the "
      "same top-1 class on 89.9–94.4% of the images, so quantization rarely changes a decision. SqueezeNet is the "
      "exception with 30 disagreements and 63.1% agreement. It has no batch normalisation layers, so its activation "
      "ranges are wide and harder to represent with 8 bits.")
    fig(IMG / "results" / "agreement_matrices.png", "Confusion matrices between FP32 and static INT8: counts of test "
        "images on which each model is right or wrong.")
    h2("Ablation study")
    header, rows = rt.ablation_table(ANALYSIS)
    table("Ablations: weight-scale granularity and dynamic versus static INT8", header, rows,
          widths=[15, 12, 15, 15, 13, 15, 15])
    p("Table IX isolates the two design decisions of the quantization stage. Per-channel weight scales matter most "
      "for MobileNetV2, whose accuracy rises from 51.4% with per-tensor scales to 57.5%, against 58.1% for FP32; its "
      "depthwise convolutions have very different weight ranges per channel. For the other models the difference is "
      "within the margin. The ResNet-50 per-tensor result of 68.2% is eight images above per-channel and was not "
      "treated as a real effect. Static INT8 is faster than dynamic INT8 on five of the six models, and dynamic INT8 "
      "is never the fastest option; on MobileNetV2 it is 2.4× slower than FP32 (30.3 ms versus 12.7 ms). These "
      "results support the planner rule that excludes dynamic INT8 for CNNs (Fig. 10).")
    fig(IMG / "results" / "ablations.png", "Ablations. Top: per-tensor versus per-channel weight scales. Bottom: "
        "latency of FP32, dynamic INT8 and static INT8.")
    h2("System-level improvements")
    table("Effect of the corrections on the service (before → after)",
          ["Aspect", "Before", "After"],
          [
              ["MobileNetV2 static INT8 top-1 (held-out)", "51.4%", "57.5% (FP32 58.1%)"],
              ["Accuracy protocol", "Scored on calibration images", "Seeded 30/70 split, margins reported"],
              ["Memory reading per model", "≈0 MB for every model", "43.0 MB FP32 / 23.5 MB INT8 (MobileNetV2)"],
              ["Effect of the target device", "None; same winner everywhere", "Winner follows target estimates"],
              ["FP32 p50 over three identical deploys", "11.95 / 18.42 / 14.17 ms", "13.11 / 14.27 / 13.14 ms"],
              ["Slowest /health reply during a deploy", "31,324 ms (3 replies)", "13.9 ms (115 replies)"],
              ["Candidates planned for a CNN", "3, including dynamic INT8", "2 (FP32, static INT8)"],
              ["Deploy time, MobileNetV2 on CPU", "≈30 s", "≈19 s"],
              ["Downloaded package", "Failed on first inference", "Runs in a clean environment and in Docker"],
              ["Automated tests", "—", "83 passed, 1 skipped"],
          ], widths=[36, 30, 34])
    p("Table X summarises the effect of the corrections on the service itself. Running each benchmark in its own "
      "process made memory measurable, and the longer, time-budgeted runs reduced the spread of FP32 latency over "
      "three identical deployments from about 50% to under 10%. Moving the heavy work to a worker thread kept the "
      "dashboard responsive during deployment: in a repeat measurement all 92 health requests issued during a "
      "deployment were answered, the slowest in 164 ms. The package was installed from its requirements file in a "
      "new Python environment, where it classified 12 of 20 sample images correctly, in line with the measured "
      "accuracy, and its container image built and ran on the host.")
    h2("Discussion and threats to validity")
    p("The results support three conclusions. Per-channel static INT8 is a safe default for CNNs in terms of size, "
      "memory and accuracy, but not always in terms of speed, so a deployment tool must measure rather than assume. "
      "A small calibration set of five to ten representative images is sufficient for the models studied, provided "
      "the weights are quantized per channel. Finally, a large part of the value of the tool lies in measurement "
      "hygiene: comparing candidates on one provider, measuring memory from a clean baseline and scoring on held-out "
      "images each changed the selected model in the prototype.")
    p("Several limitations remain. All measurements were taken on an x86-64 host; device latencies are estimates "
      "whose INT8 multipliers for ARM boards have not yet been measured. The estimates assume one core, while the "
      "packaged model uses all cores of the device. Test sets of 156–179 images give margins of about ±7–8 points, "
      "so differences of one or two points should be read as ties. TVM was evaluated without auto-tuning and built "
      "for a generic target. The models are image classifiers; MatMul-heavy models such as transformers were not "
      "evaluated.")

    # ---------------------------------------------------------------- VI
    h1("Conclusion and Future directions")
    p("EdgeMason automates the choice of a deployable model for an edge device by planning, quantizing, measuring, "
      "estimating, ranking and packaging candidates under explicit user constraints. Correcting the measurement "
      "methodology of the prototype was essential: a single execution provider per request, isolated single-thread "
      "benchmarks, peak-memory measurement and held-out accuracy each altered which model would have been shipped. "
      "On six ImageNet CNNs, per-channel static INT8 reduced size by 3.5–3.9× and peak memory by up to 3.6× with at "
      "most 2.3 points of top-1 loss, and it was up to 1.75× faster on large models but slower on the two smallest. "
      "Five to ten calibration images proved sufficient.")
    p("Future work will measure the latency multipliers on real Raspberry Pi 3, 4 and 5 boards and on a Jetson Nano, "
      "including a multi-core factor; add ARM cross-compilation and auto-tuning to the TVM path; upgrade old operator "
      "sets automatically before quantization; treat accuracy differences inside the error margin as ties; extend "
      "the evaluation to MatMul-heavy models; and accept a folder of images in place of a prepared calibration file.")

    # ---------------------------------------------------------------- appendix etc.
    h1("Appendix")
    p("All experiments can be reproduced from the project repository. The full pipeline run is produced by "
      "scripts/run_full_benchmark.py, the FP32–INT8 analysis and ablations by scripts/quantization_analysis.py, the "
      "calibration study by scripts/calibration_size_study.py, the figures by scripts/make_results_charts.py, the "
      "tables by scripts/results_tables.py and the responsiveness measurement by "
      "scripts/measure_responsiveness.py. The resulting data are stored in reports/data, and the automated test suite "
      "is run with pytest.")
    h1("Acknowledgment")
    p("The authors thank their project supervisor and the Department of Computer Science and Engineering for their "
      "guidance and support during this work.")
    B.append(("refs", None))
    return B


# --------------------------------------------------------------------------
# ODT construction
# --------------------------------------------------------------------------

AUTO_STYLES = """
<office:automatic-styles xmlns:office="{office}" xmlns:style="{style}" xmlns:fo="{fo}" xmlns:table="{table}" xmlns:text="{text}">
  <style:style style:name="EMTbl" style:family="table">
    <style:table-properties style:width="6.5in" style:rel-width="100%" table:align="margins" fo:margin-top="0.04in" fo:margin-bottom="0.08in"/>
  </style:style>
  <style:style style:name="EMCell" style:family="table-cell">
    <style:table-cell-properties fo:padding="0.03in" fo:border="0.5pt solid #000000"/>
  </style:style>
  <style:style style:name="EMHead" style:family="table-cell">
    <style:table-cell-properties fo:padding="0.03in" fo:border="0.5pt solid #000000" fo:background-color="#e7e6e6"/>
  </style:style>
  <style:style style:name="EMTblP" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:margin-top="0in" fo:margin-bottom="0in" fo:text-align="start" fo:text-indent="0in"/>
    <style:text-properties fo:font-size="8pt"/>
  </style:style>
  <style:style style:name="EMTblPH" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:margin-top="0in" fo:margin-bottom="0in" fo:text-align="center" fo:text-indent="0in"/>
    <style:text-properties fo:font-size="8pt" fo:font-weight="bold"/>
  </style:style>
  <style:style style:name="EMCode" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:margin-top="0in" fo:margin-bottom="0in" fo:margin-left="0.2in" fo:text-indent="0in"/>
    <style:text-properties style:font-name="Liberation Mono" fo:font-family="'Liberation Mono'" fo:font-size="8pt"/>
  </style:style>
  <style:style style:name="EMFigP" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:text-align="center" fo:margin-top="0.08in" fo:margin-bottom="0.02in" fo:text-indent="0in"/>
  </style:style>
  <style:style style:name="EMEq" style:family="paragraph" style:parent-style-name="Text">
    <style:paragraph-properties fo:text-align="center" fo:text-indent="0in" fo:margin-top="0.04in" fo:margin-bottom="0.04in"/>
    <style:text-properties fo:font-style="italic"/>
  </style:style>
  <style:style style:name="EMAuthor" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:text-align="center" fo:margin-top="0.06in" fo:margin-bottom="0in" fo:text-indent="0in"/>
    <style:text-properties fo:font-size="11pt"/>
  </style:style>
  <style:style style:name="EMAff" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:text-align="center" fo:margin-top="0in" fo:margin-bottom="0in" fo:text-indent="0in"/>
    <style:text-properties fo:font-size="9pt" fo:font-style="italic"/>
  </style:style>
  <style:style style:name="EMAffPlain" style:family="paragraph" style:parent-style-name="Standard">
    <style:paragraph-properties fo:text-align="center" fo:margin-top="0in" fo:margin-bottom="0in" fo:text-indent="0in"/>
    <style:text-properties fo:font-size="9pt"/>
  </style:style>
  <style:style style:name="EMAuthTbl" style:family="table">
    <style:table-properties style:width="6.3in" style:rel-width="100%" table:align="margins" fo:margin-top="0.05in"/>
  </style:style>
  <style:style style:name="EMAuthCol" style:family="table-column">
    <style:table-column-properties style:rel-column-width="21845*"/>
  </style:style>
  <style:style style:name="EMAuthCell" style:family="table-cell">
    <style:table-cell-properties fo:padding="0.02in" fo:border="none" style:vertical-align="top"/>
  </style:style>
  <style:style style:name="EMFr" style:family="graphic">
    <style:graphic-properties style:wrap="none" style:vertical-pos="top" style:vertical-rel="baseline"
        style:horizontal-pos="center" style:horizontal-rel="paragraph" fo:border="none" fo:padding="0in"/>
  </style:style>
</office:automatic-styles>
""".format(**NS)


def build():
    with zipfile.ZipFile(TEMPLATE) as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    content = etree.fromstring(files["content.xml"])
    manifest = etree.fromstring(files["META-INF/manifest.xml"])

    # Add our automatic styles.
    auto = content.find("office:automatic-styles", NS)
    for style in etree.fromstring(AUTO_STYLES):
        auto.append(style)

    sections = content.findall(".//office:text/text:section", NS)
    paper = sections[2]

    # Approval sheet.
    approval = paper[22]
    spans = approval.findall("text:span", NS)
    spans[0].text = ("This project report entitled “EdgeMason” by “Abhay Yadav and Mayank Dewangan” "
                     "is approved for V")
    spans[2].text = " Semester Minor Project/UG Research Work-V. "
    for child in list(spans[2]):
        spans[2].remove(child)

    # Title/author frames and abstract paragraph. Frame2 is laid out first, so it
    # takes the title and Frame1 (below it) takes the author block.
    abstract_p = paper[47]
    title_p = abstract_p.find(".//draw:frame[@draw:name='Frame2']//text:p", NS)
    for child in list(title_p):
        title_p.remove(child)
    title_p.set(q("text:style-name"), "P26")
    title_p.text = "EdgeMason"
    author_box = abstract_p.find(".//draw:frame[@draw:name='Frame1']/draw:text-box", NS)
    for child in list(author_box):
        author_box.remove(child)
    # IEEE conference layout: one borderless column per author.
    authors_tbl = etree.SubElement(author_box, q("table:table"))
    authors_tbl.set(q("table:name"), "EMAuthors")
    authors_tbl.set(q("table:style-name"), "EMAuthTbl")
    for _ in AUTHORS:
        col = etree.SubElement(authors_tbl, q("table:table-column"))
        col.set(q("table:style-name"), "EMAuthCol")
    row = etree.SubElement(authors_tbl, q("table:table-row"))
    for name, italic_lines, plain_lines in AUTHORS:
        cell = etree.SubElement(row, q("table:table-cell"))
        cell.set(q("table:style-name"), "EMAuthCell")
        cell.set(q("office:value-type"), "string")
        for style, lines in (("EMAuthor", [name]), ("EMAff", italic_lines), ("EMAffPlain", plain_lines)):
            for line in lines:
                cp = etree.SubElement(cell, q("text:p"))
                cp.set(q("text:style-name"), style)
                cp.text = line
    frames = abstract_p.findall("draw:frame", NS)
    for child in list(abstract_p):
        if child not in frames:
            abstract_p.remove(child)
    abstract_p.text = None
    frames[-1].tail = None
    label = etree.SubElement(abstract_p, q("text:span"))
    label.set(q("text:style-name"), "T24")
    label.text = "Abstract"
    label.tail = "—" + ABSTRACT

    impact_p = paper[49]
    first = impact_p.find("text:span", NS)
    for child in list(impact_p):
        if child is not first:
            impact_p.remove(child)
    first.set(q("text:style-name"), "T24")  # same style as "Abstract"; the template's T26 is highlighted
    first.tail = "— " + IMPACT

    index_p = paper[52]
    terms_span = index_p.find("text:span", NS)
    for child in list(index_p):
        if child is not terms_span and etree.QName(child).localname != "bookmark-start":
            index_p.remove(child)
    terms_span.tail = "—" + INDEX_TERMS

    # Drop the template's instruction text and append the report body.
    for child in list(paper)[53:]:
        paper.remove(child)

    table_no = [0]
    fig_no = [0]
    pictures = {}

    def para(style, text=None):
        el = etree.SubElement(paper, q("text:p"))
        el.set(q("text:style-name"), style)
        if text is not None:
            el.text = text
        return el

    def roman(n):
        vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        out = ""
        for v, s in vals:
            while n >= v:
                out += s
                n -= v
        return out

    for block in body_blocks():
        kind = block[0]
        if kind in ("h1", "h2"):
            el = etree.SubElement(paper, q("text:h"))
            el.set(q("text:style-name"), "Heading_20_1" if kind == "h1" else "Heading_20_2")
            el.set(q("text:outline-level"), "1" if kind == "h1" else "2")
            el.text = block[1]
        elif kind == "p":
            para("Text", block[1])
        elif kind == "eq":
            para("EMEq", block[1])
        elif kind == "code":
            for line in block[1]:
                code_p = para("EMCode")
                # Preserve leading/inner spaces.
                parts = re.split(r"( {2,})", line)
                last = None
                for part in parts:
                    if re.fullmatch(r" {2,}", part):
                        sp = etree.SubElement(code_p, q("text:s"))
                        sp.set(q("text:c"), str(len(part)))
                        last = sp
                    elif part:
                        if last is None:
                            code_p.text = (code_p.text or "") + part
                        else:
                            last.tail = (last.tail or "") + part
        elif kind == "list":
            lst = etree.SubElement(paper, q("text:list"))
            lst.set(q("text:style-name"), "WWNum41")
            for item in block[1]:
                li = etree.SubElement(lst, q("text:list-item"))
                ip = etree.SubElement(li, q("text:p"))
                ip.set(q("text:style-name"), "P29")
                ip.text = item
        elif kind == "table":
            _, caption, header, rows, widths = block
            table_no[0] += 1
            para("Table_20_Title", f"TABLE {roman(table_no[0])}")
            para("Table_20_Title", caption)
            name = f"EMTable{table_no[0]}"
            widths = widths or [1] * len(header)
            total = sum(widths)
            tbl = etree.SubElement(paper, q("table:table"))
            tbl.set(q("table:name"), name)
            tbl.set(q("table:style-name"), "EMTbl")
            for index, w in enumerate(widths):
                col_style = etree.SubElement(auto, q("style:style"))
                col_style.set(q("style:name"), f"{name}C{index}")
                col_style.set(q("style:family"), "table-column")
                props = etree.SubElement(col_style, q("style:table-column-properties"))
                props.set(q("style:column-width"), f"{6.5 * w / total:.3f}in")
                props.set(q("style:rel-column-width"), f"{int(65535 * w / total)}*")
                col = etree.SubElement(tbl, q("table:table-column"))
                col.set(q("table:style-name"), f"{name}C{index}")
            head_rows = etree.SubElement(tbl, q("table:table-header-rows"))
            for container, cells, cell_style, p_style in [(head_rows, [header], "EMHead", "EMTblPH")] + \
                    [(tbl, [r], "EMCell", "EMTblP") for r in rows]:
                for cells_row in cells:
                    tr = etree.SubElement(container, q("table:table-row"))
                    for value in cells_row:
                        tc = etree.SubElement(tr, q("table:table-cell"))
                        tc.set(q("table:style-name"), cell_style)
                        tc.set(q("office:value-type"), "string")
                        cp = etree.SubElement(tc, q("text:p"))
                        cp.set(q("text:style-name"), p_style)
                        cp.text = str(value)
        elif kind == "fig":
            _, path, caption, width = block
            fig_no[0] += 1
            data = Path(path).read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
            ext = Path(path).suffix.lower()
            pic_name = f"Pictures/edgemason_{fig_no[0]:02d}{ext}"
            pictures[pic_name] = (data, "image/png" if ext == ".png" else "image/jpeg")
            height = width * h / w
            fp = para("EMFigP")
            frame = etree.SubElement(fp, q("draw:frame"))
            frame.set(q("draw:style-name"), "EMFr")
            frame.set(q("draw:name"), f"EMFigure{fig_no[0]}")
            frame.set(q("text:anchor-type"), "as-char")
            frame.set(q("svg:width"), f"{width:.3f}in")
            frame.set(q("svg:height"), f"{height:.3f}in")
            image = etree.SubElement(frame, q("draw:image"))
            image.set(q("xlink:href"), pic_name)
            image.set(q("xlink:type"), "simple")
            image.set(q("xlink:show"), "embed")
            image.set(q("xlink:actuate"), "onLoad")
            para("Figure_20_Caption", f"Fig. {fig_no[0]}. {caption}")
        elif kind == "refs":
            h = etree.SubElement(paper, q("text:h"))
            h.set(q("text:style-name"), "Heading_20_1")
            h.set(q("text:outline-level"), "1")
            h.text = "References"
            lst = etree.SubElement(paper, q("text:list"))
            lst.set(q("text:style-name"), "WWNum12")
            for key in CITE_ORDER:
                li = etree.SubElement(lst, q("text:list-item"))
                rp = etree.SubElement(li, q("text:p"))
                rp.set(q("text:style-name"), "P44")
                rp.text = REFERENCES[key]

    unused = sorted(set(REFERENCES) - set(CITE_ORDER))
    if unused:
        raise SystemExit(f"references never cited: {unused}")

    # Register pictures in the manifest.
    for pic_name, (_, media) in pictures.items():
        entry = etree.SubElement(manifest, q("manifest:file-entry"))
        entry.set(q("manifest:full-path"), pic_name)
        entry.set(q("manifest:media-type"), media)

    files["content.xml"] = etree.tostring(content, xml_declaration=True, encoding="UTF-8")
    files["META-INF/manifest.xml"] = etree.tostring(manifest, xml_declaration=True, encoding="UTF-8")
    for pic_name, (data, _) in pictures.items():
        files[pic_name] = data

    with zipfile.ZipFile(OUTPUT, "w") as zout:
        zout.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name, data in files.items():
            zout.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    print("wrote", OUTPUT, f"({table_no[0]} tables, {fig_no[0]} figures, {len(CITE_ORDER)} references)")


if __name__ == "__main__":
    build()
