"""P2.6: pilot sweep runner (configs -> measured + additive prediction).

Builds many mixed-precision models that span the seam axis (how many layers stay
fp32, and whether they are clustered or spread), measures each end-to-end with
the existing benchmark protocol, and records one row per config in a JSON Lines
findings register.

Register schema (one JSON object per line, in config order):

    config_id          str    deterministic id, e.g. "cfg_003"
    fp32_nodes         list   pre-processed node names kept in fp32
    seam_count         int    Q/DQ boundary nodes added over the all-int8 model
    seam_bytes         int    activation elements crossing those seams (P2.7)
    stranded_fp32_bytes int   output elements of fp32 layers whose region borders int8
    measured_p50_us    float  end-to-end p50 latency in microseconds
    measured_p90_us    float  end-to-end p90 latency in microseconds
    additive_pred_us   float  sum of per-layer fp32/int8 lookup costs
    status             str    "ok" or "failed"
    error              str    failure detail, or null

Every generated config produces a row. A config whose build, fetch or run fails
is written as `status: "failed"` with the error, and the sweep continues (B-59).

Seam count is measured from the built model, relative to the pure-int8 model:
ONNX Runtime always emits a model-input QuantizeLinear and model-output
DequantizeLinear node, so the raw count is non-zero even for all-int8. Counting
only boundary nodes the split adds makes the pure-int8 config report zero seams,
which is what the spec requires. The count is read from the *optimized* graph
(the one the benchmark executes) because optimization folds, duplicates and
renames Q/DQ nodes, so the pre-optimization node list is not what runs.

Amendment A: the lookup -- the only profile the additive model uses -- runs in
the benchmark's optimization regime (ORT_ENABLE_ALL, ONNX Runtime's default),
never ORT_DISABLE_ALL. Both pure models are also measured with the benchmark
protocol, and the additive totals must match those measurements within
DEFAULT_SANITY_TOLERANCE. If they do not, the lookup is rescaled per precision
(regime ``calibrated``) so both pure configs match exactly; a lookup that still
fails the gate stops the sweep. Per-config per-layer timings no longer exist, so
the only timing in the additive model is the gated lookup.

P2.7 raises the default target to 150 and samples placements so seam count and
high-resolution fp32 placement vary independently: low-resolution-only fp32
subsets produce many seams with little stranded fp32, while high-resolution-only
subsets (``conv0``) produce the opposite. Each row also records the two
geometry features the cost model fits.
"""

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from api.optimizer import cost_model
from api.optimizer.benchmark_engine import benchmark
from api.optimizer.fetcher import BoundaryEntry, ComputeEntry, ProfileTable, fetch_profile
from api.optimizer.quantizer import CalibrationData, Quantizer

# ONNX Runtime's quantizer names QLinear nodes by appending this suffix.
_QUANT_SUFFIX = "_quant"
_QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})

DEFAULT_TARGET_CONFIGS = 150

# BenchmarkEngine leaves SessionOptions at ONNX Runtime's default, which is
# ORT_ENABLE_ALL. Profiling the lookup with ORT_DISABLE_ALL puts the per-layer
# sums and the measured latency in two different worlds: the unoptimized int8
# MaxPool alone is ~18x slower than its optimized kernel. Always use the
# benchmark's regime (Amendment A).
BENCHMARK_OPTIMIZATION_LEVEL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Allowed relative gap between an additive pure-model sum and its measurement.
DEFAULT_SANITY_TOLERANCE = 0.15


@dataclass(frozen=True)
class LayerTiming:
    """Per-layer cost from the pure fp32 and pure int8 models."""

    fp32_us: float
    int8_us: float


@dataclass(frozen=True)
class LookupTable:
    """Per-layer timings plus the pure-int8 boundary table used as seam baseline.

    ``regime`` is ``"optimized"`` when the additive sums already matched the
    measurements, or ``"calibrated"`` when they had to be rescaled.
    ``raw_sanity`` keeps the pre-calibration comparison so a finding can say
    which path was used and why.
    """

    layers: dict[str, LayerTiming]
    int8_boundary: Counter[tuple[str, str]]
    regime: str = "optimized"
    raw_sanity: tuple[dict, ...] = ()
    # Raw (unoptimized) boundary table, used to cancel the model input/output
    # conversions when extracting P2.7 seam features.
    int8_raw_boundary: dict[str, BoundaryEntry] = field(default_factory=dict)


class LookupSanityError(RuntimeError):
    """The additive lookup does not reproduce the measured pure-model latency."""


@dataclass(frozen=True)
class SweepConfig:
    """One point on the sweep: the node names to keep out of quantization."""

    config_id: str
    fp32_nodes: tuple[str, ...]


def generate_configs(
    layers: Sequence[str],
    seed: int = 0,
    target: int = DEFAULT_TARGET_CONFIGS,
) -> list[SweepConfig]:
    """Generate reproducible configs spanning counts and seam placements.

    `layers` must be in graph order. Counts range from 0 to all layers;
    clustered configs take a seeded contiguous run, spread configs take layers
    at even intervals so the two placements produce different seam counts.
    """
    ordered = list(layers)
    total = len(ordered)
    if total == 0:
        return [SweepConfig("cfg_000", ())]

    count_points = max(min(target // 2, total + 1), 1)
    if count_points == 1:
        counts = [0]
    else:
        counts = sorted({round(i * total / (count_points - 1)) for i in range(count_points)})

    rng = random.Random(seed)
    configs: list[SweepConfig] = []
    seen: set[frozenset[str]] = set()

    def add(nodes: Sequence[str]) -> None:
        key = frozenset(nodes)
        if key in seen:
            return
        seen.add(key)
        configs.append(SweepConfig(f"cfg_{len(configs):03d}", tuple(nodes)))

    for count in counts:
        if count == 0:
            add([])
            continue
        start = rng.randrange(0, total - count + 1)
        add(ordered[start : start + count])
        if 1 < count < total:
            indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
            add([ordered[index] for index in indices])

    return configs


def _spread(layers: Sequence[str], count: int) -> list[str]:
    """Take `count` layers at even intervals, maximising fp32 islands."""
    if count <= 0:
        return []
    if count >= len(layers):
        return list(layers)
    if count == 1:
        return [layers[0]]
    step = (len(layers) - 1) / (count - 1)
    return [layers[round(index * step)] for index in range(count)]


def generate_decorrelated_configs(
    layers: Sequence[str],
    layer_sizes: Mapping[str, int],
    seed: int = 0,
    target: int = DEFAULT_TARGET_CONFIGS,
) -> list[SweepConfig]:
    """Generate configs that separate seam count from high-resolution placement.

    P2.7 needs alpha and beta to be separately identifiable. Keeping only
    low-resolution layers fp32 gives many seams with little stranded fp32;
    keeping only the high-resolution head (``conv0``) gives the opposite. Both
    families are built explicitly, then seeded random subsets fill the target.

    `layers` must be in graph order and `layer_sizes` maps each name to its
    output activation elements (from `cost_model.build_geometry`).
    """
    ordered = list(layers)
    total = len(ordered)
    if total == 0:
        return [SweepConfig("cfg_000", ())]
    target = max(target, 1)

    sizes = {name: int(layer_sizes.get(name, 0)) for name in ordered}
    ranked = sorted(
        ordered, key=lambda name: (-sizes[name], ordered.index(name))
    )
    high = ranked[: max(1, total // 4)]
    high_set = set(high)
    low = [name for name in ordered if name not in high_set]

    rng = random.Random(seed)
    configs: list[SweepConfig] = []
    seen: set[frozenset[str]] = set()

    def add(nodes: Sequence[str]) -> bool:
        if len(configs) >= target:
            return False
        key = frozenset(nodes)
        if key not in seen:
            seen.add(key)
            configs.append(SweepConfig(f"cfg_{len(configs):03d}", tuple(nodes)))
        return len(configs) < target

    add(())
    add(ordered)

    # Few seams, high-resolution placement: the conv0 effect.
    for count in range(1, min(len(high), 4) + 1):
        add(high[:count])
        add(_spread(high, count))

    # Many seams, low-resolution placement: no stranded high-resolution layer.
    for fraction in (0.25, 0.5, 0.75, 1.0):
        count = max(1, min(len(low), round(len(low) * fraction)))
        add(_spread(low, count))
        start = rng.randrange(0, len(low) - count + 1)
        add(low[start : start + count])

    # Explicit grid: hold high-resolution placement fixed while low-resolution
    # islands grow, so the two features are not collinear.
    for high_count in range(0, min(len(high), 3) + 1):
        for fraction in (0.25, 0.5, 0.75):
            count = round(len(low) * fraction)
            if count == 0:
                add(high[:high_count])
                continue
            add(high[:high_count] + _spread(low, count))
            start = rng.randrange(0, len(low) - count + 1)
            add(high[:high_count] + low[start : start + count])

    # Seeded fill: half low-resolution-only, half anywhere.
    attempts = 0
    while len(configs) < target and attempts < target * 20:
        attempts += 1
        pool = low if rng.random() < 0.5 else ordered
        count = rng.randrange(0, len(pool) + 1)
        picked = set(rng.sample(pool, count))
        add([name for name in ordered if name in picked])

    return configs


def lookup_sums(layers: Mapping[str, LayerTiming]) -> tuple[float, float]:
    """Return the pure-fp32 and pure-int8 additive totals for a lookup."""
    return (
        float(sum(timing.fp32_us for timing in layers.values())),
        float(sum(timing.int8_us for timing in layers.values())),
    )


def _sanity_report(
    layers: Mapping[str, LayerTiming],
    *,
    fp32_measured_us: float,
    int8_measured_us: float,
) -> list[dict]:
    """Compare each pure config's additive sum with its measured latency."""
    if not layers:
        raise LookupSanityError("lookup table is empty, nothing to compare")
    fp32_predicted, int8_predicted = lookup_sums(layers)
    report: list[dict] = []
    for precision, predicted, measured in (
        ("fp32", fp32_predicted, fp32_measured_us),
        ("int8", int8_predicted, int8_measured_us),
    ):
        if measured <= 0:
            raise LookupSanityError(
                f"pure {precision} measured latency must be positive, got {measured}"
            )
        report.append(
            {
                "precision": precision,
                "additive_us": predicted,
                "measured_us": float(measured),
                "rel_error": abs(predicted - measured) / measured,
            }
        )
    return report


def check_lookup_sanity(
    layers: Mapping[str, LayerTiming],
    *,
    fp32_measured_us: float,
    int8_measured_us: float,
    tolerance: float = DEFAULT_SANITY_TOLERANCE,
) -> list[dict]:
    """Raise unless both pure configs' additive totals match their measurements.

    This is the check that would have caught the ORT_DISABLE_ALL regime bug: the
    additive baseline must live in the same execution regime as the benchmark.
    """
    report = _sanity_report(
        layers,
        fp32_measured_us=fp32_measured_us,
        int8_measured_us=int8_measured_us,
    )
    failures = [
        f"pure {item['precision']}: additive {item['additive_us']:.1f} us vs "
        f"measured {item['measured_us']:.1f} us "
        f"({item['rel_error']:.1%} > {tolerance:.0%})"
        for item in report
        if item["rel_error"] > tolerance
    ]
    if failures:
        raise LookupSanityError(
            "additive lookup does not match measured latency: " + "; ".join(failures)
        )
    return report


def calibrate_lookup(
    layers: Mapping[str, LayerTiming],
    *,
    fp32_measured_us: float,
    int8_measured_us: float,
) -> dict[str, LayerTiming]:
    """Rescale each precision's per-layer costs so both pure configs match.

    One scale per precision is the smallest correction that makes both ends of
    the sweep exact; a single shared scale cannot match both unless the two
    measured/predicted ratios already agree.
    """
    fp32_predicted, int8_predicted = lookup_sums(layers)
    if fp32_predicted <= 0 or int8_predicted <= 0:
        raise LookupSanityError(
            "cannot calibrate a lookup with non-positive pure sums: "
            f"fp32={fp32_predicted}, int8={int8_predicted}"
        )
    fp32_scale = fp32_measured_us / fp32_predicted
    int8_scale = int8_measured_us / int8_predicted
    return {
        name: LayerTiming(
            fp32_us=timing.fp32_us * fp32_scale,
            int8_us=timing.int8_us * int8_scale,
        )
        for name, timing in layers.items()
    }


def build_lookup_table(
    model_path: str,
    calibration_data: CalibrationData,
    *,
    quantizer: Quantizer | None = None,
    work_dir: str | Path = "temp_models/sweep/lookup",
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    time_budget_seconds: float | None = None,
    isolated: bool = True,
    tolerance: float = DEFAULT_SANITY_TOLERANCE,
) -> LookupTable:
    """Measure the pure fp32 and pure int8 models once for the additive model.

    Both are profiled in the benchmark's optimization regime, then measured with
    the same benchmark protocol. If the additive totals miss those measurements
    by more than `tolerance`, the lookup is rescaled per precision and the
    regime is reported as ``calibrated``; if the gate then still fails, the
    build stops.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    quantizer = quantizer or Quantizer(str(work / "models"))

    fp32_path = quantizer.create_fp32(model_path)
    int8_path = quantizer.create_static_int8(model_path, calibration_data)
    fp32_table = _fetch_tables(fp32_path, work, "lookup_fp32")
    int8_table = _fetch_tables(int8_path, work, "lookup_int8")

    if not fp32_table["compute"] or not int8_table["compute"]:
        raise RuntimeError("lookup build produced an empty compute table")

    int8_by_name: dict[str, ComputeEntry] = {}
    for name, entry in int8_table["compute"].items():
        canonical = name[: -len(_QUANT_SUFFIX)] if name.endswith(_QUANT_SUFFIX) else name
        if canonical in int8_by_name:
            raise RuntimeError(
                f"lookup build found two int8 layers for {canonical!r}; "
                "refusing to build an ambiguous table"
            )
        int8_by_name[canonical] = entry

    # Optimization fuses and removes nodes, so every layer that survives in one
    # model must survive in both. A partial match means an attribution we cannot
    # make honestly, so stop instead of silently dropping the layer (Amendment A).
    fp32_only = sorted(set(fp32_table["compute"]) - set(int8_by_name))
    int8_only = sorted(set(int8_by_name) - set(fp32_table["compute"]))
    if fp32_only or int8_only:
        raise RuntimeError(
            "lookup build could not map every layer between fp32 and int8 under "
            f"optimization (fp32-only={fp32_only[:5]}, int8-only={int8_only[:5]})"
        )

    layers: dict[str, LayerTiming] = {
        name: LayerTiming(
            fp32_us=float(entry["avg_us"]),
            int8_us=float(int8_by_name[name]["avg_us"]),
        )
        for name, entry in fp32_table["compute"].items()
    }
    if not layers:
        raise RuntimeError("lookup build matched no layers between fp32 and int8")

    fp32_measured_us = _measure_pure_model(
        fp32_path,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        time_budget_seconds=time_budget_seconds,
        isolated=isolated,
    )
    int8_measured_us = _measure_pure_model(
        int8_path,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        time_budget_seconds=time_budget_seconds,
        isolated=isolated,
    )
    raw_sanity = _sanity_report(
        layers,
        fp32_measured_us=fp32_measured_us,
        int8_measured_us=int8_measured_us,
    )

    regime = "optimized"
    if any(item["rel_error"] > tolerance for item in raw_sanity):
        layers = calibrate_lookup(
            layers,
            fp32_measured_us=fp32_measured_us,
            int8_measured_us=int8_measured_us,
        )
        # Calibrated costs must reproduce both pure measurements; a failure here
        # means the lookup is unusable, so let the gate stop the sweep.
        check_lookup_sanity(
            layers,
            fp32_measured_us=fp32_measured_us,
            int8_measured_us=int8_measured_us,
            tolerance=tolerance,
        )
        regime = "calibrated"

    return LookupTable(
        layers=layers,
        int8_boundary=_optimized_boundary(int8_path, work, "lookup_int8"),
        regime=regime,
        raw_sanity=tuple(raw_sanity),
        int8_raw_boundary=cost_model.fetch_raw_boundary(
            int8_path, work / "features", "lookup_int8"
        ),
    )


def compute_additive_prediction(
    layers: Mapping[str, LayerTiming],
    fp32_nodes: Sequence[str],
) -> float:
    """Sum the per-layer lookup costs for the layers in `fp32_nodes` and the rest int8.

    A config layer missing from the lookup stops the sweep: the prediction would
    be wrong, not merely noisy.
    """
    fp32 = set(fp32_nodes)
    for name in fp32:
        if name not in layers:
            raise KeyError(
                f"layer {name!r} is missing from the lookup table; "
                "the additive prediction would be wrong"
            )
    return float(
        sum(
            timing.fp32_us if name in fp32 else timing.int8_us
            for name, timing in layers.items()
        )
    )


def boundary_signatures(model: onnx.ModelProto) -> Counter[tuple[str, str]]:
    """Count Q/DQ boundary nodes in a graph by (op type, converted tensor)."""
    return Counter(
        (node.op_type, node.output[0])
        for node in model.graph.node
        if node.op_type in _QDQ_OPS
    )


def seam_count(
    mixed_boundary: Mapping[tuple[str, str], int],
    baseline_boundary: Mapping[tuple[str, str], int],
) -> int:
    """Count the Q/DQ boundary nodes this config adds over the all-int8 model.

    Both sides come from the *optimized* graphs, so the count is what the
    benchmark actually executes after ORT folds and rewrites Q/DQ nodes.
    """
    return sum(
        max(0, count - baseline_boundary[signature])
        for signature, count in mixed_boundary.items()
    )


def run_config(
    config: SweepConfig,
    *,
    model_path: str,
    calibration_data: CalibrationData,
    lookup: LookupTable,
    quantizer: Quantizer,
    work_dir: str | Path,
    geometry: cost_model.GraphGeometry,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    time_budget_seconds: float | None = None,
    isolated: bool = True,
) -> dict:
    """Run one config through build -> fetch -> benchmark, returning its register row."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    row: dict = {
        "config_id": config.config_id,
        "fp32_nodes": list(config.fp32_nodes),
        "seam_count": None,
        "seam_bytes": None,
        "stranded_fp32_bytes": None,
        "measured_p50_us": None,
        "measured_p90_us": None,
        # Deliberately outside the try: a lookup mismatch must stop the sweep.
        "additive_pred_us": compute_additive_prediction(lookup.layers, config.fp32_nodes),
        "status": "ok",
        "error": None,
    }

    mixed_path: str | None = None
    try:
        mixed_path = quantizer.create_mixed_int8(
            model_path,
            calibration_data,
            list(config.fp32_nodes),
        )
        mixed_boundary = _optimized_boundary(
            mixed_path, work, f"mixed_{config.config_id}"
        )
        row["seam_count"] = seam_count(mixed_boundary, lookup.int8_boundary)

        result = _benchmark_model(
            mixed_path,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            time_budget_seconds=time_budget_seconds,
            isolated=isolated,
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "benchmark reported failure")
        latency = result["latency_ms"]
        row["measured_p50_us"] = float(latency["p50"]) * 1000.0
        row["measured_p90_us"] = float(latency["p90"]) * 1000.0
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"

    # Feature extraction is outside the try on purpose: a missing activation
    # shape is an integrity failure (P2.7 spec #6), not a failed config.
    if mixed_path is not None:
        raw_boundary = cost_model.fetch_raw_boundary(
            mixed_path, work / "features", f"mixed_{config.config_id}"
        )
        features = cost_model.extract_features(
            geometry,
            list(config.fp32_nodes),
            raw_boundary,
            lookup.int8_raw_boundary,
            mixed_path,
        )
        row["seam_bytes"] = features.seam_bytes
        row["stranded_fp32_bytes"] = features.stranded_fp32_bytes

    return row


def run_sweep(
    model_path: str,
    calibration_data: CalibrationData,
    *,
    register_path: str | Path,
    seed: int = 0,
    target: int = DEFAULT_TARGET_CONFIGS,
    work_dir: str | Path = "temp_models/sweep",
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    time_budget_seconds: float | None = None,
    isolated: bool = True,
    quantizer: Quantizer | None = None,
) -> list[dict]:
    """Build the lookup, run every generated config and write the findings register."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    quantizer = quantizer or Quantizer(str(work / "models"))

    lookup = build_lookup_table(
        model_path,
        calibration_data,
        quantizer=quantizer,
        work_dir=work / "lookup",
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        time_budget_seconds=time_budget_seconds,
        isolated=isolated,
    )
    # The additive model needs both an fp32 and an int8 cost, so only layers
    # present in the lookup are valid config entries.
    vocabulary = [
        name for name in quantizer.excludable_node_names(model_path) if name in lookup.layers
    ]
    if not vocabulary:
        raise RuntimeError(
            "no excludable layer has a lookup entry; cannot build mixed configs"
        )

    geometry = cost_model.build_geometry(model_path)
    configs = generate_decorrelated_configs(
        vocabulary,
        cost_model.layer_output_elements(geometry, vocabulary),
        seed=seed,
        target=target,
    )
    register = Path(register_path)
    register.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with register.open("w", encoding="utf-8") as handle:
        for config in configs:
            row = run_config(
                config,
                model_path=model_path,
                calibration_data=calibration_data,
                lookup=lookup,
                quantizer=quantizer,
                work_dir=work,
                geometry=geometry,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                time_budget_seconds=time_budget_seconds,
                isolated=isolated,
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
    return rows


def _random_feeds(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    """Build one deterministic random batch for every model input."""
    feeds: dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        shape = [
            dim if isinstance(dim, int) and dim > 0 else 1 for dim in model_input.shape
        ]
        feeds[model_input.name] = np.random.default_rng(0).random(shape, dtype=np.float32)
    return feeds


def _dump_optimized_model(model_path: str, work_dir: Path, tag: str) -> Path:
    """Serialize the graph ORT runs for this model under the benchmark regime."""
    work_dir.mkdir(parents=True, exist_ok=True)
    optimized = work_dir / f"{tag}.optimized.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = BENCHMARK_OPTIMIZATION_LEVEL
    options.optimized_model_filepath = str(optimized)
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session.run(None, _random_feeds(session))
    return optimized


def _optimized_boundary(
    model_path: str, work_dir: Path, tag: str
) -> Counter[tuple[str, str]]:
    """Count Q/DQ boundary nodes in the graph the benchmark actually executes.

    Profiling a mixed model and mapping its Q/DQ events back to the original
    graph is not reliable: optimization fuses and renames them (for example
    ``squeezenet0_relu0_fwd_q``, and duplicated DequantizeLinear nodes), and
    inserts layout bookkeeping such as ``ReorderInput``. The seam count is a
    structural property, so it is read from the optimized graph instead.
    """
    optimized = _dump_optimized_model(model_path, work_dir, tag)
    return boundary_signatures(onnx.load(str(optimized)))


def _fetch_tables(model_path: str, work_dir: Path, tag: str) -> ProfileTable:
    """Profile one model and return the P2.3 fetcher's compute and boundary tables."""
    work_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Same regime as BenchmarkEngine, so per-layer costs and measured latency
    # are comparable. Fused/removed nodes are handled by the strict name check
    # in build_lookup_table (Amendment A).
    options.graph_optimization_level = BENCHMARK_OPTIMIZATION_LEVEL
    options.enable_profiling = True
    options.profile_file_prefix = str(work_dir / f"{tag}.profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    feeds = _random_feeds(session)
    for _ in range(5):
        session.run(None, feeds)
    trace = Path(session.end_profiling())

    measured = work_dir / f"{tag}.measured.json"
    measured.write_text(json.dumps({"warmup_runs": 1}), encoding="utf-8")
    return fetch_profile(str(trace), str(model_path), str(measured))


def _measure_pure_model(
    model_path: str,
    *,
    warmup_runs: int,
    benchmark_runs: int,
    time_budget_seconds: float | None,
    isolated: bool,
) -> float:
    """Benchmark a pure model with the shared protocol and return p50 in microseconds."""
    result = _benchmark_model(
        model_path,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        time_budget_seconds=time_budget_seconds,
        isolated=isolated,
    )
    if not result.get("success"):
        raise RuntimeError(
            result.get("error") or f"benchmark of pure model {model_path} failed"
        )
    return float(result["latency_ms"]["p50"]) * 1000.0


def _benchmark_model(
    model_path: str,
    *,
    warmup_runs: int,
    benchmark_runs: int,
    time_budget_seconds: float | None,
    isolated: bool,
) -> dict:
    """Call the shared benchmark protocol and return its JSON result as a dict."""
    raw = benchmark(
        model_path,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        time_budget_seconds=time_budget_seconds,
        isolated=isolated,
    )
    return json.loads(raw)
