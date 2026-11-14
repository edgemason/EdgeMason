"""P2.6: pilot sweep runner (configs -> measured + additive prediction).

Builds many mixed-precision models that span the seam axis (how many layers stay
fp32, and whether they are clustered or spread), measures each end-to-end with
the existing benchmark protocol, and records one row per config in a JSON Lines
findings register.

Register schema (one JSON object per line, in config order):

    config_id          str    deterministic id, e.g. "cfg_003"
    fp32_nodes         list   pre-processed node names kept in fp32
    seam_count         int    Q/DQ boundary nodes added over the all-int8 model
    measured_p50_us    float  end-to-end p50 latency in microseconds
    measured_p90_us    float  end-to-end p90 latency in microseconds
    additive_pred_us   float  sum of per-layer fp32/int8 lookup costs
    status             str    "ok" or "failed"
    error              str    failure detail, or null

Every generated config produces a row. A config whose build, fetch or run fails
is written as `status: "failed"` with the error, and the sweep continues (B-59).

Seam count is measured from the fetcher's boundary table, but relative to the
pure-int8 boundary set: ONNX Runtime always emits a model-input QuantizeLinear
and model-output DequantizeLinear nodes, so the raw count is non-zero even for
all-int8. Counting only boundary nodes the split adds makes the pure-int8 config
report zero seams, which is what the spec requires.
"""

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from api.optimizer.benchmark_engine import benchmark
from api.optimizer.fetcher import ComputeEntry, ProfileTable, fetch_profile
from api.optimizer.quantizer import CalibrationData, Quantizer

# ONNX Runtime's quantizer names QLinear nodes by appending this suffix.
_QUANT_SUFFIX = "_quant"

DEFAULT_TARGET_CONFIGS = 40


@dataclass(frozen=True)
class LayerTiming:
    """Per-layer cost from the pure fp32 and pure int8 models."""

    fp32_us: float
    int8_us: float


@dataclass(frozen=True)
class LookupTable:
    """Per-layer timings plus the pure-int8 boundary table used as seam baseline."""

    layers: dict[str, LayerTiming]
    int8_boundary: dict


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


def build_lookup_table(
    model_path: str,
    calibration_data: CalibrationData,
    *,
    quantizer: Quantizer | None = None,
    work_dir: str | Path = "temp_models/sweep/lookup",
) -> LookupTable:
    """Measure the pure fp32 and pure int8 models once for the additive model."""
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

    layers: dict[str, LayerTiming] = {}
    for name, entry in fp32_table["compute"].items():
        match = int8_by_name.get(name)
        if match is not None:
            layers[name] = LayerTiming(
                fp32_us=float(entry["avg_us"]),
                int8_us=float(match["avg_us"]),
            )
    if not layers:
        raise RuntimeError("lookup build matched no layers between fp32 and int8")

    return LookupTable(layers=layers, int8_boundary=int8_table["boundary"])


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


def seam_count(mixed_boundary: Mapping, baseline_boundary: Mapping) -> int:
    """Count the Q/DQ boundary nodes this config adds over the all-int8 model."""
    mixed = Counter(
        (entry["op_type"], entry["tensor"]) for entry in mixed_boundary.values()
    )
    baseline = Counter(
        (entry["op_type"], entry["tensor"]) for entry in baseline_boundary.values()
    )
    return sum(max(0, count - baseline[signature]) for signature, count in mixed.items())


def run_config(
    config: SweepConfig,
    *,
    model_path: str,
    calibration_data: CalibrationData,
    lookup: LookupTable,
    quantizer: Quantizer,
    work_dir: str | Path,
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
        "measured_p50_us": None,
        "measured_p90_us": None,
        # Deliberately outside the try: a lookup mismatch must stop the sweep.
        "additive_pred_us": compute_additive_prediction(lookup.layers, config.fp32_nodes),
        "status": "ok",
        "error": None,
    }

    try:
        mixed_path = quantizer.create_mixed_int8(
            model_path,
            calibration_data,
            list(config.fp32_nodes),
        )
        mixed_table = _fetch_tables(mixed_path, work, f"mixed_{config.config_id}")
        row["seam_count"] = seam_count(mixed_table["boundary"], lookup.int8_boundary)

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

    configs = generate_configs(vocabulary, seed=seed, target=target)
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
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                time_budget_seconds=time_budget_seconds,
                isolated=isolated,
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
    return rows


def _fetch_tables(model_path: str, work_dir: Path, tag: str) -> ProfileTable:
    """Profile one model and return the P2.3 fetcher's compute and boundary tables."""
    work_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Keep every ONNX node visible so trace names map back to the graph.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(work_dir / f"{tag}.profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    feeds = {}
    for model_input in session.get_inputs():
        shape = [
            dim if isinstance(dim, int) and dim > 0 else 1 for dim in model_input.shape
        ]
        feeds[model_input.name] = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(2):
        session.run(None, feeds)
    trace = Path(session.end_profiling())

    measured = work_dir / f"{tag}.measured.json"
    measured.write_text(json.dumps({"warmup_runs": 1}), encoding="utf-8")
    return fetch_profile(str(trace), str(model_path), str(measured))


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
