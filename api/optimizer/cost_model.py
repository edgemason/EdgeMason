"""P2.7: graph-aware cost model for mixed-precision latency.

The P2.6 additive model predicts latency as the sum of the pure per-layer
costs. It misses the cost of *placement*: an fp32 layer at a high-resolution
point costs more than its own kernel, because the surrounding precision change
adds conversion and layout work proportional to the tensor crossing it.

This module adds two geometry-derived terms to the additive base:

    predicted = additive_us
              + alpha * seam_bytes
              + beta  * stranded_fp32_bytes

* ``seam_bytes`` -- activation elements crossing every fp32<->int8 boundary the
  config adds over the pure-int8 model. The boundaries come from the P2.3
  fetcher's boundary table (profiled with optimizations off, so every Q/DQ node
  keeps a trace name that maps back to the graph); sizes come from one
  ``onnx.shape_inference.infer_shapes`` pass over the base model.
* ``stranded_fp32_bytes`` -- output activation elements of the configured fp32
  layers whose fp32 graph region touches an int8 neighbour. A region is used
  rather than immediate adjacency because ONNX Runtime leaves some nodes fp32
  even when a child was excluded (for example ``relu0``/``pool0`` after
  ``conv0``); the high-resolution layer is what makes that region expensive.

``alpha`` and ``beta`` are fitted by ordinary least squares with no intercept,
so the additive sum stays the physical anchor and a config with no seams and no
stranded layers predicts exactly the additive value.

Validation is seeded k-fold cross-validation. The integrity rule is the point
of the block: if the cost model does not beat the additive baseline
out-of-sample, that is reported, never patched by changing features or seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort

from api.optimizer.fetcher import BoundaryEntry, fetch_profile

_QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})
_PASS_THROUGH_OPS = frozenset({"Transpose"})
_QUANTIZED_SUFFIXES = ("_quantized", "_quant")
_INT8_PREFIX = "QLinear"


class ShapeError(ValueError):
    """An activation shape needed for a feature is missing or not static."""


def element_count(shape: Sequence[int | str | None]) -> int:
    """Return the product of a tensor's dimensions.

    A shape with a symbolic, dynamic or non-positive dimension cannot be
    counted, and guessing one would corrupt every fitted coefficient, so it
    raises instead.
    """
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ShapeError(f"cannot count elements of shape {tuple(shape)!r}")
        total *= dim
    return total


@dataclass(frozen=True)
class GraphGeometry:
    """Precision-independent geometry of the base model."""

    shape_map: Mapping[str, tuple[int | str | None, ...]]
    layer_outputs: Mapping[str, str]

    def tensor_elements(self, tensor: str) -> int:
        """Element count of a tensor, resolving quantizer naming where needed."""
        if tensor in self.shape_map:
            return element_count(self.shape_map[tensor])
        for suffix in _QUANTIZED_SUFFIXES:
            if tensor.endswith(suffix):
                base = tensor[: -len(suffix)]
                if base in self.shape_map:
                    return element_count(self.shape_map[base])
        raise ShapeError(
            f"no shape for tensor {tensor!r} in the base model; "
            "refusing to fabricate a feature"
        )

    def layer_elements(self, layer: str) -> int:
        """Element count of a layer's output activation."""
        tensor = self.layer_outputs.get(layer)
        if tensor is None:
            raise ShapeError(f"no output tensor recorded for layer {layer!r}")
        return self.tensor_elements(tensor)


def build_geometry(base_model_path: str) -> GraphGeometry:
    """Infer every activation shape once and record each layer's output tensor."""
    model = onnx.shape_inference.infer_shapes(onnx.load(base_model_path))
    shape_map: dict[str, tuple[int | str | None, ...]] = {}
    value_infos = [
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    ]
    for value_info in value_infos:
        dims: list[int | str | None] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(dim.dim_param or None)
        shape_map[value_info.name] = tuple(dims)
    layer_outputs = {
        node.name: node.output[0]
        for node in model.graph.node
        if node.name and node.output
    }
    return GraphGeometry(shape_map=shape_map, layer_outputs=layer_outputs)


def layer_output_elements(
    geometry: GraphGeometry, layers: Sequence[str]
) -> dict[str, int]:
    """Element count of each named layer's output; raises if any is unknown."""
    return {layer: geometry.layer_elements(layer) for layer in layers}


def _signatures(boundary: Mapping[str, BoundaryEntry]) -> Counter[tuple[str, str]]:
    return Counter((entry["op_type"], entry["tensor"]) for entry in boundary.values())


def seam_boundary_entries(
    mixed_boundary: Mapping[str, BoundaryEntry],
    baseline_boundary: Mapping[str, BoundaryEntry],
) -> list[BoundaryEntry]:
    """Boundary entries the config adds over the pure-int8 model.

    ONNX Runtime always converts the model input and output, so the raw
    boundary table is never empty even for all-int8. Cancelling the baseline
    signatures leaves only the fp32<->int8 seams the config introduces.
    """
    remaining = _signatures(baseline_boundary)
    seams: list[BoundaryEntry] = []
    for entry in mixed_boundary.values():
        signature = (entry["op_type"], entry["tensor"])
        if remaining[signature] > 0:
            remaining[signature] -= 1
        else:
            seams.append(entry)
    return seams


def seam_elements(
    geometry: GraphGeometry,
    mixed_boundary: Mapping[str, BoundaryEntry],
    baseline_boundary: Mapping[str, BoundaryEntry],
) -> int:
    """Sum the activation elements crossing every seam of this config."""
    return sum(
        geometry.tensor_elements(entry["tensor"])
        for entry in seam_boundary_entries(mixed_boundary, baseline_boundary)
    )


def _is_pass_through(node: onnx.NodeProto) -> bool:
    """Nodes that carry a tensor without computing on it."""
    return node.op_type in _QDQ_OPS or node.op_type in _PASS_THROUGH_OPS


def _is_compute(node: onnx.NodeProto) -> bool:
    return node.op_type != "Constant" and not _is_pass_through(node)


def _is_int8(node: onnx.NodeProto) -> bool:
    if node.op_type.startswith(_INT8_PREFIX):
        return True
    if node.op_type == "NhwcMaxPool":
        return True
    return any(
        output.endswith(_QUANTIZED_SUFFIXES) for output in node.output
    )


def _stranded_nodes(graph: onnx.GraphProto) -> set[int]:
    """Indices of fp32 compute nodes whose fp32 region touches int8."""
    nodes = list(graph.node)
    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for tensor in node.input:
            if tensor:
                consumers[tensor].append(index)
        for tensor in node.output:
            if tensor:
                producer[tensor] = index

    def up_neighbours(index: int) -> set[int]:
        found: set[int] = set()
        stack = [tensor for tensor in nodes[index].input if tensor]
        seen: set[int] = set()
        while stack:
            tensor = stack.pop()
            candidate = producer.get(tensor)
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            node = nodes[candidate]
            if _is_pass_through(node):
                stack.extend(item for item in node.input if item)
            elif _is_compute(node):
                found.add(candidate)
        return found

    def down_neighbours(index: int) -> set[int]:
        found: set[int] = set()
        stack = [tensor for tensor in nodes[index].output if tensor]
        seen: set[int] = set()
        while stack:
            tensor = stack.pop()
            for candidate in consumers.get(tensor, ()):
                if candidate in seen:
                    continue
                seen.add(candidate)
                node = nodes[candidate]
                if _is_pass_through(node):
                    stack.extend(item for item in node.output if item)
                elif _is_compute(node):
                    found.add(candidate)
        return found

    compute = [index for index, node in enumerate(nodes) if _is_compute(node)]
    adjacency: dict[int, set[int]] = {}
    for index in compute:
        adjacency[index] = up_neighbours(index) | down_neighbours(index)

    fp32 = {index for index in compute if not _is_int8(nodes[index])}
    int8 = {index for index in compute if _is_int8(nodes[index])}

    stranded: set[int] = set()
    remaining = set(fp32)
    while remaining:
        root = remaining.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, ()):
                if neighbour in fp32 and neighbour not in component:
                    component.add(neighbour)
                    stack.append(neighbour)
        remaining -= component
        if any(adjacency.get(index, set()) & int8 for index in component):
            stranded |= component
    return stranded


def stranded_fp32_layers(
    mixed_model_path: str, configured_fp32: Sequence[str]
) -> tuple[str, ...]:
    """Configured fp32 layers whose surrounding fp32 region borders int8.

    A configured layer can be separated from the actual precision boundary by
    other fp32 nodes ONNX Runtime left alone; the whole region is what pays the
    layout cost, so the layer counts as stranded if any node in its fp32 region
    touches an int8 neighbour.
    """
    graph = onnx.load(mixed_model_path).graph
    stranded_indices = _stranded_nodes(graph)
    stranded_names = {
        graph.node[index].name for index in stranded_indices if graph.node[index].name
    }
    return tuple(layer for layer in configured_fp32 if layer in stranded_names)


def stranded_elements(
    geometry: GraphGeometry, stranded_layers: Sequence[str]
) -> int:
    """Sum the output activation elements of the stranded fp32 layers."""
    return sum(geometry.layer_elements(layer) for layer in stranded_layers)


@dataclass(frozen=True)
class Features:
    """One config's feature vector."""

    additive_us: float = 0.0
    seam_bytes: int = 0
    stranded_fp32_bytes: int = 0


def extract_features(
    geometry: GraphGeometry,
    configured_fp32: Sequence[str],
    mixed_boundary: Mapping[str, BoundaryEntry],
    baseline_boundary: Mapping[str, BoundaryEntry],
    mixed_model_path: str,
) -> Features:
    """Compute the two graph-geometry features for one config.

    Every tensor size must resolve from the inferred base model; a missing
    shape stops extraction instead of substituting a guess.
    """
    stranded = stranded_fp32_layers(mixed_model_path, configured_fp32)
    return Features(
        seam_bytes=seam_elements(geometry, mixed_boundary, baseline_boundary),
        stranded_fp32_bytes=stranded_elements(geometry, stranded),
    )


def fetch_raw_boundary(
    model_path: str, work_dir: str | Path, tag: str = "features"
) -> dict[str, BoundaryEntry]:
    """Fetch the P2.3 boundary table without optimizations hiding Q/DQ nodes.

    Profiling in the benchmark's optimized regime renames and fuses boundary
    nodes, so feature extraction uses the authored graph's names; that is the
    graph the config defines. Timing never comes from this profile.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(work / f"{tag}.profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    feeds: dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        shape = [
            dim if isinstance(dim, int) and dim > 0 else 1 for dim in model_input.shape
        ]
        feeds[model_input.name] = np.random.default_rng(0).random(
            shape, dtype=np.float32
        )
    for _ in range(3):
        session.run(None, feeds)
    trace = Path(session.end_profiling())
    measured = work / f"{tag}.measured.json"
    measured.write_text(json.dumps({"warmup_runs": 1}), encoding="utf-8")
    return fetch_profile(str(trace), str(model_path), str(measured))["boundary"]


@dataclass(frozen=True)
class CostModel:
    """Additive base plus the fitted seam and stranded-fp32 corrections."""

    alpha: float
    beta: float
    n: int = 0

    def predict(self, features: Features) -> float:
        return (
            float(features.additive_us)
            + self.alpha * float(features.seam_bytes)
            + self.beta * float(features.stranded_fp32_bytes)
        )


def _usable_rows(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Rows that carry a measurement; failed rows stay in the register only."""
    return [
        row
        for row in rows
        if row.get("status", "ok") == "ok" and row.get("measured_p50_us") is not None
    ]


def _float_field(row: Mapping[str, object], key: str) -> float:
    return float(cast(float, row[key]))


def _int_field(row: Mapping[str, object], key: str) -> int:
    return int(cast(int, row[key]))


def _design(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [
            [_float_field(row, "seam_bytes"), _float_field(row, "stranded_fp32_bytes")]
            for row in rows
        ],
        dtype=float,
    )
    target = np.array(
        [
            _float_field(row, "measured_p50_us")
            - _float_field(row, "additive_pred_us")
            for row in rows
        ],
        dtype=float,
    )
    return features, target


def fit_cost_model(rows: Sequence[Mapping[str, object]]) -> CostModel:
    """Fit alpha and beta by ordinary least squares, with no intercept."""
    data = _usable_rows(rows)
    if len(data) < 2:
        raise ValueError(
            "need at least two measured configs to fit two coefficients, "
            f"got {len(data)}"
        )
    features, target = _design(data)
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    return CostModel(
        alpha=float(coefficients[0]), beta=float(coefficients[1]), n=len(data)
    )


def _mape(predictions: np.ndarray, actual: np.ndarray) -> float:
    if np.any(actual <= 0):
        raise ValueError("measured latencies must be positive for MAPE")
    return float(np.mean(np.abs(predictions - actual) / actual))


def cross_validate(
    rows: Sequence[Mapping[str, object]], *, k: int = 5, seed: int = 0
) -> dict:
    """Seeded k-fold CV: additive vs cost model, coefficients, and stability.

    Predictions are pooled over the held-out folds before computing MAPE, so
    every config contributes exactly once and folds of different size are
    weighted by their point count.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    data = _usable_rows(rows)
    if len(data) < k:
        raise ValueError(
            f"need at least k={k} measured configs for cross-validation, got {len(data)}"
        )
    # Two coefficients are exactly determined by two points; require one more so
    # a fold is a check on the fit, not just a solve.
    if len(data) - math.ceil(len(data) / k) < 3:
        raise ValueError(
            "a fold has fewer than three points to fit; more configs are needed"
        )

    order = np.random.default_rng(seed).permutation(len(data))
    folds = np.array_split(order, k)
    fold_reports: list[dict] = []
    cost_predictions: list[float] = []
    additive_predictions: list[float] = []
    actuals: list[float] = []

    for fold_index, test_indices in enumerate(folds):
        test_set = set(int(index) for index in test_indices)
        train = [row for index, row in enumerate(data) if index not in test_set]
        test = [row for index, row in enumerate(data) if index in test_set]
        model = fit_cost_model(train)

        fold_actual: list[float] = []
        fold_cost: list[float] = []
        fold_additive: list[float] = []
        for row in test:
            features = Features(
                additive_us=_float_field(row, "additive_pred_us"),
                seam_bytes=_int_field(row, "seam_bytes"),
                stranded_fp32_bytes=_int_field(row, "stranded_fp32_bytes"),
            )
            fold_actual.append(_float_field(row, "measured_p50_us"))
            fold_cost.append(model.predict(features))
            fold_additive.append(features.additive_us)
        cost_predictions.extend(fold_cost)
        additive_predictions.extend(fold_additive)
        actuals.extend(fold_actual)

        fold_reports.append(
            {
                "fold": fold_index,
                "n_train": len(train),
                "n_test": len(test),
                "alpha": model.alpha,
                "beta": model.beta,
                "additive_mape": _mape(np.array(fold_additive), np.array(fold_actual)),
                "cost_model_mape": _mape(np.array(fold_cost), np.array(fold_actual)),
            }
        )

    alphas = [fold["alpha"] for fold in fold_reports]
    betas = [fold["beta"] for fold in fold_reports]
    full = fit_cost_model(data)
    return {
        "n_configs": len(data),
        "n_failed": len(rows) - len(data),
        "k": k,
        "seed": seed,
        "features": ["seam_bytes", "stranded_fp32_bytes"],
        "coefficients": {"alpha_us_per_element": full.alpha, "beta_us_per_element": full.beta},
        "folds": fold_reports,
        "out_of_sample": {
            "additive_mape": _mape(
                np.array(additive_predictions), np.array(actuals)
            ),
            "cost_model_mape": _mape(
                np.array(cost_predictions), np.array(actuals)
            ),
        },
        "stability": {
            "alpha_min": float(min(alphas)),
            "alpha_max": float(max(alphas)),
            "alpha_std": float(np.std(alphas)),
            "beta_min": float(min(betas)),
            "beta_max": float(max(betas)),
            "beta_std": float(np.std(betas)),
        },
    }


def load_register(path: str | Path) -> list[dict]:
    """Read a P2.6/P2.7 JSON Lines register."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_report(report: Mapping[str, object], path: str | Path) -> None:
    """Write the CV report as indented, key-sorted JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True, help="JSON Lines sweep register")
    parser.add_argument("--report", required=True, help="output report JSON")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = cross_validate(load_register(args.register), k=args.k, seed=args.seed)
    write_report(report, args.report)
    out = report["out_of_sample"]
    print(
        f"n={report['n_configs']} k={report['k']} "
        f"additive MAPE={out['additive_mape']:.1%} "
        f"cost-model MAPE={out['cost_model_mape']:.1%}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
