"""Parse an ONNX Runtime per-node profile into per-layer timing tables.

    python -m api.optimizer.fetcher \
        --profile run_profile.json --model model.onnx \
        --measured measured.json --out table.json

The trace is the Chrome-tracing JSON that `bench.py --profile` writes through
`session.end_profiling()`. `measured.json` sits beside it and supplies the
number of warm-up runs that must be discarded.

The output holds two tables:

* ``compute``  — model ops keyed by their ONNX node name;
* ``boundary`` — QuantizeLinear/DequantizeLinear nodes keyed by their ONNX name,
  each attributed to the compute layer whose edge it converts.

All times are microseconds. ``sum(compute) + sum(boundary)`` is deliberately not
compared with end-to-end latency: the gap is the effect H1 is about.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, TypedDict

import onnx

_KERNEL_SUFFIX = "_kernel_time"
_QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})
_LAYOUT_SUFFIX_RE = re.compile(r"_(?:nchwc|nhwc|nchw|token_\d+)$")
_TOTAL_NOTE = (
    "sum(compute)+sum(boundary) is NOT end-to-end latency; "
    "that gap is what H1 predicts"
)


class ComputeEntry(TypedDict):
    op_type: str
    avg_us: float
    n: int


class BoundaryEntry(TypedDict):
    op_type: str
    avg_us: float
    serves_layer: str
    tensor: str


class TableMeta(TypedDict):
    total_kernel_us: float
    note: str


class ProfileTable(TypedDict):
    precision: str
    compute: dict[str, ComputeEntry]
    boundary: dict[str, BoundaryEntry]
    meta: TableMeta


def _load_json(path: str | Path, what: str) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {what} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} {path} is not valid JSON: {exc}") from exc


def _read_warmup_runs(path: str | Path) -> int:
    try:
        data = _load_json(path, "measured file")
    except ValueError as exc:
        raise ValueError(
            f"measured file {path} is unreadable; cannot read warmup_runs: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"measured file {path} is not a JSON object; no warmup_runs")
    warmup_runs = data.get("warmup_runs")
    if type(warmup_runs) is not int or warmup_runs < 0:
        raise ValueError(f"measured file {path} has no valid warmup_runs")
    return warmup_runs


def _kernel_events(trace: object) -> list[tuple[str, str, float]]:
    if not isinstance(trace, list):
        raise ValueError("profile trace is not valid JSON: expected a list of events")
    events: list[tuple[str, str, float]] = []
    for event in trace:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name.endswith(_KERNEL_SUFFIX):
            continue
        duration = event.get("dur")
        if not isinstance(duration, (int, float)):
            raise ValueError(f"node event {name!r} has no numeric dur")
        args = event.get("args")
        op_name = args.get("op_name") if isinstance(args, dict) else None
        if not isinstance(op_name, str):
            raise ValueError(f"node event {name!r} has no args.op_name")
        events.append((name[: -len(_KERNEL_SUFFIX)], op_name, float(duration)))
    if not events:
        raise ValueError("no node events; was --profile set?")
    return events


def _name_candidates(stem: str) -> list[str]:
    """Names an ORT event stem may carry for one ONNX node.

    ORT names events after the produced tensor and appends layout/fusion
    suffixes (``_nchwc``, ``_token_12``, ``/duplicated``). The unmodified stem is
    tried first, so a real node name always wins over a stripped alias.
    """
    candidates: list[str] = []

    def add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    add(stem)
    base = stem
    if base.startswith("fused "):
        base = base[len("fused ") :]
        add(base.split("/", 1)[0])
    if base.endswith("/duplicated"):
        base = base[: -len("/duplicated")]
        add(base)
    if "/" in base:
        add(base.split("/", 1)[0])

    current = base
    while True:
        match = _LAYOUT_SUFFIX_RE.search(current)
        if match is None:
            break
        current = current[: match.start()]
        add(current)
    return candidates


def _base_op(op_type: str) -> str:
    """Reduce an ORT fused/quantized op name to the ONNX op it implements."""
    if op_type.startswith("QLinear") and len(op_type) > len("QLinear"):
        return op_type[len("QLinear") :]
    if op_type == "NhwcMaxPool":
        return "MaxPool"
    if op_type == "NhwcConv":
        return "Conv"
    if op_type == "DynamicQuantizeMatMul":
        return "MatMul"
    if op_type.startswith("Fused") and len(op_type) > len("Fused"):
        return op_type[len("Fused") :]
    return op_type


class _GraphIndex:
    """Lookups over one ONNX graph, keyed by node and tensor name."""

    def __init__(self, model: onnx.ModelProto):
        graph = model.graph
        self.nodes = list(graph.node)
        self.by_name = {node.name: node for node in self.nodes if node.name}
        self.producer: dict[str, onnx.NodeProto] = {}
        for node in self.nodes:
            for output in node.output:
                self.producer[output] = node
        self.consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
        for node in self.nodes:
            for tensor in node.input:
                self.consumers[tensor].append(node)

    @staticmethod
    def is_compute(node: onnx.NodeProto) -> bool:
        return node.op_type not in _QDQ_OPS


def _find_compute_node(index: _GraphIndex, stem: str) -> onnx.NodeProto | None:
    for candidate in _name_candidates(stem):
        node = index.by_name.get(candidate)
        if node is not None and index.is_compute(node):
            return node
        node = index.producer.get(candidate)
        if node is not None and index.is_compute(node):
            return node
    return None


def _find_qdq_node(index: _GraphIndex, stem: str) -> onnx.NodeProto | None:
    matches: dict[str, onnx.NodeProto] = {}
    for candidate in _name_candidates(stem):
        node = index.by_name.get(candidate)
        if node is not None and node.op_type in _QDQ_OPS:
            matches[node.name] = node
            continue
        node = index.producer.get(candidate)
        if node is not None and node.op_type in _QDQ_OPS:
            matches[node.name] = node
            continue
        for consumer in index.consumers.get(candidate, []):
            if consumer.op_type in _QDQ_OPS:
                matches[consumer.name] = consumer
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None


def _reconcile_layer(
    index: _GraphIndex, node: onnx.NodeProto, op_type: str
) -> onnx.NodeProto | None:
    """Map a trace event to the ONNX compute node whose cost it measures.

    A fused Conv+Relu event is named after the Relu output but reports
    ``op_name == "Conv"``; walk one step back to the Conv that owns the cost.
    Quantized and fused op variants resolve through ``_base_op``.
    """
    base = _base_op(op_type)
    if _base_op(node.op_type) == base:
        return node
    if op_type.startswith("Fused") and node.op_type in {"MatMul", "Gemm"}:
        return node

    predecessors: list[onnx.NodeProto] = []
    for tensor in node.input:
        producer = index.producer.get(tensor)
        if producer is not None and index.is_compute(producer):
            predecessors.append(producer)
    matches = [item for item in predecessors if _base_op(item.op_type) == base]
    if len(matches) == 1:
        return matches[0]
    return None


def _compute_producer(index: _GraphIndex, tensor: str) -> onnx.NodeProto | None:
    node = index.producer.get(tensor)
    if node is not None and index.is_compute(node):
        return node
    return None


def _compute_consumers(index: _GraphIndex, tensor: str) -> list[onnx.NodeProto]:
    return [node for node in index.consumers.get(tensor, []) if index.is_compute(node)]


def _chain_up(index: _GraphIndex, tensor: str) -> onnx.NodeProto | None:
    """Follow Q/DQ producers until one compute node (or nothing) is reached."""
    seen: set[str] = set()
    current = tensor
    while True:
        node = index.producer.get(current)
        if node is None:
            return None
        if index.is_compute(node):
            return node
        if node.name in seen or not node.input:
            return None
        seen.add(node.name)
        current = node.input[0]


def _reachable_compute_consumers(
    index: _GraphIndex, tensor: str
) -> list[onnx.NodeProto]:
    """Follow Q/DQ consumers until compute nodes are reached."""
    found: dict[str, onnx.NodeProto] = {}
    seen: set[str] = set()
    queue = [tensor]
    while queue:
        current = queue.pop(0)
        for node in index.consumers.get(current, []):
            if index.is_compute(node):
                found[node.name] = node
            elif node.name not in seen:
                seen.add(node.name)
                if node.output:
                    queue.append(node.output[0])
    return list(found.values())


def _attribute_boundary(
    index: _GraphIndex, node: onnx.NodeProto
) -> tuple[onnx.NodeProto, str]:
    """Return the compute layer a Q/DQ node serves and the tensor it converts."""
    if node.op_type == "DequantizeLinear":
        consumers = _compute_consumers(index, node.output[0])
        if len(consumers) == 1:
            return consumers[0], node.output[0]
        # A branching dequantize feeds several layers; the layer it converts
        # for is the producer whose output it re-quantizes.
        producer = _chain_up(index, node.input[0])
        if producer is not None:
            return producer, node.input[0]
        raise ValueError(f"cannot attribute Q/DQ node {node.name!r} to a layer")

    producer = _compute_producer(index, node.input[0])
    if producer is not None:
        return producer, node.input[0]
    consumers = _reachable_compute_consumers(index, node.output[0])
    if len(consumers) == 1:
        return consumers[0], node.output[0]
    raise ValueError(f"cannot attribute Q/DQ node {node.name!r} to a layer")


def _is_ort_layout_event(stem: str, op_type: str) -> bool:
    """True for layout bookkeeping ORT inserts; it is not a model layer."""
    if op_type == "ReorderOutput":
        return True
    return op_type == "Transpose" and stem.startswith("Transpose")


def fetch_profile(profile: str, model: str, measured: str) -> ProfileTable:
    """Parse one ORT trace into the per-layer compute and boundary tables.

    ``measured`` is the result JSON `bench.py` wrote beside the trace; its
    ``warmup_runs`` is the only trusted source of the warm-up count.
    """
    trace = _load_json(profile, "profile trace")
    warmup_runs = _read_warmup_runs(measured)
    events = _kernel_events(trace)

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for stem, op_type, duration in events:
        grouped[stem].append((op_type, duration))

    kept: dict[str, tuple[str, list[float], float, int]] = {}
    for stem, occurrences in grouped.items():
        op_type = occurrences[0][0]
        if any(item[0] != op_type for item in occurrences):
            raise ValueError(f"node {stem!r} reports more than one op type")
        durations = [duration for _, duration in occurrences][warmup_runs:]
        if not durations:
            raise ValueError(
                f"node {stem!r} has no events after dropping "
                f"{warmup_runs} warm-up run(s)"
            )
        kept[stem] = (op_type, durations, sum(durations) / len(durations), len(durations))

    model_proto = onnx.load(model)
    index = _GraphIndex(model_proto)

    compute: dict[str, ComputeEntry] = {}
    layer_source: dict[str, str] = {}
    boundary_durations: dict[str, list[float]] = {}
    boundary_meta: dict[str, tuple[str, str, str]] = {}

    for stem, (op_type, durations, avg_us, count) in kept.items():
        if op_type in _QDQ_OPS:
            qdq_node = _find_qdq_node(index, stem)
            if qdq_node is None:
                raise ValueError(f"cannot attribute Q/DQ node {stem!r} to a layer")
            if qdq_node.op_type != op_type:
                raise ValueError(
                    f"Q/DQ node {stem!r} is {op_type} in the trace but "
                    f"{qdq_node.op_type} in {model}"
                )
            layer, tensor = _attribute_boundary(index, qdq_node)
            if not layer.name:
                raise ValueError(f"layer for Q/DQ node {stem!r} has no ONNX name")
            boundary_durations.setdefault(qdq_node.name, []).extend(durations)
            boundary_meta[qdq_node.name] = (op_type, layer.name, tensor)
            continue

        node = _find_compute_node(index, stem)
        if node is None:
            if _is_ort_layout_event(stem, op_type):
                continue
            raise ValueError(f"compute node {stem!r} has no counterpart in {model}")
        layer = _reconcile_layer(index, node, op_type)
        if layer is None:
            raise ValueError(
                f"compute node {stem!r} does not match ONNX node "
                f"{node.name!r} ({node.op_type})"
            )
        if not layer.name:
            raise ValueError(f"compute node {stem!r} maps to an ONNX node with no name")
        previous = layer_source.get(layer.name)
        if previous is not None and previous != stem:
            raise ValueError(
                f"layers collide on reconciled name {layer.name!r}: "
                f"{previous!r} and {stem!r}"
            )
        layer_source[layer.name] = stem
        compute[layer.name] = {"op_type": op_type, "avg_us": avg_us, "n": count}

    boundary: dict[str, BoundaryEntry] = {}
    for name, durations in boundary_durations.items():
        op_type, serves_layer, tensor = boundary_meta[name]
        boundary[name] = {
            "op_type": op_type,
            "avg_us": sum(durations) / len(durations),
            "serves_layer": serves_layer,
            "tensor": tensor,
        }

    has_qdq = any(node.op_type in _QDQ_OPS for node in model_proto.graph.node)
    total_kernel_us = sum(entry["avg_us"] for entry in compute.values()) + sum(
        entry["avg_us"] for entry in boundary.values()
    )

    return {
        "precision": "int8" if has_qdq else "fp32",
        "compute": compute,
        "boundary": boundary,
        "meta": {"total_kernel_us": total_kernel_us, "note": _TOTAL_NOTE},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch per-layer compute and Q/DQ boundary timings from an ORT trace."
    )
    parser.add_argument("--profile", required=True, help="ORT Chrome-tracing JSON")
    parser.add_argument("--model", required=True, help="ONNX model that produced the trace")
    parser.add_argument("--measured", required=True, help="bench.py result JSON with warmup_runs")
    parser.add_argument("--out", required=True, help="output table JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    table = fetch_profile(args.profile, args.model, args.measured)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
