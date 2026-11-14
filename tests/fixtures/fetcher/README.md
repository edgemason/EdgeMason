# Fetcher fixtures (P2.3)

Real ONNX Runtime traces for **`squeezenet1.1-13.onnx`**, all-FP32 and its static
QDQ INT8 twin. They are small enough to run on x86 in milliseconds and contain
real ORT kernel events, including the Q/DQ boundary nodes.

| File | What it is |
|---|---|
| `squeezenet1.1-13.fp32.onnx` | real FP32 model (4.96 MB, opset 13) |
| `squeezenet1.1-13.int8.onnx` | `Quantizer.create_static_int8` output (QDQ format) |
| `*.fp32.trace.json` | ORT Chrome-tracing profile, 1 warm-up + 3 timed runs |
| `*.int8.trace.json` | same, for the INT8 model |
| `*.measured.json` | `bench.py`-shaped result carrying `warmup_runs: 1` |

## Why optimisations are disabled

The traces were captured with
`ort.SessionOptions.graph_optimization_level = ORT_DISABLE_ALL`.

With default optimisations ORT fuses Conv+Relu, folds 44 of the 134 Q/DQ nodes
into `QLinearConv`/`QLinearConcat`/`NhwcMaxPool`, inserts `ReorderOutput` and
`Transpose`, and renames events after layout/fused tensors. That trace cannot
satisfy the spec's "every Q/DQ node" boundary table, and it breaks the
compute-node-with-no-ONNX-twin rule. Disabling optimisations keeps every model
node visible. The fetcher still normalises layout/fusion suffixes, so it does
not crash on default traces, but the boundary table is only complete on
optimisation-disabled captures.

## Regenerating (real model, no synthetic graphs)

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
opts.enable_profiling = True
opts.profile_file_prefix = "<tag>"
session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
for _ in range(1 + 3):          # 1 warm-up + 3 timed runs
    session.run(None, feed)
trace_path = session.end_profiling()
```

The committed traces are fixed artefacts; tests read them, never regenerate
them.
