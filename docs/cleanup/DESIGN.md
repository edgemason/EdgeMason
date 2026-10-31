# Codebase Design 

## 1. ONNX-vs-TVM knowledge is spread across five modules

Each module works out the framework on its own:
- **Planner:** produces candidate names as strings.
- **Quantizer:** `Quantizer.quantize_for_candidate` (`api/optimizer/quantizer.py:193`) matches on those strings in an if/elif chain.
- **File-extension checks:** three modules each test for `.tar`/`.so` on their own — `benchmark_engine.py:584`, `pipeline.py:70` and `generator.py:266`.
- **Accuracy:** `pipeline._evaluate_candidate` picks `evaluate_tvm` or `evaluate_onnx` itself.

**The deletion test:** adding a third framework, such as TFLite, means editing every one of those places.

**Suggested fix:** put a seam with one adapter per framework, behind a small interface: build, benchmark, evaluate and describe the inputs. ONNX and TVM are two real adapters, so the seam is real, not hypothetical. The quantizer would return a `BuiltModel` (path, framework, input spec) instead of a bare path string. The pipeline and the packager would then ask the model what it is instead of checking its file extension. This one gives the most leverage.

---

## 2. `onnx_runtime.benchmark` is a pass-through, and the benchmark interface returns a JSON string

- **A pass-through:** `pipeline.py` imports `benchmark` from `api/optimizer/onnx_runtime.py`, which just forwards to `benchmark_engine.benchmark`. The extra work it does (handling `sample_path`) never runs from the pipeline.
- **Dead code:** `benchmark_tvm`, `run_inference` and the rest of that file appear unused. `benchmark_tvm` duplicates the engine's own TVM path with a different results format.
- **Strings instead of data:** `benchmark_engine.benchmark` builds a dict, renames the `memory_mb` keys, turns it into a JSON string, and the pipeline immediately parses it back. `model_report` does the same, and both of its callers parse its output.

**The deletion test:** delete `onnx_runtime.py`, have the pipeline call the engine directly, and have both functions return dicts or `BenchmarkResult`. No complexity comes back anywhere. This is cheap and low-risk.

---

## 3. The target-device estimate is computed twice from two sources of truth

- **Two calculations:** `BenchmarkEngine` computes its own `hardware_estimate` (`benchmark_engine.py:442`) using only the fp32 `latency_scaling_factor`. The pipeline then throws it away and uses `estimate_on_target` (`services/hardware.py`), which accounts for quantization.
- **Two copies of the profiles:** `benchmark_engine.py:115-170` holds hard-coded fallback hardware profiles that duplicate `profiles/hardware_profiles.json`.

**Suggested fix:** the benchmark engine should only measure. `services/hardware` should own both the profiles and the estimate, with one profile loader.

---

## 4. A candidate's result is an unwritten dict format

`_evaluate_candidate` returns a dict, and four other places read it by key: `rank_results`, `gpu_warnings`, `_public_result` and the packaging step. That means `test_ranking.py` has to build nested dicts by hand. A `CandidateResult` dataclass would turn that format into an explicit interface. This is small, and it pairs naturally with #1.

---

## Suggested order

1. **#2** (quick, removes code) — delete `onnx_runtime.py`, have the pipeline call the engine directly, and have both functions return dicts or `BenchmarkResult`.
2. **#3** — the benchmark engine should only measure; `services/hardware` should own both the profiles and the estimate, with one profile loader.
3. **#1 + #4 together** — put a seam with one adapter per framework, behind a small interface: build, benchmark, evaluate and describe the inputs. Pair with a `CandidateResult` dataclass.

In each case, tests should target the new interface and the old unit tests on the modules being replaced should go.
