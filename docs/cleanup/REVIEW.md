# Code Review: `main...HEAD`

**Scope:** 69 files, commits `86044ad` (dashboard UI) and `f0b563d` (service layer). Reviewed against `reports/EdgeMason_Report.md` (spec) and Fowler's code-smell baseline (standards).

---

## Standards

The repo has no written coding standards, no CONTRIBUTING file and no linter config, so nothing here is a hard violation. Every finding is a judgement call based on Fowler's code-smell list.

### Possible Data Clumps / Duplicated Code
The same six deploy options travel together through three places:
- `_validate_form` (`api/routes/deploy.py:28-51`)
- The `DeployRequest` keyword arguments (`:93-104`)
- The `"constraints"` dict (`pipeline.py:220-226`)

**Fix:** a single `DeployOptions` type, validated once.

### Possible Primitive Obsession / Repeated Switches
- The priority strings are defined in `config.py`, branched on in `ranking.py:30-51`, and hard-coded again in `frontend/app.py:80`.
- The hardware list is hard-coded at `frontend/app.py:67`, even though `hardware_ids()` exists.
- Defaults like `"fp32"` and `"onnxruntime"` are bare strings in `pipeline.py:100,156`, `generator.py:187` and `frontend/app.py:256`, even though `QUANT_FP32` exists.

**Fix:** a `Priority` enum and shared constants.

### Possible Repeated Switches
TVM is detected by checking for `.tar`/`.so` at `pipeline.py:70` and `generator.py:266` (REVIEW.md #1).

### Possible Feature Envy
`_public_result`, `gpu_warnings` and `ranking._latency/_memory` all reach into nested result dicts by key (REVIEW.md #4).

### Possible Duplicated Code
- The success and failure branches at `pipeline.py:103-121` each build the same 7-key dict.
- `analyze.py` and `deploy.py` share the same work-dir, threadpool and error-handling shape, but handle errors differently: analyze catches every `Exception`, deploy only `ValueError`.
- The `/` and `/ui` handlers in `pages.py:16-24` are identical apart from the fallback text.

### Possible Middle Man
`onnx_runtime.benchmark` only forwards to the engine, and its JSON output is decoded straight away at `pipeline.py:84` (REVIEW.md #2).

### Possible Mysterious Name / Naming Drift
- The app is called "EdgeForge" in `pages.py`, `frontend/app.py` and `static/index.html`, but "EdgeMason" everywhere else.
- In `DeployRequest`, the pair `requested_provider` / `execution_provider` doesn't say which one is the resolved provider.

### Minor Issues
- `frontend/app.py:14-15` has unused `io` and `json` imports.
- `frontend/app.py:156` has an unused `e` and a broad `except Exception` that hides every error behind "Connect to backend".
- `pipeline.py:73` imports `evaluate_tvm` inside the function.
- The research scripts in `scripts/` repeat their predict/quantize loops.

---

## Spec

The core numbers in the report match the code: run counts and time budget, single-thread scoring, memory measurement, data split, hardware multipliers, provider rules, planner rules, quantization settings, ranking, and package contents. None of the mismatches are blocking.

### Missing or Partial

1. **The commit status in the report is out of date.** Line 4 says "changes not committed yet" and line 552 says "All changes are uncommitted on `abhay`". Both are false now.

2. **The reproduction table is incomplete.** Section 9 (lines 597–607) leaves out `calibration_scale_study.py`, `make_calibration_scale_chart.py` and `download_imagenetv2_large.py`. Section 5.6.1 depends on all three.

### Not Asked For by the Report

3. `scripts/build_report_odt.py` (956 lines) is not mentioned anywhere in the report.

4. Warm-up also stops after 25% of the time budget (`benchmark_engine.py:46`). The README documents this; the report doesn't (lines 190–191).

### Implemented, but Contradicts the Report

5. **The calibration warning recommends an impossible amount.** The warning says "100-500 are recommended" (`validation.py:121`), but calibration is capped at 300 images (`validation.py:17`, report lines 170–171). A user can never reach 500.

6. **The old latency multiplier still produces an estimate.** `BenchmarkEngine` still builds a `hardware_estimate` from the single fp32 multiplier (`benchmark_engine.py` ~436–449). The report says each device has its own multiplier per quantization type (lines 140–141). The pipeline overwrites the old estimate, so it's harmless today, but it leaves a second, contradicting estimate in the benchmark output (REVIEW.md #3).

7. **TVM is on by default, and the Streamlit deploy will likely time out.**
   - Both dashboards turn TVM on by default (`frontend/app.py:97`, `index.html:254`), even though the report says "Keep [TVM] off in demos" (line 542).
   - The new problem: Streamlit gives `/deploy` only 120 s (`frontend/app.py:182`).
   - By the report's own figures (lines 314 and 549), scoring ResNet-50 on untuned TVM takes about 45 minutes. So a default Streamlit deploy will probably time out.

### Verified as Matching the Spec
- **Benchmark runs:** 10 warm-up / 100 timed runs, a 10 s budget and at least 10 timed runs (`config.py:24-26`, `benchmark_engine.py:44`).
- **Threads:** one thread, in both the benchmark and accuracy scoring.
- **Memory:** VmHWM with `clear_refs`, measured in a fresh subprocess.
- **Data split:** seed-0 split, 30% capped at 300, 1000-sample limit, and the 200-test-image ± margin warning.
- **Hardware profiles:** the multipliers match the table, and execution providers are `cpu` only for Raspberry Pis and `cuda,cpu` for Jetson Nano and x86_64.
- **Execution provider:** "auto" skips CUDA when TVM is included, and a `cuda` request is rejected for CPU-only targets.
- **Planner:** CNNs never get dynamic INT8; a CNN without calibration data gets FP32 only, plus a warning.
- **Quantization:** per-channel QInt8 with `quant_pre_process`.
- **Ranking:** uses the target estimates for both the limits and the priority.
- **Responsiveness:** `run_in_threadpool` plus `Semaphore(1)`.
- **Package:** the ZIP has 6 files, `python:3.11-slim`, and the CPU `onnxruntime` pinned to the installed version.

---

## Summary

**Standards:** 10 judgement-call smells and no hard violations. The worst: result dicts read by key in four places, plus framework strings repeated across files (Feature Envy and Primitive Obsession).

**Spec:** 7 findings; the worst is #7 above, where the default Streamlit deploy with TVM on will likely time out after 120 s.
