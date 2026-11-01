# EdgeMason cleanup: Backlog

**Spec:** `docs/specs/P0.2-docs-scaffold.md`
**Sources:** `docs/cleanup/PHASE0_PLAN.md`, `docs/cleanup/REVIEW.md`, `docs/cleanup/DESIGN.md`, and `docs/executor/P0.4-report.md` (Appendix A, from D1).
**Companion:** `docs/cleanup/INVENTORY.md` maps every module to the ids below.

Every item carries a **Kind** (`refactor`, `bug`, `measurement`, `product`, `docs`, `deps`) and a
**Gate class** (`identical` means the output must be byte-for-byte unchanged; `intended improvement`
means the item exists to change a number, and the user has to sign off on that number). Items marked
**Status: done** were closed by P0.4 and are listed so the coverage check stays complete without
anything reading as pending work.

A finding raised by an executor belongs here even before its fix is specced. B-51 and B-52 came
from P0.8; the complete evidence is in `docs/executor/P0.8-report.md`.

---

## Measurement (research-sensitive)

None of these are cleanup. Each is a change to how a number is measured, each gets its own spec, and
each passes the gate as an intentional change rather than as a refactor.

| id | Title | Source | Kind | Gate class | Depends on |
|---|---|---|---|---|---|
| B-01 | Pin TVM thread settings to match the single-thread ONNX Runtime runs, so ORT-vs-TVM comparisons are fair | PLAN C.4 (research issues #1) | measurement | intended improvement | – |
| B-02 | ~~Accuracy is always scored on CPU, even for CUDA runs~~ **Superseded by D1** (2026-10-14, CPU-only): there are no CUDA runs left to score incorrectly. Kept so the finding is not lost. | PLAN C.4 (#2) | measurement | intended improvement | – |
| B-03 | Give `x86_64` per-quantization scaling factors, so INT8 estimates are not derived from the fp32 multiplier | PLAN C.4 (#3) | measurement | intended improvement | B-07 |
| B-04 | Seed benchmark inputs, so a repeat run measures the same work | PLAN C.4 (#4) | measurement | intended improvement | – |
| B-05 | ~~Memory counts CPU RSS only, so GPU memory under CUDA is not counted~~ **Superseded by D1** (2026-10-14, CPU-only). Kept for the record. | PLAN C.4 (#5) | measurement | intended improvement | – |
| B-54 | The TVM benchmark path ignores `input_data` (`api/optimizer/benchmark_engine.py:585`) and times random inputs, while the ONNX Runtime path times the calibration image | P1.1 planning | measurement | intended improvement | – |

Each of these needs user sign-off before its spec is written, and each must record the new value in
`docs/research/FINDINGS.md` (CODING_STANDARDS rule 6).

## Refactor

`DESIGN #2 → #3 → #4 → #1` is the order DESIGN.md itself recommends, then the REVIEW smells.

| id | Title | Source | Kind | Gate class | Depends on |
|---|---|---|---|---|---|
| B-06 | ~~Delete `api/optimizer/onnx_runtime.py`; the pipeline calls the engine directly~~ **Status: done (P1.1)**. Still open: `benchmark`/`model_report` return data instead of JSON strings | DESIGN #2 | refactor | identical | – |
| B-07 | Make the benchmark engine only measure: `services/hardware.py` owns the profiles and the estimate, with one profile loader. Removes the second, contradicting `hardware_estimate` | DESIGN #3, REVIEW Spec #6 | refactor | identical | – |
| B-08 | Introduce a `CandidateResult` type so the result dict stops being an unwritten format | DESIGN #4 | refactor | identical | – |
| B-09 | Put the framework decision (onnx/tvm) behind one seam with one adapter per framework, so the quantizer returns a model description rather than a path string | DESIGN #1 | refactor | identical | B-08 |
| B-10 | Bundle the six deploy options into one `DeployOptions` type, validated once | REVIEW Standards (Data Clumps / Duplicated Code) | refactor | identical | – |
| B-11 | A `Priority` enum plus shared constants, replacing the bare `"fp32"`, `"onnxruntime"` and priority strings | REVIEW Standards (Primitive Obsession / Repeated Switches) | refactor | identical | – |
| B-12 | Stop reaching into nested result dicts by key in `_public_result`, `gpu_warnings` and `ranking._latency/_memory` | REVIEW Standards (Feature Envy) | refactor | identical | B-08 |
| B-13 | Build the candidate result once instead of twice, in the success and failure branches | REVIEW Standards (Duplicated Code) | refactor | identical | B-08 |
| B-14 | Give `/analyze` and `/deploy` the same error-handling shape (today analyze catches every `Exception`, deploy only `ValueError`) | REVIEW Standards (Duplicated Code) | refactor | identical | – |
| B-15 | Collapse the identical `/` and `/ui` handlers in `routes/pages.py` | REVIEW Standards (Duplicated Code) | refactor | identical | – |
| B-16 | ~~Remove the `onnx_runtime.benchmark` pass-through~~ **Status: done (P1.1)** | REVIEW Standards (Middle Man) | refactor | identical | B-06 |
| B-17 | Rename "EdgeForge" to "EdgeMason", and rename the `requested_provider` / `execution_provider` pair so it says which one is resolved | REVIEW Standards (Mysterious Name / Naming Drift) | refactor | identical | – |
| B-18 | Small cleanups: unused `io`/`json` imports, the unused `e` behind a broad `except`, the function-level `evaluate_tvm` import | REVIEW Standards (Minor Issues) | refactor | identical | – |
| B-19 | Remove the dead code the deep-dive named: `benchmark_tvm`, `run_inference`, the unreachable `sample_path` branch, `tvm_runtime.run_model`, the overwritten `hardware_estimate`, the four unread planner fields, and most hardware-profile metadata. **Partly done (P1.1):** `benchmark_tvm`, `run_inference` and the `sample_path` branch went with `onnx_runtime.py`; the rest is still open | PLAN C.1 (dead code) | refactor | identical | B-06 |
| B-20 | Define `MAX_INPUT_ELEMENTS` and the 4096 dimension limit once, not twice | PLAN C.4 (literals) | refactor | identical | – |
| B-21 | The isolated and in-process benchmarks return different memory keys (`baseline/peak/delta` vs `baseline_mb/peak_mb/delta_mb`). Lock the quirk in a characterization test, then decide which name wins | PLAN C (memory keys) | refactor | identical | – |
| B-22 | Characterize the loosely-tested areas: the planner beyond two cases, ranking (no accuracy-priority or tie-break test), `estimate_on_target` beyond rpi4, quantization outputs, and the `/deploy` response shape | PLAN E (tested loosely) | refactor | identical | – |
| B-23 | Cover the untested surface: `/analyze`, `validate_model` limits, artifact TTL cleanup and bad ids, the 1000-sample and 640 MB limits, 413 errors, "accuracy priority requires .npz", and TVM accuracy | PLAN E (not tested) | refactor | identical | – |
| B-24 | Neutralise test nondeterminism: unseeded RNG, `system_info` and installed versions in the ZIP, temp paths, `uuid4`, and timing-sensitive assertions | PLAN E (nondeterminism) | refactor | identical | – |
| B-25 | Add pytest markers (`slow`, `tvm`, `cuda`, `research`) and a timeout, and split the one shared fixture | PLAN E (infrastructure) | refactor | identical | – |
| B-56 | CI still diffs the lint ratchet against `BASE_REF=origin/cleanup`; point it at `origin/main` before the `cleanup` branch is ever deleted | P1.1 planning | refactor | identical | – |

## Product

| id | Title | Source | Kind | Gate class | Depends on |
|---|---|---|---|---|---|
| B-26 | Port the Streamlit-only features to `index.html`: configurable backend URL, throughput bar chart, inline failure reason, `/analyze` errors in the UI, operator chart capped at the top 8 | PLAN G | product | identical | – |
| B-27 | Delete `frontend/app.py` once its features are ported | PHASE0_PLAN decisions, P0.2 verdict | product | identical | B-26 |
| B-28 | Make TVM opt-in and off by default; both dashboards currently turn it on | PHASE0_PLAN decisions, REVIEW Spec #7 | product | intended improvement | – |
| B-29 | Unify the hardware-id default, which is `rpi4` in the route, `rpi5` in the UI and `x86_64` in the functions | PLAN C.3 (profiles) | product | identical | – |
| B-30 | Fix the calibration warning that recommends "100-500" images when calibration is capped at 300, so the upper bound is impossible to reach | REVIEW Spec #5 | bug | identical | – |
| B-31 | **GPU strip**: remove every GPU/CUDA code path, test and documentation claim. The full inventory — 10 product locations, 6 tests, 5 README claims and 4 open decisions — is in `docs/executor/P0.4-report.md` Appendix A. Cite it, do not copy it | D2, P0.4 Appendix A | product | identical | – |
| B-32 | A default Streamlit deploy with TVM on will exceed the 120 s the frontend allows `/deploy`; the report's own figures put it near 45 minutes | REVIEW Spec #7 | bug | identical | B-28, B-27 |
| B-33 | Extend `TVM_TARGETS`, which today covers only rpi4, rpi5 and x86_64 | PLAN C.3 (profiles) | product | intended improvement | B-09 |
| B-51 | `create_static_int8` emits a model ONNX Runtime cannot load for **any** source model below opset 11: `per_channel=True` emits `DequantizeLinear` with `axis` (opset 13+), but ORT auto-upgrades a sub-11 graph to opset 11. Loading fails with `Unrecognized attribute: axis for operator DequantizeLinear`. Affects `resnet18-v1-7`, `squeezenet1.1-7`, `shufflenet-v2-10` and any user-supplied pre-11 model. There is **no opset guard in `api/`**; the bug stayed hidden because `assets.json` ships an opset-13 conversion of each. Fix: raise the pre-processed opset to ≥13, or reject opset-<11 with a clear error | P0.8 T2 finding | bug | identical | – |
| B-52 | Top-5 is tie-unstable: `api/optimizer/validation.py` uses `np.argpartition`, which resolves ties arbitrarily, while `scripts/quantization_analysis.py` uses `np.argsort`. On statically-quantized models whole class groups collapse onto one quantized value, so on `squeezenet1.1-13` 123 of 179 test images have a bit-identical 5th/6th logit and the two scorers disagree on every one. Thread count is a second, independent confound (the product pins 1 thread, the research script uses all cores). Fix: tie-stable top-5 | P0.8 T3 finding | bug | intended improvement | – |
| B-53 | EfficientNet B0 dynamic INT8 quantization produces **0.0% top1 accuracy** (fp32 baseline: 64.2%); static INT8 is also severely degraded at 13.9%. Depthwise separable convolutions are known to be quantization-sensitive, but 0% means the quantized model is completely non-functional. Research-relevant: if the paper discusses quantization across architectures, this result must be reported. Evidence in `baseline/full/accuracy.json` | P0.8 review finding | bug | intended improvement | – |
| B-55 | Decide whether users may set `batch_size`. Today every caller passes the default 1, so the parameter is plumbed through but never used | P1.1 planning | product | identical | – |
| B-57 | `tests/test_gate.py` `_git()` inherits the parent's `GIT_*` variables. Inside a git hook `GIT_DIR` points at the real repo and overrides `-C`, so the fixture's `init`/`config`/`commit` hit the real repo: it set `core.bare = true`, wrote `[user] gate`, and committed "base tree" onto `spec/P1.1`. Fix: run `_git` with every `GIT_*` variable removed, plus a test that sets `GIT_DIR` to a decoy repo and asserts the decoy is untouched | P1.1 push incident | bug | identical | – |

## Deps

| id | Title | Source | Kind | Gate class | Depends on |
|---|---|---|---|---|---|
| B-34 | Verify `tensorflow` and `keras` are unused, then remove them | PLAN F (possibly unused) | deps | identical | – |
| B-35 | Remove `streamlit`, `requests` and `pandas` once Streamlit is gone | PLAN F, P0.2 §Deps | deps | identical | B-27 |
| B-36 | Make `apache-tvm` an optional extra rather than a default runtime dependency | PHASE0_PLAN (TVM opt-in) | deps | identical | B-28 |
| B-37 | ~~README states the wrong Python version~~ **Status: done (P0.4)** — now `>=3.11,<3.12` | PLAN F (environment) | docs | identical | – |
| B-38 | ~~`apache-tvm` was undeclared and unpinned~~ **Status: done (P0.4)** — declared and pinned to `0.26.0` | PLAN F (missing deps) | deps | identical | – |
| B-39 | ~~`torch`, `torchvision`, `matplotlib` and `lxml` were undeclared~~ **Status: done (P0.4)** — declared in the `research` dependency group | PLAN F (missing deps) | deps | identical | – |
| B-40 | ~~`onnxruntime-gpu` and `nvidia-cudnn-cu13` pulled in CUDA libraries for CPU-only benchmarking~~ **Status: done (P0.4, D1)** — removed; a single `onnxruntime` remains | P0.4 D1 | deps | identical | – |

## Docs/Report

| id | Title | Source | Kind | Gate class | Depends on |
|---|---|---|---|---|---|
| B-41 | Give every unsourced or contradicted claim in `reports/EdgeMason_Report.md` a data file, a producing script and a command, or drop the claim. About 12 claims have no source, and one states 11 deploys where the data has 8 | PLAN B | docs | identical | – |
| B-42 | Resolve the three items of unclear provenance, the main research-integrity risk: `validation_run_2026-10-05.json` (no script produces it), `calibration_scale_study_256.json` and `calibration_scale_256.png` (unclear, uncited) | PLAN B | docs | identical | – |
| B-43 | Remove the two stale "changes are uncommitted" claims in the report | REVIEW Spec #1 | docs | identical | – |
| B-44 | Complete the reproduction table, which omits `calibration_scale_study.py`, `make_calibration_scale_chart.py` and `download_imagenetv2_large.py` even though section 5.6.1 depends on all three | REVIEW Spec #2 | docs | identical | – |
| B-45 | Document `scripts/build_report_odt.py`, which the report never mentions, and either add the missing ODT template or say where it comes from | REVIEW Spec #3, PLAN D | docs | identical | – |
| B-46 | Document that warm-up also stops after 25% of the time budget; the README says so, the report does not | REVIEW Spec #4 | docs | identical | – |
| B-47 | Document that `scripts/quantization_analysis.py` re-implements scoring, latency and quantization with different settings from `api/`, so the research numbers were measured on different code paths than the product. Document it; do not deduplicate it | PLAN D | docs | identical | – |
| B-48 | Give `scripts/run_full_benchmark.py` a dated output filename, so the dated file is not produced by renaming the file by hand | PLAN D | docs | identical | – |
| B-49 | ~~The suite size and counts in the plan are stale~~ **Status: done (P0.4)** — the plan's "87 cases, 83 passed and 1 skipped" is now 101 passed, 0 skipped | PLAN E | docs | identical | – |
| B-50 | Document that a TVM deployment package needs a C compiler on the target device, because TVM builds the module at load time. `README.md` tells users to `pip install -r requirements.txt` and says nothing about a toolchain | P0.4 T7 finding | docs | identical | – |

---

## Coverage checklist

Every finding in the three source documents, and the ids that carry it. `docs/cleanup/PHASE0_PLAN.md`
section A is out of scope here (it is the asset-provisioning input for P0.5).

### PLAN sections B–G

| Source finding | Ids |
|---|---|
| B. ~12 claims with no data source | B-41 |
| B. Line 524 says 11 deploys, the data has 8 | B-41 |
| B. `validation_run_2026-10-05.json` has no producing script | B-42 |
| B. `calibration_scale_study_256.json` / `.png` unclear and uncited | B-42 |
| C.1 Dead code (7 named items) | B-19 |
| C.1 Framework decided in 7+ places | B-09, B-11, B-33 |
| C.2 Hard-coded profile fallbacks that swallow load errors; profiles re-read per call | B-07 |
| C.2 `TVM_TARGETS` covers only rpi4, rpi5, x86_64 | B-33 |
| C.2 Hardware-id default differs in three places | B-29 |
| C.3 Literals scattered: priorities, `"fp32"`, framework names, provider names | B-11 |
| C.3 `MAX_INPUT_ELEMENTS` and the 4096 limit each defined twice | B-20 |
| C.4 Memory keys differ by benchmark mode | B-21 |
| C. Static INT8 emits an unloadable model below opset 11 | B-51 |
| C. Top-5 tie resolution is non-deterministic under static INT8 | B-52 |
| C.4 Measurement issue 1: TVM threads unpinned | B-01 |
| C.4 Measurement issue 2: accuracy always on CPU | B-02 (superseded by D1) |
| C.4 Measurement issue 3: x86_64 has no per-quantization factors | B-03 |
| C.4 Measurement issue 4: benchmark inputs unseeded | B-04 |
| C.4 Measurement issue 5: memory counts CPU RSS only | B-05 (superseded by D1) |
| C.5 `POST /deploy` runs a 14-step chain | `INVENTORY.md` § POST /deploy call path (documentation, not an item) |
| D. `quantization_analysis.py` re-implements product logic with different settings | B-47 |
| D. `build_report_odt.py` needs a template that is missing | B-45 |
| D. `run_full_benchmark.py` writes an undated filename | B-48 |
| D. Scripts repeat their predict/quantize loops | B-18 |
| E. Suite counts stale | B-49 (done, P0.4) |
| E. Tested loosely (planner, ranking, estimates, quantization, response shape) | B-22 |
| E. Not tested at all (7 areas) | B-23 |
| E. Nondeterminism to neutralise | B-24 |
| E. No markers, no timeout, one shared fixture | B-25 |
| F. Wrong Python version in the README | B-37 (done, P0.4) |
| F. `apache-tvm` undeclared and unpinned | B-38 (done, P0.4) |
| F. `torch`, `torchvision`, `matplotlib`, `lxml` undeclared | B-39 (done, P0.4) |
| F. `tensorflow` / `keras` possibly unused | B-34 |
| G. Five Streamlit features `index.html` lacks | B-26 |

### DESIGN

| Source finding | Ids |
|---|---|
| #1 ONNX-vs-TVM knowledge spread across five modules | B-09 |
| #2 `onnx_runtime` is a pass-through and returns a JSON string | B-06, B-16 |
| #3 Target estimate computed twice from two sources of truth | B-07 |
| #4 Candidate result is an unwritten dict format | B-08 |

### REVIEW — Standards

| Source finding | Ids |
|---|---|
| Data Clumps / Duplicated Code: six deploy options in three places | B-10 |
| Primitive Obsession / Repeated Switches: priorities, hardware list, `"fp32"`, `"onnxruntime"` | B-11 |
| Repeated Switches: TVM detected by `.tar`/`.so` in two modules | B-09 |
| Feature Envy: `_public_result`, `gpu_warnings`, `ranking._latency/_memory` | B-12 |
| Duplicated Code: the same 7-key dict in both branches | B-13 |
| Duplicated Code: `analyze.py` and `deploy.py` diverge on errors | B-14 |
| Duplicated Code: `/` and `/ui` handlers identical | B-15 |
| Middle Man: `onnx_runtime.benchmark` | B-16 |
| Mysterious Name: "EdgeForge" vs "EdgeMason" | B-17 |
| Mysterious Name: `requested_provider` / `execution_provider` pair | B-17 |
| Minor Issues: unused imports, unused `e`, broad except, function-level import, repeated script loops | B-18 |

### REVIEW — Spec

| Source finding | Ids |
|---|---|
| #1 Commit status in the report is out of date | B-43 |
| #2 Reproduction table is incomplete | B-44 |
| #3 `build_report_odt.py` not mentioned in the report | B-45 |
| #4 Warm-up 25% cap undocumented in the report | B-46 |
| #5 Calibration warning recommends an impossible amount | B-30 |
| #6 The old latency multiplier still produces a contradicting estimate | B-07 |
| #7 TVM on by default; Streamlit deploy likely times out | B-28, B-32 |
| "Verified as Matching the Spec" (11 items) | no action — confirmations, not findings |

### From the planning decisions and this spec

These two come from the plan's stated decisions rather than from a B–G finding, but they are
listed so every id in this file traces to something.

| Origin | Ids |
|---|---|
| `PHASE0_PLAN.md` decisions: TVM is experimental, opt-in and off by default | B-36 |
| P0.2 spec §Deps: drop `streamlit`, `requests` and `pandas` once Streamlit is deleted | B-35 |

### From the P0.8 baseline capture

| Source finding | Ids |
|---|---|
| Static INT8 emits a model ONNX Runtime cannot load for any source below opset 11 | B-51 |
| Top-5 computed with `np.argpartition` is tie-unstable, so it cannot reproduce the research script's `np.argsort` figures | B-52 |
| EfficientNet B0 dynamic INT8 quantization produces 0% top1 accuracy; static INT8 severely degraded at 13.9% (fp32 baseline 64.2%) | B-53 |
| `reports/data/quantization_analysis.json` **reproduces**: all 18 `top1` and all 18 `size_mb` match to the last digit under the research script's own scoring. The apparent disagreements are tie-resolution and thread-count artefacts, not stale data | no action — confirms B-41's data is sound |

### From the P0.4 run

| Source finding | Ids |
|---|---|
| D1 CPU-only: GPU runtime dependencies removed | B-40 (done) |
| D2 GPU code paths are now dead pending the strip | B-31 |
| `uv run pip list` reports the system, not the venv | documentation note in `CODING_STANDARDS.md` |
| A TVM deployment package needs a C compiler on the device | B-50 |
| README GPU claims are stale | B-31 |
