# EdgeMason cleanup: Inventory

**Spec:** `docs/specs/P0.2-docs-scaffold.md`
**Generated for:** branch `cleanup` @ `f5f9388`
**Companion:** `docs/cleanup/BACKLOG.md` — every id in the last column is defined there.

One row per module under `api/` and `scripts/`, plus the three non-package files the spec names.
The six empty `__init__.py` files are excluded; they carry no code and exist only to make the
packages importable.

**Verdict** is one of `keep`, `refactor`, `merge-into:<path>`, `delete`, `move-to:<path>`. It is
assigned from a named finding in `PHASE0_PLAN.md`, `REVIEW.md` or `DESIGN.md`, and defaults to
`keep` where those documents are silent. Only two verdicts were decided in advance: `frontend/app.py`
and `api/optimizer/onnx_runtime.py` both delete.

| Path | Lines | Purpose | Called by | Dead code | Research-sensitive | Verdict | Backlog ids |
|---|---|---|---|---|---|---|---|
| `api/main.py` | 10 | Builds the FastAPI app and mounts the four routers. | `uv run python -m api.main`; `tests/test_api.py:8` | – | – | keep | – |
| `api/config.py` | 26 | Size limits, artifact TTL and directory, and the domain string sets. | `routes/analyze.py`, `routes/deploy.py`, `routes/pages.py`, `services/artifacts.py`, `services/model_validation.py`, `services/pipeline.py` | – | `BENCHMARK_WARMUP_RUNS`/`TIMED_RUNS`/`TIME_BUDGET_SECONDS` (:24-26) | refactor | B-11, B-20 |
| `api/routes/deploy.py` | 114 | The `POST /deploy` endpoint: validates the form and runs the pipeline on a worker thread. | `api/main.py:4` | – | – | refactor | B-10, B-32 |
| `api/routes/analyze.py` | 34 | The `POST /analyze` endpoint: validates and reports on a model without optimizing it. | `api/main.py:4` | – | – | refactor | B-14 |
| `api/routes/pages.py` | 33 | Serves the static dashboard at `/` and `/ui`. | `api/main.py:4` | – | – | refactor | B-15 |
| `api/routes/artifacts.py` | 32 | Serves a generated deployment ZIP by its one-time artifact id. | `api/main.py:4` | – | – | keep | – |
| `api/services/pipeline.py` | 260 | The deploy pipeline: plan, quantize, benchmark, evaluate, rank, package. | `routes/deploy.py:18` | – | – | refactor | B-06, B-09, B-12, B-13 |
| `api/services/hardware.py` | 85 | Hardware targets, provider selection, and the target-device estimate. | `routes/deploy.py:17`, `services/pipeline.py:30`, `tests/test_api.py:1` | – | Consumes the scaling factors in `profiles/hardware_profiles.json` | refactor | B-07, B-29 |
| `api/services/ranking.py` | 51 | Picks the winner from target estimates rather than host numbers. | `services/pipeline.py:32`, `tests/test_ranking.py:2` | – | Priority semantics | refactor | B-11, B-12 |
| `api/services/uploads.py` | 23 | Suffix checks and size-limited saving of uploaded files. | `routes/analyze.py`, `routes/deploy.py` | – | – | keep | – |
| `api/services/artifacts.py` | 26 | Allocates, finds and expires one-time artifact paths. | `routes/artifacts.py:5`, `routes/deploy.py:16`, `services/pipeline.py:29` | – | – | keep | – |
| `api/services/model_validation.py` | 35 | Rejects models over the node, element or size limits before any work. | `routes/analyze.py:5`, `services/pipeline.py:31` | – | – | keep | – |
| `api/analyzer/model_analyzer.py` | 120 | Static ONNX analysis: validity, size, parameter count, operator counts. | `routes/analyze.py:4`, `services/pipeline.py:11`, `tests/test_model_analyzer.py:5` | – | – | refactor | B-06 |
| `api/optimizer/benchmark_engine.py` | 748 | Measures latency, memory and throughput in a fresh subprocess per measurement. | `optimizer/benchmark_worker.py:12`, `services/hardware.py:10`, `optimizer/onnx_runtime.py:164`, `tests/test_benchmark_engine.py:9` | `BenchmarkResult.hardware_estimate`, computed then overwritten | `MIN_TIMED_RUNS` (:44), `WARMUP_BUDGET_SHARE` (:46), thread settings (:369-370), unseeded inputs (:307, :498) | refactor | B-04, B-07, B-19 |
| `api/optimizer/benchmark_worker.py` | 41 | Subprocess entry point that runs exactly one benchmark. | `benchmark_engine.benchmark`, as a spawned process | – | – | keep | – |
| `api/optimizer/onnx_runtime.py` | 199 | A pass-through layer over the benchmark engine. | `services/pipeline.py:17` | `benchmark_tvm` (:19), `run_inference`, the unreachable `sample_path` branch | – | **delete** | B-06, B-16, B-19 |
| `api/optimizer/quantizer.py` | 225 | Builds each planned candidate: ONNX quantization, and TVM compilation when asked. | `services/pipeline.py:18`, `scripts/calibration_scale_study.py:35`, `scripts/calibration_size_study.py:29`, `tests/test_quantizer.py:7` | – | Per-channel QInt8 with `quant_pre_process` | refactor | B-09, B-11 |
| `api/optimizer/validation.py` | 251 | Loads calibration and validation data, splits it, and scores accuracy for both frameworks. | `config.py:4`, `optimizer/quantizer.py`, `services/pipeline.py:19`, `scripts/calibration_size_study.py:30`, `tests/test_validation.py:6` | – | `MAX_VALIDATION_SAMPLES` (:11), `CALIBRATION_FRACTION` (:16), `MAX_CALIBRATION_SAMPLES` (:17), `SPLIT_SEED` (:20), thread settings (:214-215) | keep | B-01, B-04, B-30 |
| `api/optimizer/tvm_runtime.py` | 132 | Shared TVM Relax helpers: compile an ONNX model, load the VM, run it. | `onnx_runtime.py:37`, `quantizer.py:150` and `:174`, `validation.py:143` (all lazy) | `run_model` | Thread pinning (B-01) | keep | B-19 |
| `api/planner/rule_based.py` | 267 | Turns model analysis into ranked candidate names using fixed rules. | `services/pipeline.py:28`, `tests/test_rule_based.py:1` | Fields nothing reads: `requires_calibration`, `candidate_count`, `planner`, `input` | `SIZE_THRESHOLD_MB` (:33), `LATENCY_THRESHOLD_MS` (:34) | refactor | B-09, B-19 |
| `api/packager/generator.py` | 282 | Builds the self-contained deployment ZIP for the winning candidate. | `services/pipeline.py:27`, `tests/test_packager.py:10` | – | `system_info`, installed versions, `uuid4` and temp paths written into the ZIP (:110) | refactor | B-09, B-24, B-50 |
| `api/static/index.html` | 740 | The static dashboard: upload, configure, deploy, download. | `routes/pages.py:10-11` via `STATIC_INDEX` | – | – | refactor | B-26, B-28, B-29, B-31 |
| `profiles/hardware_profiles.json` | 66 | Per-target size, latency and compute scaling factors plus supported providers. | `benchmark_engine.load_hardware_profiles`, reached via `services/hardware.py:10` | Most metadata fields | The scaling values are the frozen source of truth for every target estimate | refactor | B-03, B-07, B-19, B-31 |
| `frontend/app.py` | 278 | The Streamlit dashboard. | nothing — it is standalone and not imported by the API | – | – | **delete** | B-26, B-27, B-32 |
| `scripts/run_full_benchmark.py` | 140 | Runs the full pipeline for several models and writes the results JSON. | run by hand | – | Produces `reports/data/full_run.json` | keep | B-48 |
| `scripts/quantization_analysis.py` | 162 | Per-model quantization accuracy and size analysis for the results slide. | run by hand | – | Re-implements scoring and latency with different settings from `api/` | keep | B-47 |
| `scripts/calibration_size_study.py` | 88 | Accuracy as a function of calibration set size (20, 50, 256). | run by hand; imported by no script | – | Uses `api.optimizer.quantizer` and `api.optimizer.validation` | keep | B-47 |
| `scripts/calibration_scale_study.py` | 134 | Accuracy as a function of the number of calibration samples. | run by hand; imported by no script | – | Uses `api.optimizer.quantizer` | keep | B-47 |
| `scripts/make_calibration_scale_chart.py` | 107 | Renders the calibration-scale chart and table from the study JSON. | run by hand | – | – | keep | B-44 |
| `scripts/make_results_charts.py` | 375 | Renders the results charts for the slides. | run by hand | – | – | keep | – |
| `scripts/results_tables.py` | 190 | Prints the markdown results tables and renders the summary PNG. | run by hand | – | – | keep | – |
| `scripts/measure_responsiveness.py` | 127 | Measures server responsiveness while a deploy is running. | run by hand | – | Responsiveness claims in the report | keep | – |
| `scripts/download_imagenetv2_large.py` | 128 | Downloads the 1000-image ImageNetV2 subset from Hugging Face. | run by hand | – | Row order from Hugging Face is not checksum-verified | keep | B-44 |
| `scripts/prepare_imagenetv2_subset.py` | 186 | Builds the 20/50/256-image `.npz` subsets. | `scripts/download_imagenetv2_large.py:25`, `tests/test_imagenetv2_subset.py:4` | – | Row order from Hugging Face is not checksum-verified | keep | B-44 |
| `scripts/build_report_odt.py` | 956 | Fills the institute's ODT report template with the EdgeMason report. | run by hand | – | – | keep | B-45 |

## POST /deploy call path

Fourteen steps from the HTTP request to the response. Rebuilt by reading `api/routes/deploy.py` and
`api/services/pipeline.py`; `PHASE0_PLAN.md` section C records the count but not the chain, so the
numbering below is the one that falls out of the code at this granularity.

1. `POST /deploy` enters `deploy_model` (`api/routes/deploy.py:53`).
2. The model upload must be `.onnx` (`api/routes/deploy.py:64`, `api/services/uploads.py:6`).
3. `_validate_form` checks the hardware id, priority, provider and both limits, then resolves the
   provider (`api/routes/deploy.py:65` → `:37-50` → `api/services/hardware.py`).
4. The calibration upload, if present, must be `.npy` or `.npz` (`api/routes/deploy.py:74-79`).
5. The handler takes `DEPLOY_SEMAPHORE` and hands the blocking work to a worker thread
   (`api/routes/deploy.py:108-110`).
6. `cleanup_stale_artifacts` expires anything past its TTL (`api/routes/deploy.py:84` →
   `api/services/artifacts.py:22`).
7. A temp work dir is created; the model, then the calibration file, are saved under their size
   limits (`api/routes/deploy.py:85-92` → `api/services/uploads.py:13`).
8. `run_deploy` is called with a `DeployRequest` carrying the six deploy options
   (`api/routes/deploy.py:93` → `api/services/pipeline.py:175`).
9. `validate_model` rejects the model if it is over a node, element or size limit
   (`api/services/pipeline.py:176` → `api/services/model_validation.py:8`).
10. `model_report` produces the size, parameter count and operator counts, parsed straight back from
    a JSON string (`api/services/pipeline.py:177` → `api/analyzer/model_analyzer.py`).
11. Labelled `.npz` data is split into calibration and test sets with warnings; unlabelled `.npy`
    calibrates only (`api/services/pipeline.py:182-190` → `api/optimizer/validation.py`).
12. `generate_candidates` turns the analysis and the limits into candidate names
    (`api/services/pipeline.py:194` → `api/planner/rule_based.py`).
13. Each candidate is evaluated: `Quantizer` builds it, the `.tar`/`.so` check picks the framework,
    `benchmark` runs it in a subprocess and returns a JSON string, then `evaluate_onnx`/`evaluate_tvm`
    scores accuracy and `estimate_on_target` projects the numbers onto the target
    (`api/services/pipeline.py:204-207` → `:57-122`, via `api/optimizer/quantizer.py:38`,
    `api/optimizer/onnx_runtime.py:164` → `api/optimizer/benchmark_engine.py`,
    `api/optimizer/validation.py`, `api/services/hardware.py`).
14. `rank_results` picks a winner, `create_deployment_package` writes the ZIP, and the response is
    assembled from `_public_result` over every candidate
    (`api/services/pipeline.py:209` → `:219-230` → `:232-260`, via `api/services/ranking.py:2`,
    `api/packager/generator.py`, `api/services/artifacts.py:8`).
