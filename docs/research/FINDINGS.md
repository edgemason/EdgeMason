# EdgeMason findings register

**Spec:** `docs/specs/P0.3-findings-register.md`
**Purpose:** every quantitative claim in `reports/EdgeMason_Report.md` must be traceable as
claim → data file → script → command. This register records the gaps. It does not fix them.

---

## Claims

One row per quantitative claim in the report. Status is exactly one of `sourced` (the number
can be recomputed from the data file), `unsourced` (no data file contains it), `contradicted`
(a data file holds a different value) or `manual` (a hand measurement with no stored data).

| ID | Report line(s) | Claim (quoted, short) | Data file | JSON path / how derived | Producing script + command | Status |
|---|---|---|---|---|---|---|
| F-01 | 39 | Static INT8 51.4% → 57.5% (FP32 58.1%) | `quantization_analysis.json` | `mobilenetv2.onnx.variants.static_int8_per_tensor.top1` = 0.51397; `static_int8_per_channel.top1` = 0.57542; `fp32.top1` = 0.58101 | `scripts/quantization_analysis.py` → `uv run python scripts/quantization_analysis.py --out reports/data/quantization_analysis.json` | sourced |
| F-02 | 41, 185 | 43.0 / 23.5 MB; 255 / 77 MB | `full_run_x86_64_2026-10-05.json` | `models[mobilenetv2].candidates[onnx_fp32_baseline].memory_delta_mb` = 42.98; `[onnx_static_int8].memory_delta_mb` = 23.508; `models[resnet50]…` = 254.973 / 76.629 | `scripts/run_full_benchmark.py` → `uv run python scripts/run_full_benchmark.py --out reports/data/full_run_x86_64.json` | sourced |
| F-03 | 43, 193 | p50 11.9/18.4/14.2 → 13.1/14.3/13.1 ms | `validation_run_2026-10-05.json` | **Report says 13.1/14.3/13.1. Data says `x86_repeats[0].p50.onnx_fp32_baseline` = 13.2727, `[1]` = 13.4102, `[2]` = 12.532. Run 3 winner is `onnx_fp32_baseline`, not INT8.** | No producing script (see F-16) | **contradicted** |
| F-04 | 44, 216–220 | slowest `/health` 31,324 → 13.9 ms; 3 → 115 requests | `validation_responsiveness_2026-10-05.json` | **Report says 31,324→13.9 ms and 3→115 requests. Data says `health_during_deploy.max_ms` = 163.6, `requests` = 92. `dashboard_during_deploy.max_ms` = 70.0, `requests` = 92.** | `scripts/measure_responsiveness.py` → `uv run python scripts/measure_responsiveness.py > reports/data/validation_responsiveness_2026-10-05.json` | **contradicted** |
| F-05 | 45, 160, 480 | Dynamic INT8 2.4× slower than FP32 | `quantization_analysis.json` | `mobilenetv2.onnx.variants.dynamic_int8.latency_p50_ms_1thread` = 30.272; `fp32.latency_p50_ms_1thread` = 12.706; ratio = 2.38 | `scripts/quantization_analysis.py` (as F-01) | sourced |
| F-06 | 45 | 7.6× with QInt8; 53 `DynamicQuantizeLinear` nodes | — | No data file contains a node count or a QInt8 latency for this claim. | — | unsourced |
| F-07a | 46, 163 | Deploy time ~19 s (the "after" figure) | `validation_run_2026-10-05.json` | `deploys[0].seconds` = 18.1, `[1]` = 18.2, `[2]` = 20.8; mean ≈ 19.0 | No producing script (see F-16) | sourced |
| F-07b | 46, 163 | Deploy time ~30 s (the "before" figure) | — | No data file contains a pre-optimization deploy time. | — | unsourced |
| F-08 | 49, 508, 601 | 83 passed, 1 skipped | — | A test run on 2026-10-13, before torch was installed. **Now superseded: the suite gives 101 passed, 0 skipped after P0.4.** | — | manual |
| F-09 | 52–61, 342–346, 559–561 | 3.5–3.9× smaller; within 2.3 points; 1.17–1.75× faster; 0.75× / 0.87×; RAM 1.8–3.6× less | `quantization_analysis.json`, `full_run_x86_64_2026-10-05.json` | Per-model `variants.*.size_mb` ratios and `memory_delta_mb` ratios across all 6 models in both files | `scripts/quantization_analysis.py` + `scripts/run_full_benchmark.py` | sourced |
| F-10 | 58–59, 388–392, 583 | 5–10 calibration images are enough; −6–8 points with fewer | `calibration_size_study.json` | `points[calibration_size=5].test.top1` vs `points[calibration_size=300].test.top1` per model | `scripts/calibration_size_study.py` → `uv run python scripts/calibration_size_study.py --out reports/data/calibration_size_study.json` | sourced |
| F-11 | 61, 488–489 | TVM 21–57× slower | `full_run_x86_64_2026-10-05.json` | `models[resnet152].candidates[tvm_opt_level_3].latency_p50_ms` = 20896.9 / `[onnx_fp32_baseline]` = 368.0 → 56.8×; `models[mobilenetv2]…` = 375.0 / 17.5 → 21.4× | `scripts/run_full_benchmark.py` (as F-02) | sourced |
| F-12 | 186 | Server memory 429 → 524 MB | — | No data file contains server process memory. The report's own text says "the server's own memory grew", which is an observation, not a stored measurement. | — | unsourced |
| F-13 | 194 | ResNet-152 ~16 s instead of ~42 s per candidate | — | `full_run_x86_64_2026-10-05.json` has ResNet-152 candidates but the claim is about a time cap (16 s), not a measured latency. The stored latencies are 368/217/20897/19092 ms. | — | unsourced |
| F-14 | 206, 529 | 12/20 images correct | — | No data file contains a 12/20 accuracy count. | — | unsourced |
| F-15 | 208 | ZIPs of 3.1 / 20.1 / 47.4 MB pass the self-test | `full_run_x86_64_2026-10-05.json` | `models[mobilenetv2].package.zip_mb` = 3.13, `self_test_ok` = true; resnet50 and resnet152 packages also present with `self_test_ok` = true | `scripts/run_full_benchmark.py` (as F-02) | sourced |
| F-16 | 228–235 | Parameter counts and sizes | `full_run_x86_64_2026-10-05.json` | `models[*].parameters`, `models[*].model_size_mb` | `scripts/run_full_benchmark.py` (as F-02) | sourced |
| F-17 | 298–305 | Summary table | `quantization_analysis.json` | All 6 models × 4 variants: top1, top5, latency, size | `scripts/quantization_analysis.py` (as F-01) | sourced |
| F-18 | 310–323 | Pipeline table | `full_run_x86_64_2026-10-05.json` | `models[*].candidates[*]` flattened results | `scripts/run_full_benchmark.py` (as F-02) | sourced |
| F-19 | 333–339 | Target estimates | `validation_run_2026-10-05.json` | `deploys[*].candidates[*].estimated_latency_p50_ms`, `estimated_memory_mb` | No producing script (see F-16) | sourced |
| F-20 | 374–383, 397–398 | Calibration-size table; 63 vs 56%, 56 vs 51% | `calibration_size_study.json` | Per-model `points[*].test.top1` at each calibration size | `scripts/calibration_size_study.py` (as F-10) | sourced |
| F-21 | 417–424 | Scale table | `calibration_scale_study.json` | Per-model `points[*].test.top1` at each calibration size | `scripts/calibration_scale_study.py` → `uv run python scripts/calibration_scale_study.py --data models/samples/imagenetv2_1000.npz --test-size 700 --sizes 30,77,150,225,300 --draws 3 --out reports/data/calibration_scale_study.json` | sourced |
| F-22 | 446–451 | Agreement table | `quantization_analysis.json` | `agreement_fp32_vs_static_int8.{both_right, fp32_right_int8_wrong, fp32_wrong_int8_right, both_wrong, same_top1_prediction}` per model | `scripts/quantization_analysis.py` (as F-01) | sourced |
| F-23 | 464–469 | Ablations | `quantization_analysis.json` | Per-variant top1/top5 comparisons | `scripts/quantization_analysis.py` (as F-01) | sourced |
| F-24 | 518 | `/health` 5–9 ms during the Chrome dashboard run | — | A browser observation with no stored data. | — | manual |
| F-25 | 524 | "11 deploys + 3 repeats" | `validation_run_2026-10-05.json` | **Report says 11 deploys. Data says `deploys` has length 8.** `x86_repeats` has length 3. | No producing script (see F-16) | **contradicted** |
| F-26 | 526 | 92/92 requests, 164 ms | `validation_responsiveness_2026-10-05.json` | `health_during_deploy.requests` = 92, `max_ms` = 163.6 → rounds to 164 | `scripts/measure_responsiveness.py` (as F-04) | sourced |
| F-27 | 544 | ~632 MB of CUDA libraries | — | No data file contains a CUDA library size. The report says "on CUDA, memory includes the CUDA libraries (~632 MB)" — this was true when CUDA was in use but is now moot under D1 (CPU-only). | — | unsourced |
| F-28 | 602–604 | Run times of ~70, ~10 and ~17 min | — | The report's reproduction table lists approximate wall-clock times. No data file stores these; they are observations from the runs that produced the JSONs. | — | unsourced |

### Additional claims found beyond the deep-dive list

| ID | Report line(s) | Claim (quoted, short) | Data file | JSON path / how derived | Producing script + command | Status |
|---|---|---|---|---|---|---|
| F-29 | 170–171 | Calibration is capped at 300 images | `validation_run_2026-10-05.json` | `deploys[*].candidates[*].test_samples` ≤ 179; the cap is in code at `api/optimizer/validation.py:17` (`MAX_CALIBRATION_SAMPLES = 300`) | Code constant, not a data file | sourced |

---

## Data provenance

Every file in `reports/data/` and `reports/img/results/`. Status is `clear` (the producing script
and command are known) or `unclear` (the producer cannot be determined from the code).

| Data file or chart | Producing script | Command | Inputs | Status |
|---|---|---|---|---|
| `quantization_analysis.json` | `scripts/quantization_analysis.py` | `uv run python scripts/quantization_analysis.py --out reports/data/quantization_analysis.json` | `models/samples/imagenetv2_mobilenetv2_256.npz` (77 calibration / 179 test) | clear |
| `calibration_size_study.json` | `scripts/calibration_size_study.py` | `uv run python scripts/calibration_size_study.py --out reports/data/calibration_size_study.json` | `models/samples/imagenetv2_mobilenetv2_256.npz` | clear |
| `calibration_scale_study.json` | `scripts/calibration_scale_study.py` | `uv run python scripts/calibration_scale_study.py --data models/samples/imagenetv2_1000.npz --test-size 700 --sizes 30,77,150,225,300 --draws 3 --out reports/data/calibration_scale_study.json` | `models/samples/imagenetv2_1000.npz` | clear |
| `calibration_scale_study_256.json` | `scripts/calibration_scale_study.py` | `uv run python scripts/calibration_scale_study.py --data models/samples/imagenetv2_mobilenetv2_256.npz --test-size 156 --sizes 15,30,45,60,77 --draws 2 --out reports/data/calibration_scale_study_256.json` | `models/samples/imagenetv2_mobilenetv2_256.npz` | clear (reconstructed — arguments recovered from stored `test_samples` = 156, `calibration_size` values 15/30/45/60/77, and `draw` count = 2) |
| `full_run_x86_64_2026-10-05.json` | `scripts/run_full_benchmark.py` | `uv run python scripts/run_full_benchmark.py --out reports/data/full_run_x86_64.json` then `mv reports/data/full_run_x86_64.json reports/data/full_run_x86_64_2026-10-05.json` | `models/mobilenetv2.onnx`, `models/resnet50.onnx`, `models/resnet152.onnx`, `models/samples/imagenetv2_mobilenetv2_256.npz` | clear (script writes undated filename; the dated file was renamed by hand) |
| `validation_run_2026-10-05.json` | — | — | — | **unclear: no script produces this file.** Contains 8 deploys plus `x86_repeats`. The deep-dive flagged this as the main research-integrity risk. |
| `validation_responsiveness_2026-10-05.json` | `scripts/measure_responsiveness.py` | `uv run python scripts/measure_responsiveness.py > reports/data/validation_responsiveness_2026-10-05.json` | `models/mobilenetv2.onnx`, `models/samples/imagenetv2_mobilenetv2_256.npz` | clear (stdout redirect; the script has no `--out` argument) |
| `img/results/testing_accuracy_top1.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/full_run_x86_64_2026-10-05.json` | clear |
| `img/results/latency_p50.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/full_run_x86_64_2026-10-05.json` | clear |
| `img/results/size_and_memory.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/full_run_x86_64_2026-10-05.json` | clear |
| `img/results/agreement_matrices.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/quantization_analysis.json` | clear |
| `img/results/ablations.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/quantization_analysis.json` | clear |
| `img/results/target_latency_estimates.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/validation_run_2026-10-05.json` | clear |
| `img/results/calibration_size_accuracy.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/calibration_size_study.json` | clear |
| `img/results/calibration_vs_test_accuracy.png` | `scripts/make_results_charts.py` | `uv run python scripts/make_results_charts.py` | `reports/data/calibration_size_study.json` | clear |
| `img/results/summary_table.png` | `scripts/results_tables.py` | `uv run python scripts/results_tables.py` | `reports/data/quantization_analysis.json` | clear |
| `img/results/calibration_scale.png` | `scripts/make_calibration_scale_chart.py` | `uv run python scripts/make_calibration_scale_chart.py` | `reports/data/calibration_scale_study.json` | clear |
| `img/results/calibration_scale_256.png` | `scripts/make_calibration_scale_chart.py` | `uv run python scripts/make_calibration_scale_chart.py --data reports/data/calibration_scale_study_256.json --out reports/img/results/calibration_scale_256.png` | `reports/data/calibration_scale_study_256.json` | clear (reconstructed — same script, `--data` pointing at the 256 variant) |

---

## Measurement method notes

These differences are intentional until a spec decides otherwise. Any paper text must say which
code path produced each number.

1. **Accuracy scoring: all cores vs 1 thread.**
   `scripts/quantization_analysis.py:41-45` — `_session(path, threads=0)` leaves ONNX Runtime at
   its default (all cores). The API's `evaluate_onnx` pins to 1 thread at
   `api/optimizer/validation.py:214-215` (`intra_op_num_threads = 1`). The script's own docstring
   at `:9` mentions only latency, which *does* use 1 thread (`:69`, `threads=1`). This asymmetry
   is easy to miss: the docstring is about latency, not accuracy.

2. **Latency: in-process fixed runs vs subprocess under a time budget.**
   `scripts/quantization_analysis.py:68-74` — `latency_p50` runs `for _ in range(10)` warm-up
   then `for _ in range(50)` timed, all in the same process, no subprocess, no time budget.
   The API runs a fresh subprocess per measurement under a 10 s budget
   (`api/config.py:24-26`, `api/optimizer/benchmark_worker.py`).

3. **Per-tensor INT8.**
   `scripts/quantization_analysis.py:100` — `create_static_int8(model_path, split.calibration)`
   with `activation_type=QuantType.QInt8, per_channel=False`. The API's quantizer
   (`api/optimizer/quantizer.py`) uses per-channel by default.

4. **Calibration split: own seeded split vs the API's 30% split.**
   `scripts/calibration_size_study.py:42` — `np.random.default_rng(0).permutation(len(data.inputs))`
   creates the split. `:53` — per-draw subsets use `default_rng(1000 + draw)`. The API uses
   `CALIBRATION_FRACTION = 0.3` and `SPLIT_SEED = 0`
   (`api/optimizer/validation.py:16-20`). The studies' split is seeded differently and uses a
   different fraction.
