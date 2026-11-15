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

---

## P2.4 measurement (2026-11-14): static INT8 now uses integer Conv kernels

Host: Intel Core i7-13620H (`x86_64`, `avx2` + `avx_vnni`), ONNX Runtime 1.27.0, CPU EP.
Model: `squeezenet1.1-13.onnx`. Data: `imagenetv2_mobilenetv2_256.npz`, split by
`split_calibration_and_test` (`CALIBRATION_FRACTION=0.3`, `SPLIT_SEED=0`) into 77 calibration and
179 test images.

Shipped config: `weight_type=QInt8`, `activation_type=QUInt8`,
`quant_format=QuantFormat.QOperator`, `per_channel=True`, `reduce_range=False`.

| Measure | FP32 | Static INT8 (new) | Change |
|---|---|---|---|
| Conv kernels in saved graph | 26 `Conv` | 26 `QLinearConv` | all integer |
| Conv kernel op types in ORT trace | 26 `Conv` | 26 `QLinearConv` | all integer |
| Sum of Conv kernel time | 6079.7 µs | 2272.7 µs | **−62.6%** (0.374×) |
| `squeezenet0_conv0_fwd` kernel time | 554.7 µs | 536.0 µs | −3.4% (input-bound outlier) |
| top-1 (179 test images) | 0.363128 | 0.363128 | **0.000000** |
| top-5 (179 test images) | 0.636872 | 0.620112 | −0.016760 |
| end-to-end p50, 1 thread | 5.240 ms | 2.139 ms | **−59.2%** |
| model size | 4.959 MB | 1.295 MB | −73.9% |

**Tolerance:** the recorded claim is 2.3 top-1 points (`reports/EdgeMason_Report.md:52`, sourced by
F-09 from `reports/data/quantization_analysis.json`). P2.4's drop is **0.000**, i.e. the new INT8
model matches FP32 top-1 on this split.

**What made the difference.** A config matrix on the same model/split showed the activation type,
not `reduce_range`, controls fusion:

| Config | Conv kernels after ORT optimisation | top-1 |
|---|---|---|
| `QInt8` + QDQ + per-channel (old shipped) | 2 `QLinearConv`, 24 fp32 `Conv` | 0.340782 |
| `QUInt8` + QDQ + per-channel | 26 `QLinearConv` | 0.346369 |
| `QInt8` + QOperator + per-channel | 26 `QLinearConv` (slow: `conv0` ≈ 3.1 ms) | — |
| **`QUInt8` + QOperator + per-channel** | **26 `QLinearConv`** | **0.363128** |
| `QUInt8` + QOperator + `reduce_range=True` | 26 `QLinearConv` (no speed gain) | 0.335196 |

`reduce_range=True` does not improve speed here and costs almost three top-1 points, so it stays
off. `QOperator` is preferred over QDQ because the integer kernel is present even when ORT graph
optimisations are disabled (the setting the P2.3 fetcher needs to see every node).

**Commands.** Kernel times come from the P2.3 fetcher on optimisation-disabled traces (1 warm-up +
3 timed runs, 1 thread), with a random input; `conv0` is input-bound, so the 26-kernel sum is the
meaningful comparison. End-to-end p50 uses `scripts/quantization_analysis.py`'s method (10 warm-up
+ 50 timed runs, 1 thread, one real test image). Accuracy uses `api.optimizer.validation.evaluate_onnx`
on the same 179-image test split as `reports/data/quantization_analysis.json`.

---

## P2.6 sweep (2026-11-15): 38 mixed-precision configs, additive model with a sanity gate

Host: Intel Core i7-13620H (`x86_64`, `avx2` + `avx_vnni`), ONNX Runtime 1.27.0, CPU EP, 1 intra-op
and 1 inter-op thread. Model: `squeezenet1.1-13.fp32.onnx` (39 excludable layers). Calibration: 8
random `224×224` images (`float32`, seed 0), matching the P2.5 generator and P2.3 fetch fixtures.
Sweep: `seed=0`, `target=40`, 5 warm-up + 30 timed runs, 2 s budget per model, isolated (fresh
process) measurement.

**Method.** Two pure models (fp32, static int8) are built once. Each is profiled for per-layer
costs and then measured with the shared benchmark protocol; the lookup is accepted only if each
pure config's additive sum is within 15% of its measurement. That gate fails a lookup profiled in
a different optimization regime, which is what happened in the first pass of this spec (see
Amendment A in `docs/specs/P2.6-sweep-runner.md`). Mixed models are then predicted as
`sum(fp32_us for layers kept in fp32) + sum(int8_us for the rest)`; nothing per-config is fetched
except the structural seam count.

| Pure config | Additive sum (µs) | Measured p50 (µs) | Relative error | Gate |
|---|---:|---:|---:|---|
| fp32 | 5170.8 | 5141.3 | 0.57% | pass |
| static int8 | 2096.3 | 2066.3 | 1.45% | pass |

Regime recorded: `optimized` (`ORT_ENABLE_ALL`, ONNX Runtime's default — the level
`BenchmarkEngine` leaves in place). No calibration rescale was needed on this run.

**Seam count is structural.** The spec's original plan counted boundary nodes from each config's
own profile. That is not possible under the benchmark regime: optimization renames, folds and
duplicates Q/DQ nodes (`squeezenet0_relu0_fwd_q`, `/duplicated` DequantizeLinear) and inserts
layout bookkeeping (`ReorderInput`, `ReorderOutput`), so the P2.3 fetcher cannot attribute them
against the pre-optimization graph — 35 of 38 configs raised on the first amended run. Since a
seam is a property of the graph and not of the timing, the count is taken from the **optimized**
graph the benchmark actually executes, as Q/DQ nodes beyond the all-int8 baseline (which itself
has 2: the model-input `QuantizeLinear` and model-output `DequantizeLinear`). This is the one
deviation from the spec's letter; no timing is read from per-config profiles.

**Agreement.** All 38 configs completed (`status: ok`, 0 failed).

| Subset | n | MAPE | Median relative error | Median residual (measured − predicted) | `corr(pred, p50)` |
|---|---:|---:|---:|---:|---:|
| All configs | 38 | 24.6% | 11.5% | +410 µs | +0.395 |
| `conv0` kept fp32 ("head") | 20 | 41.0% | 46.1% | +2724 µs | +0.124 |
| `conv0` quantized ("no-head") | 18 | 6.4% | 4.9% | +154 µs | +0.920 |

Least-squares fits on measured p50 (all configs): count + seams gives 50 µs per fp32 layer and
99 µs per seam (R² = 0.419); head alone gives +2638 µs (R² = 0.599); all three give 48 µs/layer,
31 µs/seam and +2212 µs for head (R² = 0.734). The two extremes bracket correctly: `cfg_000`
(all int8, 0 seams) measures 2053 µs against 2098 predicted (−2.1%) and `cfg_037` (all fp32,
0 seams) measures 5407 µs against 5092 predicted (+6.2%).

**Finding: seam count is not the dominant term; placement of `conv0` is.** Every config where
`squeezenet0_conv0_fwd` is quantized lands within ~6% of the additive prediction, with only 0–4
seams and no systematic gap. Every config that keeps `conv0` in fp32 pays a large fixed penalty
that the additive model does not represent: the low-seam head configs (`cfg_002`–`cfg_014`, 1–4
seams) all measure 5604–5725 µs regardless of whether 2 or 14 fp32 layers are kept. Keeping `conv0`
fp32 forces a layout conversion around the NCHWc
conv kernel, which ORT inserts as `ReorderInput`/`ReorderOutput`; the cost is in the reorders, not
in the layer's own kernel time, so the additive sum cannot see it. Seam count still adds ~30–100 µs
per boundary, but placement of the head conv dominates the residual.

**Register.** Every config's row from `tmp/p2.6-sweep/register.jsonl` (gitignored; this table is the
durable copy). `fp32 layers` is the number of the 39 excludable layers kept in fp32; `seams` is the
extra Q/DQ boundary count over the all-int8 model.

| config | fp32 layers | seams | measured p50 (µs) | measured p90 (µs) | additive pred (µs) | status |
|---|---:|---:|---:|---:|---:|---|
| `cfg_000` | 0 | 0 | 2053 | 2094 | 2098 | ok |
| `cfg_001` | 2 | 3 | 2139 | 2192 | 2144 | ok |
| `cfg_002` | 2 | 1 | 5638 | 5759 | 1980 | ok |
| `cfg_003` | 4 | 4 | 2420 | 2477 | 2288 | ok |
| `cfg_004` | 4 | 1 | 5623 | 5807 | 2244 | ok |
| `cfg_005` | 6 | 3 | 4197 | 4339 | 2505 | ok |
| `cfg_006` | 6 | 1 | 5647 | 5807 | 2506 | ok |
| `cfg_007` | 8 | 3 | 2661 | 2688 | 2544 | ok |
| `cfg_008` | 8 | 1 | 5604 | 5767 | 2192 | ok |
| `cfg_009` | 10 | 4 | 2582 | 2626 | 2591 | ok |
| `cfg_010` | 10 | 1 | 5661 | 5794 | 3458 | ok |
| `cfg_011` | 12 | 3 | 2811 | 2874 | 2776 | ok |
| `cfg_012` | 12 | 1 | 5725 | 5865 | 2354 | ok |
| `cfg_013` | 14 | 4 | 2996 | 3039 | 2920 | ok |
| `cfg_014` | 14 | 4 | 5699 | 5934 | 2398 | ok |
| `cfg_015` | 16 | 3 | 3168 | 3252 | 2957 | ok |
| `cfg_016` | 16 | 17 | 6479 | 6634 | 2890 | ok |
| `cfg_017` | 18 | 2 | 3298 | 3352 | 3122 | ok |
| `cfg_018` | 18 | 20 | 3382 | 3609 | 3854 | ok |
| `cfg_019` | 21 | 2 | 3434 | 3556 | 3370 | ok |
| `cfg_020` | 21 | 23 | 6572 | 6788 | 4615 | ok |
| `cfg_021` | 23 | 2 | 3716 | 3929 | 3473 | ok |
| `cfg_022` | 23 | 27 | 5788 | 5940 | 3610 | ok |
| `cfg_023` | 25 | 3 | 3998 | 4236 | 3682 | ok |
| `cfg_024` | 25 | 19 | 6173 | 6268 | 4316 | ok |
| `cfg_025` | 27 | 2 | 4042 | 4137 | 3779 | ok |
| `cfg_026` | 27 | 24 | 8918 | 9164 | 3386 | ok |
| `cfg_027` | 29 | 3 | 4450 | 4544 | 4034 | ok |
| `cfg_028` | 29 | 22 | 7576 | 7693 | 3511 | ok |
| `cfg_029` | 31 | 2 | 4433 | 4657 | 4088 | ok |
| `cfg_030` | 31 | 18 | 8089 | 8205 | 3786 | ok |
| `cfg_031` | 33 | 1 | 5106 | 5384 | 4886 | ok |
| `cfg_032` | 33 | 14 | 7199 | 7483 | 4891 | ok |
| `cfg_033` | 35 | 2 | 4651 | 4867 | 4248 | ok |
| `cfg_034` | 35 | 11 | 5816 | 5946 | 5026 | ok |
| `cfg_035` | 37 | 1 | 5309 | 5438 | 5192 | ok |
| `cfg_036` | 37 | 5 | 6899 | 7043 | 5060 | ok |
| `cfg_037` | 39 | 0 | 5407 | 5521 | 5092 | ok |
