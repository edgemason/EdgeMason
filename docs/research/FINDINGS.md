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
fp32 forces an unfused fp32 `Conv` plus a layout conversion (`ReorderInput`/`ReorderOutput`) around
the int8 neighbours. But the penalty is **not** explained by reorder count: counting ops in the
saved optimized graphs (`tmp/p2.6-sweep/work-amended/*.optimized.onnx`), `cfg_002` (only `conv0`
fp32) has **1** reorder and a −65% miss, while `cfg_013` (14 mid-layers fp32, `conv0` int8) has
**4** reorders and only −3%. The distinguishing factor is that `conv0` sits at the network's
**highest-resolution feature map**, so its unfused fp32 kernel and the reorder of its large output
both act on the biggest activations in the net; a reorder deeper in the network is cheap. Seam
count still adds ~30–100 µs per boundary, but placement of `conv0` at the input resolution — not
reorder or seam count — dominates the residual.

Op counts backing this (optimized graphs ORT executes): `cfg_000` all-int8 = 26 `QLinearConv`,
0 reorders; `cfg_002` conv0-fp32 = 1 `Conv` + 25 `QLinearConv` + 1 `ReorderOutput`; `cfg_013`
= 10 `Conv` + 16 `QLinearConv` + 4 reorders; `cfg_037` all-fp32 = 26 `Conv` + 1 `ReorderOutput`.

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

---

## P2.7 cost model (2026-11-15): seam/stranded byte terms do not beat additive out-of-sample

**Result: no-win.** On a 150-config decorrelated sweep, the spec's graph-aware cost model scores
**25.4%** out-of-sample MAPE against the additive baseline's **21.2%** (5-fold CV, seed 0, all 150
configs, 0 failed). The two geometry features are real and deterministic, but the measured residual
is not monotone in either of them, so a positive linear per-element correction cannot model it and
the fit places `alpha` near zero. This section reports the comparison as required by spec
§6; no feature was added, no config was dropped, and no seed was changed after seeing it.

Host: Intel Core i7-13620H (`x86_64`), ONNX Runtime 1.27.0, CPU EP, 1 intra-op and 1 inter-op
thread. Model: `squeezenet1.1-13.fp32.onnx` (65 excludable pre-processed nodes; 39 survive the
lookup intersection). Calibration: the same 8 random `224×224` images (`float32`, seed 0) as P2.6.
Sweep: `seed=0`, `target=150`, 10 warm-up + 100 timed runs, 2 s budget per model, isolated (fresh
process) measurement. (P2.6 used 5 + 30; the longer run was chosen to keep the larger register's
per-row noise down, and its effect is quantified below.)

**Features.** One `onnx.shape_inference.infer_shapes` pass on the base model gives 121 activation
shapes and 66 layer-output tensors; all 65 excludable layers resolve, so no shape was fabricated
(a missing shape would raise). Per config:

| Feature | Source | Definition used |
|---|---|---|
| `additive_us` | P2.6 gated lookup | `sum(fp32_us for fp32 layers) + sum(int8_us for the rest)` |
| `seam_bytes` | P2.3 boundary table, profiled with `ORT_DISABLE_ALL` so Q/DQ keep authored names; timing never read from it | Σ elements of the tensors of the boundary entries a config adds over the all-int8 baseline (model-input `Q` / model-output `DQ` cancel out) |
| `stranded_fp32_bytes` | base-model shape map + built mixed graph | Σ output elements of configured fp32 layers whose fp32 **region** touches an int8 node. The region (not direct adjacency) is used because ORT leaves `relu0`/`pool0` fp32 after an fp32 `conv0`; without the region, the `conv0` effect would not be counted at all |

Model: `predicted = additive + alpha·seam_bytes + beta·stranded_fp32_bytes`, no intercept; `alpha`,
`beta` by ordinary least squares on `measured_p50 − additive`. Validation: 5-fold CV, seed 0,
predictions pooled over held-out folds before computing MAPE.

**Out-of-sample result.** Fitted on all 150 configs: `alpha = −1.83e−5 µs/element`,
`beta = 4.50e−4 µs/element`.

| Predictor (5-fold CV, seed 0) | Out-of-sample MAPE |
|---|---:|
| Additive (P2.6 baseline) | **21.2%** |
| Spec model: additive + α·seam + β·stranded | 25.4% |
| Spec model + intercept γ (optional variant, §2) | 25.0% |
| Seam term only, no intercept | 26.2% |
| Stranded term only, no intercept | 25.4% |

The two-feature model reduces to the stranded-only model because `alpha` fits at ~0 (and negative
overall). The intercept variant does not rescue it (`gamma = −183.5 µs`, `alpha = 5.86e−5`,
`beta = 4.88e−4`). No variant beats additive.

**Fold-to-fold coefficients** (fit on 120, test on 30):

| fold | α (µs/element) | β (µs/element) | additive MAPE | cost-model MAPE |
|---:|---:|---:|---:|---:|
| 0 | +5.11e−5 | 4.48e−4 | 19.3% | 24.7% |
| 1 | +2.95e−6 | 4.10e−4 | 24.7% | 25.4% |
| 2 | −2.06e−5 | 3.93e−4 | 25.5% | 27.5% |
| 3 | −5.74e−5 | 4.59e−4 | 18.3% | 22.1% |
| 4 | −8.86e−5 | 5.70e−4 | 18.2% | 27.5% |
| **all 150** | **−1.83e−5** | **4.50e−4** | **21.2%** | **25.4%** |

`alpha` changes sign across folds (range −8.86e−5 … +5.11e−5, std 4.83e−5, larger than its full-fit
magnitude of 1.83e−5); `beta` keeps its sign but moves over 3.93e−4 … 5.70e−4 (std 6.17e−5). Per
spec §6 that is the "coefficients swing ⇒ features collinear or data too small" flag, reported
rather than hidden. The cost model loses in **all 5 folds**, from a 0.7-point gap in fold 1 to a
9.3-point gap in fold 4.

**Why the byte terms fail: the residual is not a function of summed boundary bytes.**
Across the 150 rows, `corr(measured − additive, seam_bytes) = 0.078` and
`corr(measured − additive, stranded_fp32_bytes) = 0.324`. The `conv0` penalty is bimodal and
*anti*-monotone in the features:

- **8 configs keep `conv0` fp32 with ≤4 fp32 layers total** (an isolated high-resolution island
  ending at `pool0`): every one measures 5,659–6,086 µs, residual **+3,128 … +3,853 µs** — even
  though their `seam_bytes` span 193,600–1,448,512 (7.5×) and `stranded_fp32_bytes`
  788,544–1,756,544 (2.2×). Within this family latency is flat while the features vary by an order
  of magnitude.
- The **7 configs that keep `conv0` fp32 and still measure < 3,500 µs** have *larger* median
  features than that slow island family (seam 1,076,568 vs 808,848; stranded 1,644,416 vs
  1,172,256): more boundary bytes, less latency.
- The cleanest pair is `cfg_025` vs `cfg_026`: both keep 10 layers fp32 and both include `conv0`,
  but `cfg_025` also keeps the adjacent `conv1` (the 55×55 squeeze) and stays fast — **2,160 µs**,
  residual −1,147 — while `cfg_026` does not and **6,341 µs**, residual +3,397. `cfg_025` has the
  *larger* feature values (seam 1,246,360 vs 1,119,424; stranded 1,563,192 vs 1,521,856), so any
  positive linear weight ranks them backwards.

This refines P2.6 rather than contradicting it. Placement still dominates, but the operative
quantity is not a sum of boundary elements: it is whether an fp32 island ends on a large
high-resolution tensor (`pool0`, 55×55×64) instead of continuing through the adjacent squeeze
(`conv1`, whose output is 4× smaller). Summed elements across every boundary also add deep,
low-resolution seams that are nearly free, diluting the one boundary that matters. To carry the
effect, the model would need the size of the tensor the fp32 region actually ends on (or a
per-island fixed cost / topology indicator) — a different feature set, which spec §6 forbids adding
after seeing the result. For scale: the observed `conv0`-only residual of +3,853 µs would require
`beta ≈ 4.9e−3 µs/element`, **11×** the fitted 4.50e−4; the fitted model corrects `cfg_002` by only
+351 µs.

**Robustness to measurement noise.** The identical sweep (same seed, same sampler output, same
protocol) was run twice; all 150 configs were reproduced exactly and per-config measured p50
differed by 3.0% median / 8.3% p90 / 17.6% max, while the re-measured additive prediction differed
by 8.2% median (per-layer profiling noise accumulates across the sum). Repeating the CV on the
second register and on the per-config mean of the two gives:

| Register | Additive MAPE | Cost-model MAPE | Full-fit α | Full-fit β |
|---|---:|---:|---:|---:|
| Primary (reported above) | 21.2% | 25.4% | −1.83e−5 | 4.50e−4 |
| Repeat | 20.9% | 24.1% | +8.64e−5 | 5.30e−4 |
| Mean of the two | 20.4% | 24.5% | +3.41e−5 | 4.91e−4 |

The no-win is stable across all three, and `alpha` even changes sign between sweeps. Noise inflates
both models' error but does not explain the gap: the `cfg_025`/`cfg_026` difference is deterministic
graph geometry, and an independent rebuild plus benchmark reproduced the two extremes (`cfg_002`:
5,545 µs; `cfg_025`: 2,138 µs, against sweep values 6,075 and 2,160).

**Integrity statement.** Single pre-declared analysis: seed 0, `k=5`, all 150 rows, 0 dropped, no
features added, no re-seed. The cost model does not beat additive out-of-sample; that is the
finding. A model that adds a topology term would be a new spec/block, not a patch to this one.

**Reproduce** (from the worktree root; `tmp/` is gitignored):

```bash
uv run python - <<'PY'
import numpy as np
from api.optimizer import cost_model, sweep_runner

calibration = np.random.default_rng(0).random((8, 3, 224, 224), dtype=np.float32)
rows = sweep_runner.run_sweep(
    "tests/fixtures/fetcher/squeezenet1.1-13.fp32.onnx",
    calibration,
    register_path="tmp/p2.7-sweep/register.jsonl",
    seed=0, target=150, work_dir="tmp/p2.7-sweep/work",
    warmup_runs=10, benchmark_runs=100, time_budget_seconds=2.0, isolated=True,
)
report = cost_model.cross_validate(rows, k=5, seed=0)
cost_model.write_report(report, "tmp/p2.7-sweep/cost_model_report.json")
print(report["out_of_sample"], report["coefficients"], report["stability"])
PY
```

**Register.** Every config's row from `tmp/p2.7-sweep/register.jsonl` (gitignored; this table is the
durable copy). `fp32 layers` is the number of excludable layers kept fp32; `seams` is the structural
Q/DQ count over the all-int8 model (P2.6 method); `seam bytes` and `stranded bytes` are the P2.7
geometry features. All 150 rows are `status: ok`.

| config | fp32 layers | conv0 fp32 | seams | seam bytes | stranded bytes | p50 (µs) | p90 (µs) | additive (µs) | status |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| `cfg_000` | 0 | no | 0 | 0 | 0 | 2264 | 2348 | 2345 | ok |
| `cfg_001` | 39 | yes | 0 | 0 | 0 | 5713 | 5994 | 5882 | ok |
| `cfg_002` | 1 | yes | 1 | 193,600 | 788,544 | 6075 | 6327 | 2222 | ok |
| `cfg_003` | 2 | yes | 1 | 968,000 | 1,175,744 | 5697 | 5830 | 2238 | ok |
| `cfg_004` | 2 | yes | 1 | 566,848 | 975,168 | 5659 | 5827 | 2226 | ok |
| `cfg_005` | 3 | yes | 1 | 1,448,512 | 1,562,944 | 5694 | 6091 | 2254 | ok |
| `cfg_006` | 3 | yes | 1 | 808,848 | 1,168,768 | 5668 | 5983 | 2242 | ok |
| `cfg_007` | 4 | yes | 1 | 1,448,512 | 1,756,544 | 5846 | 6022 | 2298 | ok |
| `cfg_008` | 4 | yes | 1 | 808,848 | 1,362,368 | 5666 | 5794 | 2538 | ok |
| `cfg_009` | 8 | no | 0 | 665,560 | 387,448 | 2335 | 2394 | 3414 | ok |
| `cfg_010` | 8 | no | 3 | 18,928 | 278,512 | 2560 | 2666 | 2756 | ok |
| `cfg_011` | 15 | no | 12 | 1,947,904 | 746,096 | 4180 | 4348 | 4645 | ok |
| `cfg_012` | 15 | no | 3 | 209,560 | 771,992 | 4335 | 4477 | 4397 | ok |
| `cfg_013` | 22 | no | 19 | 1,805,704 | 960,784 | 4424 | 4640 | 3818 | ok |
| `cfg_014` | 22 | no | 6 | 1,524,576 | 1,213,056 | 3857 | 3997 | 3990 | ok |
| `cfg_015` | 30 | no | 4 | 1,438,048 | 1,654,456 | 5280 | 5528 | 5349 | ok |
| `cfg_016` | 8 | no | 3 | 18,928 | 324,480 | 2675 | 2776 | 2916 | ok |
| `cfg_017` | 15 | no | 1 | 64,896 | 676,648 | 4007 | 4154 | 4255 | ok |
| `cfg_018` | 22 | no | 2 | 355,624 | 1,252,392 | 4489 | 4618 | 4826 | ok |
| `cfg_019` | 9 | yes | 1 | 471,960 | 1,175,992 | 2142 | 2184 | 3291 | ok |
| `cfg_020` | 9 | yes | 5 | 420,784 | 1,196,896 | 5905 | 6154 | 2478 | ok |
| `cfg_021` | 16 | yes | 13 | 1,754,304 | 1,534,640 | 4278 | 4521 | 4522 | ok |
| `cfg_022` | 16 | yes | 2 | 258,496 | 1,465,192 | 7738 | 7966 | 4132 | ok |
| `cfg_023` | 23 | yes | 20 | 1,612,104 | 1,749,328 | 4356 | 4579 | 3695 | ok |
| `cfg_024` | 23 | yes | 3 | 496,864 | 1,965,248 | 7504 | 7723 | 4003 | ok |
| `cfg_025` | 10 | yes | 1 | 1,246,360 | 1,563,192 | 2160 | 2212 | 3307 | ok |
| `cfg_026` | 10 | yes | 5 | 1,119,424 | 1,521,856 | 6341 | 6609 | 2944 | ok |
| `cfg_027` | 17 | yes | 13 | 2,528,704 | 1,921,840 | 4092 | 4180 | 4538 | ok |
| `cfg_028` | 17 | yes | 4 | 1,208,704 | 1,945,472 | 6474 | 6591 | 3092 | ok |
| `cfg_029` | 24 | yes | 23 | 1,612,104 | 2,136,528 | 4500 | 4595 | 3710 | ok |
| `cfg_030` | 24 | yes | 3 | 1,084,640 | 2,313,496 | 7982 | 8511 | 4636 | ok |
| `cfg_031` | 11 | yes | 1 | 1,726,872 | 1,950,392 | 2208 | 2301 | 3323 | ok |
| `cfg_032` | 11 | yes | 3 | 1,853,200 | 2,197,520 | 6295 | 6724 | 2986 | ok |
| `cfg_033` | 18 | yes | 13 | 2,528,704 | 2,309,040 | 4090 | 4240 | 4554 | ok |
| `cfg_034` | 18 | yes | 4 | 1,645,952 | 2,263,328 | 6793 | 7112 | 3066 | ok |
| `cfg_035` | 25 | yes | 24 | 1,905,992 | 2,523,728 | 4613 | 4936 | 3726 | ok |
| `cfg_036` | 25 | yes | 7 | 1,739,264 | 2,776,544 | 7362 | 7695 | 3814 | ok |
| `cfg_037` | 9 | no | 7 | 1,352,808 | 1,151,336 | 2960 | 3112 | 3874 | ok |
| `cfg_038` | 7 | no | 5 | 1,291,280 | 840,192 | 2665 | 2767 | 2984 | ok |
| `cfg_039` | 24 | no | 19 | 2,344,032 | 1,946,768 | 4659 | 4823 | 5011 | ok |
| `cfg_040` | 37 | yes | 5 | 246,432 | 4,213,816 | 5574 | 5949 | 5858 | ok |
| `cfg_041` | 26 | no | 5 | 1,405,488 | 1,366,408 | 4724 | 4835 | 4988 | ok |
| `cfg_042` | 19 | no | 21 | 1,294,608 | 1,064,560 | 4136 | 4250 | 4020 | ok |
| `cfg_043` | 11 | yes | 7 | 1,216,144 | 1,382,384 | 6351 | 6587 | 3518 | ok |
| `cfg_044` | 23 | no | 14 | 1,746,304 | 1,383,056 | 4905 | 5271 | 4863 | ok |
| `cfg_045` | 7 | no | 6 | 300,144 | 367,744 | 2839 | 2930 | 2810 | ok |
| `cfg_046` | 27 | no | 13 | 1,630,032 | 1,567,928 | 4818 | 5007 | 5246 | ok |
| `cfg_047` | 7 | no | 3 | 446,304 | 1,036,736 | 4384 | 4512 | 3146 | ok |
| `cfg_048` | 2 | no | 0 | 358,640 | 141,712 | 2304 | 2353 | 2680 | ok |
| `cfg_049` | 35 | yes | 5 | 173,056 | 4,081,080 | 5457 | 5591 | 5488 | ok |
| `cfg_050` | 3 | no | 0 | 286,992 | 112,240 | 2250 | 2322 | 2473 | ok |
| `cfg_051` | 19 | no | 15 | 1,633,456 | 1,269,440 | 4463 | 4644 | 3965 | ok |
| `cfg_052` | 9 | no | 3 | 848,144 | 467,840 | 2381 | 2439 | 3075 | ok |
| `cfg_053` | 25 | no | 12 | 1,799,032 | 1,304,288 | 4609 | 4837 | 4048 | ok |
| `cfg_054` | 26 | no | 10 | 1,567,840 | 1,492,216 | 4905 | 5160 | 5272 | ok |
| `cfg_055` | 12 | no | 7 | 930,616 | 390,368 | 2965 | 3041 | 3356 | ok |
| `cfg_056` | 14 | yes | 6 | 1,292,440 | 2,151,976 | 2880 | 3079 | 3836 | ok |
| `cfg_057` | 22 | no | 11 | 1,630,720 | 1,354,952 | 5020 | 5200 | 4720 | ok |
| `cfg_058` | 33 | yes | 9 | 890,704 | 3,856,928 | 5926 | 6174 | 5644 | ok |
| `cfg_059` | 29 | no | 4 | 1,368,064 | 1,561,144 | 4700 | 4978 | 5056 | ok |
| `cfg_060` | 6 | no | 0 | 731,792 | 500,528 | 2354 | 2465 | 2575 | ok |
| `cfg_061` | 28 | no | 7 | 1,535,392 | 1,600,376 | 4900 | 5054 | 5281 | ok |
| `cfg_062` | 26 | no | 13 | 1,779,120 | 1,420,536 | 4986 | 5231 | 5022 | ok |
| `cfg_063` | 15 | no | 12 | 1,144,416 | 931,288 | 3798 | 3886 | 4004 | ok |
| `cfg_064` | 14 | yes | 13 | 1,426,072 | 1,834,904 | 6379 | 6590 | 3880 | ok |
| `cfg_065` | 22 | no | 9 | 1,351,632 | 1,301,512 | 3994 | 4090 | 4612 | ok |
| `cfg_066` | 18 | yes | 14 | 1,616,104 | 1,810,952 | 3744 | 3847 | 3794 | ok |
| `cfg_067` | 14 | yes | 4 | 1,025,512 | 2,204,728 | 3166 | 3227 | 4401 | ok |
| `cfg_068` | 23 | no | 12 | 1,558,720 | 1,288,640 | 5213 | 5468 | 4954 | ok |
| `cfg_069` | 36 | yes | 6 | 339,344 | 4,119,208 | 8868 | 9111 | 5734 | ok |
| `cfg_070` | 23 | no | 11 | 1,843,008 | 1,212,280 | 3777 | 3847 | 4382 | ok |
| `cfg_071` | 18 | no | 10 | 1,838,928 | 1,207,472 | 3278 | 3361 | 3948 | ok |
| `cfg_072` | 6 | no | 6 | 320,440 | 243,032 | 2378 | 2438 | 2952 | ok |
| `cfg_073` | 15 | no | 13 | 1,979,216 | 875,888 | 4352 | 4690 | 4491 | ok |
| `cfg_074` | 20 | no | 8 | 1,592,872 | 1,334,296 | 4088 | 4269 | 4670 | ok |
| `cfg_075` | 8 | no | 4 | 1,212,056 | 430,112 | 2524 | 2950 | 2849 | ok |
| `cfg_076` | 2 | no | 0 | 94,640 | 75,712 | 2236 | 2315 | 2767 | ok |
| `cfg_077` | 38 | yes | 3 | 73,008 | 4,363,912 | 5684 | 5960 | 5842 | ok |
| `cfg_078` | 4 | yes | 7 | 399,104 | 929,152 | 6086 | 6317 | 2294 | ok |
| `cfg_079` | 25 | no | 12 | 1,684,080 | 1,476,320 | 5549 | 5730 | 4844 | ok |
| `cfg_080` | 20 | yes | 20 | 1,894,872 | 2,604,968 | 9896 | 10244 | 4373 | ok |
| `cfg_081` | 13 | no | 6 | 1,524,736 | 693,792 | 3109 | 3267 | 3279 | ok |
| `cfg_082` | 7 | no | 3 | 704,480 | 642,288 | 2334 | 2375 | 3833 | ok |
| `cfg_083` | 9 | no | 3 | 541,536 | 584,160 | 2451 | 2493 | 3044 | ok |
| `cfg_084` | 20 | no | 9 | 2,497,688 | 1,841,768 | 3684 | 3812 | 4012 | ok |
| `cfg_085` | 22 | yes | 16 | 1,164,768 | 3,097,448 | 7108 | 7258 | 4513 | ok |
| `cfg_086` | 12 | no | 8 | 1,049,824 | 814,696 | 4129 | 4256 | 3832 | ok |
| `cfg_087` | 14 | no | 5 | 1,104,192 | 915,528 | 3813 | 3925 | 4614 | ok |
| `cfg_088` | 9 | yes | 3 | 1,076,568 | 1,561,048 | 2768 | 3089 | 2562 | ok |
| `cfg_089` | 20 | yes | 18 | 1,449,552 | 2,936,080 | 9058 | 9402 | 4462 | ok |
| `cfg_090` | 34 | yes | 9 | 702,464 | 3,772,728 | 5800 | 6366 | 5267 | ok |
| `cfg_091` | 4 | no | 0 | 857,840 | 195,792 | 2413 | 2473 | 2786 | ok |
| `cfg_092` | 24 | yes | 20 | 1,660,200 | 3,315,112 | 5274 | 5945 | 4129 | ok |
| `cfg_093` | 34 | yes | 10 | 813,648 | 4,065,424 | 5628 | 5776 | 5542 | ok |
| `cfg_094` | 22 | yes | 13 | 2,098,856 | 3,022,304 | 7818 | 8448 | 3511 | ok |
| `cfg_095` | 10 | no | 9 | 1,359,368 | 375,624 | 3996 | 4443 | 3455 | ok |
| `cfg_096` | 27 | yes | 17 | 1,504,304 | 3,124,648 | 8996 | 9426 | 4965 | ok |
| `cfg_097` | 9 | no | 4 | 1,655,368 | 769,672 | 2635 | 2721 | 3678 | ok |
| `cfg_098` | 10 | no | 11 | 1,803,616 | 640,496 | 3629 | 3717 | 3535 | ok |
| `cfg_099` | 4 | no | 0 | 552,008 | 213,960 | 2190 | 2242 | 3508 | ok |
| `cfg_100` | 6 | no | 3 | 890,528 | 488,384 | 2271 | 2302 | 2714 | ok |
| `cfg_101` | 16 | yes | 19 | 1,739,896 | 2,286,040 | 8911 | 9262 | 4620 | ok |
| `cfg_102` | 24 | yes | 20 | 1,786,128 | 2,687,368 | 6802 | 6968 | 4806 | ok |
| `cfg_103` | 26 | no | 10 | 1,524,896 | 1,512,472 | 5195 | 5291 | 4838 | ok |
| `cfg_104` | 22 | no | 25 | 3,094,464 | 2,187,288 | 4408 | 4618 | 4604 | ok |
| `cfg_105` | 2 | no | 0 | 293,376 | 201,712 | 2298 | 2332 | 2639 | ok |
| `cfg_106` | 25 | no | 14 | 2,247,992 | 1,139,296 | 5118 | 5341 | 4332 | ok |
| `cfg_107` | 12 | no | 10 | 865,392 | 612,576 | 2845 | 2921 | 2898 | ok |
| `cfg_108` | 12 | no | 7 | 1,462,736 | 759,160 | 3810 | 3911 | 4096 | ok |
| `cfg_109` | 11 | no | 15 | 724,056 | 634,832 | 3718 | 3783 | 3433 | ok |
| `cfg_110` | 25 | no | 14 | 2,184,096 | 1,337,040 | 4648 | 4871 | 5146 | ok |
| `cfg_111` | 16 | no | 9 | 2,339,840 | 1,795,088 | 3255 | 3364 | 4351 | ok |
| `cfg_112` | 10 | no | 3 | 562,840 | 572,968 | 2703 | 2767 | 3446 | ok |
| `cfg_113` | 24 | no | 28 | 2,704,248 | 2,079,528 | 5525 | 5744 | 4499 | ok |
| `cfg_114` | 15 | no | 8 | 1,538,416 | 739,072 | 3013 | 3083 | 3409 | ok |
| `cfg_115` | 25 | no | 14 | 2,483,920 | 2,386,352 | 5700 | 6183 | 4986 | ok |
| `cfg_116` | 4 | no | 0 | 232,544 | 102,752 | 2300 | 2407 | 2468 | ok |
| `cfg_117` | 3 | no | 3 | 326,592 | 209,952 | 2632 | 2762 | 2427 | ok |
| `cfg_118` | 20 | yes | 19 | 969,200 | 2,587,328 | 4921 | 5129 | 4826 | ok |
| `cfg_119` | 29 | yes | 9 | 573,984 | 3,767,368 | 6032 | 6375 | 5111 | ok |
| `cfg_120` | 28 | no | 7 | 1,315,312 | 1,598,680 | 5328 | 5632 | 5134 | ok |
| `cfg_121` | 1 | no | 0 | 242,000 | 48,400 | 2511 | 2624 | 2387 | ok |
| `cfg_122` | 11 | no | 6 | 1,037,768 | 615,528 | 3480 | 3628 | 4062 | ok |
| `cfg_123` | 14 | no | 7 | 3,029,304 | 944,672 | 2694 | 3017 | 3261 | ok |
| `cfg_124` | 21 | no | 10 | 1,548,224 | 1,340,104 | 5249 | 5376 | 4576 | ok |
| `cfg_125` | 26 | no | 7 | 1,325,840 | 1,496,896 | 5544 | 5770 | 5300 | ok |
| `cfg_126` | 18 | yes | 23 | 2,094,528 | 2,401,568 | 10570 | 10934 | 3728 | ok |
| `cfg_127` | 35 | yes | 4 | 274,456 | 4,329,464 | 5900 | 6002 | 5819 | ok |
| `cfg_128` | 16 | no | 10 | 299,504 | 1,015,440 | 4388 | 4594 | 4248 | ok |
| `cfg_129` | 14 | no | 11 | 1,049,640 | 780,520 | 2966 | 3012 | 3902 | ok |
| `cfg_130` | 36 | yes | 5 | 159,536 | 4,231,416 | 5791 | 5902 | 5840 | ok |
| `cfg_131` | 19 | no | 13 | 1,453,376 | 930,968 | 5064 | 5168 | 4581 | ok |
| `cfg_132` | 27 | yes | 15 | 1,016,104 | 3,837,736 | 5662 | 5789 | 4979 | ok |
| `cfg_133` | 16 | no | 14 | 1,044,256 | 845,032 | 4221 | 4371 | 4272 | ok |
| `cfg_134` | 2 | no | 5 | 105,456 | 97,344 | 2194 | 2230 | 2356 | ok |
| `cfg_135` | 26 | no | 7 | 986,544 | 1,446,152 | 4687 | 5299 | 4866 | ok |
| `cfg_136` | 9 | no | 10 | 1,366,120 | 1,435,752 | 4304 | 4489 | 2491 | ok |
| `cfg_137` | 27 | no | 11 | 1,364,064 | 1,504,632 | 6113 | 6232 | 5253 | ok |
| `cfg_138` | 36 | yes | 5 | 278,160 | 4,284,496 | 5752 | 5921 | 5576 | ok |
| `cfg_139` | 7 | no | 8 | 543,544 | 344,776 | 2547 | 2629 | 2443 | ok |
| `cfg_140` | 23 | yes | 22 | 2,256,392 | 2,780,088 | 4688 | 4811 | 3935 | ok |
| `cfg_141` | 19 | no | 10 | 1,627,968 | 1,138,632 | 4309 | 4387 | 4446 | ok |
| `cfg_142` | 14 | no | 12 | 1,948,280 | 1,337,784 | 3102 | 3612 | 2930 | ok |
| `cfg_143` | 16 | no | 10 | 1,319,000 | 678,472 | 2930 | 3026 | 3118 | ok |
| `cfg_144` | 16 | yes | 8 | 1,645,488 | 2,403,264 | 7099 | 7403 | 3228 | ok |
| `cfg_145` | 11 | no | 3 | 685,000 | 470,536 | 2440 | 2519 | 3342 | ok |
| `cfg_146` | 18 | no | 14 | 1,678,728 | 788,376 | 3880 | 4034 | 3527 | ok |
| `cfg_147` | 22 | yes | 16 | 1,756,240 | 3,049,408 | 5653 | 5742 | 3952 | ok |
| `cfg_148` | 12 | no | 12 | 1,724,376 | 1,219,768 | 2980 | 3140 | 2860 | ok |
| `cfg_149` | 11 | yes | 7 | 789,360 | 1,644,416 | 3179 | 3262 | 3941 | ok |
