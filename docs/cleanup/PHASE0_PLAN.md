# EdgeMason Cleanup: Phase 0 Plan (Baseline)

## Context

EdgeMason grew quickly, partly through AI-assisted code dumps. `REVIEW.md` (the two-part code review) and `DESIGN.md` (four deepening opportunities) catalogue the mess. The cleanup has to serve three goals:
1. A research paper, preferably at an MLSys workshop in 2027. The claim is still undecided.
2. A foreign internship, preferably in industry.
3. Real users.

**How we work:**
- The planner (Opus) writes specs and TDD test cases, and writes no code.
- Cheaper agents (Haiku, Grok) carry out each spec in an **isolated git worktree or sandbox**, never in the main tree.
- A change is adopted **only if the results gate passes**.

**Phase 0 builds the ground truth that every later spec depends on.** It changes no product behaviour. Its only output is tests, test infrastructure, tooling, documentation and asset provisioning.

## Decisions made
- **Dashboard:** only the static `api/static/index.html` stays. Streamlit `frontend/app.py` will be deleted in a later phase, after its unique features are ported (see the inventory).
- **TVM:** experimental, opt-in and off by default. It stays as the second framework adapter.
- **Branches:** first merge `abhay` into `main` through a PR, which needs the user's go-ahead. After that, each spec gets its own branch off `main`.
- **Paper claim:** deferred. Until it is decided, every quantitative finding in `reports/` is protected.

## Results gate (agreed)

| Output | Gate |
|---|---|
| Deterministic outputs: planner candidates, ranking order, target estimates, data split, ZIP contents, API response format | Identical to the reference outputs, field for field |
| Accuracy (top-1/top-5, fixed seed) | Identical within 1e-6 |
| Timing and memory | Run baseline and change on the same machine in the same session, alternating several times. The winner must not change, and p50 must stay within the measured noise band |
| Tests, lint, types | All pass. No existing test is deleted unless the spec names it |

"Better" only counts where a spec explicitly aims for an improvement. A refactor must produce identical results.

---

## Inventory findings (Phase 0 deep-dive, branch `abhay` @ f0b563d)

### A. Correction to an assumption: models are not script-downloadable
- **Datasets: scriptable.**
  - `scripts/prepare_imagenetv2_subset.py` builds the `.npz` files for 20, 50 and 256 images.
  - `scripts/download_imagenetv2_large.py` builds the 1000-image file. Its source is Hugging Face (`vaishaal/ImageNetV2`).
  - Neither verifies checksums, and their output depends on the row order Hugging Face returns.
- **Models: only 3 of 9 are scriptable.**
  - `models/model_generators.py` rebuilds mobilenetv2, resnet50 and resnet152, but it needs **torch and torchvision**, which are not declared anywhere.
  - The other 6 (squeezenet1.1, shufflenet-v2 and resnet18, each in two opset versions) were **downloaded by hand** from the ONNX Model Zoo and converted with `onnx.version_converter`.
- **Consequence:** a fresh sandbox cannot provision itself today. Phase 0 has to fix this before any results gate can run.

### B. Findings to protect, and the report's sourcing gaps
**What has a data source.** Most tables in `reports/EdgeMason_Report.md` trace back to a JSON file in `reports/data/`:
- `quantization_analysis`
- `calibration_size_study`
- `calibration_scale_study`
- `full_run`
- `validation_run`

**What doesn't.** About 12 claims have **no data source**:
- p50 11.9/18.4/14.2 → 13.1/14.3/13.1 ms (L43, L193). The stored repeats say 13.27/13.41/12.53.
- `/health` 31,324 → 13.9 ms (L44, L216–220). The JSON says 163.6 ms.
- 7.6× slower and 53 `DynamicQuantizeLinear` nodes (L45).
- ~30 s deploy "before" (L46).
- 429 → 524 MB (L186).
- ResNet-152 at ~16 s vs ~42 s (L194).
- 12/20 correct (L206).
- `/health` at 5–9 ms (L518).
- ~632 MB of CUDA libraries (L544).
- Run times (L602–604).

**Where data and report contradict.** Line 524 says "11 deploys", but the data has **8**.

**Unclear provenance.** Three items:
- `validation_run_2026-10-05.json`: no script produces it.
- `calibration_scale_study_256.json` and `calibration_scale_256.png`: unclear, and not cited in the report.
- **These are the main research-integrity risk for the paper.**

### C. Product code (`api/`)
- **Dead code:**
  - `onnx_runtime.benchmark_tvm` and `run_inference`
  - the unreachable `sample_path` branch
  - `tvm_runtime.run_model`
  - `BenchmarkResult.hardware_estimate`, which is computed and then overwritten
  - the planner fields `requires_calibration`, `candidate_count`, `planner` and `input`, which nothing reads
  - most hardware-profile metadata fields
- **Framework (onnx vs tvm) decided in 7+ places:**
  - pipeline.py:70/156
  - quantizer.py:202-217 (name strings that duplicate the `STRATEGY_*` constants)
  - benchmark_engine.py:584
  - onnx_runtime.py:181
  - generator.py:187/266
  - hardware.py:46
  - index.html:649
- **Profiles:**
  - Hard-coded fallbacks in benchmark_engine.py:115-176 silently swallow load errors.
  - The profiles file is re-read on every call.
  - `TVM_TARGETS` covers only rpi4, rpi5 and x86_64.
  - The default hardware id differs: `rpi4` in the route, `rpi5` in the UI, `x86_64` in the functions.
- **Literals scattered across the code:**
  - priority names
  - `"fp32"`
  - framework names (no constant exists)
  - ORT provider names (about 12 places)
  - `MAX_INPUT_ELEMENTS` and the 4096 dimension limit, each defined twice
- **Memory keys differ by benchmark mode.** The isolated benchmark returns `baseline/peak/delta`; the in-process one returns `baseline_mb/peak_mb/delta_mb`.
- **Research-sensitive measurement issues.** These are not cleanup: each gets its own spec later and passes the gate as an intentional change.
  1. **TVM threads are not pinned,** while ORT runs on 1 thread. That makes ORT-vs-TVM comparisons unfair.
  2. **Accuracy is always scored on CPU,** even for CUDA runs.
  3. **x86_64 has no per-quantization scaling factors.**
  4. **Benchmark inputs are unseeded.**
  5. **Memory counts CPU RSS only,** so GPU memory under CUDA is not counted.
- **POST /deploy runs a 14-step call chain.** It is recorded in the inventory output; copy it into `INVENTORY.md`.

### D. Research code (`scripts/`)
- `quantization_analysis.py` re-implements logic that already exists in `api/`, with **different settings**:
  - accuracy scoring, using all cores (the API uses 1 thread);
  - latency (10 warm-up + 50 timed runs, in-process);
  - per-tensor quantization.

  The two calibration studies import these helpers. **The research numbers were therefore measured with code paths that differ from the product's.** That is fine, but it must be documented, not "deduplicated" by accident.
- `build_report_odt.py` needs an ODT template that is missing from git and from disk.
- `run_full_benchmark.py` writes an undated filename; the dated file was renamed by hand.
- No script hard-codes an absolute path; they all use `PROJECT_ROOT`.

### E. Tests
- **The suite has 87 cases.** The last run was 83 passed and 1 skipped; the skip is the whole of `test_model_generators.py` (4 tests), because torch is missing. No test needs the model files on disk.
- **Tested exactly:**
  - `auto` provider selection
  - data split counts and seed
  - onnx top-1/top-5
  - benchmark run counts, time budget and warm-up cap
  - the ZIP's file set
- **Tested only loosely:**
  - the planner (exact lists only for the CNN and matmul cases)
  - ranking (no test for accuracy priority, no tie-breaks)
  - `estimate_on_target` (exact only for rpi4)
  - quantization outputs (QInt8 and `quant_pre_process` unchecked)
  - the `/deploy` response format
- **Not tested at all:**
  - `/analyze`
  - the `validate_model` limits
  - artifact TTL cleanup and bad IDs
  - the 1000-sample and 640 MB limits
  - 413 errors
  - "accuracy priority requires .npz"
  - TVM accuracy
- **Nondeterminism to neutralise in tests:**
  - unseeded RNG (`conftest.py:20`, `benchmark_engine.py:307,498`, `generator.py:110`)
  - `system_info` and installed versions written into the ZIP
  - temp paths
  - `uuid4`
  - timing-sensitive assertions (`test_api.py:265`, `test_benchmark_engine.py:323,362`)
- **Infrastructure:** no markers, no timeout plugin, one shared fixture.

### F. Environment
- **Wrong Python version in the README.** `pyproject.toml` requires Python `>=3.11,<3.12`; the README says `>=3.9,<3.13`.
- **Missing dependencies.** These are undeclared:
  - `apache-tvm`, unpinned (the report says 0.26.0)
  - `torch` and `torchvision`
  - `matplotlib` and `lxml`, which the scripts use through `uv run --with`
- **Possibly unused dependencies:** `tensorflow==2.15.1` and `keras` look unused. Verify before removing.

### G. Streamlit features that `index.html` lacks
Port these before Streamlit is deleted (later phase):
- a configurable backend URL
- a throughput bar chart
- an inline failure reason
- `/analyze` errors shown in the UI
- the operator chart capped at the top 8

---

## Phase 0 work packages

Each package becomes one spec under `docs/specs/` (template in P0.2). An executor agent carries it out in its own worktree. Every package except P0.1 changes **no product behaviour**.

| # | Package | Output | Depends on |
|---|---|---|---|
| P0.1 | **Merge `abhay` → `main`** through a PR. This needs the user's go-ahead. Commit `REVIEW.md` and `DESIGN.md` into `docs/cleanup/`. | `main` becomes the baseline | – |
| P0.2 | **Docs scaffold:** <ul><li>`docs/cleanup/INVENTORY.md`, a per-module verdict and risk (from sections C–G)</li><li>`docs/cleanup/BACKLOG.md`, which merges REVIEW, DESIGN and these findings into ordered items</li><li>`docs/specs/TEMPLATE.md`</li><li>`CODING_STANDARDS.md`, which turns the review smells into checkable rules</li></ul> | Docs only | P0.1 |
| P0.3 | **Findings register:** `docs/research/FINDINGS.md`. Every quantitative claim in the report gets its line number, its data file, the producing script, the exact command and its status (sourced, unsourced or contradicted). | Integrity map | P0.1 |
| P0.4 | **Reproducible environment:** <ul><li>declare optional extras: `tvm` (pinned 0.26.0), `research` (torch, torchvision, matplotlib, lxml) and `dev`</li><li>fix the README's Python version</li></ul> No dependency is removed. | `uv sync --extra …` works in a clean sandbox | P0.1 |
| P0.5 | **Asset provisioning:** <ul><li>an `assets.json` manifest listing each model and dataset with its source URL, conversion step (opset converter) and sha256</li><li>`scripts/provision_assets.py`, which is idempotent, verifies checksums, and covers the 6 Model Zoo models plus the existing generators and dataset scripts</li></ul> The checksums are recorded from the **current local files**, which are the ones that produced the findings. | A fresh sandbox can provision itself, and the provisioning is verifiable | P0.4 |
| P0.6 | **Test infrastructure:** <ul><li>pytest markers `slow`, `tvm`, `cuda` and `research`</li><li>`pytest-timeout`</li><li>seeded fixtures</li><li>one output normaliser that strips paths, uuid, `system_info` and versions</li></ul> | A fast gate and a full gate | P0.1 |
| P0.7 | **Characterization tests** for every "loose" and "not tested" item in E, asserting **current** behaviour, quirks included (e.g. the memory-key mismatch). | Behaviour locked | P0.6 |
| P0.8 | **Baseline capture.** In a fresh sandbox (P0.5): run the deterministic pipeline outputs and the research scripts' deterministic parts. Store the reference outputs in `baseline/` together with the asset manifest hash and the environment (package versions, CPU model). | The reference outputs | P0.5, P0.7 |
| P0.9 | **Gate script** `scripts/gate.py`: provision assets; run tests, lint and types; diff deterministic outputs against `baseline/`; run the A/B timing protocol (alternating runs, noise band); emit a pass/fail report. | Automated gate | P0.8 |
| P0.10 | **Lint, types and CI:** <ul><li>ruff and pyright, configured so they **ratchet**: they enforce on changed lines and files, with existing violations allow-listed</li><li>GitHub Actions running the fast gate</li></ul> | CI | P0.6 |

**P0.5 risk to settle when writing its spec.**

The problem: models re-exported from torch (mobilenetv2, resnet50, resnet152) and the Model Zoo models re-run through `version_converter` might not come out byte-identical to the current files. If so, sha256 verification fails.

The spec's first test checks exactly that. If the bytes differ, there are two options:
- **(a) Host the current files ourselves,** as private GitHub release assets. Model licences allow redistribution. The ImageNetV2 `.npz` files must **not** be published publicly.
- **(b) Verify by output equivalence:** a fixed input must give outputs within 1e-5 of the reference outputs.

The user decides between them at spec time.

**Explicitly out of scope for Phase 0:** queued in `BACKLOG.md` for Phase 1 and later.
- DESIGN #1–#4
- the REVIEW smells
- removing dead code
- deleting Streamlit
- the TVM default and timeout
- the five measurement issues
- removing unused dependencies
- fixing unsourced report claims

## Spec template (for P0.2 and every later spec)
Each spec has these sections:
- **Why**
- **Scope:** the files the agent may touch; it may touch nothing else.
- **Out of scope**
- **Target interface**
- **Test cases:** Given/When/Then, written failing first.
- **Acceptance:** exact commands that must pass.
- **Tests that may be deleted**
- **Gate class:** "identical" or "intended improvement: metric X must improve by Y".
- **Rollback:** discard the worktree.

**Executor rules:**
- One spec per worktree branch.
- Never edit outside the declared scope.
- Stop and report instead of improvising when the spec is ambiguous.

## Verification of Phase 0 itself
1. **Provisioning:** in a brand-new worktree, `provision_assets.py` reproduces every sha256 in `assets.json`.
2. **Tests:** the full suite passes with no new skips other than the declared `tvm`, `cuda` and `research` markers. Running the characterization tests twice gives identical results.
3. **Gate self-test:** running the gate with `main` against itself → **pass**, and this run records the noise band.
4. **Gate mutation test:** in a throwaway worktree, change `SPLIT_SEED` from 0 to 1, and separately the rpi4 scaling factor. The gate must **fail** both times. That proves the gate actually catches research-relevant changes.
5. **Isolation:** after all of the above, `git status` in the main working tree is unchanged.
