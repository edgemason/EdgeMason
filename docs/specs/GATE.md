# The P0.9 results gate

**Spec:** `docs/specs/P0.9-gate-script.md`
**Script:** `scripts/gate.py`
**Purpose:** every later spec is adopted only if "results match or beat the baseline". This gate turns that rule into a command with an exit code, so no agent and no tired human has to judge it by eye.

The gate never modifies either tree it is given. It runs commands inside them with bytecode writing and pytest's cache disabled, and the only tree it writes to is a throwaway detached worktree of the candidate's merge-base, which it removes again in a `finally`.

---

## 1. How to run it

### The two commands

```bash
# Deterministic results: assets, tests, goldens, outputs, lint/types.
uv run python scripts/gate.py check --candidate . --level full

# Timing and memory: alternating A/B benchmarks of two trees.
uv run python scripts/gate.py ab --baseline . --candidate .
```

### `check`

| Flag | Default | What it does |
|------|---------|--------------|
| `--candidate PATH` | required | The git worktree root to judge. |
| `--level fast\|full` | `fast` | `fast` runs `pytest -m "not slow and not tvm and not cuda and not research"` and the two quick models; `full` runs everything. |
| `--baseline-dir PATH` | `baseline` | Directory holding the stored `quick/` and `full/` captures. Relative paths resolve against the gate's own tree. |
| `--report OUT.json` | none | Write the JSON report here. |
| `--allow-deleted FILE` | none | A file of test node ids the candidate is allowed to have deleted, one per line. Blank lines and `#` comments are ignored. |
| `--allow-golden-change` | off | Accept changes under `tests/golden/` and `baseline/`. For approved specs only. |

### `ab`

| Flag | Default | What it does |
|------|---------|--------------|
| `--baseline PATH` | required | The tree to measure against. |
| `--candidate PATH` | required | The tree to measure. |
| `--models NAME ...` | `squeezenet1.1-13 mobilenetv2 resnet50` | Which models to compare. |
| `--rounds N` | `15` | Alternating B/C rounds per model × variant. Fewer rounds cannot resolve the band; see §5. |
| `--band-pct PCT` | `5.0` | Allowed **mean** paired difference, in percent. The spec's threshold. |
| `--report OUT.json` | none | Write the JSON report here. |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Pass. |
| `1` | Fail. The report says which step and why. |
| `2` | Usage or environment error: the gate cannot judge. A worker crashed or timed out, the candidate is not a directory, the baseline directory is missing, or `--rounds` is below 1. |

A timeout or crash inside either tree's benchmark subprocess is **exit 2**, never a silent pass and never a fail computed from numbers that were never measured.

---

## 2. What each `check` step does, and how to read a failure

The steps run **in order and stop at the first failure**. Later steps appear in the report with status `not run`, so the report always has the same five rows.

| # | Step | What it does | What a failure means |
|---|------|--------------|---------------------|
| 1 | `assets` | `provision_assets.py --dest <candidate> --verify-only`. Exit 3 means files are missing: it provisions them once, then verifies again. Exit 2 is a mismatch and fails immediately. | The candidate cannot run without its models. Provision it: `uv run python scripts/provision_assets.py --dest <candidate>`. |
| 2 | `tests` | Runs the suite, then collects node ids in **both** the candidate and a throwaway worktree of the candidate's merge-base. Any id collected in the baseline but missing from the candidate fails. | Either a test broke, or a test was deleted. The detail names every missing node id. Waived ids appear as "allowed deletion(s)". |
| 3 | `goldens` | `git diff --name-only <merge-base> -- tests/golden baseline` must be empty. | A golden or baseline file moved. That is the user's decision, not the gate's; pass `--allow-golden-change` only for a spec that approved it. |
| 4 | `deterministic-outputs` | Runs `capture_baseline.py` in the candidate into a temp dir and diffs against `baseline/quick` (fast) or `baseline/full`. Everything but `environment.json` must be byte-identical; `accuracy.json` may differ by 1e-6 per number. | A deterministic output changed. The detail names the file **and the JSON path**, e.g. `plans.json:plans.mobilenetv2.true[0].name`. |
| 5 | `lint-types` | P0.10's ratchet, `scripts/lint_ratchet.sh`, with `BASE_REF` set to the same merge-base. | New code broke ruff or pyright. |

The merge-base is derived by the gate itself: `cleanup` or `origin/cleanup` if either exists, otherwise the candidate's `HEAD`. It is materialised as a temporary detached worktree and removed in a `finally`. If it cannot be resolved, step 2 records `skipped` with the reason rather than passing silently.

---

## 3. How to read the report

Both subcommands write the same top-level shape:

```json
{
  "verdict": "pass" | "fail",
  "steps":  [{"name": "...", "status": "pass|fail|skipped|not run", "details": "..."}],
  "ab":     { }
}
```

For `check`, `ab` is `{"status": "not run", "reason": "..."}`. For `ab` it carries the full measurement.

The `ab` object contains:

| Key | What it is |
|-----|-----------|
| `rounds`, `models`, `variants` | What was run. `variants` maps `fp32`/`dynamic`/`static` to the `onnx_*` candidate names the tree's own `Quantizer` routes on. |
| `band_pct` | The threshold, 5.0 by default. |
| `warmup_runs_per_side` | Throwaway runs per model × variant per tree before round 0. |
| `order` | Per model × variant, the order of each round, so the balancing is verifiable. |
| `rounds_led_by` | How many rounds each side went first in. An odd total shows here. |
| `results` | Per model × variant, `p50` and `memory_delta` (below), plus a combined `pass`. |
| `power` | Per model × variant, the sensitivity of the verdict (below). |
| `winners` | Per model, the fastest variant on each side and whether it is the same. |

Each metric block in `results` carries:

| Field | Meaning |
|-------|---------|
| `baseline`, `candidate` | The raw per-round samples, in round order. |
| `pct_diffs` | The paired per-round percentage differences. This is what the verdict is computed from. |
| `n`, `mean_pct_diff`, `sd_pct_diff`, `max_abs_pct_diff`, `band_pct`, `within_band` | The verdict. **Pass iff `abs(mean_pct_diff) <= band_pct`.** |
| `informational.median_pct_diff` | Reported for auditability. **Decides nothing.** |
| `informational.median_baseline`, `median_candidate`, `mad_baseline` | Reported for auditability. **Decide nothing.** |

`median_pct_diff` and `MAD_B` are informational only. They were once the pass rule and both are misleading on this machine: the paired differences are bimodal, and the median flips to ±7% as soon as three of five rounds straddle a CPU state change. See §5.

### The `power` block

```json
"power": {
  "statistic": "mean of the paired per-round percentage differences",
  "results": {
    "mobilenetv2/fp32": {
      "p50":         {"n": 15, "se": 1.21, "mde_2se": 2.42, "band_pct": 5.0, "powered_for_band": true},
      "memory_delta": {"...": "..."}
    }
  }
}
```

`se` is `sd/sqrt(n)` and `mde_2se` is `2 * se`, the smallest effect roughly 95% detectable. `powered_for_band` is `mde_2se <= band_pct`: **if it is false, the run could not have resolved the band at all and a pass proves very little.** Check it rather than trusting the verdict.

---

## 4. Self-test result

Both commands run against the unmodified tree, baseline and candidate the same path, so anything they report is measurement noise by construction.

| Command | Verdict | Exit | Wall time |
|---------|---------|------|-----------|
| `check --candidate . --level full` | **PASS** — all 5 steps | 0 | 267 s |
| `ab --baseline . --candidate .` | **PASS** — 18/18 comparisons inside the band | 0 | 835 s |

Baseline and candidate are the same path in both, so every number the gate reports here is measurement noise by construction.

---

## 5. The measured noise band

From the `ab` self-test, `n = 15` rounds per model × variant, band 5.0%:

| model | variant | mean % | sd % | MDE (2 SE) % | verdict | memory mean % |
|-------|---------|-------:|-----:|-------------:|---------|--------------:|
| squeezenet1.1-13 | fp32 | +1.53 | 4.96 | 2.56 | pass | −0.02 |
| squeezenet1.1-13 | dynamic | −0.47 | 4.43 | 2.29 | pass | +0.01 |
| squeezenet1.1-13 | static | +0.45 | 3.59 | 1.85 | pass | −0.04 |
| mobilenetv2 | fp32 | −0.57 | 4.48 | 2.31 | pass | −0.01 |
| mobilenetv2 | dynamic | +0.62 | 5.13 | 2.65 | pass | +0.00 |
| mobilenetv2 | static | +0.13 | 4.12 | 2.13 | pass | +0.02 |
| resnet50 | fp32 | −1.56 | 3.82 | 1.97 | pass | +0.02 |
| resnet50 | dynamic | −0.48 | 3.35 | 1.73 | pass | +0.01 |
| resnet50 | static | −0.51 | 2.60 | 1.34 | pass | −0.01 |

Agreed winners: **squeezenet1.1-13 → fp32, mobilenetv2 → static, resnet50 → static**, identical on both sides. Rounds led 68/67. Tightest headroom 3.44 points (squeezenet fp32, +1.53%); worst single round 12.74%. Memory is essentially exact: every row within ±0.04%, sd under 0.10%.

**Why per-round sd is 2.6–5.1% on an idle machine.** The paired differences are bimodal, not normal: they sit near 0% when both runs of a round shared a CPU frequency/thermal state, and near ±7% when the round straddled a change between them. The mean of 15 paired rounds is what absorbs that — the straddles partly cancel. Two earlier gate versions used the median of independent samples and were flaky against an unchanged tree, passing one run and failing the next with nothing changed.

**What the gate can and cannot detect.** At `n = 15` the mean resolves **2.65% or better** (the worst row, mobilenetv2 dynamic), comfortably inside the 5% band, and every row reported `powered_for_band: true`. A real regression of 5% or more is caught. A regression of **under ~2.7% is not detectable** and will pass. That is a deliberate trade: the band is the spec's 5%, and rounds buy sensitivity without loosening the threshold.

The default of 15 rounds is chosen for this reason. The mean of 5 has a standard error near 2.1%, so its minimum detectable effect is **no smaller than the band being judged** — it cannot resolve 5% at all. Raising it to 15 costs about three times the wall time and buys the sensitivity.

---

## 6. Mutation test reports

Each mutation ran in its own throwaway git worktree at `HEAD`, provisioned with its own `scripts/provision_assets.py --dest`, and removed afterwards. `git worktree list` confirmed no leak.

### Mutation 1 — `api/optimizer/validation.py`, `SPLIT_SEED = 0` → `1`

```bash
uv run python scripts/gate.py check --candidate <mut1> --level full --report /tmp/mut1.json
```

**Result: FAIL, exit 1, at the `tests` step** (41 s).

```
tests/helpers.py:119: AssertionError
FAILED tests/test_characterize_logic.py::test_split_matrix - AssertionError
1 failed, 197 passed in 38.78s
```

**The spec predicted `deterministic-outputs`; the gate failed one step earlier at `tests`.** That is a better result, not a worse one. P0.7a's characterisation test `test_split_matrix` locks the calibration/test split against a golden, and a different seed changes that split immediately. The gate therefore caught the mutation in 41 s instead of after a full capture, and did so through a test whose intent is to pin the split. The later steps are recorded as `not run`.

This is worth stating plainly: `deterministic-outputs` would also have caught it, because `accuracy.json` is computed from that split, but the gate never got there and never had to.

### Mutation 2 — `profiles/hardware_profiles.json`, rpi4 `latency_scaling_factor` 11.8 → 12.0

```bash
uv run python scripts/gate.py check --candidate <mut2> --level fast --report /tmp/mut2.json
```

**Result: FAIL, exit 1, at the `tests` step** (37 s), as predicted.

```
tests/test_ranking.py:26: AssertionError
FAILED tests/test_characterize_logic.py::test_estimate_matrix - AssertionError
FAILED tests/test_ranking.py::test_int8_uses_its_own_multiplier_on_rpi4 - AssertionError
2 failed, 188 passed, 8 deselected in 34.48s
```

P0.7a's `estimate_matrix` golden locks the rpi4 estimates, and `test_ranking.py` independently pins the rpi4 INT8 multiplier. Two independent tests caught the change, at the predicted step.

### Conclusion

**Neither mutation passed.** The gate is not blind: it catches a changed calibration seed and a changed hardware profile, both at the `tests` step, in under a minute each.

---

## 7. Known limits

- **The band is 5%, and that is the whole threshold.** Sensitivity is bought from the statistic (mean of paired diffs) and the round count (15), never from a looser band. `--band-pct` exists for an approved spec, not for convenience.
- **A single bad thermal stretch can tighten the margin.** Per-round swings of ±7% and worse occur when a round straddles a CPU state change. At `n = 15` this is absorbed, but an unusually bad stretch would widen the standard error and reduce the MDE below 2.4%. The `power` block shows the MDE for every run, so a run that was not powered is visible rather than silently weak. Do not run `ab` while the machine is busy; a concurrent build inflated the sd to 10.4% and pushed one row's MDE past the band.
- **`ab` takes about 14 minutes** at the default 15 rounds across three models × three variants. `check --level full` takes about 4.5 minutes.
- **`check` needs provisioned assets in the candidate.** Step 1 provisions them itself if they are missing, but that means a network download inside the gate run. Provision ahead of time when the candidate is a fresh worktree.
- **Two trees only.** `ab` compares exactly two trees. A candidate is judged against its merge-base, not against every other branch.
- **Timing is machine-specific.** The numbers in §5 are this host's. A quieter or better-cooled machine will show smaller sd and a lower MDE; a laptop on battery will not.