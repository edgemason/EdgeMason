# Specs

**Rules for executors:** `EXECUTOR.md`
**Template:** `TEMPLATE.md`
**The plan these come from:** `../cleanup/PHASE0_PLAN.md`

## Phase 0: Baseline (no product behaviour changes)

| Spec | Title | Depends on | Who | Status |
|---|---|---|---|---|
| [P0.1](P0.1-merge-abhay.md) | Merge `abhay` → `main`, commit the planning docs | – | executor (user merges) | todo |
| [P0.2](P0.2-docs-scaffold.md) | INVENTORY, BACKLOG, CODING_STANDARDS | none | executor | **agreed, ready** |
| [P0.3](P0.3-findings-register.md) | Findings register (claims → data → script) | none | executor | **agreed, ready** |
| [P0.4](P0.4-reproducible-env.md) | Pin TVM; dev/research dependency groups; README Python version | none | executor | **done** |
| [P0.5](P0.5-asset-provisioning.md) | Asset provisioning plus checksums (**two stages; stops for a user decision**) | P0.4 | executor | todo |
| [P0.6](P0.6-test-infrastructure.md) | Markers, timeout, seeds, normaliser | P0.4 | executor | **agreed, ready** |
| [P0.7a](P0.7a-characterize-pure-logic.md) | Characterize planner, ranking, estimates, provider, split, validation | P0.6 | executor | todo |
| [P0.7b](P0.7b-characterize-http.md) | Characterize HTTP API and artifacts | P0.6 | executor | todo |
| [P0.7c](P0.7c-characterize-optimizer.md) | Characterize quantization, benchmark format, packaging | P0.6 | executor | todo |
| [P0.8](P0.8-baseline-capture.md) | Baseline on real models, plus the reproduction check | P0.5, P0.7* | executor | todo |
| [P0.9](P0.9-gate-script.md) | The results gate (`check` + `ab`), with self-test and mutation tests | P0.8 | executor | todo |
| [P0.10](P0.10-lint-types-ci.md) | ruff/pyright ratchet plus GitHub Actions | P0.6 | executor | todo |

## Execution waves (specs within a wave can run in parallel worktrees)
1. **Wave 1:** P0.1
2. **Wave 2:** P0.2, P0.3, P0.4, P0.6
3. **Wave 3:** P0.5 (Stage 1 → **user decision** → Stage 2), P0.7a, P0.7b, P0.7c, P0.10
4. **Wave 4:** P0.8
5. **Wave 5:** P0.9

**After each spec:** the planner reviews the executor's report and diff against the spec (Standards + Spec). The branch merges only if every acceptance command passed and the scope check is clean.
