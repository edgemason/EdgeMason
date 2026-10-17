# EdgeMason coding standards

**Spec:** `docs/specs/P0.2-docs-scaffold.md`
**Purpose:** review needs something to cite. Without written rules, every review is a judgement
call; with them, a reviewer can point at a rule.

These rules apply to `api/` and `scripts/`. Where a rule and a research constraint collide, the
research constraint wins and the exception is written down in the spec that needs it.

---

1. **Typed data, not JSON strings, between internal modules.**
   *Why:* `benchmark` builds a dict, renames the memory keys, serialises it, and `pipeline.py:84`
   parses it straight back — the format is lost and regained in one hop. *How to check:* no
   `json.loads` on the output of a function in the same package.
   *Exception:* at a process or HTTP boundary — the subprocess worker, FastAPI responses, files on disk.

2. **One place decides the framework (onnx/tvm).**
   *Why:* the knowledge is spread over five modules, and adding a third framework means editing
   every one of them (DESIGN #1). *How to check:* no new file-extension or candidate-name check
   outside the seam. DESIGN #1 creates that one place; until then, no new ones may appear.

3. **Domain strings have one definition.**
   *Why:* priorities, quantization strategies, framework names and hardware ids are bare literals
   in several files at once. *How to check:* no new bare string literal for a domain concept; it
   comes from a constant or enum. `QUANT_FP32` and `VALID_PRIORITIES` already exist.

4. **Limits and config constants are defined once,** in `api/config.py` or in the module that owns
   them. *Why:* `MAX_INPUT_ELEMENTS` and the 4096 dimension limit are each defined twice. *How to
   check:* one definition per constant; a second is a finding.

5. **No silent exception swallowing.**
   *Why:* `analyze.py` catches every `Exception` while `deploy.py` catches only `ValueError`, so
   the two endpoints fail differently for the same fault. *How to check:* catch narrowly; only
   routes map exceptions to HTTP errors.

6. **Research-sensitive constants are frozen.**
   *Why:* a number that produced a published finding must not move without that being recorded.
   *How to check:* `grep` for each name below before and after any change; a diff that moves one
   needs a spec whose gate class is `intended improvement`, and the new value recorded in
   `docs/research/FINDINGS.md`.

   | Constant | Where |
   |---|---|
   | `BENCHMARK_WARMUP_RUNS`, `BENCHMARK_TIMED_RUNS`, `BENCHMARK_TIME_BUDGET_SECONDS` | `api/config.py:24-26` |
   | `MIN_TIMED_RUNS`, `WARMUP_BUDGET_SHARE` | `api/optimizer/benchmark_engine.py:44`, `:46` |
   | Benchmark thread settings | `api/optimizer/benchmark_engine.py:369-370` |
   | `CALIBRATION_FRACTION`, `MAX_CALIBRATION_SAMPLES`, `SPLIT_SEED`, `MAX_VALIDATION_SAMPLES` | `api/optimizer/validation.py:16`, `:17`, `:20`, `:11` |
   | Accuracy-scoring thread settings | `api/optimizer/validation.py:214-215` |
   | `SIZE_THRESHOLD_MB`, `LATENCY_THRESHOLD_MS` | `api/planner/rule_based.py:33-34` |
   | Quantizer settings (per-channel QInt8, `quant_pre_process`) | `api/optimizer/quantizer.py` |
   | Target scaling values | `profiles/hardware_profiles.json` |

7. **Tests target public interfaces.**
   *Why:* internal helpers move during the redesign, so tests written against them carry no weight
   for the gate. *How to check:* the test imports a public entry point. *Exception:* a
   characterization test that locks a known quirk, which must say so in a comment.

8. **Golden and baseline files** (`tests/golden/`, `baseline/`) change only in a spec that names
   them, with the user's approval. *Why:* changing one changes the baseline every later spec is
   measured against. *How to check:* `git status` shows no golden or baseline file in a spec that
   did not ask for it.

9. **No dead code.** Every new function, parameter and field has a caller. *Why:* the deep-dive
   already found a pass-through module, two unused runtime functions, an overwritten estimate and
   four planner fields nothing reads. *How to check:* a new symbol with no reference is not merged.

10. **The product name is "EdgeMason"** everywhere. *Why:* "EdgeForge" still appears in the
    Streamlit app, the pages route and the dashboard. *How to check:* `grep -ri edgeforge`.

11. **Imports go at module top.** *Why:* scattered imports hide real dependencies.
    *How to check:* imports at the top of the module. *Exception:* optional heavy dependencies
    (`tvm`, `torch`), which are imported lazily and must raise a clear error message when missing.

12. **New code is clean.** Public functions have type hints, and new files pass ruff and pyright
    under the P0.10 config. *Why:* the repo has no linter config today, so quality only ratchets if
    new code is held to the standard. *How to check:* the P0.10 CI job, and the gate for changed files.

13. **Research scripts are not product code.** `scripts/` may differ from `api/` on purpose.
    *Why:* `quantization_analysis.py` re-implements scoring, latency and quantization with
    different settings, and the research numbers were measured on those paths (PLAN D). *How to
    check:* do not "deduplicate" a script into `api/` without a spec; the divergence is documented
    in `docs/cleanup/BACKLOG.md` (B-47), not silently removed.

### Environment note

`uv run pip list` reports the **system** interpreter, not the project venv, because the venv has no
`pip` in it and the name resolves on `PATH`. Use `uv pip list` to inspect the project environment.
This has already cost one false alarm (a phantom numpy 2.5.1 during P0.4).

---

## Review heuristics

The twelve smells below are Fowler's, copied verbatim from the code-review baseline
(`/home/dark/.agents/skills/code-review/SKILL.md`). Each reads *what it is* → *how to fix*.

**These are judgement calls, not violations.** A review may raise one of these against a change and
say why; the author may reasonably disagree, and then it is a discussion rather than a defect. The
numbered rules above are the ones a reviewer can cite without arguing about taste.

- **Mysterious Name** — a function, variable, or type whose name doesn't reveal what it does or holds. → rename it; if no honest name comes, the design's murky.
- **Duplicated Code** — the same logic shape appears in more than one hunk or file in the change. → extract the shared shape, call it from both.
- **Feature Envy** — a method that reaches into another object's data more than its own. → move the method onto the data it envies.
- **Data Clumps** — the same few fields or params keep travelling together (a type wanting to be born). → bundle them into one type, pass that.
- **Primitive Obsession** — a primitive or string standing in for a domain concept that deserves its own type. → give the concept its own small type.
- **Repeated Switches** — the same `switch`/`if`-cascade on the same type recurs across the change. → replace with polymorphism, or one map both sites share.
- **Shotgun Surgery** — one logical change forces scattered edits across many files in the diff. → gather what changes together into one module.
- **Divergent Change** — one file or module is edited for several unrelated reasons. → split so each module changes for one reason.
- **Speculative Generality** — abstraction, parameters, or hooks added for needs the spec doesn't have. → delete it; inline back until a real need shows.
- **Message Chains** — long `a.b().c().d()` navigation the caller shouldn't depend on. → hide the walk behind one method on the first object.
- **Middle Man** — a class or function that mostly just delegates onward. → cut it, call the real target direct.
- **Refused Bequest** — a subclass or implementer that ignores or overrides most of what it inherits. → drop the inheritance, use composition.
