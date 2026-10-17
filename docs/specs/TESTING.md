# Testing Strategy (applies to every spec)

**Status:** interviewing. The items marked *agreed* were settled with the user on 2026-10-14.

## Principles (agreed)
1. **Test what the system does, not how it is written.** Tests sit at the seams that survive the redesign:
   - **System:** model + target + constraints → `/deploy` → ranked candidates, chosen winner, ZIP contents.
   - **Stage contracts:** each pipeline stage's input → output (plan, quantize, benchmark, estimate, score, rank, package).

   Internal helpers may be rewritten freely, so their tests carry no weight for the gate.
2. **Interface-based contracts.** Each stage gets one contract test suite, written against its interface and run against every adapter behind it (onnx today, tvm too). A future framework or target only has to pass the same suite. This shapes the seams; it adds no feature.
3. **Small pieces.** The redesign is never a rewrite. Each spec moves one stage behind one interface, so a cheap agent can pick it up and finish it in one session.
4. **TDD in a refactor:**
   - **Lock:** system and contract tests are written first and must **pass on today's code**. The agent proves each one can fail by breaking the code on purpose.
   - **Red → green:** tests for a new interface are written first and fail. Code then moves behind that interface while every lock test stays green.

## TVM (agreed)
TVM stays installed (P0.4) but is **not exercised during this cleanup**. Tests that need TVM are marked `tvm` and are excluded from both gate tiers. The contract suites are written so that a TVM adapter can be plugged in later.

## Time budget (agreed)
- **Core tier:** at most about **10 minutes**, run on every spec.
- **Full tier:** no time limit; run once before a spec branch merges.

## Models (agreed)
- **Full tier:** **every model we have** runs the system tests end to end. That is squeezenet1.1 (-7, -13), shufflenet-v2 (-10, -13), resnet18-v1 (-7, -13), mobilenetv2, resnet50 and resnet152.
- **Core tier** (every spec's quick gate): **mobilenetv2, resnet50, EfficientNet.**
- **Open:** EfficientNet is not in the repo yet. Its source, variant and checksum must be decided, then added to P0.5 provisioning. It has **no historical baseline** in `reports/`, so its first capture in P0.8 *becomes* its baseline.

## Proving the tests work (agreed: score; threshold open)
- Mutation testing (`mutmut`) on the core logic: planner, ranking, target estimates and the quantize decisions.
- The gate enforces a **minimum mutation score**. The threshold is still open.

## Timing and estimates (agreed)
- Tests check **structure and determinism**: the same inputs give the same plan, ranking and ZIP file list. They never check absolute latency; speed is checked only by the A/B gate.
- Target estimates come from the **scaling factors in `profiles/hardware_profiles.json`**. Tests read the factors from that JSON, never from copies hard-coded in tests, and assert `estimate == host measurement × factor`.
- Real on-device runs come later and will replace or calibrate the JSON factors in their own spec.
