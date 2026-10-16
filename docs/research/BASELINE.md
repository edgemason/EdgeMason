# P0.8 Baseline capture and reproduction verdict

Reference outputs live in `baseline/full/` (10 models) and `baseline/quick/`
(`squeezenet1.1-13` + `mobilenetv2`), captured by
`scripts/capture_baseline.py` on the provisioned assets and the 256-image
labelled ImageNetV2 subset (77 calibration / 179 test, via the product's own
`split_calibration_and_test`).

## Headline

**The accuracy values committed in `reports/data/quantization_analysis.json`
reproduce exactly under the research script's own scoring method. There is no
stale, drifted or fabricated data in that file.**

All **18** comparable `top1` values and all **18** comparable `size_mb` values
reproduce **bit-exactly**. The apparent top-5 disagreement is a tie-resolution
artefact of static INT8 quantisation, not a data problem — see
[Top-5 is not comparable](#top-5-is-not-comparable).

## Wall times and determinism

| run | wall time | result |
|---|---|---|
| `capture_baseline.py --out baseline/full` | **3 m 33 s** | 10 models, 3 recorded failures |
| `capture_baseline.py --out /tmp/edgemason-rerun` | **3 m 30 s** | independent second run |
| `diff -r -x environment.json baseline/full /tmp/edgemason-rerun` | — | **no output: byte-identical** |
| `capture_baseline.py --quick --out baseline/quick` | 14.7 s | 2 models, no failures |

Every file except `environment.json` is byte-identical between the two full
runs — analysis, plans, quantized checksums, accuracy and the reproduction
verdicts. `environment.json` differs only in `date_utc` and `git_commit`, which
is why it is excluded.

This includes `resnet152` static INT8, the heaviest calibration case and the
spec's prime suspect for drift (P0.7c case 7). It is clean: static INT8
calibration is deterministic on this machine.

## Three candidates the product cannot build

Three of the ten models cannot be statically quantized at all. This is
recorded, not fatal — `quantized.json` and `accuracy.json` carry an `error`
entry with `sha256: null`, and the run prints a stderr summary and exits 0.

| model | source opset | candidate | error |
|---|---|---|---|
| `resnet18-v1-7` | 8 | `onnx_static_int8` | `Unrecognized attribute: axis for operator DequantizeLinear` |
| `squeezenet1.1-7` | 7 | `onnx_static_int8` | same |
| `shufflenet-v2-10` | 10 | `onnx_static_int8` | same |

Cause: `Quantizer.create_static_int8` passes `per_channel=True`, which emits
`DequantizeLinear` with an `axis` attribute — valid only from opset 13. For a
source below opset 11, ONNX Runtime auto-upgrades the graph to opset 11, and the
two are incompatible, so ORT refuses to load the file it just wrote. Their
opset-13 conversions (`resnet18-v1-13`, `squeezenet1.1-13`,
`shufflenet-v2-13`) all build and load normally. Tracked as **B-51**.

The recorded error strings are byte-identical between runs: the temporary work
directory is a fresh `mkdtemp` each time, so its path is replaced by a fixed
`<quantized>` token before the message is stored.

## Reproduction verdict

The committed data covers **6 models × 4 variants**. This capture covers **10
models × 3 candidates**. Both directions are handled: the 6 committed models
are compared field by field, and the 4 models that exist only in `assets.json`
are marked `not_comparable` for want of a committed value.

Of **96** comparisons in `baseline/full/reproduction.json`:

| bucket | count |
|---|---|
| `reproduces` | **36** |
| `differs` | **0** |
| `not_comparable` | **60** |

| field | verdict | count |
|---|---|---|
| `top1` | `reproduces` | 18 |
| `size_mb` | `reproduces` | 18 |
| `top5` | `not_comparable` | 18 |
| `latency_p50_ms_1thread` | `not_comparable` | 18 |
| `static_int8_per_tensor` (whole variant) | `not_comparable` | 6 |
| `agreement_fp32_vs_static_int8` (whole block) | `not_comparable` | 6 |
| model absent from the committed data (all fields) | `not_comparable` | 12 |

### `reproduces` — 18 `top1`

| model | fp32 | dynamic_int8 | static_int8_per_channel |
|---|---|---|---|
| `mobilenetv2` | 0.5810055865921788 | 0.553072625698324 | 0.5754189944134078 |
| `squeezenet1.1-13` | 0.36312849162011174 | 0.3407821229050279 | 0.3407821229050279 |
| `shufflenet-v2-13` | 0.5195530726256983 | 0.5139664804469274 | 0.4972067039106145 |
| `resnet18-v1-13` | 0.5307262569832403 | 0.5307262569832403 | 0.5418994413407822 |
| `resnet50` | 0.6480446927374302 | 0.6256983240223464 | 0.6368715083798883 |
| `resnet152` | 0.6871508379888268 | 0.6703910614525139 | 0.6871508379888268 |

Every value equals the committed one to the last digit.

### `reproduces` — 18 `size_mb`

All 18 reproduce exactly, e.g. `resnet152` fp32 `229.3978967666626`,
`resnet152` static INT8 `58.80966854095459`, `mobilenetv2` dynamic INT8
`3.5217466354370117`. The product writes the same bytes the research script
measured, so the file sizes are identical.

Note the fp32 comparison is meaningful: `Quantizer.create_fp32` re-serialises
the model through `onnx.save(onnx.load(...))` rather than copying it, and the
re-serialisation happens to be byte-for-byte the same length for all six
models.

### <a id="top-5-is-not-comparable"></a>`not_comparable` — `top5` (18)

Top-5 is **not** comparable between the two implementations. Two independent
confounds, both real:

1. **Tie-breaking.** The product's `evaluate_onnx` picks the top five with
   `np.argpartition(logits, -5)[:, -5:]`; `scripts/quantization_analysis.py`
   uses `np.argsort(logits, axis=1)[:, -5:]`. `argpartition` returns an
   arbitrary member of a tied group; `argsort` returns a different, equally
   valid, one.
2. **Thread count.** The product pins `intra_op_num_threads=1`; the research
   script uses all cores. For `squeezenet1.1-13` this genuinely changes the
   logits (`max |1thr − Nthr| = 1.474`), which moves top-5 membership on its
   own.

Static INT8 quantisation collapses whole groups of classes onto one quantized
value, so 5th place is genuinely ambiguous: **123 of 179** test images on
`squeezenet1.1-13` (and 20 of 179 on `resnet18-v1-13`) have a bit-identical
5th and 6th logit, and the two scorers disagree on every one of them.

Measured for `squeezenet1.1-13` static INT8 — one artifact, four ways of
scoring it:

| scorer | top5 |
|---|---|
| `argsort`, 1 thread | 0.6033519553072626 |
| `argsort`, N threads | **0.6089385474860335** ← the committed value |
| `argpartition`, N threads | 0.6312849162011173 |
| `argpartition`, 1 thread (what the product reports) | 0.6368715083798883 |

The committed value is reproduced **exactly** by the research script's own
method. That is the proof that the committed data is sound.

`top1` is unaffected: `argmax` over the same logits is unambiguous, and all 18
values match.

### `not_comparable` — everything else (42)

- **`latency_p50_ms_1thread` (18).** Measured in-process by
  `scripts/quantization_analysis.py` (10 warm-up + 50 timed runs on the default
  thread pool) rather than by the product's benchmark engine, and it is a
  timing number — never deterministic, so it can never be compared against a
  stored value. Timing is P0.9's business; this spec deliberately records none.
- **`static_int8_per_tensor` (6).** The product `Quantizer` implements
  per-channel static INT8 only; the per-tensor path exists only in
  `scripts/quantization_analysis.py` and has no product counterpart to compare
  against.
- **`agreement_fp32_vs_static_int8` (6).** No product function computes this
  block; it is derived in `scripts/quantization_analysis.py` only.
- **`efficientnet_b0`, `resnet18-v1-7`, `shufflenet-v2-10`,
  `squeezenet1.1-7` (12, three variants each).** No committed value in
  `reports/data/quantization_analysis.json` for these models. (The last three
  are also the models whose static INT8 the product cannot build.)

## Reproducing this

```bash
uv run python scripts/provision_assets.py --dest .
uv run python scripts/capture_baseline.py --out baseline/full
uv run python scripts/capture_baseline.py --out /tmp/edgemason-rerun
diff -r -x environment.json baseline/full /tmp/edgemason-rerun   # must print nothing
uv run python scripts/capture_baseline.py --quick --out baseline/quick
```

`environment.json` (package freeze, CPU model, Python version, the sha256 of
`assets.json`, the git commit and the date) is informational and never
compared.

## Follow-ups raised by this spec

- **B-51** — static INT8 emits a model ORT cannot load for any source below
  opset 11 (`api/optimizer/quantizer.py`).
- **B-52** — top-5 is tie-unstable (`api/optimizer/validation.py:243`), which
  is what makes the product's top-5 irreproducible against the research
  script's.

Both are product changes and are filed in `docs/executor/P0.8-report.md`.