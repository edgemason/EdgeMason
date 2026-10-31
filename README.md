# EdgeMason

**Edge AI model optimization toolkit** — upload an ONNX model, choose an edge hardware target and constraints, and get back a quantized, benchmarked, ready-to-deploy package (model + Dockerfile + inference script + report).

> Also referenced as **EdgeForge** in the source.

---

## What it does

A single POST request runs the whole pipeline:

```
Upload model + constraints
        │
        ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Analyze    │──▶│    Plan      │──▶│  Quantize    │
│  model stats │   │ candidates   │   │ fp32/int8    │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                             ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Package    │◀──│    Rank      │◀──│  Benchmark   │
│   ZIP        │   │  constraints │   │  latency/mem │
└──────────────┘   └──────────────┘   └──────────────┘
```

1. **Analyze** — inspects the ONNX model: file size (MB), parameter count, operator type histogram, IR version and opsets.
2. **Plan** — a rule-based planner decides *which* optimization candidates are worth trying (see [Planning rules](#planning-rules)).
3. **Quantize** — generates each candidate model file (FP32 copy, dynamic INT8, static INT8).
4. **Benchmark** — runs each candidate with ONNX Runtime on the server CPU and records latency (avg/p50/p95/p99) and memory delta.
5. **Rank** — filters candidates that miss constraints and sorts by your priority (latency / memory / balanced).
6. **Package** — zips the winning model with a Dockerfile, inference script, requirements, benchmark report and README for download.

---

## Architecture

```
edgemason/
├── api/
│   ├── main.py                     # FastAPI app: wires the routers together
│   ├── config.py                   # upload limits, artifact dir, valid options
│   ├── routes/                     # HTTP layer only: parse, validate, call services
│   │   ├── pages.py                # /, /ui, /api, /health
│   │   ├── analyze.py              # POST /analyze
│   │   ├── deploy.py               # POST /deploy
│   │   └── artifacts.py            # GET /download/{artifact_id}
│   ├── services/                   # business logic, no FastAPI routing
│   │   ├── pipeline.py             # plan -> quantize -> benchmark -> rank -> package
│   │   ├── ranking.py              # constraint filtering + priority ordering
│   │   ├── hardware.py             # execution provider selection per hardware profile
│   │   ├── model_validation.py     # ONNX upload checks
│   │   ├── uploads.py              # streaming upload with size limit
│   │   └── artifacts.py            # deployment ZIP storage + TTL cleanup
│   ├── analyzer/
│   │   └── model_analyzer.py       # ONNX model inspection (size, params, ops)
│   ├── planner/
│   │   └── rule_based.py           # candidate strategy selection from rules
│   ├── optimizer/
│   │   ├── onnx_runtime.py         # inference sessions + latency/memory benchmark
│   │   └── quantizer.py            # fp32 / dynamic int8 / static int8 model gen
│   └── packager/
│       └── generator.py            # deployment ZIP (Dockerfile, inference.py, report)
├── models/                         # uploads / working models (gitignored)
├── profiles/                       # hardware profiles (e.g. RPi scaling factors)
├── scripts/                        # utility scripts
├── tests/
├── pyproject.toml                  # project metadata + dependencies
├── requirements.txt
└── uv.lock
```

---

## API

| Method | Endpoint    | Description                                                              |
|--------|-------------|--------------------------------------------------------------------------|
| GET    | `/`         | Health check                                                             |
| POST   | `/analyze`  | Upload an ONNX model and get its analysis report (JSON)                  |
| POST   | `/deploy`   | Full pipeline: analyze → plan → quantize → benchmark → rank → package    |
| GET    | `/download` | Download the generated deployment ZIP (`?path=...`)                      |

### `POST /analyze`

**Form fields:** `file` (the `.onnx` model)

Returns: filename + full analysis report (`model_size_mb`, `total_parameters`, `num_nodes`, `operator_types`, `ir_version`, `opsets`).

### `POST /deploy`

**Form fields:**

| Field            | Type            | Default | Description                                    |
|------------------|-----------------|---------|------------------------------------------------|
| `file`           | file            | —       | The `.onnx` model                              |
| `hardware_id`    | string          | `rpi4`  | Target: `rpi4`, `rpi5`, `x86_64`               |
| `max_latency_ms` | float (optional)| `None`  | Hard latency constraint (p50 must be ≤ this)   |
| `max_memory_mb`  | float (optional)| `None`  | Max estimated memory on the target (MB)        |
| `priority`       | string          | `latency` | Ranking: `latency`, `memory`, `balanced`    |
| `calibration_file`| file (optional)| `None`  | `.npy` calibration data enabling static INT8    |
| `execution_provider` | string | `auto` | Benchmark provider: `auto`, `cpu`, or `cuda` |
| `include_tvm` | boolean | `false` | Include slower TVM compilation candidates |

Returns: plan summary, per-candidate benchmark results, winner, and a `download_url` to the deployment ZIP.

---

## Calibration and accuracy data

A labeled `.npz` (`inputs` + `labels`, up to 1000 images) is split once per
deploy, shuffled with a fixed seed so the split is the same every run:

| Part | Share | Used for |
|---|---|---|
| Calibration | 30%, at most 300 images | static INT8 scales |
| Test | the rest | accuracy of **every** candidate |

Accuracy is never measured on the images used to calibrate, which would make
static INT8 look better than it is. An unlabeled `.npy` has no accuracy to
score, so all of it is used for calibration.

The response includes `data_split` and `warnings`. Small datasets still run, but
warn when there are fewer than 100 calibration images or fewer than 200 test
images (with the approximate ± range of the accuracy).

## Planning rules

The rule-based planner (`api/planner/rule_based.py`) decides the candidate set:

The planner first classifies the model from the analyzer's operator counts:
**CNN** (more `Conv` than `MatMul`/`Gemm`), **MatMul-heavy** (transformer-style), or
**unknown**. Candidates are chosen by **quantization method** (`fp32`, `static_int8`,
`dynamic_int8`), so the same rules cover ONNX Runtime and TVM.

| Rule | Condition | Candidate added |
|------|-----------|-----------------|
| 1    | always                                         | `onnx_fp32_baseline` |
| 2    | calibration data                               | `onnx_static_int8` (+ `onnx_dynamic_int8` for comparison) |
| 3    | no calibration and `model_size_mb > 5`         | `onnx_dynamic_int8`  |
| 4    | `max_latency_ms < 100`                         | `onnx_dynamic_int8`  |
| 5    | `include_tvm`                                  | `tvm_opt_level_3` (fp32), and `tvm_int8_quantized` (static INT8) with calibration |
| 6    | model is a **CNN**                             | **no dynamic INT8** from rules 2–4; a CNN without calibration gets FP32 only, plus a warning to upload calibration images |

Rationale:

- **FP32 baseline is always kept** for comparison.
- **Static INT8** is the INT8 option for CNNs (ONNX Runtime recommends it for CNNs).
- **Dynamic INT8** quantizes activations at run time before every layer. That suits
  MatMul-heavy (transformer-style) models, but on CNNs it is slower than FP32
  (MobileNetV2: 2.4× slower), so CNNs never get it, in any framework.

Each candidate carries a `reason` string so the plan is self-documenting, plus a `skipped` list explaining why candidates were *not* selected.

---

## Benchmarking: estimate vs real hardware

Each candidate gets up to **10 warm-up runs** (ignored: buffers, caches and CPU
clock are still settling) and up to **100 timed runs**. Timing stops after about
**10 s**, but never before 10 timed runs, and warm-up may use at most a quarter
of that, so a large model like ResNet-152 finishes in ~15 s while small models
get all 100 runs. The response reports `warmup_runs` and `timed_runs` for each
candidate. Limits live in `api/config.py`.

Each hardware profile lists the execution providers the device supports, in
order of preference (`"execution_providers": ["cuda", "cpu"]` for Jetson and
x86_64 hosts, `["cpu"]` for Raspberry Pis). The provider is resolved **once per
deploy** and every candidate is benchmarked on it, so FP32 and INT8 always race
on the same device:

- `auto` picks the first provider in the profile that is installed on the host
  (GPU when available, CPU otherwise). It stays on CPU when TVM candidates are
  included, because TVM is compiled for CPU targets.
- An explicit `cuda` request is rejected for CPU-only targets, when
  `CUDAExecutionProvider` is not installed, or together with `include_tvm`.
- On CUDA, a candidate whose operators cannot run on the GPU fails visibly
  instead of silently falling back to CPU.

For an edge target like a Raspberry Pi there are two strategies:

### 1. Estimate from hardware profile (current model)
The server measures each candidate on the host, then multiplies by a factor
from the target's profile to **estimate** latency on the device. Each model type
has its own factor, because INT8 speeds up differently on each CPU (e.g. the
Cortex-A72 in an RPi 4 has no fast INT8 dot-product instructions):

```json
"rpi4": {
  "latency_scaling_factor": 11.8,                                          // FP32
  "latency_scaling_by_quantization": { "dynamic_int8": 15.3, "static_int8": 15.3 }
}
```

```
FP32:         12.0 ms on host × 11.8 = ~141 ms on RPi 4
Static INT8:  11.0 ms on host × 15.3 = ~168 ms on RPi 4
```

Ranking and the `max_latency_ms` limit use this **estimated target latency**.
Memory priority and `max_memory_mb` use **estimated memory** = measured
memory × `memory_overhead_scaling` (a small factor, 1.0–1.2, because the same
model needs about the same RAM on the host and the device).

Measured memory is taken in a **fresh process per candidate**, one candidate at
a time:

```
baseline = RSS before the model is loaded   (Python + onnxruntime only)
load model, warm-up, timed runs
peak     = RSS high-water mark since baseline (Linux VmHWM, reset at baseline)
memory   = peak - baseline                   (weights + activations + ORT buffers)
```

The fresh process keeps leftovers from earlier candidates (and the server's own
memory) out of the number. The runtime itself is in the baseline, so on the
device add roughly 50–100 MB for Python + ONNX Runtime. The peak is exact on
Linux; other platforms fall back to `ru_maxrss`, which can overstate it.

> The INT8 factors are placeholders, not measurements. Replace them with real
> numbers from the device (strategy 2) before trusting the ranking.

### 2. Benchmark on real edge hardware (planned)
SSH into the actual device, copy the model, run the benchmark on-device, and collect **real** measurements:

```
$ scp model.onnx pi@rpi4.local:/tmp/
$ ssh pi@rpi4.local "python benchmark.py model.onnx"
Candidate 3 (Static INT8): 89ms  ← REAL measurement
```

Most accurate; requires network access to the device.

---

## Quick start

```bash
# 1. Install dependencies (uv): product, dev and research groups, with CPU-only torch
uv sync

# 2. Run the API server
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### Example: analyze

```bash
curl -F "file=@models/my_model.onnx" http://127.0.0.1:8000/analyze
```

### Example: deploy

```bash
curl -F "file=@models/my_model.onnx" \
     -F "hardware_id=rpi4" \
     -F "max_latency_ms=200" \
     -F "priority=latency" \
     -F "calibration_file=@calib.npy" \
     http://127.0.0.1:8000/deploy
```

---

## Deployment package

`POST /deploy` returns a download link (valid once, for now) to a ZIP that runs on the
target as is. The layout and commands are the same whichever candidate won, so the
user does not need to know which framework or quantization was used:

```
edgemason-deployment.zip
├── model.onnx               # winning model (model_tvm.tar + tvm_inputs.json if TVM won)
├── inference.py             # finds the model, loads the right runtime, runs it
├── requirements.txt         # exact versions the server tested with (CPU onnxruntime)
├── Dockerfile               # python:3.11-slim, multi-arch (arm64 on a Pi, amd64 on x86)
├── benchmark_report.json    # full benchmark results
└── README.md                # how to run, input shape, expected latency/memory/accuracy
```

On the device (Python 3.11, 64-bit OS):

```bash
pip install -r requirements.txt
python inference.py              # self-test with a random input
python inference.py input.npy    # your own float32 input
```

or with Docker:

```bash
docker build -t edgemason-model .
docker run --rm edgemason-model
```

From Python: `from inference import Model; Model().predict(array)`.

---

## Development

```bash
uv sync                 # install deps (uses uv.lock)
uv run pytest           # run tests (tests/)
uv run python -m api.main   # sanity-check imports
```

### Dependencies

- Python `>=3.11,<3.12` (managed venv)
- `fastapi`, `uvicorn`, `pydantic`, `python-multipart`
- `onnx`, `onnxruntime`
- `numpy`, `psutil`
- `tensorflow`, `keras`

---

## Roadmap

- [ ] TVM candidates (FP32 / INT8 / AutoTVM) with estimated compile time
- [ ] Real-hardware benchmarking via SSH (scp + on-device benchmark)
- [ ] Hardware profiles with measured scaling factors (`profiles/`)
- [ ] Accuracy evaluation of quantized candidates (not just latency)
- [ ] Accuracy/latency trade-off ranking (`priority=accuracy`)
- [ ] Dockerfile / docker-compose for the backend itself
