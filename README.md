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
│   ├── main.py                     # FastAPI app: /analyze, /deploy, /download
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
| `max_memory_mb`  | float (optional)| `None`  | Hard memory constraint (delta must be ≤ this)  |
| `priority`       | string          | `latency` | Ranking: `latency`, `memory`, `balanced`    |
| `calibration_file`| file (optional)| `None`  | `.npy` calibration data enabling static INT8    |

Returns: plan summary, per-candidate benchmark results, winner, and a `download_url` to the deployment ZIP.

---

## Planning rules

The rule-based planner (`api/planner/rule_based.py`) decides the candidate set:

| Rule | Condition | Candidate added |
|------|-----------|-----------------|
| 1    | always                                    | `onnx_fp32_baseline` |
| 2    | `model_size_mb > 50` and calibration data | `onnx_static_int8`   |
| 3    | `model_size_mb > 50` and no calibration   | `onnx_dynamic_int8`  |
| 4    | `max_latency_ms < 100`                    | `onnx_dynamic_int8`  |

Rationale:

- **FP32 baseline is always kept** for comparison.
- **Static INT8** is best for large models when calibration data is available — biggest size/latency win.
- **Dynamic INT8** is a cheap, calibration-free win for latency-constrained or large models.

Each candidate carries a `reason` string so the plan is self-documenting, plus a `skipped` list explaining why candidates were *not* selected.

---

## Benchmarking: estimate vs real hardware

The backend currently benchmarks on the **server CPU** (x86, `CPUExecutionProvider`). For an edge target like a Raspberry Pi there are two strategies:

### 1. Estimate from hardware profile (current model)
The server measures each candidate on x86, then applies a per-hardware scaling factor from `profiles/` to **estimate** edge latency:

```
Candidate 1 (FP32):        15ms on x86 × 12 (RPi4 factor) = 180ms (est.)
Candidate 2 (Dynamic INT8):  8ms on x86 × 12                =  96ms (est.)
Candidate 3 (Static INT8):   7ms on x86 × 12                =  84ms (est.)  ← winner
```

Fast and convenient, but approximate — real edge latency will differ.

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
# 1. Install dependencies (uv)
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

`POST /deploy` returns a ZIP containing:

```
edgeforge-deployment.zip
├── model.onnx               # winning (quantized) model
├── Dockerfile               # ARM64 (arm64v8/python:3.11-slim)
├── inference.py             # ONNX Runtime inference entrypoint
├── requirements.txt
├── benchmark_report.json    # latency / memory metrics
└── README.md                # quick-start for the target device
```

On the edge device:

```bash
docker build -t my-model .
docker run my-model
```

---

## Development

```bash
uv sync                 # install deps (uses uv.lock)
uv run pytest           # run tests (tests/)
uv run python -m api.main   # sanity-check imports
```

### Dependencies

- Python `>=3.9,<3.13` (managed venv)
- `fastapi`, `uvicorn`, `pydantic`, `python-multipart`
- `onnx`, `onnxruntime`
- `numpy`, `psutil`
- `tensorflow`, `keras`, `tf2onnx` (for converting `.h5` → `.onnx`)

---

## Roadmap

- [ ] TVM candidates (FP32 / INT8 / AutoTVM) with estimated compile time
- [ ] Real-hardware benchmarking via SSH (scp + on-device benchmark)
- [ ] Hardware profiles with measured scaling factors (`profiles/`)
- [ ] Accuracy evaluation of quantized candidates (not just latency)
- [ ] Accuracy/latency trade-off ranking (`priority=accuracy`)
- [ ] Dockerfile / docker-compose for the backend itself
