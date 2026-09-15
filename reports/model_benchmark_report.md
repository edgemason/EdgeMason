# EdgeMason Model Benchmark Report

Updated: 2026-09-16

## Test setup

### Original CPU comparison — 2026-09-12

- API server: `uv run uvicorn api.main:app --host 127.0.0.1 --port 8000`
- Endpoint: `POST /deploy`
- Hardware profile: `x86_64`
- Execution provider: `CPUExecutionProvider`
- Priority: `latency`
- Calibration dataset: `models/samples/imagenetv2_mobilenetv2_50.npz`
- Calibration samples: **50** (`inputs` shape `(50, 3, 224, 224)`)
- Candidates explored: five, including two TVM candidates

### ResNet-152 GPU rerun — 2026-09-16

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GiB
- Driver: 580.178.04
- Host CUDA: 13.0
- Runtime: ONNX Runtime GPU 1.27.0
- Execution-provider mode: `auto`
- TVM candidates: disabled
- Candidates explored: three ONNX candidates
- End-to-end deployment time: **49.771 seconds**
- Dedicated latency method: 10 warm-up runs followed by 100 measured runs
- CUDA measurements disable CPU fallback so mixed execution is not reported as GPU

## Model analysis

| Model | Size (MiB) | Parameters | Nodes | Analyze request |
|---|---:|---:|---:|---|
| MobileNetV2 | 13.344 | 3,487,816 | 170 | 200 OK |
| ResNet50 | 97.414 | 25,530,472 | 122 | 200 OK |
| ResNet152 | 229.398 | 60,117,096 | 360 | 200 OK |

## Original CPU deployment results

| Model | Status | Winner | Provider | Winner p50 (ms) | Avg (ms) | FPS | Size (MiB) | Top-1 | Top-5 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 200 OK | `onnx_static_int8` | CPU | 5.2971 | 5.2849 | 189.12 | 3.500 | 50% | 74% |
| ResNet50 | 200 OK | `onnx_dynamic_int8` | CPU | 29.4124 | 29.5867 | 33.80 | 24.534 | 64% | 82% |
| ResNet152 | Timed out after 10 minutes | — | — | — | — | — | — | — | — |

## Candidate details

### MobileNetV2 — CPU

| Candidate | p50 (ms) | Avg (ms) | FPS | Size (MiB) | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| `onnx_fp32_baseline` | 5.9000 | 6.1046 | 163.63 | 13.344 | 54% | 78% |
| `onnx_static_int8` | 5.2971 | 5.2849 | 189.12 | 3.500 | 50% | 74% |
| `onnx_dynamic_int8` | 11.2826 | 11.2993 | 88.48 | 3.522 | 52% | 76% |
| `tvm_opt_level_3` | 128.4861 | 128.6576 | 7.77 | 12.604 | 54% | 78% |
| `tvm_int8_quantized` | 129.5978 | 130.0866 | 7.69 | 2.905 | 50% | 72% |

### ResNet50 — CPU

| Candidate | p50 (ms) | Avg (ms) | FPS | Size (MiB) | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| `onnx_fp32_baseline` | 60.6078 | 60.2457 | 16.60 | 97.414 | 64% | 86% |
| `onnx_static_int8` | 38.7910 | 38.7952 | 25.77 | 24.530 | 64% | 88% |
| `onnx_dynamic_int8` | 29.4124 | 29.5867 | 33.80 | 24.534 | 64% | 82% |
| `tvm_opt_level_3` | 3064.4816 | 3068.7827 | 0.33 | 90.761 | 64% | 86% |
| `tvm_int8_quantized` | 3295.2738 | 3545.0521 | 0.28 | 14.882 | 64% | 88% |

### ResNet152 — corrected deployment run

These figures come from the end-to-end deployment's two warm-up and five
measured runs per candidate.

| Candidate | Provider | p50 (ms) | Avg (ms) | FPS | Size (MiB) | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|
| `onnx_fp32_baseline` | CUDA | 8.5322 | 8.5395 | 116.98 | 229.398 | 76% | 88% |
| `onnx_static_int8` | CPU | 105.3545 | — | — | 57.854 | — | — |
| `onnx_dynamic_int8` | CPU | 171.7407 | — | — | 57.868 | — | — |

### ResNet152 — dedicated 100-run latency comparison

| Candidate | Provider | Build (s) | Size (MiB) | p50 (ms) | Avg (ms) | p95 (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | CUDA | 1.014 | 229.398 | 7.3904 | 7.5896 | 8.5094 | 131.70 |
| FP32 | CPU | 0.470 | 229.398 | 168.7225 | 166.8892 | 172.9076 | 5.99 |
| Dynamic INT8 | CPU | 1.813 | 57.868 | 164.4200 | 164.7827 | 169.6974 | 6.07 |
| Static INT8 | CPU | 9.707 | 57.854 | 97.7383 | 101.3138 | 108.7450 | 9.87 |

## ResNet-152 timeout diagnosis and resolution

ResNet-152 was always quantizable. Dynamic and static INT8 both produce valid
approximately 57.9 MiB models. The original request timed out because it also
compiled, benchmarked, and evaluated two TVM candidates. Their fixed batch size
of one caused 114 slow TVM inference calls across benchmarking and accuracy
evaluation, taking more than the client's 10-minute timeout.

TVM is now opt-in with `include_tvm=false` by default. The corrected deployment
evaluated only FP32, static INT8, and dynamic INT8 ONNX candidates and completed
in 49.771 seconds.

The ONNX Runtime CUDA provider executes FP32 ResNet-152 entirely on the RTX 4050.
It cannot execute every operator in either INT8 graph. Strict CUDA mode rejects
those graphs rather than hiding CPU fallback; `execution_provider=auto` retries
them on CPU and reports `CPUExecutionProvider`.

Genuine GPU INT8 measurements require a backend with full INT8 graph support,
such as a configured TensorRT quantization and engine-build path.

## Summary

- MobileNetV2 selected static INT8 on CPU and reduced model size by about 74%.
- ResNet50 selected dynamic INT8 on CPU and reduced p50 latency by about 51%.
- ResNet152 deployment now completes in under one minute instead of exceeding
  ten minutes.
- ResNet152 FP32 CUDA reached 7.3904 ms p50 in the dedicated benchmark, about
  22.8 times faster than its 168.7225 ms CPU FP32 result.
- ResNet152 INT8 remains a CPU path with the current ONNX Runtime CUDA provider.

Raw response captures and the detailed ResNet-152 GPU report are stored in this
directory.
