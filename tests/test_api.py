import io
import zipfile

import httpx
import numpy as np
import pytest

from api.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint_reports_ready():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_deploy_runs_all_onnx_candidates_and_downloads_package(
    tiny_onnx,
    tmp_path,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(tiny_onnx.read_bytes())

    calibration = io.BytesIO()
    np.savez(
        calibration,
        inputs=np.ones((4, 4), dtype=np.float32),
        labels=np.zeros(4, dtype=np.int64),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=60,
    ) as client:
        response = await client.post(
            "/deploy",
            data={
                "hardware_id": "x86_64",
                "priority": "accuracy",
                "execution_provider": "cpu",
            },
            files={
                "file": (
                    "model.onnx",
                    model_path.read_bytes(),
                    "application/octet-stream",
                ),
                "calibration_file": (
                    "validation.npz",
                    calibration.getvalue(),
                    "application/octet-stream",
                ),
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        results = {result["name"]: result for result in body["all_results"]}
        assert set(results) == {
            "onnx_fp32_baseline",
            "onnx_dynamic_int8",
            "onnx_static_int8",
        }
        assert body["candidates_explored"] == 3
        assert body["constraints"]["execution_provider"] == "cpu"
        assert body["execution_provider"] == "cpu"
        for name in results:
            assert results[name]["success"] is True
            assert results[name]["provider"] == "CPUExecutionProvider"
            # 4 images -> 1 to calibrate, 3 to test; all scored on the same 3.
            assert results[name]["accuracy"]["sample_count"] == 3
        assert body["data_split"] == {"calibration_samples": 1, "test_samples": 3}
        assert any("calibration images" in warning for warning in body["warnings"])
        assert any("test images" in warning for warning in body["warnings"])
        assert body["winner"]["name"] in results

        download = await client.get(body["download_url"])
        assert download.status_code == 200
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            assert set(archive.namelist()) >= {
                "model.onnx",
                "inference.py",
                "requirements.txt",
                "benchmark_report.json",
            }


@pytest.mark.anyio
async def test_deploy_rejects_invalid_execution_provider(tiny_onnx):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/deploy",
            data={"execution_provider": "quantum"},
            files={
                "file": (
                    "model.onnx",
                    tiny_onnx.read_bytes(),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid execution_provider"}


@pytest.mark.anyio
async def test_deploy_rejects_unavailable_cuda_provider(tiny_onnx, monkeypatch):
    from api.services import hardware

    monkeypatch.setattr(
        hardware.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/deploy",
            data={"hardware_id": "x86_64", "execution_provider": "cuda"},
            files={
                "file": (
                    "model.onnx",
                    tiny_onnx.read_bytes(),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "CUDAExecutionProvider is not available"}


@pytest.mark.anyio
async def test_deploy_rejects_cpu_tvm_candidates_for_cuda_request(
    tiny_onnx,
    monkeypatch,
):
    from api.services import hardware

    monkeypatch.setattr(
        hardware.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/deploy",
            data={
                "hardware_id": "x86_64",
                "execution_provider": "cuda",
                "include_tvm": "true",
            },
            files={
                "file": (
                    "model.onnx",
                    tiny_onnx.read_bytes(),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "include_tvm cannot be combined with execution_provider=cuda"
    }


@pytest.mark.anyio
async def test_deploy_rejects_cuda_for_cpu_only_target(tiny_onnx, monkeypatch):
    from api.services import hardware

    monkeypatch.setattr(
        hardware.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/deploy",
            data={"hardware_id": "rpi4", "execution_provider": "cuda"},
            files={
                "file": (
                    "model.onnx",
                    tiny_onnx.read_bytes(),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "rpi4 does not support execution_provider=cuda"}


@pytest.mark.parametrize(
    ("hardware_id", "installed", "include_tvm", "expected"),
    [
        ("rpi4", ["CUDAExecutionProvider", "CPUExecutionProvider"], False, "cpu"),
        ("jetson_nano", ["CUDAExecutionProvider", "CPUExecutionProvider"], False, "cuda"),
        ("x86_64", ["CUDAExecutionProvider", "CPUExecutionProvider"], False, "cuda"),
        ("x86_64", ["CPUExecutionProvider"], False, "cpu"),
        ("x86_64", ["CUDAExecutionProvider", "CPUExecutionProvider"], True, "cpu"),
    ],
)
def test_auto_provider_follows_hardware_profile(
    hardware_id,
    installed,
    include_tvm,
    expected,
    monkeypatch,
):
    from api.services import hardware

    monkeypatch.setattr(hardware.ort, "get_available_providers", lambda: installed)

    assert hardware.select_execution_provider(hardware_id, "auto", include_tvm) == expected


@pytest.mark.anyio
async def test_health_answers_while_deploy_is_running(tiny_onnx, monkeypatch):
    import asyncio
    import time

    from api.routes import deploy

    def slow_deploy(request):
        time.sleep(2)  # blocking, like quantization and benchmarking
        return {"ok": True}

    monkeypatch.setattr(deploy, "run_deploy", slow_deploy)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
        deploy_task = asyncio.create_task(
            client.post(
                "/deploy",
                data={"hardware_id": "x86_64", "execution_provider": "cpu"},
                files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
            )
        )
        await asyncio.sleep(0.3)  # deploy is now inside slow_deploy

        start = time.perf_counter()
        health = await client.get("/health")
        health_seconds = time.perf_counter() - start

        assert health.status_code == 200
        assert not deploy_task.done()
        assert health_seconds < 0.5
        assert (await deploy_task).json() == {"ok": True}


def test_gpu_warning_explains_cpu_only_candidates():
    from api.services.pipeline import gpu_warnings

    def result(name, quantization, success, error=None):
        return {
            "candidate": {"name": name, "quantization": quantization},
            "benchmark": {"success": success},
            "error": error,
        }

    cpu_only = (
        "[ONNXRuntimeError] : 1 : FAIL : This session contains graph nodes that are assigned "
        "to the default CPU EP, but fallback to CPU EP has been explicitly disabled by the user."
    )
    results = [
        result("onnx_fp32_baseline", "fp32", True),
        result("onnx_static_int8", "static_int8", False, cpu_only),
        result("onnx_dynamic_int8", "dynamic_int8", False, "some other failure"),
    ]

    warnings = gpu_warnings(results, "cuda")
    assert len(warnings) == 1
    assert warnings[0].startswith("onnx_static_int8 cannot run on the GPU")
    assert "Choose the CPU execution provider" in warnings[0]
    assert gpu_warnings(results, "cpu") == []
