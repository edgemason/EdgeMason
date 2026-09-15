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
        for name in results:
            assert results[name]["success"] is True
            assert results[name]["provider"] == "CPUExecutionProvider"
            assert results[name]["accuracy"]["sample_count"] == 4
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
    from api import main

    monkeypatch.setattr(
        main.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/deploy",
            data={"execution_provider": "cuda"},
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
    from api import main

    monkeypatch.setattr(
        main.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/deploy",
            data={"execution_provider": "cuda", "include_tvm": "true"},
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
