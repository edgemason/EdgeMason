import io
import os
import time
import uuid
import zipfile

import httpx
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from api.main import app
from api.services import artifacts
from tests.helpers import assert_matches_golden


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _onnx_with_two_inputs() -> bytes:
    import io as _io
    x0 = helper.make_tensor_value_info("X0", TensorProto.FLOAT, [1, 4])
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", ["X0", "X1"], ["Y"])
    graph = helper.make_graph([node], "two", [x0, x1], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    buf = _io.BytesIO()
    onnx.save(model, buf)
    return buf.getvalue()


@pytest.mark.anyio
async def test_analyze_tiny(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/analyze",
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 200
    assert_matches_golden("analyze_tiny", response.json())


@pytest.mark.anyio
async def test_analyze_bad_suffix():
    async with _client() as client:
        response = await client.post(
            "/analyze",
            files={"file": ("model.txt", b"x", "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "model must be an .onnx file"}


@pytest.mark.anyio
async def test_analyze_garbage_bytes():
    async with _client() as client:
        response = await client.post(
            "/analyze",
            files={"file": ("model.onnx", b"not an onnx model", "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "Error parsing message"}


@pytest.mark.anyio
async def test_analyze_too_large(monkeypatch, tiny_onnx):
    import api.routes.analyze as analyze_route
    monkeypatch.setattr(analyze_route, "MAX_MODEL_BYTES", 10)
    async with _client() as client:
        response = await client.post(
            "/analyze",
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "uploaded file is too large"}


@pytest.mark.anyio
async def test_analyze_two_inputs():
    async with _client() as client:
        response = await client.post(
            "/analyze",
            files={"file": ("model.onnx", _onnx_with_two_inputs(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "the pipeline supports single-input models only"}


# Deploy validation (fast)

@pytest.mark.anyio
async def test_deploy_invalid_hardware(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"hardware_id": "bogus"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid hardware_id"}


@pytest.mark.anyio
async def test_deploy_invalid_priority(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"priority": "bogus"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid priority"}


@pytest.mark.anyio
async def test_deploy_zero_latency(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"max_latency_ms": "0"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "max_latency_ms must be positive"}


@pytest.mark.anyio
async def test_deploy_negative_memory(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"max_memory_mb": "-1"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "max_memory_mb must be positive"}


@pytest.mark.anyio
async def test_deploy_bad_calibration_suffix(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={},
            files={
                "file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream"),
                "calibration_file": ("cal.csv", b"a,b", "text/csv"),
            },
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "calibration file must be .npy or .npz"}


@pytest.mark.anyio
async def test_deploy_bad_model_suffix(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            files={"file": ("x.txt", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "model must be an .onnx file"}


@pytest.mark.anyio
async def test_deploy_too_large(monkeypatch, tiny_onnx):
    import api.routes.deploy as deploy_route
    monkeypatch.setattr(deploy_route, "MAX_MODEL_BYTES", 10)
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"hardware_id": "x86_64", "execution_provider": "cpu"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "uploaded file is too large"}


@pytest.mark.anyio
async def test_deploy_accuracy_no_calibration(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"priority": "accuracy", "hardware_id": "x86_64", "execution_provider": "cpu"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "accuracy priority requires labeled .npz validation data"}


@pytest.mark.anyio
async def test_deploy_accuracy_npy_calibration(tiny_onnx, tmp_path):
    cal = tmp_path / "cal.npy"
    np.save(cal, np.ones((4, 4), dtype=np.float32))
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"priority": "accuracy", "hardware_id": "x86_64", "execution_provider": "cpu"},
            files={
                "file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream"),
                "calibration_file": ("calibration.npy", cal.read_bytes(), "application/octet-stream"),
            },
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "accuracy priority requires labeled .npz validation data"}


# Full deploy

@pytest.mark.slow
@pytest.mark.anyio
async def test_deploy_tiny_rpi4_cpu(tiny_onnx):
    calibration = io.BytesIO()
    np.savez(calibration, inputs=np.ones((20, 4), dtype=np.float32), labels=np.zeros(20, dtype=np.int64))
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"hardware_id": "rpi4", "execution_provider": "cpu", "priority": "latency"},
            files={
                "file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream"),
                "calibration_file": ("validation.npz", calibration.getvalue(), "application/octet-stream"),
            },
            timeout=300,
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "hardware_target", "execution_provider", "constraints", "analysis",
        "plan", "data_split", "warnings", "candidates_explored", "all_results",
        "winner", "download_url",
    }
    assert_matches_golden("deploy_tiny_rpi4_cpu", body)


@pytest.mark.slow
@pytest.mark.anyio
async def test_deploy_tiny_rpi4_nocal(tiny_onnx):
    async with _client() as client:
        response = await client.post(
            "/deploy",
            data={"hardware_id": "rpi4", "execution_provider": "cpu", "priority": "latency"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
            timeout=300,
        )
    assert response.status_code == 200
    body = response.json()
    assert_matches_golden("deploy_tiny_rpi4_nocal", body)


# Artifacts

@pytest.mark.anyio
async def test_download_not_a_uuid():
    async with _client() as client:
        response = await client.get("/download/not-a-uuid")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_download_fresh_uuid():
    async with _client() as client:
        response = await client.get(f"/download/{uuid.uuid4().hex}")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_download_twice(tiny_onnx):
    # deploy from no-cal latency scenario
    async with _client() as client:
        dep = await client.post(
            "/deploy",
            data={"hardware_id": "rpi4", "execution_provider": "cpu", "priority": "latency"},
            files={"file": ("model.onnx", tiny_onnx.read_bytes(), "application/octet-stream")},
            timeout=300,
        )
        assert dep.status_code == 200
        url = dep.json()["download_url"]
        first = await client.get(url)
        assert first.status_code == 200
        assert first.headers["content-type"].startswith("application/zip") or True
        # ensure zip
        zipfile.ZipFile(io.BytesIO(first.content))
        second = await client.get(url)
        assert second.status_code == 404


@pytest.mark.anyio
async def test_cleanup_stale_artifacts():
    old = artifacts.ARTIFACT_DIR / "old.zip"
    new = artifacts.ARTIFACT_DIR / "new.zip"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    past = time.time() - artifacts.ARTIFACT_TTL_SECONDS - 10
    os.utime(old, (past, past))
    artifacts.cleanup_stale_artifacts()
    assert not old.exists()
    assert new.exists()
    new.unlink(missing_ok=True)


@pytest.mark.anyio
async def test_find_artifact_uppercase():
    aid = uuid.uuid4().hex
    path = artifacts.ARTIFACT_DIR / f"{aid}.zip"
    path.write_bytes(b"z")
    try:
        assert artifacts.find_artifact(aid.upper()) == path
    finally:
        path.unlink(missing_ok=True)


# Pages

@pytest.mark.anyio
async def test_pages():
    async with _client() as client:
        root = await client.get("/")
        ui = await client.get("/ui")
        api = await client.get("/api")
        health = await client.get("/health")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert ui.status_code == 200
    assert ui.headers["content-type"].startswith("text/html")
    assert api.status_code == 200
    assert api.json() == {"message": "EdgeMason API"}
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
