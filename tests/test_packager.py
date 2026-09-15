import zipfile

from api.packager.generator import create_deployment_package


def test_create_deployment_package(tiny_onnx, tmp_path):
    output = create_deployment_package(
        model_path=str(tiny_onnx),
        benchmark_report={"latency_ms": {"p50": 12.3}},
        hardware_id="rpi4",
        output_path=str(tmp_path / "deploy.zip"),
    )

    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        assert "model.onnx" in names
        assert "Dockerfile" in names
        assert "inference.py" in names
        assert "requirements.txt" in names
        assert "benchmark_report.json" in names
        assert "README.md" in names
