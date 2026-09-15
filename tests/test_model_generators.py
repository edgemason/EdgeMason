from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
models = pytest.importorskip("torchvision.models")

from models import model_generators


def test_model_input_uses_the_weights_transform():
    transformed = torch.ones(3, 224, 224)

    class Weights:
        def transforms(self):
            return lambda image: transformed

    model_input = model_generators.make_model_input(Weights())

    torch.testing.assert_close(model_input, transformed.unsqueeze(0))


def test_model_inputs_match_pretrained_weight_requirements():
    weights_by_model = (
        models.MobileNet_V2_Weights.DEFAULT,
        models.ResNet50_Weights.DEFAULT,
        models.ResNet152_Weights.DEFAULT,
    )

    for weights in weights_by_model:
        model_input = model_generators.make_model_input(weights)

        assert model_input.shape == (1, 3, 224, 224)
        assert model_input.dtype == torch.float32


def test_export_passes_model_input_as_positional_arguments(monkeypatch, tmp_path):
    model = torch.nn.Identity()
    model_input = torch.zeros(1, 3, 224, 224)
    received = None

    def fake_export(exported_model, args, output_path, **kwargs):
        nonlocal received
        received = (exported_model, args, output_path)

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = model_generators.export_to_onnx(
        model, model_input, tmp_path / "model.onnx"
    )

    assert received == (model, (model_input,), str(tmp_path / "model.onnx"))
    assert output_path == tmp_path / "model.onnx"


def test_generate_all_accepts_string_output_directory(monkeypatch, tmp_path):
    seen_output_dirs = ()
    model_input = torch.zeros(1, 3, 224, 224)

    def record_output_dir(name, weights, input_tensor, output_dir):
        nonlocal seen_output_dirs
        seen_output_dirs = (*seen_output_dirs, output_dir)

    monkeypatch.setattr(
        model_generators,
        "load_torchvision_model",
        lambda name: (torch.nn.Identity(), object()),
    )
    monkeypatch.setattr(
        model_generators, "make_model_input", lambda weights: model_input
    )
    monkeypatch.setattr(
        model_generators,
        "export_to_onnx",
        lambda model, input_tensor, output_path, opset_version: Path(output_path),
    )
    monkeypatch.setattr(
        model_generators,
        "save_sample_inputs",
        record_output_dir,
    )
    monkeypatch.setattr(Path, "stat", lambda self: type("Stat", (), {"st_size": 0})())

    model_generators.generate_all(str(tmp_path))

    assert seen_output_dirs == (tmp_path, tmp_path, tmp_path)
