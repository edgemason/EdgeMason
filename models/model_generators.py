"""Generate ONNX versions of 3 pretrained TorchVision classification models.

Models:
    - MobileNetV2
    - ResNet50
    - ResNet152

Each model is loaded with pretrained ImageNet weights, given an input produced
by its own preprocessing transforms, and exported under the ``models/`` directory.

Usage:
    uv run python -m models.model_generators
    # or
    uv run python models/model_generators.py
"""

from pathlib import Path
from typing import Any

import torch

OUTPUT_DIR = Path(__file__).resolve().parent

MODELS = ("mobilenetv2", "resnet50", "resnet152")


def load_torchvision_model(name: str) -> tuple[torch.nn.Module, Any]:
    """Load a pretrained TorchVision model in eval mode.

    Args:
        name: One of ``mobilenetv2``, ``resnet50``, ``resnet152``.

    Returns:
        A tuple containing the pretrained model in eval mode and its weights.
    """
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required. Install it with: "
            "uv add torch torchvision  (or pip install torch torchvision)"
        ) from exc

    name = name.lower()
    if name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    elif name == "resnet152":
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
    else:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODELS}.")

    model.eval()
    return model, weights


def export_to_onnx(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    output_path: str | Path,
    opset_version: int = 13,
) -> Path:
    """Export a torch model to ONNX format.

    Args:
        model: Torch model in eval mode.
        model_input: Preprocessed input expected by the model.
        output_path: Destination ``.onnx`` file.
        opset_version: ONNX opset version.

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # dynamo=False -> legacy single-file exporter (no .onnx.data sidecar),
        # honours opset_version and stays compatible with onnxruntime 1.18.
        torch.onnx.export(
            model,
            (model_input,),
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=opset_version,
            dynamo=False,
        )
    return output_path


def make_model_input(weights: Any) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    image_data = np.linspace(
        0, 255, 224 * 224 * 3, dtype=np.uint8
    ).reshape(224, 224, 3)
    image = Image.fromarray(image_data)
    return weights.transforms()(image).unsqueeze(0)


def save_sample_inputs(
    model_name: str,
    weights: Any,
    model_input: torch.Tensor,
    output_dir: Path,
) -> None:
    import numpy as np
    import json

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    np.save(samples_dir / f"{model_name}_sample.npy", model_input.numpy())

    # Shared imagenet labels
    labels_path = samples_dir / "imagenet_labels.json"
    if not labels_path.exists():
        with open(labels_path, "w") as f:
            json.dump(weights.meta["categories"], f)


def generate_all(
    output_dir: str | Path = OUTPUT_DIR,
    opset_version: int = 13,
) -> dict[str, Path]:
    """Load all 3 pretrained models and save them as ONNX.

    Args:
        output_dir: Directory to write ``.onnx`` files into.
        opset_version: ONNX opset version for the export.

    Returns:
        Mapping of model name -> exported ONNX path.
    """
    output_dir = Path(output_dir)
    exported: dict[str, Path] = {}
    for name in MODELS:
        print(f"[1/4] Loading pretrained {name} ...")
        model, weights = load_torchvision_model(name)
        model_input = make_model_input(weights)
        out_path = output_dir / f"{name}.onnx"
        print(f"[2/4] Exporting {name} -> {out_path} ...")
        export_to_onnx(model, model_input, out_path, opset_version=opset_version)
        print(f"[3/4] Saving sample for {name} ...")
        save_sample_inputs(name, weights, model_input, output_dir)
        print(f"[4/4] Done: {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
        exported[name] = out_path
    return exported


if __name__ == "__main__":
    generate_all()
