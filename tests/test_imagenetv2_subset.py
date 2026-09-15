import numpy as np
from PIL import Image

from scripts.prepare_imagenetv2_subset import build_local_subset, preprocess_image


def test_preprocessing_matches_imagenet_weights():
    image = Image.new("RGB", (300, 250), (255, 0, 0))

    model_input = preprocess_image(image, "mobilenetv2")

    assert model_input.shape == (3, 224, 224)
    assert model_input.dtype == np.float32
    np.testing.assert_allclose(
        model_input[:, 0, 0],
        np.array([2.2489083, -2.0357141, -1.8044444], dtype=np.float32),
        rtol=1e-5,
    )


def test_local_subset_keeps_numeric_imagenet_labels(tmp_path):
    source = tmp_path / "imagenetv2"
    (source / "10").mkdir(parents=True)
    (source / "2").mkdir()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(source / "10" / "red.jpg")
    Image.new("RGB", (32, 32), (0, 0, 0)).save(source / "2" / "black.jpg")
    output = tmp_path / "subset.npz"

    build_local_subset(source, output, model_name="resnet50", sample_count=2)

    with np.load(output, allow_pickle=False) as subset:
        assert subset["inputs"].shape == (2, 3, 224, 224)
        assert subset["labels"].tolist() == [2, 10]
