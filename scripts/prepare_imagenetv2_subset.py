import argparse
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


DATASET_API = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "vaishaal/ImageNetV2"
VARIANT = "imagenetv2-matched-frequency-format-val"
MAX_SAMPLES = 256
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
MODEL_NAMES = {"mobilenetv2", "resnet50", "resnet152"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {sorted(MODEL_NAMES)}")

    image = image.convert("RGB")
    width, height = image.size
    if width < height:
        resized = (232, int(232 * height / width))
    else:
        resized = (int(232 * width / height), 232)
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left = round((image.width - 224) / 2)
    top = round((image.height - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))

    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((array - MEAN) / STD).astype(np.float32, copy=False)


def _validate_sample_count(sample_count: int) -> None:
    if not 1 <= sample_count <= MAX_SAMPLES:
        raise ValueError(f"sample_count must be between 1 and {MAX_SAMPLES}")


def _save_subset(
    samples: list[tuple[int, Image.Image]],
    output_path: str | Path,
    model_name: str,
) -> Path:
    inputs = np.stack(
        [preprocess_image(image, model_name) for _, image in samples]
    ).astype(np.float32, copy=False)
    labels = np.asarray([label for label, _ in samples], dtype=np.int64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, inputs=inputs, labels=labels)
    return output_path


def build_local_subset(
    input_dir: str | Path,
    output_path: str | Path,
    model_name: str,
    sample_count: int,
) -> Path:
    _validate_sample_count(sample_count)
    input_dir = Path(input_dir)
    candidates: list[tuple[int, Path]] = []

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            label = int(path.parent.name)
        except ValueError as exc:
            raise ValueError("ImageNetV2 class directories must be numeric") from exc
        if not 0 <= label < 1000:
            raise ValueError("ImageNetV2 labels must be between 0 and 999")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError(f"image is too large: {path.name}")
        candidates.append((label, path))

    candidates.sort(key=lambda item: (item[0], item[1].name))
    selected = candidates[:sample_count]
    if len(selected) != sample_count:
        raise ValueError(f"found {len(selected)} images, expected {sample_count}")

    samples: list[tuple[int, Image.Image]] = []
    for label, path in selected:
        with Image.open(path) as image:
            samples.append((label, image.copy()))
    return _save_subset(samples, output_path, model_name)


def _read_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "EdgeMason/0.1"})
    with urlopen(request, timeout=30) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("dataset metadata response is too large")
    return json.loads(data)


def _read_image(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": "EdgeMason/0.1"})
    with urlopen(request, timeout=30) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("downloaded image is too large")
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def download_subset(
    output_path: str | Path,
    model_name: str,
    sample_count: int,
) -> Path:
    _validate_sample_count(sample_count)
    rows: list[tuple[int, str]] = []
    seen_labels: set[int] = set()

    for offset in range(0, 10_000, 100):
        query = urlencode(
            {
                "dataset": DATASET_NAME,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        page = _read_json(f"{DATASET_API}?{query}")
        for item in page.get("rows", []):
            row = item.get("row", {})
            key = row.get("__key__", "")
            parts = key.split("/")
            if len(parts) < 3 or parts[-3] != VARIANT:
                continue
            label = int(parts[-2])
            if label in seen_labels or not 0 <= label < 1000:
                continue
            image_url = row.get("jpeg", {}).get("src")
            if not isinstance(image_url, str) or not image_url.startswith(
                "https://datasets-server.huggingface.co/cached-assets/"
            ):
                raise ValueError("dataset returned an invalid image URL")
            rows.append((label, image_url))
            seen_labels.add(label)
            if len(rows) == sample_count:
                samples = [(label, _read_image(url)) for label, url in rows]
                return _save_subset(samples, output_path, model_name)

    raise ValueError(f"found {len(rows)} labeled images, expected {sample_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="mobilenetv2")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--input-dir")
    args = parser.parse_args()

    if args.input_dir:
        path = build_local_subset(
            args.input_dir,
            args.output,
            model_name=args.model,
            sample_count=args.samples,
        )
    else:
        path = download_subset(
            args.output,
            model_name=args.model,
            sample_count=args.samples,
        )
    print(path)


if __name__ == "__main__":
    main()
