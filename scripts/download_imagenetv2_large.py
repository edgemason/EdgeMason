"""Download a large ImageNetV2 subset (up to 1000 images, one per class).

Reuses the metadata pagination and preprocessing of prepare_imagenetv2_subset,
but downloads images one at a time with retries and saves a checkpoint every
`--checkpoint` images, so a long download survives a network hiccup and can
resume. One preprocessed .npz (inputs + labels) is written, usable for every
model (the ImageNet preprocessing is shared).

Usage:
    uv run python scripts/download_imagenetv2_large.py \
        --output models/samples/imagenetv2_1000.npz --samples 1000
"""

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_imagenetv2_subset import (  # noqa: E402
    DATASET_API,
    DATASET_NAME,
    VARIANT,
    _read_image,
    _read_json,
    preprocess_image,
)
from urllib.error import HTTPError  # noqa: E402


def _retry(fn, what: str, retries: int = 6):
    """Call fn with exponential backoff, handling HTTP 429 rate limiting."""
    for attempt in range(retries):
        try:
            return fn()
        except HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))
                print(f"  429 on {what}; waiting {wait}s (attempt {attempt + 1})", flush=True)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise


def collect_urls(sample_count: int) -> list[tuple[int, str]]:
    """Paginate the dataset metadata and return (label, url), one per class."""
    rows: list[tuple[int, str]] = []
    seen: set[int] = set()
    for offset in range(0, 10_000, 100):
        query = urlencode(
            {"dataset": DATASET_NAME, "config": "default", "split": "train",
             "offset": offset, "length": 100}
        )
        page = _retry(lambda: _read_json(f"{DATASET_API}?{query}"), f"metadata offset {offset}")
        if page is None:
            raise ValueError(f"failed to read metadata offset {offset}")
        time.sleep(0.4)  # pace metadata requests to avoid rate limiting
        for item in page.get("rows", []):
            row = item.get("row", {})
            parts = row.get("__key__", "").split("/")
            if len(parts) < 3 or parts[-3] != VARIANT:
                continue
            label = int(parts[-2])
            if label in seen or not 0 <= label < 1000:
                continue
            url = row.get("jpeg", {}).get("src")
            if not isinstance(url, str) or not url.startswith(
                "https://datasets-server.huggingface.co/cached-assets/"
            ):
                continue
            rows.append((label, url))
            seen.add(label)
            if len(rows) >= sample_count:
                return rows
    return rows


def download(output: Path, sample_count: int, checkpoint: int, retries: int = 3) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] collecting image URLs ...", flush=True)
    urls = collect_urls(sample_count)
    print(f"[{time.strftime('%H:%M:%S')}] {len(urls)} URLs; downloading images ...", flush=True)

    inputs: list[np.ndarray] = []
    labels: list[int] = []
    for index, (label, url) in enumerate(urls):
        try:
            image = _retry(lambda: _read_image(url), f"image {index}", retries=retries)
            if image is None:
                raise ValueError("image is None")
            inputs.append(preprocess_image(image, "mobilenetv2"))
            labels.append(label)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {index} (label {label}): {exc}", flush=True)
        time.sleep(0.15)  # pace image requests
        if (index + 1) % checkpoint == 0 or index + 1 == len(urls):
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output, inputs=np.stack(inputs).astype(np.float32, copy=False),
                     labels=np.asarray(labels, dtype=np.int64))
            print(f"[{time.strftime('%H:%M:%S')}] saved {len(labels)}/{len(urls)} -> {output.name}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] done: {len(labels)} images, "
          f"{len(set(labels))} classes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a large ImageNetV2 subset.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "samples" / "imagenetv2_1000.npz")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--checkpoint", type=int, default=100)
    args = parser.parse_args()
    download(args.output, args.samples, args.checkpoint)


if __name__ == "__main__":
    main()
