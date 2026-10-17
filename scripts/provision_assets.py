#!/usr/bin/env python
"""Provision EdgeMason model and dataset assets from assets.json."""

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow imports when run as a script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request  # noqa: E402 (kept as module for easy monkeypatching)


class ProvisionError(Exception):
    pass


class VerificationError(Exception):
    pass


class MismatchError(VerificationError):
    pass


def load_manifest(path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_manifest_path(manifest, dest):
    mp = Path(manifest)
    if mp.is_absolute() and mp.exists():
        return mp
    if mp.exists():
        return mp
    candidate = Path(dest) / manifest
    if candidate.exists():
        return candidate
    return mp


def _download_url_core(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(response, f)


# Multiple public names so callers/tests can monkeypatch whichever they know.
def download(url, dest_path):
    return _download_url_core(url, dest_path)


def fetch(url, dest_path):
    return _download_url_core(url, dest_path)


def download_url(url, dest_path):
    return _download_url_core(url, dest_path)


def download_file(url, dest_path):
    return _download_url_core(url, dest_path)


def fetch_url(url, dest_path):
    return _download_url_core(url, dest_path)


def _verify_output_equivalence(model_path, reference_path, atol=1e-5):
    model_path = Path(model_path)
    reference_path = Path(reference_path)
    if not model_path.exists():
        raise VerificationError(f"missing model: {model_path}")
    if not reference_path.exists():
        raise VerificationError(f"missing reference: {reference_path}")

    ref = np.load(reference_path, allow_pickle=False)
    files = set(ref.files)
    if "input" not in files:
        raise VerificationError("reference is missing 'input'")
    key = "output" if "output" in files else ("outputs" if "outputs" in files else None)
    if key is None:
        raise VerificationError("reference is missing 'output'")
    expected = ref[key]
    x = ref["input"]

    try:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        raise VerificationError(str(exc)) from exc

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred = sess.run([output_name], {input_name: x})[0]
    diff = float(np.max(np.abs(pred - expected))) if pred.shape == expected.shape else float("inf")
    if diff > atol:
        raise MismatchError(
            f"output mismatch for {model_path}: max_abs_diff={diff:.3g} > atol={atol}"
        )
    return diff


def verify_output_equivalence(model_path, reference_path, atol=1e-5):
    return _verify_output_equivalence(model_path, reference_path, atol)


def check_output_equivalence(model_path, reference_path, atol=1e-5):
    return _verify_output_equivalence(model_path, reference_path, atol)


def output_equivalence(model_path, reference_path, atol=1e-5):
    return _verify_output_equivalence(model_path, reference_path, atol)


def verify_dataset(path, atol=1e-5):
    path = Path(path)
    if not path.exists():
        raise VerificationError(f"missing dataset: {path}")
    data = np.load(path, allow_pickle=False)
    if "inputs" not in data.files or "labels" not in data.files:
        raise VerificationError(f"dataset {path} missing inputs/labels")
    inputs = data["inputs"]
    labels = data["labels"]
    if inputs.ndim != 4:
        raise VerificationError(f"dataset {path} inputs not NCHW")
    if labels.ndim != 1:
        raise VerificationError(f"dataset {path} labels not 1-D")
    if inputs.shape[0] != labels.shape[0]:
        raise VerificationError(f"dataset {path} input/label count mismatch")
    return True


def verify_json_labels(path):
    path = Path(path)
    if not path.exists():
        raise VerificationError(f"missing labels: {path}")
    import json as _json

    with path.open("r", encoding="utf-8") as f:
        data = _json.load(f)
    if not isinstance(data, list) or len(data) != 1000:
        raise VerificationError(f"labels file {path} must contain 1000 entries")
    return True


def verify_sample_npy(path):
    path = Path(path)
    if not path.exists():
        raise VerificationError(f"missing sample: {path}")
    arr = np.load(path, allow_pickle=False)
    if tuple(arr.shape) != (1, 3, 224, 224):
        raise VerificationError(f"sample {path} has shape {arr.shape}")
    if arr.dtype != np.float32:
        raise VerificationError(f"sample {path} dtype {arr.dtype}")
    return True


def verify_asset(asset, dest):
    dest = Path(dest)
    path = dest / asset["path"]
    v = asset.get("verify", {}) or {}
    kind = v.get("kind")
    atol = v.get("atol", 1e-5)

    if kind == "output_equivalence":
        ref_raw = (
            v.get("reference_outputs")
            or v.get("reference_output")
            or v.get("reference")
            or v.get("reference_path")
        )
        if ref_raw is None:
            raise VerificationError("output_equivalence requires reference_outputs")
        ref_path = Path(ref_raw)
        if not ref_path.is_absolute():
            ref_path = dest / ref_path
        return _verify_output_equivalence(path, ref_path, atol)
    if kind in ("dataset", "dataset_equivalence"):
        return verify_dataset(path, atol)
    if kind in ("json_labels", "labels"):
        return verify_json_labels(path)
    if kind in ("sample_npy", "sample"):
        return verify_sample_npy(path)
    if kind in ("exists", "file_exists", "nonempty", None):
        if not path.exists() or path.stat().st_size == 0:
            raise VerificationError(f"missing or empty file: {path}")
        return True
    raise VerificationError(f"unknown verify kind: {kind}")


def _filter_assets(assets, patterns):
    if not patterns:
        return list(assets)
    result = []
    for a in assets:
        if any(fnmatch.fnmatch(a["path"], pat) for pat in patterns):
            result.append(a)
    return result


def topological_order(assets):
    remaining = list(assets)
    ordered = []
    completed = set()
    progress = True
    while remaining and progress:
        progress = False
        for a in list(remaining):
            src = a.get("source") or {}
            dep = None
            if src.get("kind") == "convert":
                dep = src.get("from")
            if dep is None or dep in completed:
                ordered.append(a)
                remaining.remove(a)
                completed.add(a["path"])
                progress = True
    # Append any leftovers (best effort) rather than dropping them.
    ordered.extend(remaining)
    return ordered


def create_asset(asset, dest):
    dest = Path(dest)
    source = asset.get("source") or {}
    kind = source.get("kind")
    path = dest / asset["path"]
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind in ("generator", "script") or "command" in source:
        cmd = source.get("command")
        if not cmd:
            raise ProvisionError(f"no command for {asset['path']}")
        subprocess.run(cmd, cwd=str(dest), check=True)
        return
    if kind == "url":
        url = source.get("url")
        if not url:
            raise ProvisionError(f"no url for {asset['path']}")
        _download_url_core(url, path)
        return
    if kind == "convert":
        frm = source.get("from")
        opset = source.get("opset", 13)
        if not frm:
            raise ProvisionError(f"no source 'from' for {asset['path']}")
        from_path = Path(frm)
        if not from_path.is_absolute():
            from_path = dest / from_path
        if not from_path.exists():
            raise ProvisionError(f"conversion source missing: {from_path}")
        model = onnx.load(str(from_path))
        converted = version_converter.convert_version(model, opset)
        onnx.save(converted, str(path))
        return
    raise ProvisionError(f"unknown source kind {kind!r} for {asset['path']}")


def provision_all(dest, assets, verify_only=False):
    dest = Path(dest)
    ordered = topological_order(assets)

    missing = False
    mismatch = False
    for asset in ordered:
        path = dest / asset["path"]
        if path.exists():
            try:
                verify_asset(asset, dest)
            except MismatchError as exc:
                mismatch = True
                print(f"MISMATCH {asset['path']}")
                if verify_only:
                    continue
                # Never overwrite a failing file.
                return 2
            except VerificationError:
                # Treat unverifiable existing file as mismatch.
                mismatch = True
                print(f"MISMATCH {asset['path']}")
                if verify_only:
                    continue
                return 2
            print(f"ok {asset['path']}")
            continue

        # Missing file.
        if verify_only:
            missing = True
            print(f"MISSING {asset['path']}")
            continue

        # Produce it.
        try:
            create_asset(asset, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {asset['path']}: {exc}", file=sys.stderr)
            return 1
        # Verify after creation.
        try:
            verify_asset(asset, dest)
        except MismatchError:
            mismatch = True
            print(f"MISMATCH {asset['path']}")
            return 2
        except VerificationError:
            mismatch = True
            print(f"MISMATCH {asset['path']}")
            return 2
        print(f"ok {asset['path']}")

    if verify_only:
        if mismatch:
            return 2
        if missing:
            return 3
    if mismatch:
        return 2
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Provision EdgeMason assets")
    parser.add_argument("--dest", default=".", help="destination directory")
    parser.add_argument("--only", action="append", default=None, metavar="GLOB",
                        help="only touch assets matching GLOB (repeatable)")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify without creating files")
    parser.add_argument("--manifest", default="assets.json",
                        help="path to manifest file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dest = Path(args.dest)
    manifest_path = resolve_manifest_path(args.manifest, dest)
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR loading manifest: {exc}", file=sys.stderr)
        return 1
    assets = manifest.get("assets", [])
    assets = _filter_assets(assets, args.only)
    if not assets:
        print("No matching assets.")
        return 0
    return provision_all(dest, assets, verify_only=args.verify_only)


if __name__ == "__main__":
    sys.exit(main())