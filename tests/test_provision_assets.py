import io
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import pytest

import scripts.provision_assets as pa


def _tiny_onnx(path, weight_value=0.5):
    w = np.full((4, 8), weight_value, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["h"]),
        helper.make_node("Add", ["h", "B"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten().tolist()),
            helper.make_tensor("B", TensorProto.FLOAT, b.shape, b.flatten().tolist()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))
    return path


def _run_tiny(path, x):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    return sess.run([out], {inp: x})[0]


def test_provision_url_asset(tmp_path, monkeypatch):
    manifest = {"version": 1, "assets": [
        {"path": "data/hello.txt",
         "verify": {"kind": "exists"},
         "source": {"kind": "url", "url": "http://example/hello.txt"}},
    ]}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    def fake_urlopen(url):
        return io.BytesIO(b"hello")

    monkeypatch.setattr("scripts.provision_assets.urllib.request.urlopen", fake_urlopen)

    rc = pa.main(["--dest", str(tmp_path), "--manifest", str(mp)])
    assert rc == 0
    assert (tmp_path / "data" / "hello.txt").read_bytes() == b"hello"


def test_already_exists_ok(tmp_path, monkeypatch):
    target = tmp_path / "data" / "hello.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"hello")

    manifest = {"version": 1, "assets": [
        {"path": "data/hello.txt",
         "verify": {"kind": "exists"},
         "source": {"kind": "url", "url": "http://example/hello.txt"}},
    ]}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    called = {"n": 0}

    def fake_urlopen(url):
        called["n"] += 1
        return io.BytesIO(b"nope")

    monkeypatch.setattr("scripts.provision_assets.urllib.request.urlopen", fake_urlopen)

    import io as _io
    import contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pa.main(["--dest", str(tmp_path), "--manifest", str(mp)])
    assert rc == 0
    assert called["n"] == 0
    assert "ok data/hello.txt" in buf.getvalue()


def test_mismatch_exits_2_file_unchanged(tmp_path):
    # Prepare reference + original model output
    models = tmp_path / "models"
    models.mkdir()
    src = tmp_path / "src.onnx"
    _tiny_onnx(src, 0.5)
    x = np.ones((1, 4), dtype=np.float32)
    y = _run_tiny(src, x)

    refdir = tmp_path / "assets" / "reference"
    refdir.mkdir(parents=True)
    np.savez(refdir / "m_ref.npz", input=x, output=y)

    # Provision target already exists but with perturbed weights (diff > 1e-5)
    target = models / "m.onnx"
    _tiny_onnx(target, 0.501)
    before = target.read_bytes()

    manifest = {"version": 1, "assets": [
        {"path": "models/m.onnx",
         "verify": {"kind": "output_equivalence", "atol": 1e-5, "reference_outputs": "assets/reference/m_ref.npz"},
         "source": {"kind": "url", "url": "http://x/m.onnx"}},
    ]}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    rc = pa.main(["--dest", str(tmp_path), "--manifest", str(mp)])
    assert rc == 2
    assert target.read_bytes() == before


def test_convert_dependency_order(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    # from asset will be listed after convert asset in manifest order
    base = models / "base.onnx"
    _tiny_onnx(base, 0.25)
    # convert to same model output at models/out.onnx (opset 13 identity-ish)
    manifest = {"version": 1, "assets": [
        {"path": "models/out.onnx",
         "verify": {"kind": "exists"},
         "source": {"kind": "convert", "from": "models/base.onnx", "opset": 13}},
        {"path": "models/base.onnx",
         "verify": {"kind": "exists"},
         "source": {"kind": "url", "url": "http://x/base.onnx"}},
    ]}
    # Actually base exists already; make url not needed. Use generator? Instead place base as existing by making its source "exists"? Not supported. We'll monkeypatch convert dependency: remove base from manifest? Instead mark base existing and no source? Add source kind exists not supported. Simpler: base already exists; convert asset source from base. But provision would try to verify base exists (yes) skip. Convert depends on base asset object. Include base asset with source kind url monkeypatched to not called (file exists).
    # Already base exists.
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    import io as _io
    def fake_urlopen(url):
        return _io.BytesIO(b"")
    # Not needed because base exists. but url source skip. noop.
    rc = pa.main(["--dest", str(tmp_path), "--manifest", str(mp)])
    assert rc == 0
    assert (models / "out.onnx").exists()


def test_verify_only_missing(tmp_path):
    manifest = {"version": 1, "assets": [
        {"path": "models/missing.onnx",
         "verify": {"kind": "exists"},
         "source": {"kind": "url", "url": "http://x/m.onnx"}},
    ]}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    # ensure nothing pre-exists
    rc = pa.main(["--dest", str(tmp_path), "--manifest", str(mp), "--verify-only"])
    assert rc == 3
    assert not (tmp_path / "models" / "missing.onnx").exists()


def test_only_filter(tmp_path, monkeypatch):
    # two assets, one onnx one txt
    (tmp_path / "models").mkdir()
    # Precreate txt to make it existing+ok? Actually filter should skip touching txt entirely.
    manifest = {"version": 1, "assets": [
        {"path": "models/a.onnx", "verify": {"kind": "exists"}, "source": {"kind": "url", "url": "http://x/a.onnx"}},
        {"path": "docs/readme.txt", "verify": {"kind": "exists"}, "source": {"kind": "url", "url": "http://x/r.txt"}},
    ]}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    def fake_urlopen(url):
        # returns content based on url
        if url.endswith("a.onnx"):
            return io.BytesIO(b"onnx")
        return io.BytesIO(b"txt")

    monkeypatch.setattr("scripts.provision_assets.urllib.request.urlopen", fake_urlopen)

    rc = pa.main(["--dest", str(tmp_path), "--manifest", str(mp), "--only", "models/*.onnx"])
    assert rc == 0
    assert (tmp_path / "models" / "a.onnx").exists()
    assert not (tmp_path / "docs" / "readme.txt").exists()


def test_output_equivalence_perturbed(tmp_path):
    src = tmp_path / "orig.onnx"
    pert = tmp_path / "pert.onnx"
    _tiny_onnx(src, 0.5)
    _tiny_onnx(pert, 0.501)
    x = np.ones((1, 4), dtype=np.float32)
    y = _run_tiny(src, x)

    ref = tmp_path / "ref.npz"
    np.savez(ref, input=x, output=y)

    # original passes
    assert pa.verify_output_equivalence(src, ref, atol=1e-5) is not None or True
    try:
        pa.verify_output_equivalence(src, ref, atol=1e-5)
    except Exception:
        pytest.fail("original should pass")

    with pytest.raises(pa.MismatchError):
        pa.verify_output_equivalence(pert, ref, atol=1e-5)


def test_real_manifest_schema():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = pa.load_manifest(repo_root / "assets.json")
    assert "version" in manifest
    assert isinstance(manifest["assets"], list)
    assert len(manifest["assets"]) > 0
    required = {"path", "verify", "source"}
    for asset in manifest["assets"]:
        assert required.issubset(asset.keys())
        assert isinstance(asset["path"], str)
        assert isinstance(asset["verify"], dict) and "kind" in asset["verify"]
        assert isinstance(asset["source"], dict) and "kind" in asset["source"]
