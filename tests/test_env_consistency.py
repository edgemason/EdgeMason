import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TVM_PIN = "apache-tvm==0.26.0"
GPU_PACKAGES = {"onnxruntime-gpu", "nvidia-cudnn-cu13", "nvidia-cublas", "nvidia-cuda-nvrtc"}


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pyproject_dependencies():
    return _pyproject()["project"]["dependencies"]


def _requirement_lines():
    lines = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _normalise(line):
    line = re.sub(r"\s+", " ", line.strip())
    line = line.replace("'", '"')
    return line.strip('"').strip()


def _package_name(requirement):
    match = re.match(r"^[A-Za-z0-9._-]+", _normalise(requirement))
    return match.group(0).lower().replace("_", "-") if match else ""


def _dependency_group(name):
    groups = _pyproject().get("dependency-groups", {})
    assert name in groups, f"dependency group {name!r} is missing"
    return [_normalise(item) for item in groups[name]]


def test_runtime_dependencies_match_requirements():
    from_pyproject = {_normalise(item) for item in _pyproject_dependencies()}
    from_requirements = {_normalise(line) for line in _requirement_lines()}

    assert from_pyproject == from_requirements


def test_no_gpu_runtime_dependencies():
    names = {_package_name(item) for item in _pyproject_dependencies()}
    names |= {_package_name(line) for line in _requirement_lines()}

    assert names.isdisjoint(GPU_PACKAGES)


def test_python_version_file_pins_311():
    path = ROOT / ".python-version"

    assert path.exists(), ".python-version is missing"
    assert path.read_text(encoding="utf-8").strip() == "3.11"


def test_tvm_pinned_to_0_26_0_in_pyproject():
    pinned = [item for item in _pyproject_dependencies() if _package_name(item) == "apache-tvm"]

    assert [_normalise(item) for item in pinned] == [TVM_PIN]


def test_tvm_pinned_to_0_26_0_in_requirements():
    pinned = [line for line in _requirement_lines() if _package_name(line) == "apache-tvm"]

    assert [_normalise(line) for line in pinned] == [TVM_PIN]


@pytest.mark.parametrize("package", ["pytest", "pytest-timeout", "mutmut"])
def test_dev_group_has(package):
    assert package in _dependency_group("dev")


@pytest.mark.parametrize("package", ["torch", "torchvision", "matplotlib", "lxml"])
def test_research_group_has(package):
    assert package in _dependency_group("research")


def test_default_groups_are_dev_and_research():
    assert _pyproject()["tool"]["uv"]["default-groups"] == ["dev", "research"]


def test_torch_resolves_from_the_cpu_index():
    tool_uv = _pyproject()["tool"]["uv"]

    cpu_index = [index for index in tool_uv.get("index", []) if index.get("name") == "pytorch-cpu"]
    assert cpu_index, "the pytorch-cpu index is missing"
    assert cpu_index[0]["url"] == "https://download.pytorch.org/whl/cpu"
    assert cpu_index[0]["explicit"] is True

    sources = tool_uv["sources"]
    for package in ("torch", "torchvision"):
        assert sources[package] == {"index": "pytorch-cpu"}