"""Tests for scripts/gate.py's `check` subcommand.

Every tree here is a tmp git repo holding a handful of trivial files plus tiny
fakes for provision_assets.py and capture_baseline.py: no real assets, no
models, no network, and no writes outside tmp_path. The baseline side of the
comparison is the merge-base commit materialised by the gate itself, which is
what the real gate does too.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import scripts.gate as gate

TESTS_SOURCE = '''\
def test_one():
    assert 1 + 1 == 2


def test_two():
    assert "a" == "a"


def test_three():
    assert [1, 2] == [1, 2]
'''

PROVISION_SOURCE = '''\
#!/usr/bin/env python
"""Fake provision_assets.py: one marker file, no downloads, no real assets."""

import argparse
from pathlib import Path

ASSET = "models/fake.onnx"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=".")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.dest) / ASSET
    if path.is_file():
        print(f"ok {ASSET}")
        return 0
    if args.verify_only:
        print(f"MISSING {ASSET}")
        return 3
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake asset", encoding="utf-8")
    print(f"created {ASSET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

CAPTURE_SOURCE = '''\
#!/usr/bin/env python
"""Fake capture_baseline.py: replays a recorded payload into --out."""

import argparse
import shutil
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--assets-root", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    level = "quick" if args.quick else "full"
    source = args.assets_root / "gate_fixture" / "payload" / level
    if not source.is_dir():
        print(f"capture_baseline: no recorded payload at {source}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, args.out, dirs_exist_ok=True)
    print("wrote 6 files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

RATCHET_SOURCE = "#!/usr/bin/env bash\necho 'lint ratchet: nothing to check'\nexit 0\n"

# What the fake capture writes, and what the stored baseline holds.
PAYLOAD = {
    "environment.json": {"cpu_model": "fake cpu", "git_commit": "unknown"},
    "analysis.json": {"mobilenetv2": {"model_size_mb": 14.0, "operator_types": {"Conv": 52}}},
    "plans.json": {
        "plans": {
            "mobilenetv2": {
                "false": [{"name": "onnx_fp32_baseline", "priority": 3}],
                "true": [{"name": "onnx_static_int8", "priority": 1}],
            }
        }
    },
    "quantized.json": {"mobilenetv2": {"onnx_static_int8": {"sha256": "abc123", "error": None}}},
    "accuracy.json": {
        "mobilenetv2": {"onnx_static_int8": {"top1_accuracy": 0.612345, "top5_accuracy": 0.801234}}
    },
    "reproduction.json": {"comparisons": [], "committed_source": "reports/data/quant.json"},
}


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@dataclass(frozen=True)
class GateEnv:
    """Two tiny trees and the stored baseline the gate will judge against."""

    candidate: Path
    baseline_dir: Path
    reports: Path

    def run(self, name: str, *extra: str) -> tuple[int, dict]:
        report_path = self.reports / f"{name}.json"
        argv = [
            "check",
            "--candidate", str(self.candidate),
            "--level", "fast",
            "--baseline-dir", str(self.baseline_dir),
            "--report", str(report_path),
            *extra,
        ]
        code = gate.main(argv)
        return code, json.loads(report_path.read_text(encoding="utf-8"))

    def edit_json(self, relative: str, change) -> None:
        """Apply `change` to one recorded JSON payload file in the candidate."""
        path = self.candidate / "gate_fixture" / "payload" / "quick" / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        _write(path, _json_text(payload))


@pytest.fixture
def gate_env(tmp_path) -> GateEnv:
    """An origin repo, a candidate clone of it, and the stored baseline."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin.parent, "init", "-q", "-b", "cleanup", str(origin))
    for key, value in (("user.email", "gate@example.invalid"), ("user.name", "gate")):
        _git(origin, "config", key, value)

    _write(origin / "tests" / "test_alpha.py", TESTS_SOURCE)
    _write(origin / "tests" / "golden" / "values.json", _json_text({"threshold": 0.5}))
    _write(origin / "scripts" / "provision_assets.py", PROVISION_SOURCE)
    _write(origin / "scripts" / "capture_baseline.py", CAPTURE_SOURCE)
    _write(origin / "scripts" / "lint_ratchet.sh", RATCHET_SOURCE)
    _write(origin / "pytest.ini", "[pytest]\n")
    for level in ("quick", "full"):
        for name, payload in PAYLOAD.items():
            _write(origin / "gate_fixture" / "payload" / level / name, _json_text(payload))
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "base tree")

    baseline_dir = tmp_path / "baseline"
    for level in ("quick", "full"):
        for name, payload in PAYLOAD.items():
            _write(baseline_dir / level / name, _json_text(payload))

    candidate = tmp_path / "candidate"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )
    for key, value in (("user.email", "gate@example.invalid"), ("user.name", "gate")):
        _git(candidate, "config", key, value)
    _git(candidate, "checkout", "-q", "-b", "feature")

    reports = tmp_path / "reports"
    reports.mkdir()
    return GateEnv(candidate=candidate, baseline_dir=baseline_dir, reports=reports)


def _step(report: dict, name: str) -> dict:
    steps = [step for step in report["steps"] if step["name"] == name]
    assert steps, f"no step called {name} in {report}"
    return steps[0]


def _delete_test_three(env: GateEnv) -> str:
    """Remove one test function from the candidate's working tree."""
    path = env.candidate / "tests" / "test_alpha.py"
    text = path.read_text(encoding="utf-8")
    assert "def test_three()" in text
    path.write_text(
        text.replace('\n\ndef test_three():\n    assert [1, 2] == [1, 2]\n', ""), encoding="utf-8"
    )
    return "tests/test_alpha.py::test_three"


def test_check_1_passes_on_two_identical_trees(gate_env, capsys) -> None:
    """Case 1: nothing changed, so every step passes."""
    code, report = gate_env.run("case1")

    assert code == gate.EXIT_PASS
    assert report["verdict"] == "pass"
    assert [step["status"] for step in report["steps"]] == ["pass"] * len(gate.STEP_NAMES)
    assert "**PASS**" in capsys.readouterr().out


def test_check_2_fails_when_the_candidate_deleted_a_test(gate_env) -> None:
    """Case 2: a deleted test fails the tests step and is named in the report."""
    node_id = _delete_test_three(gate_env)

    code, report = gate_env.run("case2")

    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "fail"
    tests = _step(report, "tests")
    assert tests["status"] == "fail"
    assert node_id in tests["details"]
    # The run stops at the first failure; later steps are recorded, not run.
    assert [step["name"] for step in report["steps"]] == list(gate.STEP_NAMES)
    assert _step(report, "goldens")["status"] == "not run"
    assert _step(report, "deterministic-outputs")["status"] == "not run"


def test_check_3_allow_deleted_lets_the_tests_step_pass(gate_env) -> None:
    """Case 3: the same deletion, listed in --allow-deleted."""
    node_id = _delete_test_three(gate_env)
    allowed = gate_env.reports / "allowed.txt"
    allowed.write_text(f"# a comment\n{node_id}\n\n", encoding="utf-8")

    code, report = gate_env.run("case3", "--allow-deleted", str(allowed))

    assert code == gate.EXIT_PASS
    assert report["verdict"] == "pass"
    tests = _step(report, "tests")
    assert tests["status"] == "pass"
    assert node_id in tests["details"]


def test_check_4_fails_when_a_golden_file_changed(gate_env) -> None:
    """Case 4: a change under tests/golden/ fails the goldens step."""
    _write(gate_env.candidate / "tests" / "golden" / "values.json",
           _json_text({"threshold": 0.6}))

    code, report = gate_env.run("case4")

    assert code == gate.EXIT_FAIL
    goldens = _step(report, "goldens")
    assert goldens["status"] == "fail"
    assert "tests/golden/values.json" in goldens["details"]
    assert _step(report, "tests")["status"] == "pass"


def test_check_5_fails_on_one_changed_planner_field(gate_env) -> None:
    """Case 5: one changed planner value is reported with its JSON path."""
    gate_env.edit_json(
        "plans.json",
        lambda payload: payload["plans"]["mobilenetv2"]["true"][0].update({"name": "onnx_dynamic"}),
    )

    code, report = gate_env.run("case5")

    assert code == gate.EXIT_FAIL
    outputs = _step(report, "deterministic-outputs")
    assert outputs["status"] == "fail"
    assert "plans.json:plans.mobilenetv2.true[0].name" in outputs["details"]


def test_check_6_accuracy_within_tolerance_passes(gate_env) -> None:
    """Case 6, first half: 1e-7 is inside the 1e-6 tolerance."""
    gate_env.edit_json(
        "accuracy.json",
        lambda payload: payload["mobilenetv2"]["onnx_static_int8"].update(
            {"top1_accuracy": 0.612345 + 1e-7}
        ),
    )

    code, report = gate_env.run("case6_tiny")

    assert code == gate.EXIT_PASS
    assert _step(report, "deterministic-outputs")["status"] == "pass"


def test_check_6_accuracy_beyond_tolerance_fails(gate_env) -> None:
    """Case 6, second half: 1e-5 is outside the tolerance."""
    gate_env.edit_json(
        "accuracy.json",
        lambda payload: payload["mobilenetv2"]["onnx_static_int8"].update(
            {"top1_accuracy": 0.612345 + 1e-5}
        ),
    )

    code, report = gate_env.run("case6_large")

    assert code == gate.EXIT_FAIL
    outputs = _step(report, "deterministic-outputs")
    assert outputs["status"] == "fail"
    assert "accuracy.json:mobilenetv2.onnx_static_int8.top1_accuracy" in outputs["details"]


def test_check_goldens_step_can_be_waived(gate_env) -> None:
    """The --allow-golden-change escape hatch names the files it accepted."""
    _write(gate_env.candidate / "tests" / "golden" / "values.json",
           _json_text({"threshold": 0.6}))

    code, report = gate_env.run("waived", "--allow-golden-change")

    assert code == gate.EXIT_PASS
    goldens = _step(report, "goldens")
    assert goldens["status"] == "pass"
    assert "tests/golden/values.json" in goldens["details"]


def test_check_lint_step_is_skipped_without_the_ratchet(gate_env) -> None:
    """Without P0.10 the gate says so instead of quietly passing."""
    (gate_env.candidate / "scripts" / "lint_ratchet.sh").unlink()

    code, report = gate_env.run("no_ratchet")

    assert code == gate.EXIT_PASS
    lint = _step(report, "lint-types")
    assert lint["status"] == "skipped"
    assert "lint_ratchet.sh" in lint["details"]


def test_check_exit_2_on_a_missing_baseline_dir(gate_env) -> None:
    """A broken environment is exit 2, not a verdict."""
    code = gate.main([
        "check", "--candidate", str(gate_env.candidate), "--level", "fast",
        "--baseline-dir", str(gate_env.baseline_dir / "absent"),
    ])

    assert code == gate.EXIT_USAGE


def test_check_report_names_the_level_and_the_ab_placeholder(gate_env) -> None:
    """The report shape is the one the spec fixes."""
    _code, report = gate_env.run("shape")

    assert set(report) == {"verdict", "steps", "ab"}
    assert report["ab"]["status"] == "not run"
    assert [step["name"] for step in report["steps"]] == list(gate.STEP_NAMES)


# --- ab -------------------------------------------------------------------

# The worker runs the *tree's* `api` package, so the fixture gives each tree a
# tiny one that answers from canned numbers. The gate under test is real; only
# the product code it calls is a fake, and nothing here times a model.
CONFIG_SOURCE = '''\
BENCHMARK_WARMUP_RUNS = 10
BENCHMARK_TIMED_RUNS = 100
BENCHMARK_TIME_BUDGET_SECONDS = 10.0
'''

BENCHMARK_SOURCE = '''\
"""A benchmark_engine that answers from canned numbers instead of a model."""

import json
from pathlib import Path

TIMINGS = "ab_timings.json"


def benchmark(artifact, warmup_runs=10, benchmark_runs=100, time_budget_seconds=None,
              isolated=True, **kwargs):
    variant = Path(artifact).stem
    timings = json.loads(Path(TIMINGS).read_text(encoding="utf-8"))
    # The counter sits beside the artifact, which is in the gate's temp dir, so
    # nothing is ever written into the tree being measured.
    counter = Path(artifact).parent / "calls.json"
    counts = json.loads(counter.read_text(encoding="utf-8")) if counter.is_file() else {}
    index = counts.get(variant, 0)
    counts[variant] = index + 1
    counter.write_text(json.dumps(counts), encoding="utf-8")
    p50s = timings["p50"][variant]
    mems = timings["memory_delta"][variant]
    return json.dumps({
        "success": True,
        "error": None,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "latency_ms": {"p50": p50s[index] if index < len(p50s) else p50s[-1]},
        "memory_mb": {"delta": mems[index] if index < len(mems) else mems[-1]},
    })
'''

QUANTIZER_SOURCE = '''\
"""A Quantizer that writes a placeholder artifact named after the variant."""

from pathlib import Path

VARIANT_BY_CANDIDATE = {
    "onnx_fp32_baseline": "fp32",
    "onnx_dynamic_int8": "dynamic",
    "onnx_static_int8": "static",
}


class Quantizer:
    def __init__(self, output_dir="./temp_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def quantize_for_candidate(self, model_path, candidate_name, calibration_data=None,
                               hardware_id="x86_64"):
        if candidate_name == "onnx_static_int8" and calibration_data is None:
            raise ValueError("Static INT8 requires calibration data")
        variant = VARIANT_BY_CANDIDATE[candidate_name]
        artifact = self.output_dir / f"{variant}.onnx"
        artifact.write_text("placeholder", encoding="utf-8")
        return str(artifact)
'''

VALIDATION_SOURCE = '''\
"""A validation module that splits a canned fixture instead of an npz."""

from pathlib import Path
from types import SimpleNamespace


def load_validation_data(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return SimpleNamespace(inputs=[0.0] * 8, labels=[0] * 8)


def split_calibration_and_test(data):
    half = len(data.inputs) // 2
    return SimpleNamespace(
        calibration=data.inputs[:half],
        test=SimpleNamespace(inputs=data.inputs[half:], labels=data.labels[half:]),
    )
'''

AB_MODEL = "tiny"


def _fake_tree(root: Path, timings: dict) -> Path:
    """A tree whose `api` package answers the gate's worker from canned numbers."""
    _write(root / "api" / "__init__.py", "")
    _write(root / "api" / "config.py", CONFIG_SOURCE)
    _write(root / "api" / "optimizer" / "__init__.py", "")
    _write(root / "api" / "optimizer" / "benchmark_engine.py", BENCHMARK_SOURCE)
    _write(root / "api" / "optimizer" / "quantizer.py", QUANTIZER_SOURCE)
    _write(root / "api" / "optimizer" / "validation.py", VALIDATION_SOURCE)
    _write(root / "ab_timings.json", _json_text(timings))
    _write(root / "models" / f"{AB_MODEL}.onnx", "placeholder")
    _write(root / "models" / "samples" / "imagenetv2_mobilenetv2_256.npz", "placeholder")
    return root


def _canned(p50: dict[str, list[float]], memory: dict[str, list[float]] | None = None) -> dict:
    """A timings file; memory defaults to a constant that never moves the verdict."""
    memory = memory or {variant: [100.0] * 5 for variant in p50}
    return {"p50": p50, "memory_delta": memory}


# Case 7's numbers, and one variant's worth of the rest.
CASE7_BASELINE = _canned({
    "fp32": [10.0, 10.2, 9.9, 10.1, 10.0],
    "dynamic": [9.0, 9.2, 8.9, 9.1, 9.0],
    "static": [5.0, 5.2, 4.9, 5.1, 5.0],
})
CASE7_CANDIDATE = _canned({
    "fp32": [10.3, 10.1, 10.2, 10.0, 10.2],
    "dynamic": [9.3, 9.1, 9.2, 9.0, 9.2],
    "static": [5.3, 5.1, 5.2, 5.0, 5.2],
})
# 20% slower across the board.
CASE8_CANDIDATE = _canned({
    "fp32": [p50 * 1.2 for p50 in CASE7_BASELINE["p50"]["fp32"]],
    "dynamic": [p50 * 1.2 for p50 in CASE7_BASELINE["p50"]["dynamic"]],
    "static": [p50 * 1.2 for p50 in CASE7_BASELINE["p50"]["static"]],
})
# Every variant inside its own band, but the fastest one is no longer the same.
CASE9_BASELINE = _canned({
    "fp32": [10.02, 10.02, 10.02],
    "dynamic": [10.04, 10.04, 10.04],
    "static": [10.0, 10.0, 10.0],
})
CASE9_CANDIDATE = _canned({
    "fp32": [10.0, 10.0, 10.0],
    "dynamic": [10.04, 10.04, 10.04],
    "static": [10.02, 10.02, 10.02],
})


@dataclass(frozen=True)
class AbEnv:
    """Two trees to time against each other, and where to write the report."""

    baseline: Path
    candidate: Path
    reports: Path

    def run(self, name: str, *extra: str) -> tuple[int, dict]:
        report_path = self.reports / f"{name}.json"
        code = gate.main([
            "ab",
            "--baseline", str(self.baseline),
            "--candidate", str(self.candidate),
            "--models", AB_MODEL,
            "--report", str(report_path),
            *extra,
        ])
        return code, json.loads(report_path.read_text(encoding="utf-8"))

    def set_timings(self, role: str, timings: dict) -> None:
        """Re-record one tree's canned numbers."""
        tree = self.baseline if role == "baseline" else self.candidate
        _write(tree / "ab_timings.json", _json_text(timings))


@pytest.fixture
def ab_env(tmp_path) -> AbEnv:
    """Two fake trees; by default they hold case 7's canned timings."""
    reports = tmp_path / "reports"
    reports.mkdir()
    return AbEnv(
        baseline=_fake_tree(tmp_path / "baseline_tree", CASE7_BASELINE),
        candidate=_fake_tree(tmp_path / "candidate_tree", CASE7_CANDIDATE),
        reports=reports,
    )


def _result(report: dict, variant: str) -> dict:
    found = [
        entry for entry in report["ab"]["results"] if entry["variant"] == variant
    ]
    assert found, f"no result for {variant} in {report['ab']['results']}"
    return found[0]


def test_ab_7_pass_within_five_percent(ab_env, capsys) -> None:
    """Case 7: a 2% median shift is inside the noise band."""
    code, report = ab_env.run("case7", "--rounds", "5")

    assert code == gate.EXIT_PASS
    assert report["verdict"] == "pass"
    entry = _result(report, "fp32")
    assert entry["p50"]["baseline"] == [10.0, 10.2, 9.9, 10.1, 10.0]
    assert entry["p50"]["candidate"] == [10.3, 10.1, 10.2, 10.0, 10.2]
    # Median of B's five samples; C's is 0.2 above it.
    assert entry["p50"]["mB"] == pytest.approx(10.0)
    assert entry["p50"]["mC"] == pytest.approx(10.2)
    # MAD_B is 0.1, so the band is max(0.2, 5% of 10.0) = 0.5.
    assert entry["p50"]["mad_b"] == pytest.approx(0.1)
    assert entry["p50"]["band"] == pytest.approx(0.5)
    assert entry["p50"]["within_band"] is True
    assert entry["pass"] is True
    assert _step(report, "ab:tiny")["status"] == "pass"
    assert _step(report, "ab:winners")["status"] == "pass"
    assert "**PASS**" in capsys.readouterr().out


def test_ab_8_fails_when_the_candidate_is_twenty_percent_slower(ab_env) -> None:
    """Case 8: a 20% regression is far outside the band."""
    ab_env.set_timings("candidate", CASE8_CANDIDATE)

    code, report = ab_env.run("case8", "--rounds", "5")

    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "fail"
    entry = _result(report, "fp32")
    # Every sample is 1.2x, so the median is too.
    assert entry["p50"]["mC"] == pytest.approx(12.0)
    assert entry["p50"]["within_band"] is False
    assert entry["pass"] is False
    model_step = _step(report, "ab:tiny")
    assert model_step["status"] == "fail"
    assert "OUTSIDE" in model_step["details"]


def test_ab_9_fails_when_the_winner_changes(ab_env) -> None:
    """Case 9: every median is in band, but a different variant is now fastest."""
    ab_env.set_timings("baseline", CASE9_BASELINE)
    ab_env.set_timings("candidate", CASE9_CANDIDATE)

    code, report = ab_env.run("case9", "--rounds", "3")

    assert code == gate.EXIT_FAIL
    # The per-variant rule alone would pass: this is the winner check that fails.
    for variant in gate.AB_VARIANTS:
        entry = _result(report, variant)
        assert entry["p50"]["within_band"] is True
        assert entry["memory_delta"]["within_band"] is True
    assert _step(report, "ab:tiny")["status"] == "pass"
    winners_step = _step(report, "ab:winners")
    assert winners_step["status"] == "fail"
    assert report["ab"]["winners"][AB_MODEL] == {
        "baseline": "static",
        "candidate": "fp32",
        "same": False,
    }


def test_ab_defaults_to_the_three_named_models() -> None:
    """The documented defaults are the CLI's defaults."""
    args = gate.build_parser().parse_args(["ab", "--baseline", "b", "--candidate", "c"])

    assert args.models is None
    assert args.rounds == 5
    assert gate._ab_request(args).models == gate.AB_MODELS


def test_ab_exit_2_when_a_worker_cannot_run(ab_env, capsys) -> None:
    """A crash in one tree is an environment error, never a quiet pass."""
    (ab_env.candidate / "api" / "optimizer" / "benchmark_engine.py").unlink()

    code = gate.main([
        "ab", "--baseline", str(ab_env.baseline), "--candidate", str(ab_env.candidate),
        "--models", AB_MODEL, "--rounds", "1",
    ])

    assert code == gate.EXIT_USAGE
    assert "benchmark worker" in capsys.readouterr().err


def test_ab_leaves_both_trees_untouched(ab_env) -> None:
    """Nothing the workers do may land in either tree."""
    before = {
        tree: sorted(str(path.relative_to(tree)) for path in tree.rglob("*"))
        for tree in (ab_env.baseline, ab_env.candidate)
    }

    ab_env.run("clean", "--rounds", "2")

    assert {
        tree: sorted(str(path.relative_to(tree)) for path in tree.rglob("*"))
        for tree in (ab_env.baseline, ab_env.candidate)
    } == before


def test_ab_report_carries_both_winners_and_every_sample(ab_env) -> None:
    """The report keeps the numbers a reviewer would want to recheck the maths."""
    _code, report = ab_env.run("shape", "--rounds", "3")

    assert set(report) == {"verdict", "steps", "ab"}
    assert report["ab"]["rounds"] == 3
    assert {entry["variant"] for entry in report["ab"]["results"]} == set(gate.AB_VARIANTS)
    for entry in report["ab"]["results"]:
        assert len(entry["p50"]["baseline"]) == 3
        assert len(entry["p50"]["candidate"]) == 3
        assert entry["p50"]["band"] >= gate.AB_NOISE_SHARE * entry["p50"]["mB"]
    winners = report["ab"]["winners"][AB_MODEL]
    assert set(winners) == {"baseline", "candidate", "same"}
    assert winners["baseline"] == "static"
