"""Tests for scripts/gate.py's `check` subcommand.

Every tree here is a tmp git repo holding a handful of trivial files plus tiny
fakes for provision_assets.py and capture_baseline.py: no real assets, no
models, no network, and no writes outside tmp_path. The baseline side of the
comparison is the merge-base commit materialised by the gate itself, which is
what the real gate does too.
"""

import json
import math
import statistics
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
    # nothing is ever written into the tree being measured. One call advances it
    # once, so p50 and memory stay paired from the same round.
    counter = Path(artifact).parent / "calls.json"
    counts = json.loads(counter.read_text(encoding="utf-8")) if counter.is_file() else {}
    index = counts.get(variant, 0)
    counts[variant] = index + 1
    counter.write_text(json.dumps(counts), encoding="utf-8")

    def value(field):
        series = timings[field][variant]
        return series[index] if index < len(series) else series[-1]

    return json.dumps({
        "success": True,
        "error": None,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "latency_ms": {"p50": value("p50")},
        "memory_mb": {"delta": value("memory_delta")},
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
    """A timings file whose entries are the *measured rounds*.

    One leading entry is prepended per variant because the gate burns a warm-up
    run per side per variant before round 0, and the fake serves the series in
    call order.
    """
    memory = memory or {variant: [100.0] * len(values) for variant, values in p50.items()}

    def with_warmup(series: dict[str, list[float]]) -> dict[str, list[float]]:
        return {variant: [values[0]] + list(values) for variant, values in series.items()}

    return {"p50": with_warmup(p50), "memory_delta": with_warmup(memory)}


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
# 20% slower across the board, in every round.
CASE8_CANDIDATE = _canned({
    "fp32": [p50 * 1.2 for p50 in CASE7_BASELINE["p50"]["fp32"][1:]],
    "dynamic": [p50 * 1.2 for p50 in CASE7_BASELINE["p50"]["dynamic"][1:]],
    "static": [p50 * 1.2 for p50 in CASE7_BASELINE["p50"]["static"][1:]],
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
# A machine that is ~7% slower on whichever run goes second. With four rounds
# the order alternates B,C | C,B | B,C | C,B, so the candidate is the slow one
# in the rounds it runs second and the fast one in the rounds it runs first:
# the bias cancels, and no real difference exists.
BIAS_BASELINE = _canned({
    "fp32": [100.0, 107.0, 100.0, 107.0],
    "dynamic": [90.0, 96.3, 90.0, 96.3],
    "static": [50.0, 53.5, 50.0, 53.5],
})
BIAS_CANDIDATE = _canned({
    "fp32": [107.0, 100.0, 107.0, 100.0],
    "dynamic": [96.3, 90.0, 96.3, 90.0],
    "static": [53.5, 50.0, 53.5, 50.0],
})
# A flat baseline, so the only difference in the run is the candidate's.
FLAT_BASELINE = _canned({
    "fp32": [100.0] * 4,
    "dynamic": [90.0] * 4,
    "static": [50.0] * 4,
})
# The candidate is genuinely 8% slower in every round, whichever position it
# runs in. Position cannot explain this away.
DRIFT_CANDIDATE = _canned({
    "fp32": [108.0] * 4,
    "dynamic": [97.2] * 4,
    "static": [54.0] * 4,
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
    """Case 7: a couple of percent of paired difference is inside the band."""
    code, report = ab_env.run("case7", "--rounds", "5")

    assert code == gate.EXIT_PASS
    assert report["verdict"] == "pass"
    entry = _result(report, "fp32")
    # The warm-up run is discarded, so the five measured rounds are these.
    assert entry["p50"]["baseline"] == [10.0, 10.2, 9.9, 10.1, 10.0]
    assert entry["p50"]["candidate"] == [10.3, 10.1, 10.2, 10.0, 10.2]
    assert entry["p50"]["pct_diffs"] == pytest.approx(
        [3.0, -0.98039216, 3.03030303, -0.99009901, 2.0]
    )
    assert entry["p50"]["mean_pct_diff"] == pytest.approx(1.21196237)
    assert entry["p50"]["band_pct"] == pytest.approx(5.0)
    assert entry["p50"]["within_band"] is True
    # The median is kept for auditability but must not decide the verdict.
    assert entry["p50"]["informational"]["median_pct_diff"] == pytest.approx(2.0)
    assert "median_pct_diff" not in entry["p50"]
    # MAD_B is informational only too; it must not appear in the rule.
    assert entry["p50"]["informational"]["mad_baseline"] == pytest.approx(0.1)
    assert "mad_b" not in entry["p50"]
    assert entry["pass"] is True
    assert _step(report, "ab:tiny")["status"] == "pass"
    assert _step(report, "ab:winners")["status"] == "pass"
    assert "**PASS**" in capsys.readouterr().out


def test_ab_8_fails_when_the_candidate_is_twenty_percent_slower(ab_env) -> None:
    """Case 8: a 20% regression is far outside the band, under the new maths too."""
    ab_env.set_timings("candidate", CASE8_CANDIDATE)

    code, report = ab_env.run("case8", "--rounds", "5")

    assert code == gate.EXIT_FAIL
    assert report["verdict"] == "fail"
    for variant in gate.AB_VARIANTS:
        entry = _result(report, variant)
        # Every paired round is +20%, so the mean of the differences is +20%.
        assert entry["p50"]["mean_pct_diff"] == pytest.approx(20.0)
        assert entry["p50"]["within_band"] is False
        assert entry["pass"] is False
    model_step = _step(report, "ab:tiny")
    assert model_step["status"] == "fail"
    assert "OUTSIDE" in model_step["details"]


def test_ab_9_fails_when_the_winner_changes(ab_env) -> None:
    """Case 9: every mean is in band, but a different variant is now fastest."""
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


def test_ab_positional_bias_cancels_under_paired_diffs(ab_env) -> None:
    """A machine that is 7% slower on whichever run goes second is not a change.

    On the old fixed order (B always first) this machine produced a flat
    baseline series and a flat, 7% slower candidate series, and the old
    independent-median rule called that a regression. Alternating the order and
    pairing each round shows there is nothing there.
    """
    # What the old protocol recorded on this machine: B always ran first and so
    # always got the fast position.
    old_baseline = [100.0] * 4
    old_candidate = [107.0] * 4
    median_baseline = statistics.median(old_baseline)
    mad_baseline = statistics.median([abs(x - median_baseline) for x in old_baseline])
    old_band = max(2 * mad_baseline, 0.05 * median_baseline)
    assert abs(statistics.median(old_candidate) - median_baseline) > old_band

    ab_env.set_timings("baseline", BIAS_BASELINE)
    ab_env.set_timings("candidate", BIAS_CANDIDATE)

    code, report = ab_env.run("bias", "--rounds", "4")

    assert code == gate.EXIT_PASS
    assert report["verdict"] == "pass"
    entry = _result(report, "fp32")
    # Alternating order: the pairs are (100,107), (107,100), (100,107), (107,100),
    # so the positional penalty lands on a different side in each round.
    assert entry["p50"]["baseline"] == [100.0, 107.0, 100.0, 107.0]
    assert entry["p50"]["candidate"] == [107.0, 100.0, 107.0, 100.0]
    assert entry["p50"]["pct_diffs"] == pytest.approx([7.0, -6.54205607, 7.0, -6.54205607])
    # Even numbers of +7 and -6.54 give a mean just above zero: the positional
    # penalty largely cancels.
    assert entry["p50"]["mean_pct_diff"] == pytest.approx(0.22897196)
    assert entry["p50"]["within_band"] is True
    # The raw evidence of a big positional effect is still in the report.
    assert entry["p50"]["max_abs_pct_diff"] == pytest.approx(7.0)


def test_ab_one_sided_drift_still_fails(ab_env) -> None:
    """A genuine 8% regression fails whichever position the candidate runs in."""
    ab_env.set_timings("baseline", FLAT_BASELINE)
    ab_env.set_timings("candidate", DRIFT_CANDIDATE)

    code, report = ab_env.run("drift", "--rounds", "4")

    assert code == gate.EXIT_FAIL
    entry = _result(report, "fp32")
    assert entry["p50"]["mean_pct_diff"] == pytest.approx(8.0)
    assert entry["p50"]["within_band"] is False
    assert entry["pass"] is False


def test_ab_alternating_order_is_recorded_and_balanced(ab_env) -> None:
    """The order actually alternates, and the report lets a reader verify it."""
    _code, report = ab_env.run("order", "--rounds", "4")

    # The parity does not restart per variant, so variant 2 opens with the
    # opposite order to variant 1 even though it is round 0 of that variant.
    assert report["ab"]["order"][f"{AB_MODEL}/fp32"] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ]
    # Four rounds is even, so the parity realigns and variant 2 opens the same
    # way variant 1 did. See the five-round case below for one that does not.
    assert report["ab"]["order"][f"{AB_MODEL}/dynamic"] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ]
    # 4 rounds x 3 variants = 12 rounds, split evenly.
    assert report["ab"]["rounds_led_by"] == {"baseline": 6, "candidate": 6}

    # An odd total is recorded rather than hidden: 5 rounds x 3 variants = 15,
    # and because the parity carries across variants the second variant opens
    # with the opposite order to the first.
    _odd_code, odd = ab_env.run("order5", "--rounds", "5")
    assert odd["ab"]["rounds_led_by"] == {"baseline": 8, "candidate": 7}
    assert odd["ab"]["order"][f"{AB_MODEL}/dynamic"][0] == ["candidate", "baseline"]


def test_ab_warmup_runs_are_discarded(ab_env) -> None:
    """One throwaway run per side per variant happens before round 0."""
    _code, report = ab_env.run("warmup", "--rounds", "3")

    assert report["ab"]["warmup_runs_per_side"] == gate.AB_WARMUP_RUNS_PER_SIDE
    # Six measured calls (3 rounds x 2 series) plus the warm-up = 7 in the series,
    # and only the last three are kept.
    entry = _result(report, "fp32")
    assert len(entry["p50"]["baseline"]) == 3
    assert len(entry["p50"]["candidate"]) == 3


def test_ab_mean_passes_where_the_median_would_fail(ab_env) -> None:
    """The regression test for the flaky self-test.

    These are the paired diffs from a measured self-test run that the median
    rule rejected and the mean rule accepts. Three of the five rounds straddled
    a machine state change and two did not, so the median lands on the straddles
    and reads as a 6.8% regression while the mean correctly reports -4.24%.
    Restoring the median as the pass statistic must break this test.
    """
    diffs = [-7.8, 0.2, 0.1, -6.9, -6.8]
    baseline = [100.0] * 5
    candidate = [value * (1.0 + diff / 100.0) for value, diff in zip(baseline, diffs, strict=True)]
    ab_env.set_timings("baseline", _canned({
        variant: baseline for variant in gate.AB_VARIANTS
    }))
    ab_env.set_timings("candidate", _canned({
        variant: candidate for variant in gate.AB_VARIANTS
    }))

    code, report = ab_env.run("bimodal", "--rounds", "5")

    assert code == gate.EXIT_PASS
    entry = _result(report, "fp32")
    assert entry["p50"]["pct_diffs"] == pytest.approx(diffs)
    assert entry["p50"]["mean_pct_diff"] == pytest.approx(-4.24)
    assert entry["p50"]["within_band"] is True
    # The median would have called this a failure: it is kept, and it disagrees.
    assert entry["p50"]["informational"]["median_pct_diff"] == pytest.approx(-6.8)
    assert abs(entry["p50"]["informational"]["median_pct_diff"]) > entry["p50"]["band_pct"]


def test_ab_one_sided_drift_still_fails_with_fifteen_rounds(ab_env) -> None:
    """More rounds buy sensitivity; they must not blunt a real regression."""
    ab_env.set_timings("baseline", _canned({
        variant: [100.0] * 15 for variant in gate.AB_VARIANTS
    }))
    ab_env.set_timings("candidate", _canned({
        variant: [108.0] * 15 for variant in gate.AB_VARIANTS
    }))

    code, report = ab_env.run("drift15", "--rounds", "15")

    assert code == gate.EXIT_FAIL
    entry = _result(report, "fp32")
    assert entry["p50"]["n"] == 15
    assert entry["p50"]["mean_pct_diff"] == pytest.approx(8.0)
    assert entry["p50"]["within_band"] is False
    assert entry["pass"] is False
    assert _step(report, "ab:tiny")["status"] == "fail"


def test_ab_reports_its_own_power(ab_env) -> None:
    """The report says how big an effect the run could have seen."""
    _code, report = ab_env.run("power", "--rounds", "15")

    power = report["ab"]["power"]
    assert power["statistic"] == "mean of the paired per-round percentage differences"
    for result in report["ab"]["results"]:
        key = f"{result['model']}/{result['variant']}"
        p50 = power["results"][key]["p50"]
        assert p50["n"] == result["p50"]["n"] == 15
        # se is sd/sqrt(n) and the minimum detectable effect is 2 SE.
        assert p50["se"] == pytest.approx(result["p50"]["sd_pct_diff"] / math.sqrt(15))
        assert p50["mde_2se"] == pytest.approx(2.0 * p50["se"])
        assert p50["band_pct"] == result["p50"]["band_pct"]
        assert p50["powered_for_band"] == (p50["mde_2se"] <= p50["band_pct"])
        assert set(power["results"][key]) == {"p50", "memory_delta"}


def test_ab_power_block_reports_the_standard_error(ab_env) -> None:
    """A flat comparison has no spread, so it can see a very small effect."""
    ab_env.set_timings("baseline", FLAT_BASELINE)
    ab_env.set_timings("candidate", FLAT_BASELINE)

    _code, report = ab_env.run("power_flat", "--rounds", "4")

    p50_power = report["ab"]["power"]["results"][f"{AB_MODEL}/fp32"]["p50"]
    entry = _result(report, "fp32")
    assert entry["p50"]["sd_pct_diff"] == pytest.approx(0.0)
    assert p50_power["se"] == pytest.approx(0.0)
    assert p50_power["mde_2se"] == pytest.approx(0.0)
    assert p50_power["powered_for_band"] is True


def test_ab_band_pct_is_settable(ab_env) -> None:
    """--band-pct moves the threshold without touching the measurements."""
    ab_env.set_timings("baseline", FLAT_BASELINE)
    ab_env.set_timings("candidate", DRIFT_CANDIDATE)

    code, report = ab_env.run("band", "--rounds", "4", "--band-pct", "10")

    assert code == gate.EXIT_PASS
    entry = _result(report, "fp32")
    assert entry["p50"]["band_pct"] == pytest.approx(10.0)
    assert entry["p50"]["mean_pct_diff"] == pytest.approx(8.0)
    assert entry["p50"]["within_band"] is True
    assert report["ab"]["band_pct"] == pytest.approx(10.0)


def test_ab_defaults_to_the_three_named_models() -> None:
    """The documented defaults are the CLI's defaults."""
    args = gate.build_parser().parse_args(["ab", "--baseline", "b", "--candidate", "c"])

    assert args.models is None
    # 15 rounds, because five cannot resolve the 5% band: the mean of five
    # paired diffs has a standard error near 2.1%, so the minimum detectable
    # effect is no smaller than the band being judged.
    assert args.rounds == 15
    assert args.rounds == gate.AB_ROUNDS
    assert args.band_pct == gate.AB_BAND_PCT
    assert gate._ab_request(args).models == gate.AB_MODELS
    assert gate._ab_request(args).band_pct == gate.AB_BAND_PCT


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
        assert len(entry["p50"]["pct_diffs"]) == 3
        assert entry["p50"]["band_pct"] == gate.AB_BAND_PCT
        assert entry["p50"]["n"] == 3
        assert entry["p50"]["max_abs_pct_diff"] >= abs(entry["p50"]["mean_pct_diff"])
    winners = report["ab"]["winners"][AB_MODEL]
    assert set(winners) == {"baseline", "candidate", "same"}
    assert winners["baseline"] == "static"
