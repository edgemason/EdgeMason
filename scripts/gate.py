#!/usr/bin/env python
"""P0.9 results gate: the `check` subcommand.

The gate answers one question - "does the candidate's results match the
baseline?" - and answers it with an exit code, so that no agent and no tired
human has to judge it by eye. Exit 0 is a pass, 1 a fail, 2 a usage or
environment error.

The gate never modifies either tree it is given. It runs commands inside them
with bytecode writing off, and the only tree it writes to is a throwaway
detached worktree of the candidate's merge-base, which it removes again in a
`finally`.

Usage:
    uv run python scripts/gate.py check --candidate . --level full
    uv run python scripts/gate.py check --candidate ../other --report /tmp/g.json
    uv run python scripts/gate.py ab --baseline . --candidate . --rounds 5
"""

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

PASS = "pass"
FAIL = "fail"
SKIPPED = "skipped"
NOT_RUN = "not run"

# The files the capture writes, in the order the gate inspects them.
ACCURACY_FILE = "accuracy.json"
ENVIRONMENT_FILE = "environment.json"
# The A/B protocol's defaults: three models, three candidate variants each.
AB_MODELS = ("squeezenet1.1-13", "mobilenetv2", "resnet50")
AB_VARIANTS = ("fp32", "dynamic", "static")
# The name `Quantizer.quantize_for_candidate` routes on, per protocol variant.
AB_CANDIDATES = {
    "fp32": "onnx_fp32_baseline",
    "dynamic": "onnx_dynamic_int8",
    "static": "onnx_static_int8",
}
# Static INT8 needs calibration data, and this is where it lives in both trees.
AB_DATASET = "models/samples/imagenetv2_mobilenetv2_256.npz"
AB_MODEL_DIR = "models"
AB_MODEL_SUFFIX = ".onnx"
# A pass needs the mean paired difference within this many percent. The 5% is
# the spec's number; it is a percentage of each round's own baseline value, not
# a share of a median, so it does not drift with the model's size.
AB_BAND_PCT = 5.0
# The two sides swap places every round, so each tree spends half the rounds
# first and half second. A machine that is slower the second time a model is
# benchmarked then costs both sides the same amount instead of one of them.
AB_ROUND_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"))
# One throwaway run per model x variant per tree before round 0, so the first
# measured round is not the one paying for the page cache and the cold artifact.
AB_WARMUP_RUNS_PER_SIDE = 1
# Rounds per model x variant. Five is underpowered against a 5% band: with the
# measured paired-difference spread (sd ~4.7%) the standard error of a mean of
# 15 is ~1.2%, giving a minimum detectable effect near 2.4%, while the mean of
# 5 gives ~4.2% and cannot resolve the band at all. 15 buys the sensitivity; it
# costs about three times the wall time, and that trade is deliberate.
AB_ROUNDS = 15
# Accuracy is the one output the gate allows to wobble: every number must be
# equal to within this much. Everything else must be byte-identical.
ACCURACY_TOLERANCE = 1e-6
FAST_MARKERS = "not slow and not tvm and not cuda and not research"
GOLDEN_PATHS = ("tests/golden", "baseline")
# The gate judges a candidate against the commit it branched from, never
# against itself, so a change already on `cleanup` cannot slip through.
BASE_REFS = ("cleanup", "origin/cleanup")
# A single detail never lists more than this many ids or paths.
MAX_LISTED = 25
STEP_NAMES = ("assets", "tests", "goldens", "deterministic-outputs", "lint-types")
NODE_ID = re.compile(r"^[^\s]+\.py::")


class GateError(Exception):
    """A usage or environment problem: the gate cannot judge at all."""


@dataclass(frozen=True)
class Step:
    """One gate step and what it found."""

    name: str
    status: str
    details: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "details": self.details}


@dataclass(frozen=True)
class CheckRequest:
    """Everything the `check` subcommand needs, resolved."""

    candidate: Path
    level: str
    baseline_dir: Path
    allow_deleted: frozenset[str]
    allow_golden_change: bool


@dataclass(frozen=True)
class BaselineTree:
    """A throwaway worktree of the candidate's merge-base, or nothing at all."""

    path: Path | None
    reason: str


def _run(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _child_env() -> dict[str, str]:
    """An environment that leaves no trace in the tree it runs in."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _git(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(tree), *args], cwd=tree)


def _tail(result: subprocess.CompletedProcess[str], lines: int = 5) -> str:
    """The last few lines of a failed command, as one readable line."""
    text = (result.stderr or result.stdout or "").strip().splitlines()
    return " | ".join(text[-lines:]) if text else f"exit {result.returncode}"


def _list(values: Sequence[str]) -> str:
    """A capped, comma-separated list for a step detail."""
    shown = list(values[:MAX_LISTED])
    if len(values) > MAX_LISTED:
        shown.append(f"... (+{len(values) - MAX_LISTED} more)")
    return ", ".join(shown)


def resolve_merge_base(tree: Path) -> tuple[str | None, str]:
    """The commit the candidate branched from, and why it was not found.

    `cleanup` is the branch every spec branches from; without it the candidate
    *is* the baseline, and `HEAD` is the only honest answer.
    """
    for ref in BASE_REFS:
        if _git(tree, "rev-parse", "--verify", "--quiet", ref).returncode != 0:
            continue
        result = _git(tree, "merge-base", "HEAD", ref)
        commit = result.stdout.strip()
        if result.returncode == 0 and commit:
            return commit, ""
        return None, f"`git merge-base HEAD {ref}` failed: {_tail(result)}"
    return None, "neither `cleanup` nor `origin/cleanup` exists in the candidate tree"


@contextlib.contextmanager
def baseline_worktree(candidate: Path, merge_base: str | None) -> Iterator[BaselineTree]:
    """Materialise the merge-base as a temporary detached worktree.

    Always removed, even when the caller raised. A leaked worktree is a bug.
    """
    if merge_base is None:
        yield BaselineTree(None, "the merge-base could not be resolved")
        return
    parent = Path(tempfile.mkdtemp(prefix="edgemason-gate-baseline-"))
    path = parent / "tree"
    try:
        result = _git(candidate, "worktree", "add", "--detach", str(path), merge_base)
        if result.returncode != 0:
            yield BaselineTree(None, f"could not check out {merge_base}: {_tail(result)}")
            return
        yield BaselineTree(path, "")
    finally:
        _git(candidate, "worktree", "remove", "--force", str(path))
        _git(candidate, "worktree", "prune")
        shutil.rmtree(parent, ignore_errors=True)


# --- step 1: assets -------------------------------------------------------


def step_assets(candidate: Path) -> Step:
    """Verify the candidate's assets, provisioning them once if they are missing."""
    script = candidate / "scripts" / "provision_assets.py"
    if not script.is_file():
        raise GateError(f"no provisioning script at {script}")

    verify = _run(
        [sys.executable, str(script), "--dest", str(candidate), "--verify-only"],
        cwd=candidate,
        env=_child_env(),
    )
    if verify.returncode == 0:
        return Step("assets", PASS, _tail(verify))
    if verify.returncode != 3:
        return Step(
            "assets",
            FAIL,
            f"`provision_assets.py --verify-only` reported a mismatch "
            f"(exit {verify.returncode}): {_tail(verify)}",
        )

    # 3 is the real script's "files are missing": provision them, then verify.
    provision = _run(
        [sys.executable, str(script), "--dest", str(candidate)],
        cwd=candidate,
        env=_child_env(),
    )
    if provision.returncode != 0:
        return Step(
            "assets",
            FAIL,
            f"provisioning failed (exit {provision.returncode}): {_tail(provision)}",
        )
    again = _run(
        [sys.executable, str(script), "--dest", str(candidate), "--verify-only"],
        cwd=candidate,
        env=_child_env(),
    )
    if again.returncode != 0:
        return Step(
            "assets",
            FAIL,
            f"assets are still not verified after provisioning "
            f"(exit {again.returncode}): {_tail(again)}",
        )
    return Step("assets", PASS, f"assets were missing, provisioned and verified: {_tail(again)}")


# --- step 2: tests --------------------------------------------------------


def _pytest_command(level: str, *extra: str) -> list[str]:
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra]
    if level == "fast":
        command += ["-m", FAST_MARKERS]
    return command


def _collect_node_ids(tree: Path, level: str) -> set[str]:
    result = _run(_pytest_command(level, "--collect-only"), cwd=tree, env=_child_env())
    # 5 is pytest's "no tests collected", which is a valid empty set.
    if result.returncode not in (0, 5):
        raise GateError(f"`pytest --collect-only` failed in {tree} (exit {result.returncode})")
    return {line for line in result.stdout.splitlines() if NODE_ID.match(line)}


def _deleted_ids(candidate_ids: set[str], baseline_ids: set[str]) -> list[str]:
    return sorted(baseline_ids - candidate_ids)


def step_tests(
    candidate: Path,
    level: str,
    allow_deleted: frozenset[str],
    merge_base: str | None,
) -> Step:
    """`check`'s step 2, including the `--allow-deleted` escape hatch."""
    with baseline_worktree(candidate, merge_base) as baseline:
        if baseline.path is None:
            return Step("tests", SKIPPED, f"baseline test ids unavailable: {baseline.reason}")
        run = _run(_pytest_command(level), cwd=candidate, env=_child_env())
        if run.returncode != 0:
            return Step(
                "tests",
                FAIL,
                f"`pytest` failed in the candidate (exit {run.returncode}): {_tail(run)}",
            )
        try:
            candidate_ids = _collect_node_ids(candidate, level)
            baseline_ids = _collect_node_ids(baseline.path, level)
        except GateError as exc:
            return Step("tests", FAIL, str(exc))

    missing = [node_id for node_id in _deleted_ids(candidate_ids, baseline_ids)
               if node_id not in allow_deleted]
    if missing:
        return Step(
            "tests",
            FAIL,
            f"{len(missing)} test id(s) are collected in the baseline tree but missing from "
            f"the candidate: {_list(missing)}",
        )
    waived = sorted(set(_deleted_ids(candidate_ids, baseline_ids)) & set(allow_deleted))
    details = f"{_tail(run)}; no baseline test id is missing"
    if waived:
        details += f"; {len(waived)} allowed deletion(s): {_list(waived)}"
    return Step("tests", PASS, details)


# --- step 3: goldens ------------------------------------------------------


def step_goldens(
    candidate: Path,
    merge_base: str | None,
    merge_base_reason: str,
    allow_golden_change: bool,
) -> Step:
    """Golden and baseline files must not have moved since the merge-base."""
    if merge_base is None:
        return Step("goldens", SKIPPED, f"no merge-base to diff against: {merge_base_reason}")
    result = _git(candidate, "diff", "--name-only", merge_base, "--", *GOLDEN_PATHS)
    if result.returncode != 0:
        return Step("goldens", FAIL, f"`git diff` failed: {_tail(result)}")
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not changed:
        return Step("goldens", PASS, f"no golden or baseline file changed since {merge_base[:12]}")
    if allow_golden_change:
        return Step(
            "goldens",
            PASS,
            f"{len(changed)} golden/baseline file(s) changed, allowed by --allow-golden-change: "
            f"{_list(changed)}",
        )
    return Step(
        "goldens",
        FAIL,
        f"{len(changed)} golden/baseline file(s) changed since {merge_base[:12]}, and "
        f"--allow-golden-change was not given: {_list(changed)}",
    )


# --- step 4: deterministic outputs ----------------------------------------


def _diff_json(
    expected: object,
    actual: object,
    path: str,
    tolerance: float | None,
) -> list[tuple[str, str]]:
    """Structural differences as (json path, reason) pairs, in document order.

    `tolerance` is the allowed difference per number; `None` means every value
    must be identical.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[tuple[str, str]] = []
        ordered = list(expected) + [key for key in actual if key not in expected]
        for key in ordered:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append((child, "added"))
            elif key not in actual:
                differences.append((child, "removed"))
            else:
                differences += _diff_json(expected[key], actual[key], child, tolerance)
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            child = f"{path}[{index}]" if path else f"[{index}]"
            if index >= len(expected):
                differences.append((child, "added"))
            elif index >= len(actual):
                differences.append((child, "removed"))
            else:
                differences += _diff_json(expected[index], actual[index], child, tolerance)
        return differences
    numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (expected, actual)
    )
    if numeric and tolerance is not None:
        # An int and the float it equals are the same accuracy, not a difference.
        delta = abs(float(expected) - float(actual))  # pyright: ignore [reportArgumentType]
        if delta > tolerance:
            return [(path, f"differs by {delta:.3g}, which is more than the "
                           f"{tolerance:g} tolerance")]
        return []
    if type(expected) is type(actual) and expected == actual:
        return []
    return [(path, "differs")]


def _byte_difference(expected: Path, actual: Path, name: str) -> str:
    """Why two same-named JSON files are not byte-identical."""
    try:
        expected_json = json.loads(expected.read_text(encoding="utf-8"))
        actual_json = json.loads(actual.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "content differs and is not JSON"
    differences = _diff_json(expected_json, actual_json, "", None)
    if not differences:
        return "the values are equal, so the bytes differ only in formatting"
    path, reason = differences[0]
    location = f"{name}:{path}" if path else name
    return f"differs at {location} ({reason})"


def _compare_outputs(expected_dir: Path, captured_dir: Path) -> list[str]:
    """One line per difference between the stored baseline and the capture."""
    expected_names = sorted(path.name for path in expected_dir.iterdir() if path.is_file())
    captured_names = sorted(path.name for path in captured_dir.iterdir() if path.is_file())
    problems: list[str] = []
    for name in expected_names:
        if name not in captured_names:
            problems.append(f"{name}: missing from the candidate's capture")
            continue
        if name == ENVIRONMENT_FILE:
            # Machine- and date-specific by design; P0.8 records it, nobody compares it.
            continue
        expected, actual = expected_dir / name, captured_dir / name
        if name == ACCURACY_FILE:
            try:
                expected_json = json.loads(expected.read_text(encoding="utf-8"))
                actual_json = json.loads(actual.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"{name}: unreadable as JSON ({exc})")
                continue
            problems += [
                f"{name}:{path} {reason}"
                for path, reason in _diff_json(expected_json, actual_json, "", ACCURACY_TOLERANCE)
            ]
        elif expected.read_bytes() != actual.read_bytes():
            problems.append(_byte_difference(expected, actual, name))
    problems += [f"{name}: unexpected file in the candidate's capture"
                 for name in captured_names if name not in expected_names]
    return problems


def step_outputs(candidate: Path, baseline_dir: Path, level: str) -> Step:
    """Capture the candidate's deterministic outputs and diff them."""
    expected_dir = baseline_dir / ("quick" if level == "fast" else "full")
    script = candidate / "scripts" / "capture_baseline.py"
    if not script.is_file():
        raise GateError(f"no capture script at {script}")

    out_dir = Path(tempfile.mkdtemp(prefix="edgemason-gate-capture-"))
    try:
        command = [sys.executable, str(script), "--out", str(out_dir)]
        if level == "fast":
            command.append("--quick")
        result = _run(command, cwd=candidate, env=_child_env())
        if result.returncode != 0:
            return Step(
                "deterministic-outputs",
                FAIL,
                f"`capture_baseline.py` failed in the candidate (exit {result.returncode}): "
                f"{_tail(result)}",
            )
        problems = _compare_outputs(expected_dir, out_dir)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    if problems:
        return Step(
            "deterministic-outputs",
            FAIL,
            f"{len(problems)} difference(s) against {expected_dir}: {_list(problems)}",
        )
    counted = len([name for name in expected_dir.iterdir() if name.is_file()])
    return Step(
        "deterministic-outputs",
        PASS,
        f"{counted} baseline file(s) match the candidate's capture against {expected_dir} "
        f"(accuracy within {ACCURACY_TOLERANCE:g}, {ENVIRONMENT_FILE} not compared)",
    )


# --- step 5: lint and types -----------------------------------------------


def step_lint_types(candidate: Path, merge_base: str | None) -> Step:
    """P0.10's ratchet, run against the same merge-base the gate judges by."""
    ratchet = candidate / "scripts" / "lint_ratchet.sh"
    if not ratchet.is_file():
        return Step(
            "lint-types",
            SKIPPED,
            f"no ratchet at {ratchet}: P0.10 has not merged into this tree, so ruff and "
            f"pyright were not run",
        )
    env = _child_env()
    if merge_base is not None:
        env["BASE_REF"] = merge_base
    result = _run(["bash", str(ratchet)], cwd=candidate, env=env)
    if result.returncode != 0:
        return Step(
            "lint-types",
            FAIL,
            f"`lint_ratchet.sh` failed (exit {result.returncode}): {_tail(result, 12)}",
        )
    return Step("lint-types", PASS, _tail(result, 3))


# --- the ab subcommand ----------------------------------------------------

# Every measurement runs inside the tree it measures, so the product code that
# imports must be that tree's and not this gate's. The worker is passed with
# `-c` and cwd set to the tree, which puts the tree first on sys.path, and it
# is handed every path it needs so no asset path is written down twice.
AB_WORKER = '''
import json
import sys
from pathlib import Path


def prepare(request):
    from api.optimizer.quantizer import Quantizer
    from api.optimizer.validation import load_validation_data, split_calibration_and_test

    source = Path(request["model_dir"]) / f"{request['model']}.onnx"
    if not source.is_file():
        raise SystemExit(f"missing model: {source}")
    artifact_dir = Path(request["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    calibration = None
    if request["needs_calibration"]:
        calibration = split_calibration_and_test(
            load_validation_data(request["dataset"])
        ).calibration
    built = Quantizer(str(artifact_dir)).quantize_for_candidate(
        str(source), request["candidate"], calibration_data=calibration
    )
    print(json.dumps({"artifact": built}))


def measure(request):
    from api import config
    from api.optimizer.benchmark_engine import benchmark

    # The tree's own defaults, not this gate's idea of them.
    payload = json.loads(benchmark(
        request["artifact"],
        warmup_runs=config.BENCHMARK_WARMUP_RUNS,
        benchmark_runs=config.BENCHMARK_TIMED_RUNS,
        time_budget_seconds=config.BENCHMARK_TIME_BUDGET_SECONDS,
        isolated=True,
    ))
    if not payload.get("success"):
        raise SystemExit("the benchmark did not succeed: " + str(payload.get("error")))
    print(json.dumps({
        "p50": payload["latency_ms"]["p50"],
        "memory_delta": payload["memory_mb"]["delta"],
    }))


ACTIONS = {"prepare": prepare, "measure": measure}


if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    ACTIONS[request["mode"]](request)
'''

# One subprocess per measurement, so a tree that hangs takes the gate with it
# rather than the whole run. The heaviest measured run here takes ~8 s.
AB_SUBPROCESS_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class AbRequest:
    """Everything the `ab` subcommand needs, resolved."""

    baseline: Path
    candidate: Path
    models: tuple[str, ...]
    rounds: int
    band_pct: float = AB_BAND_PCT


@dataclass(frozen=True)
class MetricComparison:
    """One measured quantity, compared round by round across the two trees.

    The verdict is the **mean** of the *paired* per-round percentage
    differences. Medians of independent samples fold in anything that moved the
    whole series, including the position a run happened to occupy, and the
    bimodal bursts seen on a shared machine make MAD useless as a band
    (measured: MAD_B 0.2% against a real spread of 6.6%). Pairing cancels the
    position, and the mean absorbs the straddles that make the median a coin
    flip. `median_pct_diff` and `mad_baseline` are reported under
    `informational` and decide nothing.
    """

    baseline: tuple[float, ...]
    candidate: tuple[float, ...]
    pct_diffs: tuple[float, ...]
    mean_pct_diff: float
    sd_pct_diff: float
    max_abs_pct_diff: float
    band_pct: float
    # Informational only; kept for auditability and deliberately not in the rule.
    median_pct_diff: float
    median_baseline: float
    median_candidate: float
    mad_baseline: float

    @property
    def n(self) -> int:
        return len(self.pct_diffs)

    @property
    def within_band(self) -> bool:
        return abs(self.mean_pct_diff) <= self.band_pct

    @property
    def standard_error(self) -> float:
        """The sampling error of the mean, so a reader can judge the verdict."""
        return self.sd_pct_diff / math.sqrt(self.n) if self.n else 0.0

    @property
    def minimum_detectable_effect(self) -> float:
        """The smallest effect two standard errors from zero: about 95%."""
        return 2.0 * self.standard_error

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": list(self.baseline),
            "candidate": list(self.candidate),
            "pct_diffs": list(self.pct_diffs),
            "n": self.n,
            "mean_pct_diff": self.mean_pct_diff,
            "sd_pct_diff": self.sd_pct_diff,
            "max_abs_pct_diff": self.max_abs_pct_diff,
            "band_pct": self.band_pct,
            "within_band": self.within_band,
            "informational": {
                "median_pct_diff": self.median_pct_diff,
                "median_baseline": self.median_baseline,
                "median_candidate": self.median_candidate,
                "mad_baseline": self.mad_baseline,
            },
        }

    def power_dict(self) -> dict[str, object]:
        """What the verdict could and could not have seen."""
        return {
            "n": self.n,
            "se": self.standard_error,
            "mde_2se": self.minimum_detectable_effect,
            "band_pct": self.band_pct,
            # True when the gate could resolve this band at all; a band far
            # inside the noise means a pass here proves very little.
            "powered_for_band": self.minimum_detectable_effect <= self.band_pct,
        }


@dataclass(frozen=True)
class VariantComparison:
    """One model × variant: two measured quantities and one verdict."""

    model: str
    variant: str
    p50: MetricComparison
    memory_delta: MetricComparison

    @property
    def passed(self) -> bool:
        return self.p50.within_band and self.memory_delta.within_band

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "variant": self.variant,
            "p50": self.p50.as_dict(),
            "memory_delta": self.memory_delta.as_dict(),
            "pass": self.passed,
        }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _median_absolute_deviation(values: Sequence[float]) -> float:
    centre = _median(values)
    return _median([abs(value - centre) for value in values])


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _sample_sd(values: Sequence[float]) -> float:
    """The sample standard deviation; 0 for a single sample, which has no spread."""
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def compare_metric(
    baseline: Sequence[float],
    candidate: Sequence[float],
    band_pct: float,
) -> MetricComparison:
    """Compare one quantity by pairing each round with the other side of it.

    `baseline` and `candidate` are the two measurements taken in each round, in
    round order and *regardless of which ran first*; the order is what
    `AB_ROUND_ORDERS` balances, not something this function needs to know.

    The verdict uses the **mean** of the paired differences, not the median.
    The differences are bimodal, sitting near zero when both runs shared a
    machine state and near ±7% when a round straddled a frequency or thermal
    change. The median flips to ±7% as soon as three of five rounds straddle
    one, which is a coin flip rather than a measurement; the straddles partly
    cancel in the mean, so the mean tracks the real effect while the median
    tracks whichever side of a state change the sample happened to land on.
    """
    if len(baseline) != len(candidate):
        raise GateError("the two sides produced a different number of measurements")
    pct_diffs = tuple(
        ((candidate_value - baseline_value) / baseline_value) * 100.0
        for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    )
    return MetricComparison(
        baseline=tuple(baseline),
        candidate=tuple(candidate),
        pct_diffs=pct_diffs,
        mean_pct_diff=_mean(pct_diffs),
        sd_pct_diff=_sample_sd(pct_diffs),
        max_abs_pct_diff=max((abs(value) for value in pct_diffs), default=0.0),
        band_pct=band_pct,
        median_pct_diff=_median(pct_diffs),
        median_baseline=_median(baseline),
        median_candidate=_median(candidate),
        mad_baseline=_median_absolute_deviation(baseline),
    )


def _ab_worker(tree: Path, request: Mapping[str, object]) -> dict[str, object]:
    """Run one worker action inside `tree` and return what it printed.

    A crash or a timeout here is an environment error, not a measurement: the
    gate cannot compare numbers it never got.
    """
    env = _child_env()
    env["PYTHONPATH"] = str(tree)
    command = [sys.executable, "-c", AB_WORKER, json.dumps(request)]
    try:
        result = _run(command, cwd=tree, env=env, timeout=AB_SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise GateError(
            f"the benchmark worker in {tree} did not finish within "
            f"{AB_SUBPROCESS_TIMEOUT_SECONDS}s"
        ) from None
    if result.returncode != 0:
        raise GateError(f"the benchmark worker in {tree} failed: {_tail(result)}")
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise GateError(f"the benchmark worker in {tree} printed no JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise GateError(f"the benchmark worker in {tree} printed an unexpected payload")
    return payload


def _ab_prepare(tree: Path, model: str, variant: str, artifact_dir: Path) -> Path:
    """Build this tree's own artifact for one variant, with its own Quantizer."""
    payload = _ab_worker(tree, {
        "mode": "prepare",
        "model": model,
        "model_dir": str(tree / AB_MODEL_DIR),
        "artifact_dir": str(artifact_dir),
        "candidate": AB_CANDIDATES[variant],
        # Only static INT8 needs calibration; fp32 and dynamic do not.
        "needs_calibration": variant == "static",
        "dataset": str(tree / AB_DATASET),
    })
    artifact = payload.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        raise GateError(f"the worker in {tree} produced no artifact for {model}/{variant}")
    return Path(artifact)


def _ab_measure(tree: Path, artifact: Path, warmup: bool = False) -> tuple[float, float]:
    """One isolated benchmark run inside `tree`; returns (p50 ms, memory delta MB).

    `warmup` marks a throwaway run: its numbers are discarded, but the flag
    travels with the request so a run can be shown to have happened.
    """
    payload = _ab_worker(tree, {
        "mode": "measure",
        "artifact": str(artifact),
        "warmup": warmup,
    })
    p50 = payload.get("p50")
    memory_delta = payload.get("memory_delta")
    if not isinstance(p50, (int, float)) or not isinstance(memory_delta, (int, float)):
        raise GateError(f"the worker in {tree} returned no usable measurement")
    return float(p50), float(memory_delta)


@dataclass(frozen=True)
class AbReport:
    """The A/B verdict, in the same shape every gate report uses.

    The typed objects are kept so the summary and the report are rendered from
    one source, not from the JSON the report happens to serialise to.
    """

    verdict: str
    steps: tuple[Step, ...]
    request: AbRequest
    results: tuple[VariantComparison, ...]
    winners: dict[str, dict[str, object]]
    orders: dict[str, list[list[str]]]

    @property
    def power(self) -> dict[str, object]:
        """Per model x variant, what the verdict could and could not have seen."""
        return {
            "statistic": "mean of the paired per-round percentage differences",
            "results": {
                f"{result.model}/{result.variant}": {
                    "p50": result.p50.power_dict(),
                    "memory_delta": result.memory_delta.power_dict(),
                }
                for result in self.results
            },
        }

    @property
    def worst_minimum_detectable_effect(self) -> float:
        """The bluntest p50 comparison in the run, in percent."""
        return max(
            (result.p50.minimum_detectable_effect for result in self.results), default=0.0
        )

    @property
    def led_by(self) -> dict[str, int]:
        """How many rounds each side went first in, across the whole run.

        An odd total leaves one side ahead by one; that is recorded rather than
        smoothed away.
        """
        return {
            role: sum(
                1
                for variant_orders in self.orders.values()
                for order in variant_orders
                if order[0] == role
            )
            for role in ("baseline", "candidate")
        }

    @property
    def rounds_per_variant(self) -> int:
        return len(next(iter(self.orders.values()))) if self.orders else 0

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "steps": [step.as_dict() for step in self.steps],
            "ab": {
                "rounds": self.request.rounds,
                "models": list(self.request.models),
                "variants": dict(AB_CANDIDATES),
                "band_pct": self.request.band_pct,
                "warmup_runs_per_side": AB_WARMUP_RUNS_PER_SIDE,
                # Per model x variant, so the balancing of each series is visible.
                "order": {key: list(map(list, value)) for key, value in self.orders.items()},
                "rounds_led_by": self.led_by,
                "results": [result.as_dict() for result in self.results],
                "power": self.power,
                "winners": {model: dict(found) for model, found in self.winners.items()},
            },
        }


def run_ab(request: AbRequest) -> AbReport:
    """Run the alternating A/B protocol and compare every model × variant."""
    for tree, flag in ((request.baseline, "--baseline"), (request.candidate, "--candidate")):
        if not tree.is_dir():
            raise GateError(f"{flag} is not a directory: {tree}")
    if request.rounds < 1:
        raise GateError(f"--rounds must be at least 1, got {request.rounds}")

    comparisons: list[VariantComparison] = []
    baseline_winners: dict[str, str] = {}
    candidate_winners: dict[str, str] = {}

    artifacts = Path(tempfile.mkdtemp(prefix="edgemason-gate-ab-"))
    tree_of = {"baseline": request.baseline, "candidate": request.candidate}
    trees = (("baseline", request.baseline), ("candidate", request.candidate))
    # The order of every round, recorded so a reader can verify the balancing.
    # The parity runs continuously across models and variants rather than
    # restarting per variant, so two consecutive measured runs are never both
    # made by the same tree. Over 3 variants x 5 rounds that is 15 rounds and
    # the baseline leads 8 of them to the candidate's 7.
    measured_round = 0
    orders: dict[str, list[list[str]]] = {}
    try:
        for model in request.models:
            # Each tree builds its own artifact per variant, once: re-quantizing
            # between rounds would measure the quantizer, not the candidate.
            prepared = {
                (role, variant): _ab_prepare(tree, model, variant, artifacts / role / variant)
                for role, tree in trees
                for variant in AB_VARIANTS
            }
            # Warm-up: throw away one run per side before round 0, so the first
            # measured round is not the one paying for the cold artifact.
            for role, tree in trees:
                for variant in AB_VARIANTS:
                    for _ in range(AB_WARMUP_RUNS_PER_SIDE):
                        _ab_measure(tree, prepared[(role, variant)], warmup=True)

            for variant in AB_VARIANTS:
                baseline_p50: list[float] = []
                candidate_p50: list[float] = []
                baseline_memory: list[float] = []
                candidate_memory: list[float] = []
                variant_orders: list[list[str]] = []
                # The order alternates so each tree occupies each position
                # equally often; the parity carries over from the previous
                # variant rather than restarting. The paired series is
                # collected in round order, not in run order.
                for _ in range(request.rounds):
                    order = AB_ROUND_ORDERS[measured_round % len(AB_ROUND_ORDERS)]
                    measured_round += 1
                    variant_orders.append(list(order))
                    for role in order:
                        p50, memory_delta = _ab_measure(
                            tree_of[role], prepared[(role, variant)]
                        )
                        if role == "baseline":
                            baseline_p50.append(p50)
                            baseline_memory.append(memory_delta)
                        else:
                            candidate_p50.append(p50)
                            candidate_memory.append(memory_delta)
                orders[f"{model}/{variant}"] = variant_orders
                comparisons.append(VariantComparison(
                    model=model,
                    variant=variant,
                    p50=compare_metric(baseline_p50, candidate_p50, request.band_pct),
                    memory_delta=compare_metric(
                        baseline_memory, candidate_memory, request.band_pct
                    ),
                ))
            measured = [comparison for comparison in comparisons if comparison.model == model]
            baseline_winners[model] = min(measured, key=lambda c: c.p50.median_baseline).variant
            candidate_winners[model] = min(measured, key=lambda c: c.p50.median_candidate).variant
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)

    steps = _ab_steps(request, comparisons, baseline_winners, candidate_winners)
    verdict = PASS if all(step.status != FAIL for step in steps) else FAIL
    return AbReport(
        verdict=verdict,
        steps=tuple(steps),
        request=request,
        results=tuple(comparisons),
        orders=orders,
        winners={
            model: {
                "baseline": baseline_winners[model],
                "candidate": candidate_winners[model],
                "same": baseline_winners[model] == candidate_winners[model],
            }
            for model in request.models
        },
    )


def _ab_steps(
    request: AbRequest,
    comparisons: Sequence[VariantComparison],
    baseline_winners: Mapping[str, str],
    candidate_winners: Mapping[str, str],
) -> list[Step]:
    """One step per model, then one for the winner, which is judged separately."""
    steps = []
    for model in request.models:
        measured = [comparison for comparison in comparisons if comparison.model == model]
        failed = [comparison for comparison in measured if not comparison.passed]
        details = "; ".join(
            f"{comparison.variant}: p50 {_metric_detail(comparison.p50)}ms, "
            f"memory {_metric_detail(comparison.memory_delta)}mb"
            for comparison in measured
        )
        steps.append(Step(
            f"ab:{model}",
            FAIL if failed else PASS,
            f"{len(measured) - len(failed)}/{len(measured)} variants within the noise band. "
            f"{details}",
        ))
    changed = [
        model for model in request.models
        if baseline_winners[model] != candidate_winners[model]
    ]
    winners = ", ".join(
        f"{model}: baseline {baseline_winners[model]}, candidate {candidate_winners[model]}"
        for model in request.models
    )
    steps.append(Step(
        "ab:winners",
        FAIL if changed else PASS,
        f"the winner changed for {len(changed)} of {len(request.models)} model(s). {winners}",
    ))
    return steps


def _metric_detail(comparison: MetricComparison) -> str:
    """One metric as its mean paired difference against the band."""
    verdict = "within" if comparison.within_band else "OUTSIDE"
    return (
        f"mean {comparison.mean_pct_diff:+.2f}% (sd {comparison.sd_pct_diff:.2f}%, "
        f"n={comparison.n}, band {comparison.band_pct:g}%, "
        f"worst round {comparison.max_abs_pct_diff:.2f}%, {verdict})"
    )


def _ab_summary(report: AbReport) -> str:
    """A short markdown table of the A/B run for stdout."""
    rows = [
        "| model | variant | p50 mean | sd | band | p50 | memory mean | band | memory |",
        "| ----- | ------ | -------- | -- | ---- | --- | ----------- | ---- | ------ |",
    ]
    for result in report.results:
        rows.append(
            f"| {result.model} | {result.variant} | "
            f"{result.p50.mean_pct_diff:+.2f}% | {result.p50.sd_pct_diff:.2f}% | "
            f"{result.p50.band_pct:g}% | {_verdict_word(result.p50)} | "
            f"{result.memory_delta.mean_pct_diff:+.2f}% | "
            f"{result.memory_delta.band_pct:g}% | {_verdict_word(result.memory_delta)} |"
        )
    led = report.led_by
    worst_mde = report.worst_minimum_detectable_effect
    lines = [
        f"## P0.9 gate `ab`: **{report.verdict.upper()}**",
        "",
        f"- baseline: `{report.request.baseline}`",
        f"- candidate: `{report.request.candidate}`",
        "- statistic: mean of the paired per-round percentage differences",
        f"- rounds: {report.request.rounds} per model x variant, order alternates "
        f"starting {AB_ROUND_ORDERS[0][0]}",
        f"- rounds led by: baseline {led['baseline']}, candidate {led['candidate']}",
        f"- smallest effect this run could see: ~{worst_mde:.2f}% "
        f"(worst 2 SE; the band is {report.request.band_pct:g}%, "
        f"{'so' if worst_mde <= report.request.band_pct else 'but'}"
        f"{' the gate is powered for it' if worst_mde <= report.request.band_pct else ''}"
        f")",
        "",
        *rows,
        "",
        *[
            f"- winner {model}: baseline {found['baseline']}, candidate {found['candidate']}"
            + ("" if found["same"] else "  <- CHANGED")
            for model, found in report.winners.items()
        ],
    ]
    details = [f"**{step.name}** - {step.details}" for step in report.steps if step.status == FAIL]
    if details:
        lines += ["", *details]
    return "\n".join(lines)


def _verdict_word(comparison: MetricComparison) -> str:
    return "pass" if comparison.within_band else "FAIL"


def _ab_request(args: argparse.Namespace) -> AbRequest:
    return AbRequest(
        baseline=args.baseline.resolve(),
        candidate=args.candidate.resolve(),
        models=tuple(args.models) if args.models else AB_MODELS,
        rounds=args.rounds,
        band_pct=args.band_pct,
    )


# --- the check subcommand -------------------------------------------------


@dataclass(frozen=True)
class CheckReport:
    """The whole verdict, in the shape the spec's report fixes."""

    verdict: str
    steps: tuple[Step, ...]
    ab: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "steps": [step.as_dict() for step in self.steps],
            "ab": dict(self.ab),
        }


def run_check(request: CheckRequest) -> CheckReport:
    """Run the five steps in order and stop at the first failure."""
    if not request.candidate.is_dir():
        raise GateError(f"--candidate is not a directory: {request.candidate}")
    if not request.baseline_dir.is_dir():
        raise GateError(f"--baseline-dir is not a directory: {request.baseline_dir}")

    merge_base, merge_base_reason = resolve_merge_base(request.candidate)
    runners = [
        lambda: step_assets(request.candidate),
        lambda: step_tests(request.candidate, request.level, request.allow_deleted, merge_base),
        lambda: step_goldens(
            request.candidate, merge_base, merge_base_reason, request.allow_golden_change
        ),
        lambda: step_outputs(request.candidate, request.baseline_dir, request.level),
        lambda: step_lint_types(request.candidate, merge_base),
    ]

    steps: list[Step] = []
    for index, runner in enumerate(runners):
        step = runner()
        steps.append(step)
        if step.status == FAIL:
            steps += [
                Step(name, NOT_RUN, "not run: an earlier step failed")
                for name in STEP_NAMES[index + 1:]
            ]
            break

    verdict = PASS if all(step.status != FAIL for step in steps) else FAIL
    return CheckReport(
        verdict=verdict,
        steps=tuple(steps),
        ab={
            "status": "not run",
            "reason": "the check subcommand does not run the A/B protocol",
        },
    )


def _summary(report: CheckReport, candidate: Path, level: str) -> str:
    """A short markdown summary for stdout."""
    rows = [
        f"| {index} | {step.name} | {step.status} |"
        for index, step in enumerate(report.steps, start=1)
    ]
    lines = [
        f"## P0.9 gate `check`: **{report.verdict.upper()}**",
        "",
        f"- candidate: `{candidate}`",
        f"- level: `{level}`",
        "",
        "| # | step | status |",
        "| - | ---- | ------ |",
        *rows,
    ]
    details = [
        f"**{step.name}** - {step.details}"
        for step in report.steps
        if step.status in (FAIL, SKIPPED)
    ]
    if details:
        lines += ["", *details]
    return "\n".join(lines)


def _read_allow_deleted(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    if not path.is_file():
        raise GateError(f"--allow-deleted is not a file: {path}")
    return frozenset(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gate.py", description="The P0.9 results gate."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("check", help="judge a candidate tree against the baseline")
    check.add_argument("--candidate", required=True, type=Path, help="git worktree root to judge")
    check.add_argument(
        "--level", choices=("fast", "full"), default="fast",
        help="fast runs the unmarked-out suite and the two quick models (default: fast)",
    )
    check.add_argument(
        "--baseline-dir", type=Path, default=Path("baseline"),
        help="directory holding the stored quick/ and full/ captures (default: baseline)",
    )
    check.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    check.add_argument(
        "--allow-deleted", type=Path, default=None, metavar="FILE",
        help="file of test node ids the candidate may have deleted, one per line",
    )
    check.add_argument(
        "--allow-golden-change", action="store_true",
        help="accept changes under tests/golden/ and baseline/ (approved specs only)",
    )

    ab = subcommands.add_parser(
        "ab", help="time two trees against each other over alternating rounds"
    )
    ab.add_argument("--baseline", required=True, type=Path, help="git worktree root to measure")
    ab.add_argument("--candidate", required=True, type=Path, help="git worktree root to measure")
    ab.add_argument(
        "--models", nargs="+", default=None, metavar="NAME",
        help=f"models to compare (default: {', '.join(AB_MODELS)})",
    )
    ab.add_argument(
        "--rounds", type=int, default=AB_ROUNDS,
        help=(
            "alternating B/C rounds per model and variant "
            f"(default: {AB_ROUNDS}; fewer rounds cannot resolve the band)"
        ),
    )
    ab.add_argument(
        "--band-pct", type=float, default=AB_BAND_PCT, metavar="PCT",
        help=(
            "allowed mean paired difference in percent "
            f"(default: {AB_BAND_PCT:g}; the spec's threshold)"
        ),
    )
    ab.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    return parser


def _check_request(args: argparse.Namespace) -> CheckRequest:
    baseline_dir = args.baseline_dir
    if not baseline_dir.is_absolute():
        baseline_dir = PROJECT_ROOT / baseline_dir
    return CheckRequest(
        candidate=args.candidate.resolve(),
        level=args.level,
        baseline_dir=baseline_dir,
        allow_deleted=_read_allow_deleted(args.allow_deleted),
        allow_golden_change=args.allow_golden_change,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ab":
            result = run_ab(_ab_request(args))
            print(_ab_summary(result))
            payload = result.as_dict()
            verdict = result.verdict
        else:
            request = _check_request(args)
            result = run_check(request)
            print(_summary(result, args.candidate, args.level))
            payload = result.as_dict()
            verdict = result.verdict
    except GateError as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return EXIT_PASS if verdict == PASS else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
