import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from api.optimizer.tvm_tuning import (
    TuningConfig,
    apply_tuning_best,
    configure_runtime_threads,
    get_default_tuning_dir,
    get_tuning_database,
)
from api.planner.rule_based import (
    STRATEGY_TVM_OPT3,
    STRATEGY_TVM_TUNED,
    PlannerInput,
    generate_candidates,
)


def test_tuning_config_defaults():
    config = TuningConfig()
    assert config.hardware_id == "x86_64"
    assert config.max_trials_global == 16
    assert config.num_trials_per_iter == 4
    assert config.strategy == "evolutionary"

    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["max_trials_global"] == 16


def test_get_default_tuning_dir(tmp_path):
    target_dir = get_default_tuning_dir("rpi4")
    assert target_dir.name == "rpi4"
    assert target_dir.exists()


def test_configure_runtime_threads():
    threads = configure_runtime_threads(4)
    assert threads == 4
    assert os.environ.get("TVM_NUM_THREADS") == "4"
    assert os.environ.get("OMP_NUM_THREADS") == "4"


def test_planner_tvm_tuning_candidates():
    # Without TVM tuning
    inp_no_tune = PlannerInput(
        model_size_mb=10.0,
        include_tvm=True,
        enable_tvm_tuning=False,
    )
    plan_no_tune = generate_candidates(inp_no_tune)
    names_no_tune = [c["name"] for c in plan_no_tune["candidates"]]
    assert STRATEGY_TVM_OPT3 in names_no_tune
    assert STRATEGY_TVM_TUNED not in names_no_tune

    # With TVM tuning
    inp_tune = PlannerInput(
        model_size_mb=10.0,
        include_tvm=True,
        enable_tvm_tuning=True,
    )
    plan_tune = generate_candidates(inp_tune)
    names_tune = [c["name"] for c in plan_tune["candidates"]]
    assert STRATEGY_TVM_OPT3 in names_tune
    assert STRATEGY_TVM_TUNED in names_tune


def test_apply_tuning_best_none():
    mock_mod = MagicMock()
    result = apply_tuning_best(mock_mod, None)
    assert result is mock_mod


def test_quantizer_tvm_tuned_routing(tmp_path):
    from api.optimizer.quantizer import Quantizer

    q = Quantizer(str(tmp_path))

    with patch.object(q, "create_tvm_tuned", return_value="dummy_tuned.tar") as mock_tune:
        res = q.quantize_for_candidate(
            model_path="dummy.onnx",
            candidate_name="tvm_tuned",
            hardware_id="rpi4",
        )
        assert res == "dummy_tuned.tar"
        mock_tune.assert_called_once_with("dummy.onnx", hardware_id="rpi4")
