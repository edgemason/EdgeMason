import pytest
import numpy as np
import onnx
from onnx import TensorProto, helper

from api.optimizer.validation import evaluate_onnx, load_validation_data


def _write_accuracy_model(path):
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["N", 3, 1, 1]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["N", 6]
    )
    weights = np.array(
        [
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.0, 0.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weights", "bias"], ["output"]),
        ],
        "accuracy",
        [model_input],
        [model_output],
        initializer=[
            helper.make_tensor(
                "weights", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True
            ),
            helper.make_tensor(
                "bias", TensorProto.FLOAT, bias.shape, bias.tobytes(), raw=True
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    onnx.save(model, path)


def test_labeled_validation_data_reports_top1_and_top5(tmp_path):
    model_path = tmp_path / "accuracy.onnx"
    data_path = tmp_path / "validation.npz"
    _write_accuracy_model(model_path)

    inputs = np.zeros((2, 3, 1, 1), dtype=np.float32)
    inputs[0, 0, 0, 0] = 1.0
    inputs[1, 0, 0, 0] = -1.0
    np.savez(data_path, inputs=inputs, labels=np.array([0, 2], dtype=np.int64))

    validation = load_validation_data(data_path)
    result = evaluate_onnx(model_path, validation)

    assert result == {
        "sample_count": 2,
        "top1_accuracy": 0.5,
        "top5_accuracy": 1.0,
    }


def test_validation_data_rejects_mismatched_labels(tmp_path):
    data_path = tmp_path / "invalid.npz"
    np.savez(
        data_path,
        inputs=np.zeros((2, 3, 1, 1), dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
    )

    try:
        load_validation_data(data_path)
    except ValueError as exc:
        assert str(exc) == "inputs and labels must contain the same number of samples"
    else:
        raise AssertionError("invalid validation data was accepted")


def _labeled(count):
    from api.optimizer.validation import ValidationData

    return ValidationData(
        inputs=np.arange(count, dtype=np.float32).reshape(count, 1, 1),
        labels=np.arange(count, dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("count", "calibration", "test"),
    [(1000, 300, 700), (256, 77, 179), (50, 15, 35), (4, 1, 3), (2, 1, 1)],
)
def test_split_takes_30_percent_capped_at_300_for_calibration(count, calibration, test):
    from api.optimizer.validation import split_calibration_and_test

    split = split_calibration_and_test(_labeled(count))

    assert len(split.calibration) == calibration
    assert len(split.test.inputs) == test


def test_split_is_disjoint_complete_and_repeatable():
    from api.optimizer.validation import split_calibration_and_test

    first = split_calibration_and_test(_labeled(100))
    second = split_calibration_and_test(_labeled(100))

    calibration_ids = set(first.calibration.ravel().tolist())
    test_ids = set(first.test.inputs.ravel().tolist())
    assert calibration_ids.isdisjoint(test_ids)
    assert calibration_ids | test_ids == set(range(100))
    # Labels stay attached to their images.
    assert np.array_equal(first.test.inputs.ravel(), first.test.labels)
    assert np.array_equal(first.calibration, second.calibration)
    # Shuffled, not just the first 30.
    assert calibration_ids != set(range(30))


def test_split_rejects_single_sample():
    from api.optimizer.validation import split_calibration_and_test

    with pytest.raises(ValueError, match="at least 2 samples"):
        split_calibration_and_test(_labeled(1))


def test_data_warnings():
    from api.optimizer.validation import data_warnings

    assert data_warnings(300, 700) == []
    small = data_warnings(77, 179)
    assert "Only 77 calibration images" in small[0]
    assert "179 test images" in small[1] and "±7 points" in small[1]
    assert len(data_warnings(50, None)) == 1
