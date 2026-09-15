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
