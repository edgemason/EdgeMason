import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from api.optimizer.quantizer import Quantizer


@pytest.fixture
def calib_npy(tmp_path):
    path = tmp_path / "calib.npy"
    np.save(path, np.random.rand(10, 4).astype(np.float32))
    return str(path)


def test_create_fp32(tiny_onnx, tmp_path):
    quantizer = Quantizer(output_dir=str(tmp_path / "out"))
    out = quantizer.create_fp32(str(tiny_onnx))
    onnx.checker.check_model(onnx.load(out))


def test_create_dynamic_int8(tiny_onnx, tmp_path):
    quantizer = Quantizer(output_dir=str(tmp_path / "out"))
    out = quantizer.create_dynamic_int8(str(tiny_onnx))
    assert onnx.load(out).graph.node


def test_dynamic_int8_convolution_runs_on_cpu(tmp_path):
    model_path = tmp_path / "conv.onnx"
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 6, 6]
    )
    weights = np.ones((2, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "weights"], ["output"])],
        "conv",
        [model_input],
        [model_output],
        initializer=[
            helper.make_tensor(
                "weights", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, model_path)

    quantized_path = Quantizer(str(tmp_path / "out")).create_dynamic_int8(
        str(model_path)
    )
    session = ort.InferenceSession(
        quantized_path,
        providers=["CPUExecutionProvider"],
    )
    result = session.run(
        None,
        {"input": np.ones((1, 3, 8, 8), dtype=np.float32)},
    )

    assert result[0].shape == (1, 2, 6, 6)


def test_create_static_int8(tiny_onnx, calib_npy, tmp_path):
    quantizer = Quantizer(output_dir=str(tmp_path / "out"))
    out = quantizer.create_static_int8(str(tiny_onnx), calib_npy)
    assert onnx.load(out).graph.node


def test_create_static_int8_accepts_labeled_validation_npz(tiny_onnx, tmp_path):
    calibration_path = tmp_path / "validation.npz"
    np.savez(
        calibration_path,
        inputs=np.random.rand(10, 4).astype(np.float32),
        labels=np.arange(10, dtype=np.int64),
    )

    output = Quantizer(str(tmp_path / "out")).create_static_int8(
        str(tiny_onnx),
        str(calibration_path),
    )

    session = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
    result = session.run(None, {"X": np.ones((1, 4), dtype=np.float32)})
    assert result[0].shape == (1, 8)


def test_static_int8_requires_calibration(tiny_onnx, tmp_path):
    quantizer = Quantizer(output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError):
        quantizer.quantize_for_candidate(str(tiny_onnx), "onnx_static_int8")


def test_unknown_candidate_raises(tiny_onnx, tmp_path):
    quantizer = Quantizer(output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError):
        quantizer.quantize_for_candidate(str(tiny_onnx), "tvm_autotvm")


def test_static_int8_uses_one_weight_scale_per_output_channel(tmp_path):
    # Conv with 4 output channels whose weights differ by 1000x in size.
    weights = np.random.rand(4, 3, 3, 3).astype(np.float32)
    weights *= np.array([1.0, 0.1, 0.01, 0.001], dtype=np.float32)[:, None, None, None]
    graph = helper.make_graph(
        [helper.make_node("Conv", ["X", "W"], ["Y"])],
        "conv",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 6, 6])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    model_path = tmp_path / "conv.onnx"
    onnx.save(model, model_path)

    output = Quantizer(str(tmp_path / "out")).create_static_int8(
        str(model_path),
        np.random.rand(8, 3, 8, 8).astype(np.float32),
    )

    quantized = onnx.load(output)
    initializers = {init.name: numpy_helper.to_array(init) for init in quantized.graph.initializer}
    weight_scales = [
        initializers[node.input[1]]
        for node in quantized.graph.node
        if node.op_type == "DequantizeLinear" and initializers.get(node.input[0]) is not None
        and initializers[node.input[0]].ndim == 4
    ]
    assert len(weight_scales) == 1
    scale = weight_scales[0]
    assert scale.shape == (4,)
    # Each channel's scale follows its own weight size.
    assert scale[0] > scale[1] > scale[2] > scale[3]
