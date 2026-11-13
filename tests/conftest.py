import os
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def tiny_onnx(tmp_path):
    """Build a tiny MatMul+Add+Relu model and write it to a temp file."""
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["h"]),
        helper.make_node("Add", ["h", "B"], ["a"]),
        helper.make_node("Relu", ["a"], ["Y"]),
    ]
    w = np.random.rand(4, 8).astype(np.float32)
    b = np.random.rand(8).astype(np.float32)
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True),
            helper.make_tensor("B", TensorProto.FLOAT, b.shape, b.tobytes(), raw=True),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    return path


@pytest.fixture(autouse=True)
def _seed_rng():
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
