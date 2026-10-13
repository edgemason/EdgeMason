from pathlib import Path

import onnx

from api.config import MAX_INPUT_ELEMENTS, MAX_MODEL_NODES


def validate_model(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.node) > MAX_MODEL_NODES:
        raise ValueError("model has too many nodes")

    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError("the pipeline supports single-input models only")

    tensor_type = inputs[0].type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("the pipeline supports float32 model inputs only")

    elements = 1
    for index, dimension in enumerate(tensor_type.shape.dim):
        if dimension.dim_value > 0:
            value = dimension.dim_value
        elif index == 0:
            value = 1
        else:
            raise ValueError("only the model batch dimension may be dynamic")
        if value > 4096:
            raise ValueError("model input dimension is too large")
        elements *= value
    if elements > MAX_INPUT_ELEMENTS:
        raise ValueError("model input tensor is too large")
