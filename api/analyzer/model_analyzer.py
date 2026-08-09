import json
import os
from collections import Counter
from pathlib import Path
import onnx
import numpy as np


def is_file_valid(filename: str) -> bool:
    """Checking if the model given is in valid onnx format

    Args:
      filename : File name given by the user , stored as str

    Output:
      Bool : True or False value
    """
    return Path(filename).suffix.lower() == ".onnx"

#ModelProto actaully open and validate the model
def load_model(filename: str) -> onnx.ModelProto:
    """Load an ONNX model from disk and validate it.

    Args:
      filename : Path to the .onnx file

    Output:
      onnx.ModelProto : The parsed and validated ONNX model
    """
    if not is_file_valid(filename):
        print("Model format not matched")
        raise ValueError(
            "The model is not in onnx format. Please convert it into onnx format"
        )
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def count_parameters(onnx_model: onnx.ModelProto) -> int:
    """Count the total number of parameters (weights) in the model.

    Parameters are the values stored in the graph's initializers.

    Args:
      onnx_model : The loaded ONNX model

    Output:
      int : Total number of scalar parameters
    """
    total = 0
    # In onnx, training weights are ususally stored in graph.initalizer
    for initializer in onnx_model.graph.initializer:
        #TODO: infer input dtype from inp.type instead of always using float32.
        total += int(np.prod(initializer.dims, dtype=np.int64))
    return total


def model_size_bytes(filename: str) -> int:
    """Get the on-disk size of the model file in bytes.

    Args:
      filename : Path to the .onnx file

    Output:
      int : File size in bytes
    """
    return os.path.getsize(filename)


def model_size_mb(filename: str) -> float:
    """Get the on-disk size of the model file in megabytes.

    Args:
      filename : Path to the .onnx file

    Output:
      float : File size in megabytes
    """
    return model_size_bytes(filename) / (1024 * 1024)


def list_operator_types(onnx_model: onnx.ModelProto) -> dict:
    """List the distinct operator types and how often each occurs.

    Args:
      onnx_model : The loaded ONNX model

    Output:
      dict : Mapping of operator type -> occurrence count
    """
    ops = Counter(node.op_type for node in onnx_model.graph.node)
    return dict(sorted(ops.items(), key=lambda item: item[1], reverse=True))


def model_report(filename: str) -> str:
    """Build a JSON report for the given ONNX model.

    Args:
      filename : Path to the .onnx file

    Output:
      str : JSON string with model stats and operator type counts
    """
    onnx_model = load_model(filename)

    report = {
        "filename": filename,
        "model_size_mb": model_size_mb(filename),
        "ir_version": onnx_model.ir_version,
        "opsets": [
            {"domain": opset.domain, "version": opset.version}
            for opset in onnx_model.opset_import
        ],
        "total_parameters": count_parameters(onnx_model),
        "num_nodes": len(onnx_model.graph.node),
        "operator_types": list_operator_types(onnx_model),
    }

    return json.dumps(report, indent=2)
