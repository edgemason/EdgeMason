from pathlib import Path
from typing import Optional, Union
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)
from onnxruntime.quantization.registry import QLinearOpsRegistry
from onnxruntime.quantization.shape_inference import quant_pre_process
import numpy as np

from api.optimizer.validation import (
    load_calibration_inputs,
)

# ---------------------------------------------------------------------------
# TVM hardware target map (dict form; CLI strings were removed in TVM 0.25+)
# ---------------------------------------------------------------------------
TVM_TARGETS: dict = {
    "rpi4": {"kind": "llvm", "mcpu": "cortex-a72"},
    "rpi5": {"kind": "llvm", "mcpu": "cortex-a76"},
    "x86_64": {"kind": "llvm", "mcpu": "core-avx2"},
}


# A .npy/.npz path, or calibration images already loaded (e.g. the
# calibration share of a split validation file).
CalibrationData = Union[str, np.ndarray]


def _tvm_target(hardware_id: str) -> dict:
    """Return a TVM target dict for the given hardware_id."""
    return TVM_TARGETS.get(hardware_id, {"kind": "llvm"})


class _NumpyDataReader(CalibrationDataReader):
    """Feed one calibration sample per call, in the caller's order."""

    def __init__(self, data: np.ndarray, input_name: str) -> None:
        self.data = data
        self.input_name = input_name
        self.idx = 0

    def get_next(self) -> dict[str, np.ndarray]:
        if self.idx >= len(self.data):
            return {}
        # ONNX Runtime expects a dict: {input_name: numpy_array}
        item = {
            self.input_name: np.asarray(
                self.data[self.idx : self.idx + 1],
                dtype=np.float32,
            )
        }
        self.idx += 1
        return item


class Quantizer:
    def __init__(self, output_dir: str = "./temp_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_fp32(self, model_path: str) -> str:
        """FP32 is just the original model, copied to temp."""
        out_path = self.output_dir / "model_fp32.onnx"
        onnx.save(onnx.load(model_path), out_path)
        return str(out_path)

    def create_dynamic_int8(self, model_path: str) -> str:
        """Dynamic INT8: weights quantized, activations quantized at runtime."""
        out_path = self.output_dir / "model_dynamic_int8.onnx"
        quantize_dynamic(
            model_input=model_path,
            model_output=str(out_path),
            weight_type=QuantType.QUInt8,
        )
        return str(out_path)

    def create_static_int8(self, model_path: str, calibration_data: CalibrationData) -> str:
        """Static INT8: both weights and activations quantized using calibration data."""
        out_path = self.output_dir / "model_static_int8.onnx"
        reader = self._calibration_reader(model_path, calibration_data)

        # Pre-processing (fuse Conv+BatchNorm, infer shapes) is recommended by
        # ONNX Runtime but optional, so quantize the original if it fails.
        source_path = self._preprocess(model_path, "model_static_int8_prepped.onnx")
        self._quantize_static(source_path, str(out_path), reader)
        return str(out_path)

    def excludable_node_names(self, model_path: str) -> list[str]:
        """Node names `create_mixed_int8` accepts in its fp32 list.

        `quantize_static` sees the graph *after* `quant_pre_process`, which
        renames and inserts nodes, so only these names are valid exclusions.
        A node is excludable when ONNX Runtime has a QLinear implementation for
        its op type; nodes without a name cannot be addressed.
        """
        source_path = self._preprocess(model_path, "model_mixed_int8_prepped.onnx")
        graph = onnx.load(source_path).graph
        return [
            node.name
            for node in graph.node
            if node.name and node.op_type in QLinearOpsRegistry
        ]

    def create_mixed_int8(
        self,
        model_path: str,
        calibration_data: CalibrationData,
        fp32_node_names: list[str],
    ) -> str:
        """Quantize all but `fp32_node_names` to INT8.

        Names must be the *pre-processed* ONNX node names returned by
        `excludable_node_names`, not the raw model's names. Any name that is
        missing, unnamed, or for an op with no QLinear implementation raises
        instead of being silently ignored: a layer the caller believes is fp32
        would otherwise be quantized, corrupting every downstream result.
        """
        out_path = self.output_dir / "model_mixed_int8.onnx"
        source_path = self._preprocess(model_path, "model_mixed_int8_prepped.onnx")
        self._validate_fp32_node_names(source_path, fp32_node_names)

        reader = self._calibration_reader(model_path, calibration_data)
        self._quantize_static(
            source_path,
            str(out_path),
            reader,
            nodes_to_exclude=fp32_node_names,
        )
        return str(out_path)

    def _preprocess(self, model_path: str, prepped_name: str) -> str:
        """Run `quant_pre_process` into the output dir, falling back to the original."""
        prepped_path = self.output_dir / prepped_name
        try:
            quant_pre_process(model_path, str(prepped_path), skip_symbolic_shape=True)
            return str(prepped_path)
        except Exception:
            return model_path

    def _calibration_reader(
        self,
        model_path: str,
        calibration_data: CalibrationData,
    ) -> CalibrationDataReader:
        """Validate calibration data and wrap it in a reader for the model input."""
        calib_data = (
            calibration_data
            if isinstance(calibration_data, np.ndarray)
            else load_calibration_inputs(calibration_data)
        )

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        model_inputs = session.get_inputs()
        if len(model_inputs) != 1:
            raise ValueError("static quantization supports single-input models only")
        input_name = model_inputs[0].name
        expected_shape = model_inputs[0].shape[1:]
        if calib_data.ndim - 1 != len(expected_shape):
            raise ValueError("calibration data rank does not match model input")
        for actual, expected in zip(calib_data.shape[1:], expected_shape):
            if isinstance(expected, int) and expected > 0 and actual != expected:
                raise ValueError("calibration data shape does not match model input")

        return _NumpyDataReader(calib_data, input_name)

    def _validate_fp32_node_names(
        self,
        source_path: str,
        fp32_node_names: list[str],
    ) -> None:
        """Reject names that `quantize_static` would silently ignore."""
        graph = onnx.load(source_path).graph
        named_ops = {node.name: node.op_type for node in graph.node if node.name}
        for name in fp32_node_names:
            if name not in named_ops:
                raise ValueError(
                    f"fp32 node {name!r} is not a named node in the pre-processed "
                    "graph; use Quantizer.excludable_node_names() for valid names"
                )
            if named_ops[name] not in QLinearOpsRegistry:
                raise ValueError(
                    f"fp32 node {name!r} has op type {named_ops[name]!r}, which has "
                    "no QLinear implementation and is never quantized; "
                    "remove it from the fp32 list"
                )

    def _quantize_static(
        self,
        model_input: str,
        model_output: str,
        reader: CalibrationDataReader,
        nodes_to_exclude: Optional[list[str]] = None,
    ) -> None:
        """Quantize with the P2.4 INT8 config, optionally keeping nodes fp32."""
        quantize_static(
            model_input=model_input,
            model_output=model_output,
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            # ORT's CPU (MLAS) QLinearConv expects uint8 activations. With
            # QInt8 activations the QDQ graph only fuses two of squeezenet's
            # 26 convolutions to integer kernels; the rest stay fp32 Conv
            # wrapped in Q/DQ (B-62). QUInt8 fuses all 26.
            activation_type=QuantType.QUInt8,
            # Emit QLinearConv directly instead of relying on ORT's QDQ
            # fusion pass, so the integer kernel is still present when graph
            # optimisations are disabled (e.g. profiling traces).
            quant_format=QuantFormat.QOperator,
            # One scale per output channel instead of one per layer, so
            # channels with small weights keep their precision. MobileNetV2
            # top-1 on held-out images: 51.4% -> 57.5% (FP32: 58.1%).
            # reduce_range is only advised with per-channel on x86 CPUs
            # without VNNI; it costs accuracy, so leave it off on VNNI x86
            # and ARM.
            per_channel=True,
            reduce_range=False,
            nodes_to_exclude=nodes_to_exclude,
        )

    # -----------------------------------------------------------------------
    # TVM methods
    # -----------------------------------------------------------------------

    def create_tvm_compiled(
        self,
        model_path: str,
        hardware_id: str = "x86_64",
        opt_level: int = 3,
    ) -> str:
        """Compile an ONNX model with TVM Relax and save as a .tar module.

        Args:
            model_path     : Path to the source .onnx file.
            hardware_id    : Edge target ('rpi4', 'rpi5', 'x86_64').
            opt_level      : Kept for API compatibility; Relax uses its
                default optimization pipeline.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_opt{opt_level}.tar'.
        """
        from api.optimizer.tvm_runtime import compile_onnx

        out_path = self.output_dir / f"model_tvm_opt{opt_level}.tar"
        return compile_onnx(out_path, model_path, _tvm_target(hardware_id))

    def create_tvm_int8(
        self,
        model_path: str,
        hardware_id: str = "x86_64",
        calibration_data: Optional[CalibrationData] = None,
    ) -> str:
        """Compile an INT8-quantized ONNX model with TVM Relax.

        The model is first statically quantized with ONNX Runtime using the
        calibration data, then compiled to a Relax VM .tar module.

        Args:
            model_path             : Path to the source .onnx file.
            hardware_id            : Edge target ('rpi4', 'rpi5', 'x86_64').
            calibration_data       : .npy/.npz path or calibration images.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_int8.tar'.
        """
        from api.optimizer.tvm_runtime import compile_onnx

        if calibration_data is None:
            raise ValueError("TVM INT8 requires calibration data")

        out_path = self.output_dir / "model_tvm_int8.tar"

        # 1. Static INT8 quantization via ONNX Runtime (reuses calibration).
        static_path = self.create_static_int8(model_path, calibration_data)
        quantized_onnx = self.output_dir / "_tvm_int8_src.onnx"
        quantized_onnx.write_bytes(Path(static_path).read_bytes())

        # 2. Compile the quantized model with Relax.
        return compile_onnx(out_path, str(quantized_onnx), _tvm_target(hardware_id))

    # -----------------------------------------------------------------------
    # Candidate router
    # -----------------------------------------------------------------------

    def quantize_for_candidate(
        self,
        model_path: str,
        candidate_name: str,
        calibration_data: Optional[CalibrationData] = None,
        hardware_id: str = "x86_64",
    ) -> str:
        """Router: creates the right model file for the candidate strategy."""
        # --- ONNX Runtime candidates ---
        if candidate_name == "onnx_fp32_baseline":
            return self.create_fp32(model_path)

        elif candidate_name == "onnx_dynamic_int8":
            return self.create_dynamic_int8(model_path)

        elif candidate_name == "onnx_static_int8":
            if calibration_data is None:
                raise ValueError("Static INT8 requires calibration data")
            return self.create_static_int8(model_path, calibration_data)

        # --- TVM candidates ---
        elif candidate_name == "tvm_opt_level_3":
            return self.create_tvm_compiled(model_path, hardware_id=hardware_id, opt_level=3)

        elif candidate_name == "tvm_int8_quantized":
            return self.create_tvm_int8(
                model_path,
                hardware_id=hardware_id,
                calibration_data=calibration_data,
            )

        else:
            raise ValueError(f"Unknown candidate: {candidate_name}")
