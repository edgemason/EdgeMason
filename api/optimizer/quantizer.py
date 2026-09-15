from pathlib import Path
from typing import Optional
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
)
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


def _tvm_target(hardware_id: str) -> dict:
    """Return a TVM target dict for the given hardware_id."""
    return TVM_TARGETS.get(hardware_id, {"kind": "llvm"})


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

    def create_static_int8(self, model_path: str, calibration_data_path: str) -> str:
        """Static INT8: both weights and activations quantized using calibration data."""
        out_path = self.output_dir / "model_static_int8.onnx"

        calib_data = load_calibration_inputs(calibration_data_path)

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

        class NumpyDataReader(CalibrationDataReader):
            def __init__(self, data, input_name):
                self.data = data
                self.input_name = input_name
                self.idx = 0

            def get_next(self):
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

        reader = NumpyDataReader(calib_data, input_name)

        quantize_static(
            model_input=model_path,
            model_output=str(out_path),
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        return str(out_path)

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
        calibration_data_path: Optional[str] = None,
    ) -> str:
        """Compile an INT8-quantized ONNX model with TVM Relax.

        The model is first statically quantized with ONNX Runtime using the
        calibration data, then compiled to a Relax VM .tar module.

        Args:
            model_path             : Path to the source .onnx file.
            hardware_id            : Edge target ('rpi4', 'rpi5', 'x86_64').
            calibration_data_path  : Path to a .npy/.npz calibration dataset.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_int8.tar'.
        """
        from api.optimizer.tvm_runtime import compile_onnx

        if calibration_data_path is None:
            raise ValueError("TVM INT8 requires calibration data")

        out_path = self.output_dir / "model_tvm_int8.tar"

        # 1. Static INT8 quantization via ONNX Runtime (reuses calibration).
        static_path = self.create_static_int8(model_path, calibration_data_path)
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
        calibration_data_path: Optional[str] = None,
        hardware_id: str = "x86_64",
    ) -> str:
        """Router: creates the right model file for the candidate strategy."""
        # --- ONNX Runtime candidates ---
        if candidate_name == "onnx_fp32_baseline":
            return self.create_fp32(model_path)

        elif candidate_name == "onnx_dynamic_int8":
            return self.create_dynamic_int8(model_path)

        elif candidate_name == "onnx_static_int8":
            if calibration_data_path is None:
                raise ValueError("Static INT8 requires calibration data")
            return self.create_static_int8(model_path, calibration_data_path)

        # --- TVM candidates ---
        elif candidate_name == "tvm_opt_level_3":
            return self.create_tvm_compiled(model_path, hardware_id=hardware_id, opt_level=3)

        elif candidate_name == "tvm_int8_quantized":
            return self.create_tvm_int8(
                model_path,
                hardware_id=hardware_id,
                calibration_data_path=calibration_data_path,
            )

        else:
            raise ValueError(f"Unknown candidate: {candidate_name}")
