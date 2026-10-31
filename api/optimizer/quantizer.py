from pathlib import Path
from typing import Optional, Union
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
import numpy as np

from api.optimizer.validation import (
    load_calibration_inputs,
)

# ---------------------------------------------------------------------------
# TVM hardware target map (dict form; CLI strings were removed in TVM 0.25+)
# Enhanced with CPU architecture vector flags (AVX2/FMA for x86, NEON for ARM)
# ---------------------------------------------------------------------------
TVM_TARGETS: dict = {
    "rpi4": {"kind": "llvm", "mcpu": "cortex-a72", "mattr": ["+neon", "+v8a"]},
    "rpi5": {"kind": "llvm", "mcpu": "cortex-a76", "mattr": ["+neon", "+v8.2a", "+dotprod"]},
    "x86_64": {"kind": "llvm", "mcpu": "core-avx2", "mattr": ["+avx2", "+fma"]},
}


# A .npy/.npz path, or calibration images already loaded (e.g. the
# calibration share of a split validation file).
CalibrationData = Union[str, np.ndarray]


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

    def create_static_int8(
        self,
        model_path: str,
        calibration_data: CalibrationData,
    ) -> str:
        """Static INT8: both weights and activations quantized using calibration data."""
        out_path = self.output_dir / "model_static_int8.onnx"

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

        # Infer dynamic shapes and symbolic dimensions so quantize_static can
        # place QDQ operators without failing on unknown shapes.
        preprocessed_path = self.output_dir / "_preprocessed_for_quant.onnx"
        try:
            quant_pre_process(
                input_model_path=model_path,
                output_model_path=str(preprocessed_path),
                skip_symbolic_shape=False,
            )
            quant_source = str(preprocessed_path)
        except Exception:
            # Fall back to raw model if preprocess fails on an uncommon op.
            quant_source = model_path

        quantize_static(
            model_input=quant_source,
            model_output=str(out_path),
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            # One scale per output channel instead of one per layer, so
            # channels with small weights keep their precision. MobileNetV2
            # top-1 on held-out images: 51.4% -> 57.5% (FP32: 58.1%).
            # ONNX Runtime advises reduce_range=True with per-channel only on
            # x86 CPUs without VNNI; not needed on VNNI x86 or ARM.
            per_channel=True,
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

        Automatically applies pre-tuned schedule databases if available in
        profiles/tvm_tuning/{hardware_id}.

        Args:
            model_path     : Path to the source .onnx file.
            hardware_id    : Edge target ('rpi4', 'rpi5', 'x86_64').
            opt_level      : Kept for API compatibility; Relax uses its
                default optimization pipeline.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_opt{opt_level}.tar'.
        """
        from api.optimizer.tvm_runtime import compile_onnx
        from api.optimizer.tvm_tuning import get_default_tuning_dir

        tuning_dir = get_default_tuning_dir(hardware_id)
        tuning_path = (
            str(tuning_dir)
            if (tuning_dir / "database_tuning_record.json").exists()
            else None
        )

        out_path = self.output_dir / f"model_tvm_opt{opt_level}.tar"
        return compile_onnx(
            out_path,
            model_path,
            _tvm_target(hardware_id),
            tune=False,
            tuning_database_path=tuning_path,
        )

    def create_tvm_tuned(
        self,
        model_path: str,
        hardware_id: str = "x86_64",
        max_trials: int = 16,
        work_dir: Optional[str] = None,
    ) -> str:
        """Compile an ONNX model with TVM Relax and MetaSchedule autotuning.

        Args:
            model_path  : Path to the source .onnx file.
            hardware_id : Target edge platform ('rpi4', 'rpi5', 'x86_64').
            max_trials  : Search budget for MetaSchedule trials.
            work_dir    : Directory for storing tuning records.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_tuned.tar'.
        """
        from api.optimizer.tvm_runtime import compile_onnx

        out_path = self.output_dir / "model_tvm_tuned.tar"
        return compile_onnx(
            out_path,
            model_path,
            _tvm_target(hardware_id),
            tune=True,
            tuning_database_path=work_dir,
            max_tuning_trials=max_trials,
        )

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

        elif candidate_name == "tvm_tuned":
            return self.create_tvm_tuned(
                model_path,
                hardware_id=hardware_id,
            )

        elif candidate_name == "tvm_int8_quantized":
            return self.create_tvm_int8(
                model_path,
                hardware_id=hardware_id,
                calibration_data=calibration_data,
            )

        else:
            raise ValueError(f"Unknown candidate: {candidate_name}")
