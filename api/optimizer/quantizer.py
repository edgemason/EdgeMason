from pathlib import Path
from typing import Optional
import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
)
import numpy as np

# ---------------------------------------------------------------------------
# TVM hardware target map
# ---------------------------------------------------------------------------
TVM_TARGETS: dict = {
    "rpi4":   "llvm -mcpu=cortex-a72",
    "rpi5":   "llvm -mcpu=cortex-a76",
    "x86_64": "llvm -mcpu=core-avx2",
}

def _tvm_target(hardware_id: str) -> str:
    """Return a TVM target string for the given hardware_id."""
    return TVM_TARGETS.get(hardware_id, "llvm")


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
            weight_type=QuantType.QInt8,
        )
        return str(out_path)

    def create_static_int8(self, model_path: str, calibration_data_path: str) -> str:
        """Static INT8: both weights and activations quantized using calibration data."""
        out_path = self.output_dir / "model_static_int8.onnx"

        # Build a CalibrationDataReader from the .npy file
        # Expects shape: (N, C, H, W) or (N, seq_len) saved as npy
        calib_data = np.load(calibration_data_path)

        # Use the model's real input names instead of a hardcoded "input"
        input_names = [inp.name for inp in onnx.load(model_path).graph.input]

        class NumpyDataReader(CalibrationDataReader):
            def __init__(self, data, input_names):
                self.data = data
                self.input_names = input_names
                self.idx = 0

            def get_next(self):
                if self.idx >= len(self.data):
                    return {}
                # ONNX Runtime expects a dict: {input_name: numpy_array}
                item = {
                    name: self.data[self.idx : self.idx + 1].astype(np.float32)
                    for name in self.input_names
                }
                self.idx += 1
                return item

        reader = NumpyDataReader(calib_data, input_names)

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
        """Compile an ONNX model with TVM Relay and save as a .tar module.

        Args:
            model_path     : Path to the source .onnx file.
            hardware_id    : Edge target ('rpi4', 'rpi5', 'x86_64').
            opt_level      : TVM optimization level (0-4). Default 3.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_opt{opt_level}.tar'.
        """
        try:
            import tvm
            from tvm import relay
            from tvm.contrib import cc  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Apache TVM is not installed. Run: pip install apache-tvm"
            ) from exc

        out_path = self.output_dir / f"model_tvm_opt{opt_level}.tar"

        # 1. Load ONNX model and infer shapes
        onnx_model = onnx.load(model_path)
        # Build a shape dict from the first graph input; fall back to batch=1
        shape_dict = {}
        for inp in onnx_model.graph.input:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            shape_dict[inp.name] = dims

        # 2. Import into Relay IR
        mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)

        # 3. Compile
        target = tvm.target.Target(_tvm_target(hardware_id))
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)

        # 4. Export as a self-contained .tar module
        lib.export_library(str(out_path))
        return str(out_path)

    def create_tvm_int8(
        self,
        model_path: str,
        hardware_id: str = "x86_64",
        calibration_data_path: Optional[str] = None,
    ) -> str:
        """Quantize an ONNX model to INT8 with TVM Relay and save as a .tar module.

        Args:
            model_path             : Path to the source .onnx file.
            hardware_id            : Edge target ('rpi4', 'rpi5', 'x86_64').
            calibration_data_path  : Optional path to a .npy calibration dataset.

        Returns:
            str : Path to the exported '<output_dir>/model_tvm_int8.tar'.
        """
        try:
            import tvm
            from tvm import relay
            from tvm.relay import quantize as qtz
        except ImportError as exc:
            raise ImportError(
                "Apache TVM is not installed. Run: pip install apache-tvm"
            ) from exc

        out_path = self.output_dir / "model_tvm_int8.tar"

        # 1. Load and import into Relay IR
        onnx_model = onnx.load(model_path)
        shape_dict = {}
        for inp in onnx_model.graph.input:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            shape_dict[inp.name] = dims

        mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)

        # 2. Relay quantization
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod_q = qtz.quantize(mod, params=params)

        # 3. Compile the quantized graph
        target = tvm.target.Target(_tvm_target(hardware_id))
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod_q, target=target, params=params)

        lib.export_library(str(out_path))
        return str(out_path)

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