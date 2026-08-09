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

    def quantize_for_candidate(
        self,
        model_path: str,
        candidate_name: str,
        calibration_data_path: Optional[str] = None,
    ) -> str:
        """Router: creates the right model file for the candidate strategy."""
        if candidate_name == "onnx_fp32_baseline":
            return self.create_fp32(model_path)

        elif candidate_name == "onnx_dynamic_int8":
            return self.create_dynamic_int8(model_path)

        elif candidate_name == "onnx_static_int8":
            if calibration_data_path is None:
                raise ValueError("Static INT8 requires calibration data")
            return self.create_static_int8(model_path, calibration_data_path)

        else:
            raise ValueError(f"Unknown candidate: {candidate_name}")