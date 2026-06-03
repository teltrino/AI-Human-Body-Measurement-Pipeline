import sys
import os
import pickle
import torch
import json
import argparse
import numpy as np
import traceback
import time

# --- Dependency Path Setup ---
# Paths are now relative to the project root or set via environment variables
smpl_anth_path = os.environ.get("SMPL_ANTHROPOMETRY_PATH", "./models/SMPL-Anthropometry")

if smpl_anth_path and os.path.isdir(smpl_anth_path):
    if smpl_anth_path not in sys.path:
        sys.path.append(smpl_anth_path)

# --- Import Core Measurement Library ---
try:
    from measure import MeasureBody
    from measurement_definitions import STANDARD_LABELS
    _measure_body_available = True
except ImportError:
    _measure_body_available = False
    class MeasureBody: pass
    STANDARD_LABELS = {}
except Exception:
    traceback.print_exc()
    _measure_body_available = False
    class MeasureBody: pass
    STANDARD_LABELS = {}

# Handle potential numpy deprecation
try:
    _ = np.infty
except AttributeError:
    np.infty = np.inf

class BodyMeasurementCalculator:
    """
    A wrapper class to calculate 3D body measurements using SMPL-Anthropometry.
    """
    def __init__(self, model_type: str = "smplx"):
        self.model_type = model_type
        self.measurer = None
        if not _measure_body_available:
             return 

        try:
            self.measurer = MeasureBody(self.model_type)
        except Exception:
            traceback.print_exc()
            self.measurer = None 

    def calculate_measurements(self, betas: np.ndarray, gender: str) -> dict:
        """
        Calculates 3D body measurements for the given shape parameters and gender.
        """
        if self.measurer is None:
            raise RuntimeError("BodyMeasurementCalculator not initialized.")

        if not isinstance(betas, np.ndarray):
             if isinstance(betas, torch.Tensor):
                 betas = betas.detach().cpu().numpy()
             else:
                 raise TypeError(f"Input 'betas' must be a NumPy array or PyTorch Tensor, got {type(betas)}")

        if betas.ndim == 2 and betas.shape[0] == 1:
            betas_tensor = torch.tensor(betas, dtype=torch.float32)
        elif betas.ndim == 1:
            betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Invalid shape for betas: {betas.shape}.")

        if not isinstance(gender, str) or gender.lower() not in ['male', 'female', 'neutral']:
            raise ValueError(f"Invalid gender: '{gender}'.")

        try:
            self.measurer.from_body_model(gender=gender, shape=betas_tensor)
            measurement_keys = list(STANDARD_LABELS.keys())
            self.measurer.measure(measurement_keys)
            self.measurer.label_measurements(STANDARD_LABELS)
            
            measurements = {}
            for label, value in self.measurer.labeled_measurements.items():
                try:
                    measurements[label] = float(value)
                except (ValueError, TypeError):
                    measurements[label] = None
            return measurements
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Measurement calculation failed: {e}") from e

# --- Global Instance Creation ---
body_measurement_calculator = None
if _measure_body_available:
    try:
        body_measurement_calculator = BodyMeasurementCalculator()
    except Exception:
         pass

def save_measurements_as_json(measurements, output_json_path):
    """Helper to save measurements to JSON for testing."""
    try:
        output_dir = os.path.dirname(output_json_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, "w") as json_file:
            json.dump(measurements, json_file, indent=4)
    except Exception:
        pass

if __name__ == "__main__":
    if body_measurement_calculator:
        print("BodyMeasurementCalculator ready.")