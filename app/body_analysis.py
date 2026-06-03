import cv2
import torch
import numpy as np
import os
import time
import traceback
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Paths are now relative to the project root or set via environment variables
DEFAULT_DENSEPOSE_CONFIG = os.environ.get("DENSEPOSE_CONFIG", "./models/densepose/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
DEFAULT_DENSEPOSE_WEIGHTS = os.environ.get("DENSEPOSE_WEIGHTS", "./models/densepose/model_final_5ad38f.pkl")

class DensePosePredictorWrapper:
    """
    A wrapper class for DensePose inference using Detectron2.
    """
    def __init__(self, cfg_path: str = DEFAULT_DENSEPOSE_CONFIG, weights_path: str = DEFAULT_DENSEPOSE_WEIGHTS, score_thresh: float = 0.8):
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.score_thresh = score_thresh
        self.predictor = None
        self._load_model()

    def _load_model(self):
        """Loads the Detectron2 model based on the provided paths."""
        if not os.path.exists(self.cfg_path) or not os.path.exists(self.weights_path):
            print(f"Warning: DensePose config or weights not found at {self.cfg_path}")
            return
            
        try:
            cfg = get_cfg()
            cfg.merge_from_file(self.cfg_path)
            cfg.MODEL.WEIGHTS = self.weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.predictor = DefaultPredictor(cfg)
        except Exception as e:
            traceback.print_exc()
            self.predictor = None

    def predict_measurements(self, image_bgr: np.ndarray, user_height_cm: float) -> dict:
        """
        Predicts DensePose keypoints and extracts approximate 2D body measurements.
        """
        if self.predictor is None:
            raise RuntimeError("DensePose predictor is not available.")
            
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("Input image_bgr must be a non-empty NumPy array.")

        with torch.no_grad():
             outputs = self.predictor(image_bgr)
        
        instances = outputs["instances"]
        if len(instances) == 0:
            raise ValueError("DensePose: No instances detected in the image.")

        target_instance = instances[0]
        if not target_instance.has("pred_keypoints"):
            raise ValueError("DensePose: Detected instance has no 'pred_keypoints'.")
            
        keypoints_tensor = target_instance.pred_keypoints 
        keypoints = keypoints_tensor[0].cpu().numpy() 

        # --- Keypoint Indices (COCO Format) ---
        nose_idx, l_shoulder_idx, r_shoulder_idx = 0, 5, 6
        l_hip_idx, r_hip_idx = 11, 12
        l_ankle_idx, r_ankle_idx = 15, 16
        
        required_indices = [nose_idx, l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx, l_ankle_idx, r_ankle_idx]
        if max(required_indices) >= keypoints.shape[0]:
            raise ValueError("Incompatible model: keypoint indices out of range.")

        try:
            shoulder_left, shoulder_right = keypoints[l_shoulder_idx][:2], keypoints[r_shoulder_idx][:2]
            hip_left, hip_right = keypoints[l_hip_idx][:2], keypoints[r_hip_idx][:2]
            head_y = keypoints[nose_idx][1]
            foot_y = max(keypoints[l_ankle_idx][1], keypoints[r_ankle_idx][1]) 
            
            total_body_height_px = abs(head_y - foot_y)
        except IndexError as e:
             raise ValueError(f"Failed accessing keypoint indices: {e}")

        if total_body_height_px < 10:
            raise ValueError(f"Detected body height in pixels ({total_body_height_px:.1f}) is too small.")

        def pixel_distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        shoulder_width_px = pixel_distance(shoulder_left, shoulder_right)
        hip_width_px = pixel_distance(hip_left, hip_right)
        pixel_to_cm_ratio = user_height_cm / total_body_height_px

        measurements = {
            "shoulder_width_cm": round(float(shoulder_width_px * pixel_to_cm_ratio), 2),
            "hip_width_cm": round(float(hip_width_px * pixel_to_cm_ratio), 2)
        }
        return measurements

if __name__ == "__main__":
    # Test execution logic preserved with generic naming
    try:
        predictor = DensePosePredictorWrapper()
        print("DensePose predictor initialized.")
    except Exception as e:
        print(f"Initialization failed: {e}")