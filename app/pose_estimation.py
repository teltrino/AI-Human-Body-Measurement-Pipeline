import sys
import os
import cv2
import json
import numpy as np
import time
import traceback

# --- Configuration Section ---
# Paths are now relative to the project root or set via environment variables
OPENPOSE_ROOT = os.environ.get("OPENPOSE_ROOT", "./models/pyopenpose")
PYOPENPOSE_MODULE_PATH = os.path.join(OPENPOSE_ROOT, "build/python/openpose")
CPP_LIBRARY_PATH = os.path.join(OPENPOSE_ROOT, "build/lib")

DEFAULT_PARAMS = {
    "model_folder": os.path.join(OPENPOSE_ROOT, "models/"),
    "hand": True,
    "face": True,
    "net_resolution": "-1x256",
    "body": 1, 
    "number_people_max": 1, 
    "disable_blending": True,
    "write_json": None,
    "write_images": None,
    "logging_level": 3,
}

_openpose_initialized = False

def setup_openpose_paths():
    """Configures sys.path and LD_LIBRARY_PATH for PyOpenPose."""
    global _openpose_initialized
    if _openpose_initialized:
        return True

    if not os.path.isdir(OPENPOSE_ROOT):
        return False

    if not os.path.isdir(PYOPENPOSE_MODULE_PATH):
        return False

    if PYOPENPOSE_MODULE_PATH not in sys.path:
        sys.path.append(PYOPENPOSE_MODULE_PATH)

    if os.path.isdir(CPP_LIBRARY_PATH):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if CPP_LIBRARY_PATH not in current_ld_path.split(':'):
             os.environ["LD_LIBRARY_PATH"] = f"{CPP_LIBRARY_PATH}:{current_ld_path}"

    _openpose_initialized = True
    return True

class OpenPoseWrapper:
    """
    Manages OpenPose initialization and inference.
    """
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params
        self.opWrapper = None
        self.op = None 

        if not setup_openpose_paths():
             raise RuntimeError("OpenPose path setup failed. Check OPENPOSE_ROOT.")

        try:
            import pyopenpose as op_module 
            self.op = op_module 
        except ImportError as e:
            raise RuntimeError("PyOpenPose import failed. Check build and paths.") from e

        model_folder = self.params.get("model_folder")
        if not model_folder or not os.path.isdir(model_folder):
            raise FileNotFoundError(f"OpenPose model folder not found: {model_folder}")

        try:
            self.opWrapper = self.op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()
        except Exception as e:
            self.opWrapper = None 
            raise RuntimeError("OpenPose WrapperPython failed to start.") from e

    def run_inference(self, image_bgr: np.ndarray) -> dict:
        """
        Runs OpenPose inference on a single image and returns keypoints.
        """
        if self.opWrapper is None or self.op is None:
            return {"people": []}

        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("Input image_bgr must be a non-empty NumPy array.")

        try:
            datum = self.op.Datum() 
            datum.cvInputData = image_bgr
            self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))

            people_list = []
            num_people = len(datum.poseKeypoints) if datum.poseKeypoints is not None else 0

            if num_people > 0:
                pose_kp = datum.poseKeypoints[0].flatten().tolist() if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0 else []
                face_kp = datum.faceKeypoints[0].flatten().tolist() if datum.faceKeypoints is not None and len(datum.faceKeypoints) > 0 else []

                hand_left_kp = []
                hand_right_kp = []
                
                if datum.handKeypoints is not None and len(datum.handKeypoints) == 2:
                     if datum.handKeypoints[0] is not None and len(datum.handKeypoints[0]) > 0:
                          hand_left_kp = datum.handKeypoints[0][0].flatten().tolist() 
                     if datum.handKeypoints[1] is not None and len(datum.handKeypoints[1]) > 0:
                          hand_right_kp = datum.handKeypoints[1][0].flatten().tolist() 

                person_data = {
                    "pose_keypoints_2d": pose_kp,
                    "face_keypoints_2d": face_kp,
                    "hand_left_keypoints_2d": hand_left_kp,
                    "hand_right_keypoints_2d": hand_right_kp,
                }
                people_list.append(person_data)

            output_data = {
                "version": 1.3,
                "people": people_list
            }
            return output_data

        except Exception as e:
            traceback.print_exc()
            return {"people": []}

openpose_instance = None
try:
    openpose_instance = OpenPoseWrapper(params=DEFAULT_PARAMS)
except Exception:
     pass

if __name__ == "__main__":
    # Test block logic preserved but paths generic
    if openpose_instance:
        print("OpenPose instance initialized successfully.")