from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
import os
import json
import uuid
import shutil
import torch
import cv2
import numpy as np
import traceback
import time
import tempfile
from pathlib import Path
import sys

# --- Import Renamed Modules ---
try:
    from body_analysis import DensePosePredictorWrapper
    from body_measurements import BodyMeasurementCalculator
    from body_reconstruction import (
        fit_smplifyx, load_smplx_model, load_vposer_model,
        load_smplify_config, smplifyx_parent_dir, DEFAULT_SMPLIFY_CONFIG_PATH
    )
    _imports_ok = True
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    traceback.print_exc()
    _imports_ok = False

# --- Global Variables & Configuration ---
app = FastAPI(title="Body Measurement API")

# Use environment variables with generic relative path fallbacks
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", "./"))
OUTPUTS_FOLDER = BASE_DIR / "outputs"
SMPL_MODELS_PATH = BASE_DIR / "models/smplify-x/models/"
VPOSER_DEFAULT_PATH = BASE_DIR / "models/smplify-x/vposer"

os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# --- OpenPose Global Setup ---
op = None
opWrapper = None
_openpose_setup_done = False

def setup_and_init_openpose_globally():
    """ Sets up paths and initializes OpenPose globally once. """
    global op, opWrapper, _openpose_setup_done
    if _openpose_setup_done: return True
    
    try:
        OPENPOSE_ROOT = os.environ.get("OPENPOSE_ROOT", "./models/pyopenpose")
        PYOPENPOSE_MODULE_PATH = os.path.join(OPENPOSE_ROOT, "build/python/openpose")
        CPP_LIBRARY_PATH = os.path.join(OPENPOSE_ROOT, "build/lib")

        if not os.path.isdir(OPENPOSE_ROOT): return False
        
        if PYOPENPOSE_MODULE_PATH not in sys.path:
            sys.path.append(PYOPENPOSE_MODULE_PATH)

        if os.path.isdir(CPP_LIBRARY_PATH):
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if CPP_LIBRARY_PATH not in current_ld_path.split(':'):
                os.environ["LD_LIBRARY_PATH"] = f"{CPP_LIBRARY_PATH}:{current_ld_path}"

        import pyopenpose as op_module
        op = op_module

        params = {
            "model_folder": os.path.join(OPENPOSE_ROOT, "models/"),
            "hand": True, "face": True, "net_resolution": "-1x256",
            "body": 1, "number_people_max": 1, "disable_blending": True,
            "write_json": None, "write_images": None, "logging_level": 3,
        }
        
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        _openpose_setup_done = True
        return True
    except Exception as e:
        traceback.print_exc()
        opWrapper = None
        return False

# --- Global Instances ---
smplx_config: dict = None
device: torch.device = None
densepose_predictor_instance: DensePosePredictorWrapper = None
vposer_model: torch.nn.Module = None
smplx_models: dict = {}
body_measurement_calculator: BodyMeasurementCalculator = None

@app.on_event("startup")
async def load_all_models_on_startup():
    """Load ALL models and components at startup."""
    global opWrapper, smplx_config, device, _imports_ok
    global densepose_predictor_instance, vposer_model, smplx_models, body_measurement_calculator

    if not _imports_ok: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup OpenPose
    setup_and_init_openpose_globally()

    # 2. Load SMPLify-X Config
    try:
        smplx_config = load_smplify_config(DEFAULT_SMPLIFY_CONFIG_PATH)
        smplx_config['use_cuda'] = (device.type == 'cuda')
    except Exception:
        smplx_config = None

    # 3. Load DensePose
    try:
        densepose_predictor_instance = DensePosePredictorWrapper()
    except Exception:
        densepose_predictor_instance = None

    # 4. Load VPoser
    if smplx_config and smplx_config.get('use_vposer', True):
        try:
            vposer_path_str = smplx_config.get('vposer_ckpt')
            vposer_path = Path(vposer_path_str) if vposer_path_str else VPOSER_DEFAULT_PATH
            if not vposer_path.is_absolute(): vposer_path = Path(smplifyx_parent_dir) / vposer_path
            vposer_model = load_vposer_model(str(vposer_path), device)
        except Exception:
            vposer_model = None

    # 5. Load SMPL-X Models
    if smplx_config:
        try:
            for gender in ['male', 'female']:
                 smplx_models[gender] = load_smplx_model(str(SMPL_MODELS_PATH), gender, device, smplx_config)
        except Exception:
            traceback.print_exc()

    # 6. Load Measurement Calculator
    try:
        body_measurement_calculator = BodyMeasurementCalculator()
    except Exception:
        body_measurement_calculator = None

async def decode_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode image '{file.filename}'.")
    return img_bgr

def save_json(data: dict, file_path: Path):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f: json.dump(data, f, indent=4)
    except Exception:
        traceback.print_exc()

@app.post("/process-all/")
async def process_all_measurements(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    height: float = Form(...),
    gender: str = Form(...),
    focal_length: float = Form(5000.0)
):
    global opWrapper, op, densepose_predictor_instance, vposer_model, smplx_models
    global body_measurement_calculator, smplx_config, device

    request_id = str(uuid.uuid4())
    gender_lower = gender.lower()
    
    if gender_lower not in ['male', 'female', 'neutral']: raise HTTPException(status_code=400, detail="Invalid gender.")
    if height <= 0: raise HTTPException(status_code=400, detail="Height must be positive.")
    if not _openpose_setup_done or opWrapper is None: raise HTTPException(status_code=503, detail="OpenPose not ready.")
    if densepose_predictor_instance is None: raise HTTPException(status_code=503, detail="DensePose not ready.")
    if body_measurement_calculator is None: raise HTTPException(status_code=503, detail="Measurement calculator not ready.")

    persistent_obj_path = OUTPUTS_FOLDER / f"{request_id}_mesh.obj"
    persistent_json_path = OUTPUTS_FOLDER / f"{request_id}_measurements.json"

    try:
        # 1. Decode Image
        image_bgr = await decode_image(image)
        img_h, img_w = image_bgr.shape[:2]

        # 2. Run OpenPose
        datum = op.Datum()
        datum.cvInputData = image_bgr
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        op_people_list = []
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            pose_kp = datum.poseKeypoints[0].flatten().tolist()
            face_kp = datum.faceKeypoints[0].flatten().tolist() if datum.faceKeypoints is not None else []
            hand_left_kp, hand_right_kp = [], []
            if datum.handKeypoints is not None and len(datum.handKeypoints) == 2:
                 if datum.handKeypoints[0] is not None and len(datum.handKeypoints[0]) > 0: hand_left_kp = datum.handKeypoints[0][0].flatten().tolist()
                 if datum.handKeypoints[1] is not None and len(datum.handKeypoints[1]) > 0: hand_right_kp = datum.handKeypoints[1][0].flatten().tolist()
            op_people_list.append({"pose_keypoints_2d": pose_kp, "face_keypoints_2d": face_kp, "hand_left_keypoints_2d": hand_left_kp, "hand_right_keypoints_2d": hand_right_kp})
        
        op_results = {"version": 1.3, "people": op_people_list}
        if not op_results.get("people"): raise ValueError("No person detected.")

        # 3. Run DensePose
        densepose_results = densepose_predictor_instance.predict_measurements(image_bgr, height)

        # 4. Run SMPLify-X
        body_model = smplx_models.get(gender_lower, smplx_models.get('male'))
        smplify_results = fit_smplifyx(
            image_height=img_h, image_width=img_w, keypoints_dict=op_results,
            body_model=body_model, vposer_model=vposer_model,
            gender=gender_lower, config=smplx_config, focal_length=focal_length,
            output_obj_path=str(persistent_obj_path),
            device=device
        )

        # 5. Calculate 3D Measurements
        measurements_3d = body_measurement_calculator.calculate_measurements(smplify_results["betas"], smplify_results["gender"])

        # 6. Finalize
        final_results = {
            "request_id": request_id, "status": "Success",
            "measurements_3d_cm": measurements_3d, "measurements_2d_cm": densepose_results,
            "mesh_file": persistent_obj_path.name
        }
        save_json(final_results, persistent_json_path)
        return final_results

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Body Measurement API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)