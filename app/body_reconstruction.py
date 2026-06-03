import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import json
import yaml
import cv2
import time
import smplx
import traceback

# --- Dynamic Path Logic ---
# Paths are now relative to the project root or set via environment variables
BASE_MODELS_DIR = os.environ.get("FULL_AR_MODELS_DIR", "./models")
smplifyx_parent_dir = os.path.join(BASE_MODELS_DIR, "smplify-x")
smplx_dir = os.path.join(BASE_MODELS_DIR, "smplx")

if smplifyx_parent_dir not in sys.path:
    sys.path.insert(0, smplifyx_parent_dir)
if smplx_dir not in sys.path:
    sys.path.insert(0, smplx_dir)

# --- Imports from the SMPLify-X library ---
try:
    from smplifyx.camera import create_camera
    from smplifyx.fitting import create_loss, FittingMonitor, guess_init
    from smplifyx.prior import create_prior, SMPLifyAnglePrior
    from smplifyx.optimizers import LBFGS
    import smplifyx.utils as smplifyx_utils
    from human_body_prior.tools.model_loader import load_vposer

    try:
        from smplifyx.losses.collision import CollisionLoss
    except ImportError:
        try:
            from pytorch_collision.collision_loss import CollisionLoss
        except ImportError:
            CollisionLoss = None
except ImportError as e:
    traceback.print_exc()
    raise SystemExit(f"Cannot import required SMPLify-X modules: {e}")

# --- Configuration & Model Loading ---
DEFAULT_SMPLIFY_CONFIG_PATH = os.path.join(smplifyx_parent_dir, "cfg_files/fit_smplx.yaml")

def load_smplify_config(config_path=None):
    """Loads the SMPLify-X configuration from a YAML file."""
    if config_path is None: config_path = DEFAULT_SMPLIFY_CONFIG_PATH
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    return config

def load_smplx_model(model_folder, gender, device, config):
    """Loads the SMPL-X body model."""
    model_type = config.get('model_type', 'smplx')
    num_betas = config.get('num_betas', 10)
    num_expr = config.get('num_expression_coeffs', 10)
    ext = config.get('ext', 'npz')
    use_pca = config.get('use_pca', True)
    num_pca = config.get('num_pca_comps', 12)
    flat_hand = config.get('flat_hand_mean', False)
    use_face_cont = config.get('use_face_contour', False)
    
    try:
        model = smplx.create(model_path=model_folder, model_type=model_type, gender=gender, ext=ext,
                             num_betas=num_betas, num_expression_coeffs=num_expr, use_pca=use_pca,
                             num_pca_comps=num_pca, flat_hand_mean=flat_hand, use_face_contour=use_face_cont).to(device=device)
        return model
    except Exception as e: raise RuntimeError(f"Failed loading SMPLX model: {e}")

def load_vposer_model(vposer_ckpt_path, device):
    """Loads the VPoser model."""
    potential_path = vposer_ckpt_path
    if not os.path.exists(vposer_ckpt_path):
         potential_path = os.path.join(smplifyx_parent_dir, vposer_ckpt_path)
         if not os.path.exists(potential_path): raise FileNotFoundError(f"VPoser not found at '{vposer_ckpt_path}'")
    try:
        vposer, _ = load_vposer(potential_path, vp_model='snapshot')
        vposer = vposer.to(device=device); vposer.eval(); return vposer
    except Exception as e: raise RuntimeError(f"Failed loading VPoser: {e}")

def keypoints_openpose_dict_to_smplifyx(openpose_keypoints_dict: dict):
    """Converts OpenPose keypoints dict to SMPLify-X compatible tensor."""
    people_data = openpose_keypoints_dict.get('people', [])
    if not people_data: return None
    person = people_data[0]
    pose_kp_flat = person.get("pose_keypoints_2d")
    if not pose_kp_flat: raise ValueError("Missing 'pose_keypoints_2d' in keypoints dict")
    
    num_joints_detected = len(pose_kp_flat) // 3
    target_num_joints = 25 
    pose_kp = np.array(pose_kp_flat, dtype=np.float32).reshape(num_joints_detected, 3)

    if num_joints_detected < target_num_joints:
        padded_kp = np.zeros((target_num_joints, 3), dtype=np.float32)
        padded_kp[:num_joints_detected, :] = pose_kp
        pose_kp = padded_kp
    elif num_joints_detected > target_num_joints:
        pose_kp = pose_kp[:target_num_joints, :]

    return torch.tensor(pose_kp, dtype=torch.float32).unsqueeze(0)

def fit_smplifyx(
    image_height: int, image_width: int, keypoints_dict: dict,
    body_model: smplx.SMPLX, vposer_model: nn.Module,
    gender: str, config: dict, focal_length: float,
    output_obj_path: str,
    device: torch.device,
    ):
    """
    Performs the SMPLify-X fitting process.
    """
    process_start_time = time.time()
    use_vposer = config.get('use_vposer', True)
    if use_vposer and vposer_model is None: raise ValueError("VPoser enabled but model not provided.")

    num_stages = len(config.get('shape_weights', [0]))
    keypoints_tensor = keypoints_openpose_dict_to_smplifyx(keypoints_dict)
    if keypoints_tensor is None: raise ValueError("Could not extract keypoints from input dictionary")
    
    keypoints_tensor = keypoints_tensor.to(device)
    batch_size = keypoints_tensor.shape[0]
    joints_conf = keypoints_tensor[:, :, -1].clone() if config.get('use_joints_conf', True) else \
                  torch.ones_like(keypoints_tensor[:, :, -1])

    img_size = torch.tensor([[image_width, image_height]], dtype=torch.float32, device=device)
    focal_length_tensor = torch.tensor([[focal_length, focal_length]], dtype=torch.float32, device=device)
    camera_center = img_size / 2.0
    camera = create_camera(focal_length_x=focal_length_tensor[:, 0], focal_length_y=focal_length_tensor[:, 1],
                           center_x=camera_center[:, 0], center_y=camera_center[:, 1],
                           img_size_x=img_size[:, 0], img_size_y=img_size[:, 1],
                           batch_size=batch_size, camera_mode='perspective', device=device)

    num_body_joints = body_model.NUM_BODY_JOINTS
    num_hand_joints = getattr(body_model, 'num_hand_joints', 15)
    num_jaw_joints = 1
    num_pca_comps = config.get('num_pca_comps', 12)
    use_pca_hands = config.get('use_pca', True) and config.get('use_hands', False)

    if use_vposer:
        pose_embedding = torch.zeros([batch_size, config.get('vposer_latent_dim', 32)], dtype=torch.float32, device=device, requires_grad=True)
        pose_params = [pose_embedding]
    else:
        init_body_pose = torch.zeros([batch_size, num_body_joints * 3], dtype=torch.float32, device=device, requires_grad=True)
        pose_params = [init_body_pose]
    
    init_betas = torch.zeros([batch_size, body_model.num_betas], dtype=torch.float32, device=device, requires_grad=True)
    shape_params = [init_betas]
    init_expression = torch.zeros([batch_size, body_model.num_expression_coeffs], dtype=torch.float32, device=device, requires_grad=True)
    init_jaw_pose = torch.zeros([batch_size, num_jaw_joints * 3], dtype=torch.float32, device=device, requires_grad=True)
    
    face_params = []
    if config.get('use_face', False): face_params.extend([init_expression, init_jaw_pose])
    
    if use_pca_hands:
        init_left_hand_pose = torch.zeros([batch_size, num_pca_comps], dtype=torch.float32, device=device, requires_grad=True)
        init_right_hand_pose = torch.zeros([batch_size, num_pca_comps], dtype=torch.float32, device=device, requires_grad=True)
    else:
        init_left_hand_pose = torch.zeros([batch_size, num_hand_joints * 3], dtype=torch.float32, device=device, requires_grad=True)
        init_right_hand_pose = torch.zeros([batch_size, num_hand_joints * 3], dtype=torch.float32, device=device, requires_grad=True)
    
    hand_params = []
    if config.get('use_hands', False): hand_params.extend([init_left_hand_pose, init_right_hand_pose])
    
    camera_translation_opt = torch.tensor([[0.0, 0.0, config.get('init_cam_z', 2.5)]], dtype=torch.float32, device=device, requires_grad=True)
    cam_params = [camera_translation_opt]

    body_pose_prior = create_prior(prior_type=config.get('body_prior_type', 'l2')).to(device)
    shape_prior = create_prior(prior_type=config.get('shape_prior_type', 'l2')).to(device)
    angle_prior = SMPLifyAnglePrior(dtype=torch.float32).to(device)
    left_hand_prior, right_hand_prior, jaw_prior, expr_prior = None, None, None, None
    if config.get('use_hands', False):
        left_hand_prior = create_prior(config.get('left_hand_prior_type', 'l2')).to(device)
        right_hand_prior = create_prior(config.get('right_hand_prior_type', 'l2')).to(device)
    if config.get('use_face', False):
        expr_prior = create_prior(config.get('expr_prior_type', 'l2')).to(device)
        jaw_prior = create_prior(config.get('jaw_prior_type', 'l2')).to(device)

    pen_distance, search_tree, tri_filtering_module = None, None, None
    use_interpenetration = config.get('interpenetration', False)
    if use_interpenetration and CollisionLoss is not None:
        try:
            model_faces = body_model.faces_tensor.to(device)
            coll_loss_func = CollisionLoss(faces=model_faces, dtype=torch.float32).to(device)
            pen_distance = getattr(coll_loss_func, 'kd_tree', None)
            search_tree = getattr(coll_loss_func, 'bvh', None)
        except Exception:
            use_interpenetration = False

    loss_func = create_loss(loss_type='smplify', rho=config.get('rho', 100),
                           use_joints_conf=config.get('use_joints_conf', True), use_face=config.get('use_face', False),
                           use_hands=config.get('use_hands', False), body_pose_prior=body_pose_prior,
                           shape_prior=shape_prior, expr_prior=expr_prior, angle_prior=angle_prior,
                           jaw_prior=jaw_prior, left_hand_prior=left_hand_prior, right_hand_prior=right_hand_prior,
                           interpenetration=use_interpenetration, search_tree=search_tree,
                           pen_distance=pen_distance, tri_filtering_module=tri_filtering_module,
                           dtype=torch.float32).to(device=device)

    optimizer_type = config.get('optim_type', 'lbfgs').lower()
    lr = float(config.get('lr', 1.0))
    gtol = float(config.get('gtol', 1e-5 if optimizer_type != 'lbfgs' else 1e-9))
    ftol = float(config.get('ftol', 1e-9))
    lbfgs_max_iter = config.get('lbfgs_maxiters', 20)
    max_iters_per_stage_lbfgs = config.get('maxiters', 30)
    lbfgs_history = config.get('lbfgs_history_size', 100)

    for stage_idx in range(num_stages):
        current_stage_params = []
        current_stage_params.extend(cam_params)
        current_stage_params.extend(pose_params)
        if stage_idx >= 1: current_stage_params.extend(shape_params)
        if stage_idx >= 2:
            if config.get('use_face', False): current_stage_params.extend(face_params)
            if config.get('use_hands', False): current_stage_params.extend(hand_params)

        active_params = [p for p in current_stage_params if p is not None and p.requires_grad]
        if not active_params: continue

        stage_lr = lr * (0.1 ** stage_idx)
        if optimizer_type == 'adam':
            optimizer = optim.Adam(active_params, lr=stage_lr, betas=(0.9, 0.999))
            max_iters_this_stage = config.get('maxiters', 100)
        elif optimizer_type == 'lbfgsls':
            optimizer = LBFGS(active_params, lr=stage_lr, max_iter=lbfgs_max_iter,
                              line_search_fn='strong_Wolfe', tolerance_grad=gtol,
                              tolerance_change=ftol, history_size=lbfgs_history)
            max_iters_this_stage = max_iters_per_stage_lbfgs
        else: raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        stage_weights = {}
        def get_stage_weight(key, default=0.0):
             raw_val_list = config.get(key, [default])
             if not isinstance(raw_val_list, list): raw_val_list = [raw_val_list]
             idx = min(stage_idx, len(raw_val_list) - 1)
             val = raw_val_list[idx]
             try: return float(val) if val is not None else default
             except: return default

        stage_weights['data_weight'] = get_stage_weight('data_weights', 1.0)
        stage_weights['body_pose_weight'] = get_stage_weight('body_pose_prior_weights', 0.0)
        stage_weights['shape_weight'] = get_stage_weight('shape_weights', 0.0)
        stage_weights['expr_weight'] = get_stage_weight('expr_weights', 0.0)
        stage_weights['hand_prior_weight'] = get_stage_weight('hand_pose_prior_weights', 0.0)
        stage_weights['jaw_prior_weight'] = get_stage_weight('jaw_pose_prior_weights', 0.0)
        stage_weights['coll_loss_weight'] = get_stage_weight('coll_loss_weights', 0.0)
        stage_weights['bending_prior_weight'] = get_stage_weight('bending_prior_weights', 0.0)

        if stage_idx == 1:
            stage_weights['data_weight'] = 0.0
            stage_weights['shape_weight'] = 0.0

        loss_func.reset_loss_weights(stage_weights)

        with FittingMonitor(visualize=False, maxiters=max_iters_this_stage, ftol=ftol, gtol=gtol,
                            model_type=config.get('model_type', 'smplx'), summary_steps=10) as monitor:
            closure_kwargs = dict(optimizer=optimizer, body_model=body_model, camera=camera,
                                  loss=loss_func, gt_joints=keypoints_tensor[:, :, :2],
                                  joints_conf=joints_conf, joint_weights=torch.ones_like(joints_conf),
                                  use_vposer=use_vposer, vposer=vposer_model,
                                  pose_embedding=pose_embedding if use_vposer else None,
                                  create_graph=(optimizer_type != 'lbfgsls'), return_verts=True, return_full_pose=True)
            closure = monitor.create_fitting_closure(**{k: v for k, v in closure_kwargs.items() if v is not None})

            prev_loss = float('inf')
            loop_iter = 0
            while loop_iter < max_iters_this_stage:
                loop_iter += 1
                camera.translation.data = camera_translation_opt.data
                optimizer.zero_grad()
                loss = optimizer.step(closure) if optimizer_type == 'lbfgsls' else closure()
                if optimizer_type == 'adam': loss.backward(); optimizer.step()

                loss_val = loss.item()
                if torch.isnan(loss).sum() > 0 or torch.isinf(loss).sum() > 0: break
                if loop_iter > 1 and abs(prev_loss - loss_val) / max(abs(prev_loss), 1e-12) < ftol: break
                prev_loss = loss_val

    final_betas_np = init_betas.detach().cpu().numpy()
    final_transl_np = camera_translation_opt.detach().cpu().numpy()[0]
    final_cam_rot_np = camera.rotation.detach().cpu().numpy()[0]

    body_model.eval()
    with torch.no_grad():
        if use_vposer:
            final_body_pose_aa = vposer_model.decode(pose_embedding.detach(), output_type='aa').view(batch_size, -1)
            final_body_pose_np = final_body_pose_aa.cpu().numpy()[0]
        else:
            final_body_pose_np = init_body_pose.detach().cpu().numpy()[0]

        final_betas_tensor = torch.tensor(final_betas_np, device=device)
        model_kwargs = {'betas': final_betas_tensor, 'global_orient': torch.zeros([batch_size, 3], device=device),
                        'body_pose': torch.tensor(final_body_pose_np, device=device).unsqueeze(0),
                        'return_full_pose': True, 'return_verts': True}
        
        final_model_output = body_model(**model_kwargs)
        final_vertices_np = final_model_output.vertices.detach().cpu().numpy()[0]
        model_faces_np = body_model.faces_tensor.cpu().numpy()

        os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
        with open(output_obj_path, 'w') as f_obj:
            for v in final_vertices_np: f_obj.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for face in model_faces_np: f_obj.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    return {'camera_rotation': final_cam_rot_np, 'camera_translation': final_transl_np,
            'betas': final_betas_np, 'body_pose': final_body_pose_np, 'gender': gender, 
            'fitting_time': time.time() - process_start_time}

if __name__ == "__main__":
    print("Module body_reconstruction loaded.")