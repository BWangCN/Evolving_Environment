"""
3DGS → AnyGrasp → TrajectoryGenerator pipeline.

Connects Gaussian Grouping's segmented 3DGS scene to AnyGrasp grasp detection,
then to our TrajectoryGenerator for trajectory synthesis.

Pipeline:
    1. Load 3DGS scene (PLY + classifier)
    2. Extract object point cloud by segmentation ID
    3. Run AnyGrasp on the object point cloud
    4. Convert AnyGrasp grasps → GraspPose
    5. Generate pick-place trajectories
    6. Compute camera poses for gsplat rendering
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Rotation conversion ───────────────────────────────────────────────

def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)  # Normalize
    if q[0] < 0:
        q = -q  # Canonical form: w > 0
    return q.astype(np.float32)


# ── Step 1: Extract object point cloud from 3DGS ─────────────────────

def extract_object_pointcloud(
    ply_path: str,
    classifier_path: str,
    object_id: int,
    threshold: float = 0.3,
    max_points: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract point cloud of a specific object from segmented 3DGS.

    Args:
        ply_path: Path to 3DGS PLY file.
        classifier_path: Path to classifier.pth (identity encoding → object ID).
        object_id: Target object ID to extract.
        threshold: Minimum classification confidence.
        max_points: Max points to return (subsample if larger).

    Returns:
        points: (N, 3) float32 positions.
        colors: (N, 3) float32 RGB colors [0, 1].
    """
    import torch
    from plyfile import PlyData

    # Load PLY
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)

    # Load object features (16-dim identity encoding per Gaussian)
    n_gaussians = len(xyz)
    if 'obj_dc_0' in v.data.dtype.names:
        features = np.stack([v[f'obj_dc_{i}'] for i in range(16)], axis=-1)
    elif 'f_rest_0' in v.data.dtype.names:
        features = np.stack([v[f'f_rest_{i}'] for i in range(16)], axis=-1)
    else:
        features = None

    # Load classifier
    classifier = torch.load(classifier_path, map_location='cpu', weights_only=True)
    # The classifier maps 16-dim features → object class probabilities

    # Get DC colors
    SH_C0 = 0.28209479177387814
    colors = np.clip(
        SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1) + 0.5,
        0, 1
    ).astype(np.float32)

    if features is not None and classifier is not None:
        # Run classifier to get object assignments
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32)
            # Classifier is a Conv2d stored as OrderedDict: weight (C_out, 16, 1, 1), bias (C_out,)
            if isinstance(classifier, dict):
                weight = classifier.get('weight', classifier.get('classifier.weight'))
                bias = classifier.get('bias', classifier.get('classifier.bias'))
                # Reshape conv weight to linear: (C_out, 16, 1, 1) → (C_out, 16)
                if weight.dim() == 4:
                    weight = weight.squeeze(-1).squeeze(-1)  # (C_out, 16)
                logits = feat_tensor @ weight.T + bias  # (N, C_out)
            else:
                logits = classifier(feat_tensor)
            probs = torch.softmax(logits, dim=-1)
            obj_prob = probs[:, object_id].numpy()

        # Select Gaussians belonging to the object
        mask = obj_prob > threshold
        obj_xyz = xyz[mask]
        obj_colors = colors[mask]
        print(f"  Object {object_id}: {mask.sum()} / {n_gaussians} Gaussians (threshold={threshold})")
    else:
        # Fallback: return all points (no classifier available)
        print(f"  WARNING: No classifier/features, returning all {n_gaussians} points")
        obj_xyz = xyz
        obj_colors = colors

    # Subsample if too many
    if len(obj_xyz) > max_points:
        idx = np.random.choice(len(obj_xyz), max_points, replace=False)
        obj_xyz = obj_xyz[idx]
        obj_colors = obj_colors[idx]

    return obj_xyz, obj_colors


def extract_scene_pointcloud(
    ply_path: str,
    max_points: int = 200000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract full scene point cloud from 3DGS PLY.

    Returns:
        points: (N, 3) float32.
        colors: (N, 3) float32.
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    SH_C0 = 0.28209479177387814
    colors = np.clip(
        SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1) + 0.5,
        0, 1
    ).astype(np.float32)

    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, colors = xyz[idx], colors[idx]

    return xyz, colors


# ── Step 2 & 3: AnyGrasp → GraspPose ─────────────────────────────────

def detect_grasps(
    points: np.ndarray,
    colors: np.ndarray,
    anygrasp_dir: str = "/home/bwang25/Desktop/Manipulation/anygrasp_sdk/grasp_detection",
    checkpoint: str = "checkpoints/checkpoint_detection.tar",
    max_gripper_width: float = 0.08,
    top_k: int = 20,
    apply_object_mask: bool = True,
    collision_detection: bool = True,
) -> list:
    """Run AnyGrasp on a point cloud and return GraspPose objects.

    Args:
        points: (N, 3) float32 point positions.
        colors: (N, 3) float32 RGB colors [0, 1].
        anygrasp_dir: Path to AnyGrasp grasp_detection directory.
        checkpoint: Relative path to checkpoint within anygrasp_dir.
        max_gripper_width: Max gripper opening (Franka Panda: 0.08m).
        top_k: Number of top grasps to return.
        apply_object_mask: Filter by objectness mask.
        collision_detection: Enable collision checking.

    Returns:
        List of GraspPose objects, sorted by score (best first).
    """
    import sys
    import os

    # Import GraspPose from our trajectory module
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.trajectory import GraspPose

    # Import AnyGrasp
    orig_dir = os.getcwd()
    os.chdir(anygrasp_dir)
    sys.path.insert(0, anygrasp_dir)

    # Setup AnyGrasp config
    class Cfg:
        pass
    cfg = Cfg()
    cfg.checkpoint_path = checkpoint
    cfg.max_gripper_width = max_gripper_width
    cfg.gripper_height = 0.03
    cfg.top_down_grasp = False
    cfg.debug = False

    os.environ['OMP_NUM_THREADS'] = '16'
    from gsnet import AnyGrasp
    anygrasp = AnyGrasp(cfg)
    anygrasp.load_net()

    # Set workspace limits from point cloud bounds (with margin)
    margin = 0.1
    lims = [
        points[:, 0].min() - margin, points[:, 0].max() + margin,
        points[:, 1].min() - margin, points[:, 1].max() + margin,
        points[:, 2].min() - margin, points[:, 2].max() + margin,
    ]

    # Detect grasps
    result = anygrasp.get_grasp(
        points, colors,
        lims=lims,
        apply_object_mask=apply_object_mask,
        dense_grasp=False,
        collision_detection=collision_detection,
    )

    os.chdir(orig_dir)

    gg = result[0]
    if gg is None or len(gg) == 0:
        print("  WARNING: No grasps detected")
        return []

    # NMS + sort by score
    gg = gg.nms().sort_by_score()
    n = min(top_k, len(gg))

    # Convert to GraspPose
    grasp_poses = []
    for i in range(n):
        g = gg[i]
        quat_wxyz = rotation_matrix_to_quaternion_wxyz(g.rotation_matrix)
        gp = GraspPose(
            position=g.translation.astype(np.float32),
            orientation=quat_wxyz,
            score=float(g.score),
            width=float(g.width),
        )
        grasp_poses.append(gp)

    print(f"  {len(grasp_poses)} grasps converted (top scores: {[f'{gp.score:.3f}' for gp in grasp_poses[:5]]})")
    return grasp_poses


# ── Full pipeline ─────────────────────────────────────────────────────

def run_3dgs_grasp_pipeline(
    ply_path: str,
    classifier_path: str,
    object_id: int,
    place_position: np.ndarray,
    language_instruction: str = "pick up the object",
    top_k_grasps: int = 10,
    threshold: float = 0.3,
):
    """Run the full 3DGS → AnyGrasp → Trajectory pipeline.

    Args:
        ply_path: Path to 3DGS PLY.
        classifier_path: Path to classifier.pth.
        object_id: Object ID to grasp.
        place_position: (3,) target position for placement.
        language_instruction: Language instruction for the task.
        top_k_grasps: Number of grasp candidates.
        threshold: Object classification threshold.

    Returns:
        List of dicts, each containing:
            'grasp': GraspPose
            'trajectory': Trajectory
            'actions': np.ndarray (T-1, 7)
            'camera_poses': list of (pos, quat)
            'instruction': str
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.trajectory import (
        TrajectoryGenerator, CollisionChecker,
        compute_place_target, trajectory_to_actions, compute_camera_poses,
    )

    print(f"=== 3DGS → Grasp → Trajectory Pipeline ===")
    print(f"PLY: {ply_path}")
    print(f"Object ID: {object_id}, Place target: {place_position}")

    # Step 1: Extract point clouds
    print("\nStep 1: Extracting point clouds...")
    obj_points, obj_colors = extract_object_pointcloud(
        ply_path, classifier_path, object_id, threshold
    )
    scene_points, scene_colors = extract_scene_pointcloud(ply_path)
    print(f"  Object: {len(obj_points)} points, Scene: {len(scene_points)} points")

    # Step 2 & 3: Detect grasps
    print("\nStep 2-3: Detecting grasps...")
    grasp_poses = detect_grasps(
        obj_points, obj_colors,
        top_k=top_k_grasps,
        collision_detection=False,  # We do our own collision checking
    )
    if not grasp_poses:
        print("  ERROR: No grasps detected. Aborting.")
        return []

    # Step 4: Generate trajectories
    print(f"\nStep 4: Generating trajectories for {len(grasp_poses)} grasps...")
    gen = TrajectoryGenerator()
    checker = CollisionChecker(scene_points)
    results = []

    for i, gp in enumerate(grasp_poses):
        try:
            traj = gen.generate_pick_place(gp, place_position)
            valid = gen.validate_trajectory(traj, checker)
            if not valid:
                continue

            # Step 5: Convert to training format
            actions = trajectory_to_actions(traj)
            cam_poses = compute_camera_poses(traj)

            results.append({
                'grasp': gp,
                'trajectory': traj,
                'actions': actions,
                'camera_poses': cam_poses,
                'instruction': language_instruction,
            })
        except Exception as e:
            continue

    print(f"  {len(results)} / {len(grasp_poses)} valid trajectories generated")

    return results
