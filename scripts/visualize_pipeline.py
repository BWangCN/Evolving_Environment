"""
Visualize the 3DGS → AnyGrasp → Trajectory pipeline.

Three visualization modes:
  1. `scene`     — View 3DGS scene point cloud with object segmentation highlighted
  2. `grasps`    — Run AnyGrasp on object point cloud, visualize grasp poses
  3. `trajectory` — Generate pick-place trajectory from best grasp, visualize full path

Usage (run from gaussian_grouping conda env):
    conda activate gaussian_grouping

    # 1. Visualize segmented scene (bear object highlighted)
    python scripts/visualize_pipeline.py scene

    # 2. Visualize AnyGrasp detections on the object
    python scripts/visualize_pipeline.py grasps

    # 3. Full trajectory visualization (uses mock tabletop scale for proper robot-scale demo)
    python scripts/visualize_pipeline.py trajectory

    # 4. All three in sequence
    python scripts/visualize_pipeline.py all

    # Use --mock to skip real 3DGS/AnyGrasp and use synthetic data
    python scripts/visualize_pipeline.py trajectory --mock

    # Save screenshot instead of interactive window
    python scripts/visualize_pipeline.py trajectory --save output.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Color palette ────────────────────────────────────────────────────
COLORS = {
    "scene": np.array([0.7, 0.7, 0.7]),       # light gray
    "object": np.array([1.0, 0.3, 0.1]),       # orange-red
    "grasp_frame": np.array([0.0, 0.8, 0.0]),  # green
    "gripper": np.array([0.2, 0.6, 1.0]),      # blue
    "trajectory": np.array([0.1, 0.9, 0.3]),   # green
    "place_target": np.array([1.0, 0.8, 0.0]), # yellow
}

# Phase → color mapping for trajectory
PHASE_COLORS = {
    "transit_pick":  np.array([0.2, 0.6, 1.0]),   # blue
    "pre_grasp":     np.array([0.0, 0.8, 0.4]),   # green
    "approach":      np.array([1.0, 0.6, 0.0]),   # orange
    "grasp":         np.array([1.0, 0.0, 0.0]),   # red
    "lift":          np.array([0.8, 0.0, 0.8]),   # purple
    "transit_place": np.array([0.2, 0.6, 1.0]),   # blue
    "pre_place":     np.array([0.0, 0.8, 0.4]),   # green
    "place_descend": np.array([1.0, 0.6, 0.0]),   # orange
    "release":       np.array([1.0, 0.0, 0.0]),   # red
    "retreat":       np.array([0.5, 0.5, 0.5]),   # gray
}


# ── Geometry helpers ─────────────────────────────────────────────────

def create_coordinate_frame(position, orientation_wxyz, size=0.03):
    """Create an Open3D coordinate frame at the given 6-DoF pose."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    R = quat_wxyz_to_rotation_matrix(orientation_wxyz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    frame.transform(T)
    return frame


def quat_wxyz_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def create_gripper_lines(position, orientation_wxyz, width=0.08, color=None):
    """Create a simple line-based gripper visualization at the given pose.

    Draws a T-shaped gripper: base bar + two fingers.
    """
    if color is None:
        color = COLORS["gripper"]

    R = quat_wxyz_to_rotation_matrix(orientation_wxyz)
    hw = width / 2  # half-width
    fl = 0.04       # finger length

    # Points in gripper local frame (x=right, y=forward/approach, z=up)
    # Base bar (horizontal, connecting finger bases)
    local_pts = np.array([
        [-hw, 0, 0],      # 0: left finger base
        [hw, 0, 0],       # 1: right finger base
        [-hw, -fl, 0],    # 2: left finger tip
        [hw, -fl, 0],     # 3: right finger tip
        [0, 0, 0],        # 4: center (contact point)
        [0, 0.04, 0],     # 5: wrist (behind contact)
    ])

    # Transform to world frame
    world_pts = (R @ local_pts.T).T + position

    lines = [[0, 1], [0, 2], [1, 3], [4, 5]]
    colors = [color.tolist()] * len(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_trajectory_lines(waypoints, color_by_phase=True):
    """Create line set connecting trajectory waypoints.

    Args:
        waypoints: list of Waypoint objects (from src.trajectory.generator)
        color_by_phase: if True, color each segment by its phase
    """
    positions = np.array([w.position for w in waypoints])
    n = len(positions)

    lines = [[i, i + 1] for i in range(n - 1)]

    if color_by_phase:
        colors = []
        for i in range(n - 1):
            phase = waypoints[i + 1].phase
            c = PHASE_COLORS.get(phase, np.array([0.5, 0.5, 0.5]))
            colors.append(c.tolist())
    else:
        colors = [COLORS["trajectory"].tolist()] * len(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_waypoint_spheres(waypoints, radius=0.003):
    """Create small spheres at each waypoint, colored by gripper state."""
    spheres = []
    for w in waypoints:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(w.position)
        # Green = open, Red = closed
        if w.gripper > 0.5:
            s.paint_uniform_color([0.0, 0.8, 0.0])
        else:
            s.paint_uniform_color([0.8, 0.0, 0.0])
        s.compute_vertex_normals()
        spheres.append(s)
    return spheres


def create_table_plane(center=None, size=0.6, height=0.0):
    """Create a translucent table surface."""
    if center is None:
        center = [0.3, 0.0, height]
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size, height=size, depth=0.005
    )
    mesh.translate([center[0] - size/2, center[1] - size/2, height - 0.005])
    mesh.paint_uniform_color([0.85, 0.75, 0.6])  # wood color
    mesh.compute_vertex_normals()
    return mesh


def create_place_target_marker(position, radius=0.015):
    """Create a marker sphere at the place target."""
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(position)
    s.paint_uniform_color(COLORS["place_target"].tolist())
    s.compute_vertex_normals()
    return s


# ── 3DGS data loading ───────────────────────────────────────────────

def load_3dgs_scene(ply_path, classifier_path, object_id=34,
                    threshold=0.3, max_scene_pts=100000, max_obj_pts=50000):
    """Load 3DGS scene and extract object point cloud.

    Returns:
        scene_pcd: Open3D point cloud of full scene (dimmed)
        object_pcd: Open3D point cloud of target object (highlighted)
        obj_points: (N, 3) raw object point positions
        obj_colors: (N, 3) raw object colors
    """
    from src.pipeline.gs_to_grasp import extract_object_pointcloud, extract_scene_pointcloud

    print(f"Loading 3DGS scene from {ply_path}...")
    scene_pts, scene_cols = extract_scene_pointcloud(ply_path, max_points=max_scene_pts)
    obj_pts, obj_cols = extract_object_pointcloud(
        ply_path, classifier_path, object_id, threshold, max_points=max_obj_pts
    )

    # Scene point cloud (dimmed)
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_cols * 0.4)

    # Object point cloud (highlighted)
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(obj_pts)
    object_pcd.colors = o3d.utility.Vector3dVector(obj_cols)

    return scene_pcd, object_pcd, obj_pts, obj_cols


# ── Mock data for trajectory demo ───────────────────────────────────

def create_mock_tabletop_scene():
    """Create a synthetic tabletop scene at robot scale for trajectory visualization.

    Returns:
        scene_pcd: Open3D point cloud of table + clutter
        object_pcd: Open3D point cloud of target object (cube)
        obj_points: (N, 3) object point positions
        obj_colors: (N, 3) object colors
    """
    rng = np.random.default_rng(42)

    # Table surface points
    n_table = 5000
    table_x = rng.uniform(0.0, 0.6, n_table)
    table_y = rng.uniform(-0.3, 0.3, n_table)
    table_z = np.full(n_table, 0.0) + rng.normal(0, 0.001, n_table)
    table_pts = np.stack([table_x, table_y, table_z], axis=-1).astype(np.float32)
    table_colors = np.tile([0.85, 0.75, 0.6], (n_table, 1)).astype(np.float32)

    # Target object: small cube at (0.35, 0.1, 0.025)
    obj_center = np.array([0.35, 0.1, 0.025])
    obj_size = 0.05  # 5cm cube
    n_obj = 2000
    obj_pts = obj_center + rng.uniform(-obj_size/2, obj_size/2, (n_obj, 3)).astype(np.float32)
    obj_colors = np.tile([1.0, 0.3, 0.1], (n_obj, 1)).astype(np.float32)

    # Distractor objects
    distractors = []
    for pos, color in [
        ([0.2, -0.1, 0.02], [0.2, 0.6, 1.0]),
        ([0.45, -0.15, 0.03], [0.0, 0.8, 0.4]),
    ]:
        n_d = 800
        d_pts = np.array(pos) + rng.uniform(-0.03, 0.03, (n_d, 3)).astype(np.float32)
        d_colors = np.tile(color, (n_d, 1)).astype(np.float32)
        distractors.append((d_pts, d_colors))

    # Combine scene (table + distractors, NOT including target object)
    all_scene_pts = [table_pts] + [d[0] for d in distractors]
    all_scene_cols = [table_colors] + [d[1] for d in distractors]
    scene_pts = np.concatenate(all_scene_pts)
    scene_cols = np.concatenate(all_scene_cols)

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_cols)

    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(obj_pts)
    object_pcd.colors = o3d.utility.Vector3dVector(obj_colors)

    return scene_pcd, object_pcd, obj_pts, obj_colors


def create_mock_grasp_poses(obj_points, n=5):
    """Create mock grasp poses centered on the object, approaching from above."""
    from src.trajectory.generator import GraspPose

    center = obj_points.mean(axis=0)
    rng = np.random.default_rng(42)

    grasps = []
    for i in range(n):
        # Slight position variation around object center
        pos = center.copy()
        pos[:2] += rng.uniform(-0.01, 0.01, 2)

        # Top-down grasp: gripper pointing -Z (w=1, x=0, y=0, z=0)
        # Add slight rotation variation
        angle = rng.uniform(-0.1, 0.1)
        ori = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)], dtype=np.float32)

        score = 0.9 - i * 0.1
        grasps.append(GraspPose(
            position=pos.astype(np.float32),
            orientation=ori,
            score=score,
            width=0.05 + rng.uniform(-0.01, 0.01),
        ))
    return grasps


# ── Visualization modes ─────────────────────────────────────────────

def visualize_scene(args):
    """Mode 1: Visualize 3DGS scene with object highlighted."""
    ply_path = args.ply_path
    classifier_path = args.classifier_path
    object_id = args.object_id

    scene_pcd, object_pcd, _, _ = load_3dgs_scene(
        ply_path, classifier_path, object_id, args.threshold
    )

    print(f"Scene: {len(scene_pcd.points)} points")
    print(f"Object {object_id}: {len(object_pcd.points)} points")

    # Add coordinate frame at origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    geometries = [scene_pcd, object_pcd, origin]

    _show(geometries, "3DGS Scene — Object Segmentation", args)


def visualize_grasps(args):
    """Mode 2: Visualize AnyGrasp detections on object point cloud."""
    if args.mock:
        _, object_pcd, obj_pts, obj_cols = create_mock_tabletop_scene()
        grasp_poses = create_mock_grasp_poses(obj_pts)
    else:
        ply_path = args.ply_path
        classifier_path = args.classifier_path
        _, object_pcd, obj_pts, obj_cols = load_3dgs_scene(
            ply_path, classifier_path, args.object_id, args.threshold,
        )
        # Run AnyGrasp
        from src.pipeline.gs_to_grasp import detect_grasps
        grasp_poses = detect_grasps(obj_pts, obj_cols, top_k=args.top_k)

    if not grasp_poses:
        print("ERROR: No grasps detected!")
        return

    print(f"\n{len(grasp_poses)} grasps detected:")
    for i, gp in enumerate(grasp_poses[:10]):
        print(f"  Grasp {i}: pos={gp.position}, score={gp.score:.3f}, width={gp.width:.3f}")

    # Compute scale factor from object extent (for adapting marker sizes)
    obj_extent = obj_pts.max(axis=0) - obj_pts.min(axis=0)
    scale = max(obj_extent) * 0.1  # 10% of object extent

    # Build visualization
    geometries = [object_pcd]

    # Grasp pose frames + gripper outlines
    for i, gp in enumerate(grasp_poses):
        # Coordinate frame at grasp pose (scale-adaptive)
        frame = create_coordinate_frame(gp.position, gp.orientation, size=scale * 0.5)
        geometries.append(frame)

        # Gripper outline (scale-adaptive width)
        alpha = max(0.3, 1.0 - i * 0.1)  # fade lower-ranked grasps
        color = np.array([0.0, alpha, 0.0])
        gripper_width = max(gp.width, scale * 0.5)  # ensure visible
        gripper_lines = create_gripper_lines(gp.position, gp.orientation, gripper_width, color)
        geometries.append(gripper_lines)

    # Add object center sphere (scale-adaptive)
    center = obj_pts.mean(axis=0)
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.2)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1.0, 1.0, 0.0])
    center_sphere.compute_vertex_normals()
    geometries.append(center_sphere)

    _show(geometries, f"AnyGrasp — {len(grasp_poses)} Grasp Poses", args)


def visualize_trajectory(args):
    """Mode 3: Full trajectory visualization with pick-place path."""
    from src.trajectory.generator import TrajectoryGenerator, GraspPose
    from src.trajectory.action_format import trajectory_to_actions

    if args.mock:
        scene_pcd, object_pcd, obj_pts, obj_cols = create_mock_tabletop_scene()
        grasp_poses = create_mock_grasp_poses(obj_pts)
    else:
        ply_path = args.ply_path
        classifier_path = args.classifier_path
        scene_pcd, object_pcd, obj_pts, obj_cols = load_3dgs_scene(
            ply_path, classifier_path, args.object_id, args.threshold,
        )
        from src.pipeline.gs_to_grasp import detect_grasps
        grasp_poses = detect_grasps(obj_pts, obj_cols, top_k=args.top_k)

    if not grasp_poses:
        print("ERROR: No grasps detected!")
        return

    # Use best grasp
    best_grasp = grasp_poses[0]
    print(f"\nBest grasp: pos={best_grasp.position}, score={best_grasp.score:.3f}")

    # Place target: offset from grasp position
    place_target = np.array([0.25, -0.15, 0.025], dtype=np.float32)
    print(f"Place target: {place_target}")

    # Generate trajectory
    gen = TrajectoryGenerator()
    traj = gen.generate_pick_place(best_grasp, place_target)
    actions = trajectory_to_actions(traj)

    print(f"\nTrajectory: {len(traj)} waypoints, {len(actions)} actions")
    print(f"Phases: {' → '.join(dict.fromkeys(w.phase for w in traj.waypoints))}")

    # ── Build visualization ──
    geometries = []

    # Scene + object point clouds
    geometries.append(scene_pcd)
    geometries.append(object_pcd)

    # Table surface
    geometries.append(create_table_plane())

    # Trajectory path (colored by phase)
    traj_lines = create_trajectory_lines(traj.waypoints, color_by_phase=True)
    geometries.append(traj_lines)

    # Waypoint spheres (green=open, red=closed gripper)
    geometries.extend(create_waypoint_spheres(traj.waypoints, radius=0.003))

    # Gripper at key poses: grasp contact, lift top, place release
    # Only show one gripper per key phase (first waypoint of that phase)
    shown_phases = set()
    key_phases = {"grasp", "release"}
    for w in traj.waypoints:
        if w.phase in key_phases and w.phase not in shown_phases:
            shown_phases.add(w.phase)
            width = best_grasp.width if w.gripper < 0.5 else 0.08
            color = np.array([0.8, 0.0, 0.0]) if w.gripper < 0.5 else np.array([0.0, 0.8, 0.0])
            geometries.append(create_gripper_lines(w.position, w.orientation, width, color))
            geometries.append(create_coordinate_frame(w.position, w.orientation, size=0.02))
    # Also show gripper at the highest lift point
    lift_wps = [w for w in traj.waypoints if w.phase == "lift"]
    if lift_wps:
        top_wp = max(lift_wps, key=lambda w: w.position[2])
        geometries.append(create_gripper_lines(
            top_wp.position, top_wp.orientation, best_grasp.width,
            np.array([0.6, 0.0, 0.6])  # purple for lift
        ))

    # Place target marker
    geometries.append(create_place_target_marker(place_target))

    # Home position marker
    from src.config import franka
    home_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    home_sphere.translate(franka.HOME_POSITION)
    home_sphere.paint_uniform_color([0.5, 0.5, 1.0])
    home_sphere.compute_vertex_normals()
    geometries.append(home_sphere)
    geometries.append(create_coordinate_frame(
        franka.HOME_POSITION, franka.HOME_ORIENTATION, size=0.03
    ))

    # Origin frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    # Print action statistics
    print(f"\nAction stats (delta EE format, {actions.shape}):")
    print(f"  Δpos  range: [{actions[:, :3].min():.4f}, {actions[:, :3].max():.4f}]")
    print(f"  Δrot  range: [{actions[:, 3:6].min():.4f}, {actions[:, 3:6].max():.4f}]")
    print(f"  grip  vals:  {np.unique(actions[:, 6])}")

    _show(geometries, "Pick-Place Trajectory", args)


def visualize_plot2d(args):
    """Mode 4: Matplotlib 2D plots — trajectory XZ/XY views + action distributions.

    Generates a 2x2 figure:
      Top-left:  XZ side view (trajectory path colored by phase)
      Top-right: XY top-down view
      Bottom-left: Action deltas over time (pos + rot)
      Bottom-right: Gripper state over time
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from src.trajectory.generator import TrajectoryGenerator, GraspPose
    from src.trajectory.action_format import trajectory_to_actions

    if args.mock:
        _, _, obj_pts, obj_cols = create_mock_tabletop_scene()
        grasp_poses = create_mock_grasp_poses(obj_pts)
    else:
        _, _, obj_pts, obj_cols = load_3dgs_scene(
            args.ply_path, args.classifier_path, args.object_id, args.threshold,
        )
        from src.pipeline.gs_to_grasp import detect_grasps
        grasp_poses = detect_grasps(obj_pts, obj_cols, top_k=args.top_k)

    if not grasp_poses:
        print("ERROR: No grasps detected!")
        return

    best_grasp = grasp_poses[0]
    place_target = np.array([0.25, -0.15, 0.025], dtype=np.float32)

    gen = TrajectoryGenerator()
    traj = gen.generate_pick_place(best_grasp, place_target)
    actions = trajectory_to_actions(traj)
    positions = traj.positions
    phases = [w.phase for w in traj.waypoints]
    grippers = traj.gripper_states

    # Phase → color mapping for matplotlib
    phase_cmap = {
        "transit_pick":  "#3399FF",
        "pre_grasp":     "#00CC66",
        "approach":      "#FF9900",
        "grasp":         "#FF0000",
        "lift":          "#CC00CC",
        "transit_place": "#3399FF",
        "pre_place":     "#00CC66",
        "place_descend": "#FF9900",
        "release":       "#FF0000",
        "retreat":       "#888888",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pick-Place Trajectory Analysis", fontsize=14, fontweight="bold")

    # ── Top-left: XZ side view ──
    ax = axes[0, 0]
    for i in range(len(positions) - 1):
        c = phase_cmap.get(phases[i + 1], "#888888")
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 2], color=c, linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 2], c="blue", s=80, zorder=5, label="Home")
    ax.scatter(best_grasp.position[0], best_grasp.position[2], c="red", s=80,
               marker="v", zorder=5, label="Grasp")
    ax.scatter(place_target[0], place_target[2], c="gold", s=80,
               marker="^", zorder=5, label="Place")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Side View (XZ)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── Top-right: XY top-down view ──
    ax = axes[0, 1]
    for i in range(len(positions) - 1):
        c = phase_cmap.get(phases[i + 1], "#888888")
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=c, linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], c="blue", s=80, zorder=5, label="Home")
    ax.scatter(best_grasp.position[0], best_grasp.position[1], c="red", s=80,
               marker="v", zorder=5, label="Grasp")
    ax.scatter(place_target[0], place_target[1], c="gold", s=80,
               marker="^", zorder=5, label="Place")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-Down View (XY)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: Action deltas ──
    ax = axes[1, 0]
    t = np.arange(len(actions))
    ax.plot(t, actions[:, 0], label="Δx", alpha=0.8)
    ax.plot(t, actions[:, 1], label="Δy", alpha=0.8)
    ax.plot(t, actions[:, 2], label="Δz", alpha=0.8)
    ax.plot(t, actions[:, 3], label="Δroll", linestyle="--", alpha=0.6)
    ax.plot(t, actions[:, 4], label="Δpitch", linestyle="--", alpha=0.6)
    ax.plot(t, actions[:, 5], label="Δyaw", linestyle="--", alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Delta")
    ax.set_title("Action Deltas (EE frame)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Add phase boundaries as vertical lines
    prev_phase = phases[0]
    for i, phase in enumerate(phases[1:], 1):
        if phase != prev_phase:
            ax.axvline(i - 1, color="gray", linestyle=":", alpha=0.4)
            prev_phase = phase

    # ── Bottom-right: Gripper state ──
    ax = axes[1, 1]
    colors_grip = ["green" if g > 0.5 else "red" for g in grippers]
    ax.bar(np.arange(len(grippers)), grippers, color=colors_grip, width=1.0, alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gripper State")
    ax.set_title("Gripper (1=Open, 0=Closed)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Phase color legend at bottom
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=p) for p, c in phase_cmap.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    save_path = args.save or "logs/trajectory_plot2d.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved 2D plot to {save_path}")
    plt.close()


# ── Legend / HUD ─────────────────────────────────────────────────────

def print_legend():
    """Print a color legend to the terminal."""
    print("\n── Legend ──────────────────────────────")
    print("  Trajectory colors by phase:")
    for phase, color in PHASE_COLORS.items():
        r, g, b = (int(c * 255) for c in color)
        print(f"    \033[38;2;{r};{g};{b}m██\033[0m {phase}")
    print("  Waypoint spheres:")
    print("    \033[32m●\033[0m gripper open")
    print("    \033[31m●\033[0m gripper closed")
    print("  Markers:")
    print("    \033[33m●\033[0m place target")
    print("    \033[34m●\033[0m home position")
    print("────────────────────────────────────────")


# ── Display ──────────────────────────────────────────────────────────

def _auto_viewpoint(vis, geometries):
    """Set camera to look at the centroid of all point cloud geometries."""
    all_pts = []
    for g in geometries:
        if isinstance(g, o3d.geometry.PointCloud) and len(g.points) > 0:
            all_pts.append(np.asarray(g.points))
    if not all_pts:
        return
    pts = np.concatenate(all_pts)
    center = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)
    dist = max(extent) * 1.5

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.4, -0.5, 0.7])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.4)


def _show(geometries, title, args):
    """Display or save the visualization."""
    print_legend()

    if args.save:
        # Off-screen render to image
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1920, height=1080, visible=False)
        for g in geometries:
            vis.add_geometry(g)
        pt_size = 3.0 if not args.mock else 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.15])
        vis.get_render_option().point_size = pt_size
        vis.poll_events()
        vis.update_renderer()

        _auto_viewpoint(vis, geometries)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(args.save, do_render=True)
        print(f"\nSaved screenshot to {args.save}")
        vis.destroy_window()
    else:
        # Interactive window
        pt_size = 3.0 if not args.mock else 2.0
        print(f"\nOpening interactive viewer: {title}")
        print("  Mouse: rotate (left), pan (middle), zoom (scroll)")
        print("  Press Q or Esc to close")

        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1920,
            height=1080,
            point_size=pt_size,
        )


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3DGS → AnyGrasp → Trajectory pipeline"
    )
    parser.add_argument(
        "mode", choices=["scene", "grasps", "trajectory", "plot2d", "all"],
        help="Visualization mode"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic tabletop data (skip real 3DGS/AnyGrasp)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save screenshot to file instead of interactive window"
    )
    parser.add_argument(
        "--ply-path", type=str,
        default="/home/bwang25/Desktop/Manipulation/gaussian-grouping/output/bear/point_cloud/iteration_30000/point_cloud.ply",
        help="Path to 3DGS PLY file"
    )
    parser.add_argument(
        "--classifier-path", type=str,
        default="/home/bwang25/Desktop/Manipulation/gaussian-grouping/output/bear/point_cloud/iteration_30000/classifier.pth",
        help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--object-id", type=int, default=34,
        help="Object ID to extract from segmented 3DGS"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Classification confidence threshold"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top grasps to visualize"
    )

    args = parser.parse_args()

    if args.mode == "scene":
        visualize_scene(args)
    elif args.mode == "grasps":
        visualize_grasps(args)
    elif args.mode == "trajectory":
        visualize_trajectory(args)
    elif args.mode == "plot2d":
        visualize_plot2d(args)
    elif args.mode == "all":
        visualize_scene(args)
        visualize_grasps(args)
        visualize_trajectory(args)
        visualize_plot2d(args)


if __name__ == "__main__":
    main()
