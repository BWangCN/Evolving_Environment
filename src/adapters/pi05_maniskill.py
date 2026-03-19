"""
Adapter between π0.5 (openpi) and ManiSkill 3.

Uses the LIBERO action format (delta EE pose, 7-dim) which maps directly
to ManiSkill's pd_ee_delta_pose controller.

Architecture:
    π0.5 policy server (openpi .venv, Python 3.11, port 8000)
        ↕ websocket (openpi-client)
    ManiSkill environment (gaussian_grouping env, Python 3.10)

Action format:
    LIBERO/π0.5 output: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper] (7-dim)
    ManiSkill pd_ee_delta_pose: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper] (7-dim, [-1, 1])

State format:
    LIBERO expects: [ee_pos(3), ee_axis_angle(3), gripper_qpos(2)] (8-dim)
    ManiSkill provides: tcp_pose(7, pos+quat), qpos(9, 7joints+2fingers)
"""

import numpy as np
from PIL import Image

# ── Rotation helpers ──────────────────────────────────────────────────

def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle (3-dim).

    ManiSkill uses (w, x, y, z) internally but tcp_pose stores (x, y, z, w)
    in the last 4 elements. Check and handle both conventions.
    """
    # Normalize
    quat = quat / (np.linalg.norm(quat) + 1e-8)

    # Assume (x, y, z, w) convention from ManiSkill tcp_pose
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Clamp w for numerical stability
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(w))

    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)

    s = np.sqrt(1.0 - w * w + 1e-8)
    axis = np.array([x, y, z]) / s
    if w < 0:
        axis = -axis
        angle = 2.0 * np.pi - angle

    return (axis * angle).astype(np.float32)


# ── Observation adapter ───────────────────────────────────────────────

def maniskill_obs_to_pi05(
    obs: dict,
    frame: np.ndarray,
    prompt: str = "pick up the red cube",
    image_size: int = 224,
) -> dict:
    """Convert ManiSkill observation to π0.5 LIBERO input format.

    Args:
        obs: ManiSkill observation dict with 'agent' and 'extra' keys.
        frame: RGB render from env.render(), uint8 numpy array (H, W, 3).
        prompt: Language instruction string.
        image_size: Target image size for π0.5 (default 224).

    Returns:
        Dict matching LIBERO input format for openpi-client.
    """
    # Resize image to 224x224
    img = np.array(
        Image.fromarray(frame).resize((image_size, image_size), Image.BILINEAR),
        dtype=np.uint8,
    )

    # Extract EE pose from ManiSkill
    # tcp_pose: (7,) = [x, y, z, qx, qy, qz, qw] (position + quaternion)
    tcp_pose = obs["extra"]["tcp_pose"][0].cpu().numpy()
    ee_pos = tcp_pose[:3].astype(np.float32)
    ee_quat = tcp_pose[3:].astype(np.float32)  # (qx, qy, qz, qw)
    ee_axis_angle = quat_to_axis_angle(ee_quat)

    # Gripper: ManiSkill qpos has 9 dims (7 joints + 2 finger positions)
    qpos = obs["agent"]["qpos"][0].cpu().numpy()
    gripper_qpos = qpos[7:9].astype(np.float32)  # 2 finger positions

    # LIBERO state: [ee_pos(3), ee_axis_angle(3), gripper_qpos(2)] = 8-dim
    state = np.concatenate([ee_pos, ee_axis_angle, gripper_qpos])

    return {
        "observation/image": img,
        "observation/wrist_image": img,  # No wrist cam in default ManiSkill; reuse base
        "observation/state": state,
        "prompt": prompt,
    }


# ── Action adapter ────────────────────────────────────────────────────

def pi05_action_to_maniskill(
    actions: np.ndarray,
    step_index: int = 0,
    action_scale: float = 1.0,
) -> np.ndarray:
    """Convert π0.5 LIBERO action to ManiSkill pd_ee_delta_pose format.

    π0.5 LIBERO output: (horizon, 7) = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
        - Position deltas in meters (denormalized from quantile stats)
        - Rotation deltas in radians (axis-angle)
        - Gripper: continuous value (negative=closed, positive=open in LIBERO)

    ManiSkill pd_ee_delta_pose: (7,) in [-1, 1]
        - Position deltas: normalized, 1.0 ≈ max step size
        - Rotation deltas: normalized
        - Gripper: -1=close, 1=open

    Args:
        actions: π0.5 output array, shape (horizon, 7).
        step_index: Which step from the action chunk to use.
        action_scale: Scaling factor for position/rotation deltas.
            Increase if robot moves too slowly, decrease if too fast.

    Returns:
        ManiSkill action array, shape (7,), clipped to [-1, 1].
    """
    action = actions[step_index].copy()

    # Scale position and rotation deltas
    # LIBERO deltas are in absolute units (meters, radians)
    # ManiSkill expects [-1, 1] where the controller internally scales
    # The exact mapping depends on ManiSkill's controller gains
    action[:6] *= action_scale

    # Gripper mapping:
    # LIBERO: negative=closed, positive=open
    # ManiSkill pd_ee_delta_pose: -1=close, 1=open (same convention)
    # No change needed for gripper, but clip to valid range

    return np.clip(action, -1.0, 1.0).astype(np.float32)


# ── Full integration loop ─────────────────────────────────────────────

def run_episode(
    env,
    policy_client,
    prompt: str = "pick up the red cube",
    max_steps: int = 300,
    replan_steps: int = 10,
    action_scale: float = 1.0,
    verbose: bool = True,
):
    """Run one episode with π0.5 controlling ManiSkill Franka.

    Args:
        env: ManiSkill gymnasium environment.
        policy_client: openpi WebsocketClientPolicy connected to π0.5 server.
        prompt: Language instruction.
        max_steps: Maximum environment steps.
        replan_steps: Execute this many steps from each action chunk before replanning.
        action_scale: Action scaling factor (tune empirically).
        verbose: Print step info.

    Returns:
        Dict with 'frames' (list of RGB arrays), 'rewards' (list), 'success' (bool).
    """
    import torch
    from collections import deque

    obs, info = env.reset()
    action_plan = deque()
    frames = []
    rewards = []

    for step in range(max_steps):
        # Render frame
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        frames.append(frame)

        # Replan if action queue is empty
        if not action_plan:
            pi05_input = maniskill_obs_to_pi05(obs, frame, prompt)
            result = policy_client.infer(pi05_input)
            action_chunk = result["actions"]

            for i in range(min(replan_steps, len(action_chunk))):
                ms_action = pi05_action_to_maniskill(
                    action_chunk, step_index=i, action_scale=action_scale
                )
                action_plan.append(ms_action)

        # Execute next action
        action = action_plan.popleft()
        action_tensor = torch.tensor(action, dtype=torch.float32, device="cuda").unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action_tensor)
        rewards.append(float(reward))

        if verbose and step % 20 == 0:
            print(f"Step {step}: action_pos={action[:3].round(3)} rot={action[3:6].round(3)} grip={action[6]:.2f} reward={float(reward):.3f}")

        if terminated or truncated:
            frames.append(frame)
            if verbose:
                print(f"Episode ended at step {step}, final reward={float(reward):.3f}")
            break

    success = float(rewards[-1]) > 0.9 if rewards else False
    return {"frames": frames, "rewards": rewards, "success": success}
