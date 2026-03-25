"""
Generate smooth demonstration data from ManiSkill motion planning trajectories.

Replays motion planning demos with pd_ee_delta_pose control and renders RGB.
Produces HDF5 files compatible with convert_maniskill_to_lerobot.py.

Usage:
    conda activate gaussian_grouping
    python Evolving_Environment/scripts/generate_mp_demos.py
"""

import os
import time
import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image

import gymnasium as gym
import mani_skill.envs

DEMO_DIR = Path(os.path.expanduser("~/.maniskill/demos"))
OUTPUT_DIR = Path(os.path.expanduser("~/.maniskill/demos_mp"))

TASKS = [
    "PickCube-v1",
    "StackCube-v1",
    "PullCube-v1",
    "LiftPegUpright-v1",
]

TASK_INSTRUCTIONS = {
    "PickCube-v1": "pick up the red cube and move it to the green target",
    "StackCube-v1": "pick up the red cube and stack it on the green cube",
    "PullCube-v1": "push the blue cube onto the red and white target",
    "LiftPegUpright-v1": "pick up the peg and stand it upright",
}

IMG_SIZE = 256
MAX_DEMOS = 1000


def replay_trajectory(task, max_demos=MAX_DEMOS):
    """Replay motion planning trajectories with pd_ee_delta_pose and capture RGB."""

    traj_path = DEMO_DIR / task / "motionplanning" / "trajectory.h5"
    if not traj_path.exists():
        print(f"  {task}: no motion planning demos found at {traj_path}")
        return

    instruction = TASK_INSTRUCTIONS.get(task, f"complete the {task} task")
    print(f"\nTask: {task}")
    print(f"Instruction: '{instruction}'")

    # Load source trajectories
    src = h5py.File(traj_path, "r")
    ep_keys = sorted([k for k in src.keys() if k.startswith("traj_")])
    n_episodes = min(len(ep_keys), max_demos)
    print(f"  Source episodes: {len(ep_keys)}, using {n_episodes}")

    # Create environment with delta EE control
    env = gym.make(
        task,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        max_episode_steps=200,
        human_render_camera_configs={"shader_pack": "default"},
    )

    # Output HDF5
    output_path = OUTPUT_DIR / task / f"demos_{n_episodes}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = h5py.File(output_path, "w")

    successes = 0
    total_frames = 0

    for ep_idx in range(n_episodes):
        ep = src[ep_keys[ep_idx]]

        # Get env states for replay
        if "env_states" not in ep:
            print(f"  Episode {ep_idx}: no env_states, skipping")
            continue

        env_states = ep["env_states"][:]
        src_actions = ep["actions"][:]
        n_steps = src_actions.shape[0]

        # Reset to initial state
        obs, info = env.reset()
        env.unwrapped.set_state_dict(
            {k: torch.tensor(v[0:1], device=env.unwrapped.device)
             for k, v in _parse_env_state(env_states[0]).items()}
        )

        frames = []
        delta_actions = []
        states = []
        tcp_poses = []

        # Capture initial frame
        frame = _render_frame(env, IMG_SIZE)
        frames.append(frame)
        state, tcp = _extract_state(obs)
        states.append(state)
        tcp_poses.append(tcp)

        # Replay using env states to compute delta actions
        episode_success = False
        for t in range(n_steps):
            # Set to state t, get observation
            if t + 1 < len(env_states):
                target_state = env_states[t + 1]
                current_state = env_states[t]

                # Compute EE delta from consecutive env states
                current_tcp = tcp_poses[-1]
                env.unwrapped.set_state_dict(
                    {k: torch.tensor(v[0:1], device=env.unwrapped.device)
                     for k, v in _parse_env_state(target_state).items()}
                )
                obs_next = env.unwrapped.get_obs()
                next_state, next_tcp = _extract_state(obs_next)

                # Delta position
                dp = next_tcp[:3] - current_tcp[:3]
                # Delta rotation (approximate from axis-angle diff)
                from src.adapters.pi05_maniskill import quat_to_axis_angle
                cur_aa = quat_to_axis_angle(current_tcp[3:7])
                nxt_aa = quat_to_axis_angle(next_tcp[3:7])
                dr = nxt_aa - cur_aa

                # Gripper from source actions
                gripper = src_actions[t, -1] if src_actions.shape[1] > 7 else src_actions[t, 6]

                # Normalize to [-1, 1] range for pd_ee_delta_pose
                # ManiSkill pd_ee_delta_pose has internal scaling
                # Position: typically ±0.1m per step maps to ±1.0
                # Rotation: typically ±0.5rad per step maps to ±1.0
                action = np.concatenate([
                    np.clip(dp / 0.05, -1, 1),      # pos scale
                    np.clip(dr / 0.25, -1, 1),       # rot scale
                    [np.clip(gripper, -1, 1)],        # gripper
                ]).astype(np.float32)

                delta_actions.append(action)

                # Step env with computed delta action
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action_tensor)

                frame = _render_frame(env, IMG_SIZE)
                frames.append(frame)
                state, tcp = _extract_state(obs)
                states.append(state)
                tcp_poses.append(tcp)

                if terminated or truncated:
                    episode_success = bool(info.get("success", [False])[0]) if isinstance(info.get("success"), (list, torch.Tensor)) else bool(info.get("success", False))
                    break

        if len(delta_actions) < 3:
            continue

        # Save episode
        grp = out.create_group(f"traj_{successes}")
        grp.create_dataset("rgb", data=np.array(frames[:len(delta_actions) + 1], dtype=np.uint8))
        grp.create_dataset("actions", data=np.array(delta_actions, dtype=np.float32))
        grp.create_dataset("state", data=np.array(states[:len(delta_actions) + 1], dtype=np.float32))
        grp.create_dataset("tcp_pose", data=np.array(tcp_poses[:len(delta_actions) + 1], dtype=np.float32))
        grp.attrs["instruction"] = instruction
        grp.attrs["num_steps"] = len(delta_actions)
        grp.attrs["success"] = episode_success

        successes += 1
        total_frames += len(delta_actions)

        if (ep_idx + 1) % 100 == 0:
            print(f"  [{ep_idx + 1}/{n_episodes}] converted, {successes} successful, {total_frames} frames")

    out.attrs["total_episodes"] = successes
    out.attrs["total_frames"] = total_frames
    out.attrs["control_mode"] = "pd_ee_delta_pose"
    out.close()
    src.close()
    env.close()

    print(f"  Done: {successes} episodes, {total_frames} frames → {output_path}")


def _render_frame(env, img_size):
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame[0].cpu().numpy()
    elif frame.ndim == 4:
        frame = frame[0]
    return np.array(Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR))


def _extract_state(obs):
    """Extract LIBERO-format state from ManiSkill obs."""
    if isinstance(obs, dict):
        if "extra" in obs and "tcp_pose" in obs["extra"]:
            tcp = obs["extra"]["tcp_pose"]
        else:
            tcp = obs.get("tcp_pose", np.zeros(7))
        if "agent" in obs and "qpos" in obs["agent"]:
            qpos = obs["agent"]["qpos"]
        else:
            qpos = obs.get("qpos", np.zeros(9))
    else:
        return np.zeros(8, dtype=np.float32), np.zeros(7, dtype=np.float32)

    if hasattr(tcp, "cpu"):
        tcp = tcp[0].cpu().numpy()
    if hasattr(qpos, "cpu"):
        qpos = qpos[0].cpu().numpy()

    from src.adapters.pi05_maniskill import quat_to_axis_angle
    ee_pos = tcp[:3].astype(np.float32)
    ee_aa = quat_to_axis_angle(tcp[3:7].astype(np.float32))
    gripper_qpos = qpos[7:9].astype(np.float32)
    state = np.concatenate([ee_pos, ee_aa, gripper_qpos])
    return state, tcp.astype(np.float32)


def _parse_env_state(state_array):
    """Parse env state array into dict format for set_state_dict."""
    # This is task-dependent; for simple envs, env_states stores the full sim state
    # We'll use ManiSkill's built-in replay mechanism instead
    return {"env_state": state_array}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    for task in TASKS:
        try:
            replay_trajectory(task)
        except Exception as e:
            print(f"  {task} failed: {e}")
            import traceback
            traceback.print_exc()
