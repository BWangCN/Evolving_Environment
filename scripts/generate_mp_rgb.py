"""
Re-render converted MP trajectories with RGB images.

Reads the pd_ee_delta_pose converted trajectories (state-only),
replays them in ManiSkill, captures RGB frames, and saves HDF5 files
compatible with convert_maniskill_to_lerobot.py.

Usage:
    conda activate gaussian_grouping
    python Evolving_Environment/scripts/generate_mp_rgb.py
"""

import os
import sys
import time
import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image

import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.adapters.pi05_maniskill import quat_to_axis_angle

DEMO_DIR = Path(os.path.expanduser("~/.maniskill/demos"))
OUTPUT_DIR = Path(os.path.expanduser("~/.maniskill/demos_mp"))

TASKS = {
    "PickCube-v1": "pick up the red cube and move it to the green target",
    "StackCube-v1": "pick up the red cube and stack it on the green cube",
}

IMG_SIZE = 256


def render_frame(env):
    """Get RGB frame from env.render()."""
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame[0].cpu().numpy()
    elif frame.ndim == 4:
        frame = frame[0]
    return np.array(Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


def extract_state(obs):
    """Extract LIBERO-format state [ee_pos(3), ee_axis_angle(3), gripper_qpos(2)]."""
    tcp = obs["extra"]["tcp_pose"]
    qpos = obs["agent"]["qpos"]
    if hasattr(tcp, "cpu"):
        tcp = tcp[0].cpu().numpy()
    if hasattr(qpos, "cpu"):
        qpos = qpos[0].cpu().numpy()

    ee_pos = tcp[:3].astype(np.float32)
    ee_aa = quat_to_axis_angle(tcp[3:7].astype(np.float32))
    gripper_qpos = qpos[7:9].astype(np.float32)
    return np.concatenate([ee_pos, ee_aa, gripper_qpos])


def process_task(task, instruction):
    """Replay converted trajectories and capture RGB."""
    converted_path = DEMO_DIR / task / "motionplanning" / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    env_states_path = DEMO_DIR / task / "motionplanning" / "trajectory.h5"

    if not converted_path.exists():
        print(f"  {task}: no converted trajectory found, skipping")
        return

    # Load converted trajectories (has actions in pd_ee_delta_pose)
    conv = h5py.File(converted_path, "r")
    # Load original trajectories (has env_states for resetting)
    orig = h5py.File(env_states_path, "r")

    ep_keys = sorted([k for k in conv.keys() if k.startswith("traj_")])
    print(f"  {task}: {len(ep_keys)} converted episodes")

    # Create environment
    env = gym.make(
        task,
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        max_episode_steps=300,
        human_render_camera_configs={"shader_pack": "default"},
    )

    # Output
    output_path = OUTPUT_DIR / task / f"demos_{len(ep_keys)}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = h5py.File(output_path, "w")

    successes = 0
    total_frames = 0
    start_time = time.time()

    for ep_idx, ep_key in enumerate(ep_keys):
        conv_ep = conv[ep_key]
        orig_ep = orig[ep_key]
        actions = conv_ep["actions"][:]
        n_steps = actions.shape[0]

        # Reset env using original env_states (stored as nested HDF5 groups)
        obs, info = env.reset()
        env_states_grp = orig_ep["env_states"]

        # Build state dict matching ManiSkill's expected format
        state_dict = {}
        for category in env_states_grp.keys():  # 'actors', 'articulations'
            state_dict[category] = {}
            cat_grp = env_states_grp[category]
            for name in cat_grp.keys():  # e.g., 'cube', 'panda'
                data = cat_grp[name][:]  # (T+1, dim)
                state_dict[category][name] = torch.tensor(
                    data[0:1], dtype=torch.float32, device=env.unwrapped.device
                )
        env.unwrapped.set_state_dict(state_dict)
        obs = env.unwrapped.get_obs()

        # Collect frames and states
        frames = []
        states = []

        # Initial observation
        frames.append(render_frame(env))
        states.append(extract_state(obs))

        # Step through actions
        episode_success = False
        valid_actions = []
        for t in range(n_steps):
            action_tensor = torch.tensor(
                actions[t], dtype=torch.float32
            ).unsqueeze(0).to(env.unwrapped.device)

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            frames.append(render_frame(env))
            states.append(extract_state(obs))
            valid_actions.append(actions[t])

            if terminated or truncated:
                success_val = info.get("success", False)
                if isinstance(success_val, torch.Tensor):
                    success_val = success_val[0].item()
                episode_success = bool(success_val)
                break

        if len(valid_actions) < 3:
            continue

        # Save episode in same format as generate_fresh_demos.py
        grp = out.create_group(f"traj_{successes}")
        grp.create_dataset("rgb", data=np.array(frames[:len(valid_actions) + 1], dtype=np.uint8))
        grp.create_dataset("actions", data=np.array(valid_actions, dtype=np.float32))
        grp.create_dataset("state", data=np.array(states[:len(valid_actions) + 1], dtype=np.float32))
        grp.attrs["instruction"] = instruction
        grp.attrs["num_steps"] = len(valid_actions)
        grp.attrs["success"] = episode_success

        successes += 1
        total_frames += len(valid_actions)

        if (ep_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep_idx + 1) / elapsed
            print(f"    [{ep_idx + 1}/{len(ep_keys)}] {successes} saved, "
                  f"{total_frames} frames, {eps_per_sec:.1f} ep/s")

    out.attrs["total_episodes"] = successes
    out.attrs["total_frames"] = total_frames
    out.attrs["control_mode"] = "pd_ee_delta_pose"
    out.close()
    conv.close()
    orig.close()
    env.close()

    elapsed = time.time() - start_time
    print(f"  Done: {successes} episodes, {total_frames} frames, "
          f"{elapsed:.0f}s → {output_path}")


if __name__ == "__main__":
    print("=== Generating MP demos with RGB ===\n")
    for task, instruction in TASKS.items():
        try:
            process_task(task, instruction)
        except Exception as e:
            print(f"  {task} FAILED: {e}")
            import traceback
            traceback.print_exc()
    print("\nAll tasks done.")
