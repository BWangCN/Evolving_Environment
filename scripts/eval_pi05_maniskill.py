"""Evaluate finetuned pi0.5 on ManiSkill PickCube.

Runs multiple episodes and prints raw action values for debugging action scale.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from openpi_client import websocket_client_policy as wcp
from src.adapters.pi05_maniskill import run_episode, maniskill_obs_to_pi05
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

# --- First: inspect raw action magnitudes ---
print("=== Inspecting raw pi0.5 action output ===")
env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    num_envs=1,
    sim_backend="cpu",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=300,
)
obs, info = env.reset(seed=42)
frame = env.render()
if isinstance(frame, torch.Tensor):
    frame = frame.cpu().numpy()
if frame.ndim == 4:
    frame = frame[0]

client = wcp.WebsocketClientPolicy("localhost", 8000)
pi05_input = maniskill_obs_to_pi05(obs, frame, "pick up the red cube and move it to the green target")
result = client.infer(pi05_input)
raw_actions = result["actions"]
print(f"Action chunk shape: {raw_actions.shape}")
print(f"Action step 0 (raw): {raw_actions[0].round(4)}")
print(f"Action pos range: [{raw_actions[:, :3].min():.4f}, {raw_actions[:, :3].max():.4f}]")
print(f"Action rot range: [{raw_actions[:, 3:6].min():.4f}, {raw_actions[:, 3:6].max():.4f}]")
print(f"Action grip range: [{raw_actions[:, 6].min():.4f}, {raw_actions[:, 6].max():.4f}]")

# --- Determine appropriate action_scale ---
# ManiSkill pd_ee_delta_pose expects [-1, 1]
# If raw actions are ~0.01-0.1 (meters), scale ~1.0 is fine
# If raw actions are already ~1.0, we need scale ~0.1
pos_max = np.abs(raw_actions[:, :3]).max()
print(f"\nMax abs position delta: {pos_max:.4f}")
if pos_max > 0.5:
    suggested_scale = 0.1
elif pos_max > 0.1:
    suggested_scale = 0.3
else:
    suggested_scale = 1.0
print(f"Suggested action_scale: {suggested_scale}")

# --- Run evaluation ---
# Training data uses pd_ee_delta_pose actions already in [-1, 1]
# No additional scaling needed (action_scale=1.0)
scale = 1.0
print(f"\n=== Evaluating with action_scale={scale}, max_steps=300 ===")
successes = 0
n_episodes = 5
for ep in range(n_episodes):
    result = run_episode(
        env, client,
        prompt="pick up the red cube and move it to the green target",
        max_steps=300,
        replan_steps=5,
        action_scale=scale,
        verbose=(ep == 0),
    )
    successes += int(result["success"])
    print(f"  Episode {ep}: success={result['success']}, steps={len(result['rewards'])}, final_reward={result['rewards'][-1]:.3f}")
print(f"  Success rate: {successes}/{n_episodes}")

env.close()
print("\nDone.")
