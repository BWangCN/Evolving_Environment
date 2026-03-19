"""Generate ManiSkill demos by running pre-trained RL policies with RGB rendering.

Features:
- Fresh rollouts (no replay issues)
- Configurable image resolution (default 256x256 for VLA training)
- Language instructions per task
- h5py gzip compression
- All fields needed for π0.5 fine-tuning (rgb, actions, tcp_pose, qpos, language)

Usage:
    python scripts/generate_fresh_demos.py --task PickCube-v1 --num-demos 2000
    python scripts/generate_fresh_demos.py --task all --num-demos 2000
"""

import argparse
import json
import os
import time

import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ── Language instructions per task ────────────────────────────────────

TASK_INSTRUCTIONS = {
    "PickCube-v1": "pick up the red cube and move it to the green target",
    "StackCube-v1": "pick up the red cube and stack it on the green cube",
    "PullCube-v1": "push the blue cube onto the red and white target",
    "LiftPegUpright-v1": "pick up the peg and stand it upright",
}

ALL_TASKS = list(TASK_INSTRUCTIONS.keys())


# ── Policy loader ─────────────────────────────────────────────────────

class MLPPolicy(nn.Module):
    """MLP policy matching ManiSkill's PPO baseline (Tanh activations)."""

    def __init__(self, layers_list):
        super().__init__()
        self.net = nn.Sequential(*layers_list)

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        layers = []
        i = 0
        while f"actor_mean.{i}.weight" in ckpt:
            w = ckpt[f"actor_mean.{i}.weight"]
            b = ckpt[f"actor_mean.{i}.bias"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data = w
            layer.bias.data = b
            layers.append(layer)
            if f"actor_mean.{i+2}.weight" in ckpt:
                layers.append(nn.Tanh())
            i += 2
        policy = cls(layers).to(device)
        policy.eval()
        return policy

    def forward(self, obs):
        return self.net(obs)


# ── Quaternion to axis-angle ──────────────────────────────────────────

def quat_to_axis_angle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3-dim)."""
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    x, y, z, w = quat
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


# ── Demo generation ───────────────────────────────────────────────────

def generate_demos(task, num_demos, ckpt_path, output_path, max_steps=50,
                   img_size=256, device="cuda"):
    """Generate demos by running RL policy with RGB rendering."""

    instruction = TASK_INSTRUCTIONS.get(task, f"complete the {task} task")
    print(f"Task: {task}")
    print(f"Instruction: '{instruction}'")
    print(f"Loading policy from {ckpt_path}")
    policy = MLPPolicy.from_checkpoint(ckpt_path, device=device)

    print(f"Creating environment (max_steps={max_steps}, img={img_size}x{img_size})...")
    env = gym.make(
        task,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        human_render_camera_configs={"shader_pack": "rt-fast"},
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h5file = h5py.File(output_path, "w")

    successes = 0
    attempts = 0
    start_time = time.time()
    demo_idx = 0

    while demo_idx < num_demos:
        obs, info = env.reset()

        frames = []
        actions_list = []
        tcp_poses = []
        qpos_list = []

        # Initial observation
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame[0].cpu().numpy()
        elif frame.ndim == 4:
            frame = frame[0]
        # Resize
        frame = np.array(Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR))
        frames.append(frame)

        # Get initial state
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'agent'):
            tcp = unwrapped.agent.tcp.pose.raw_pose[0].cpu().numpy()
            tcp_poses.append(tcp)
            qp = unwrapped.agent.robot.qpos[0].cpu().numpy()
            qpos_list.append(qp)

        success = False
        for step in range(max_steps):
            with torch.no_grad():
                action = policy(obs.cuda()).clamp(-1, 1)

            obs, reward, terminated, truncated, info = env.step(action)

            # Record action
            actions_list.append(action[0].cpu().numpy().astype(np.float32))

            # Record frame
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame[0].cpu().numpy()
            elif frame.ndim == 4:
                frame = frame[0]
            frame = np.array(Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR))
            frames.append(frame)

            # Record state
            if hasattr(unwrapped, 'agent'):
                tcp = unwrapped.agent.tcp.pose.raw_pose[0].cpu().numpy()
                tcp_poses.append(tcp)
                qp = unwrapped.agent.robot.qpos[0].cpu().numpy()
                qpos_list.append(qp)

            if terminated or truncated:
                s = info.get("success", False)
                if isinstance(s, torch.Tensor):
                    s = s.item()
                success = bool(s)
                break

        attempts += 1

        if not success:
            if attempts % 200 == 0:
                print(f"  Attempt {attempts}: {demo_idx}/{num_demos} demos collected")
            continue

        successes += 1

        # Save to h5 with compression
        grp = h5file.create_group(f"traj_{demo_idx}")
        grp.create_dataset("actions", data=np.array(actions_list), compression="gzip", compression_opts=4)
        grp.create_dataset("rgb", data=np.array(frames, dtype=np.uint8), compression="gzip", compression_opts=4)
        if tcp_poses:
            grp.create_dataset("tcp_pose", data=np.array(tcp_poses, dtype=np.float32), compression="gzip")
        if qpos_list:
            grp.create_dataset("qpos", data=np.array(qpos_list, dtype=np.float32), compression="gzip")

        # Compute LIBERO-style state: [ee_pos(3), ee_axis_angle(3), gripper_qpos(2)]
        if tcp_poses and qpos_list:
            states = []
            for tcp, qp in zip(tcp_poses, qpos_list):
                ee_pos = tcp[:3]
                ee_quat = tcp[3:]  # x,y,z,w from SAPIEN
                ee_aa = quat_to_axis_angle(ee_quat)
                gripper = qp[7:9]
                state = np.concatenate([ee_pos, ee_aa, gripper]).astype(np.float32)
                states.append(state)
            grp.create_dataset("state", data=np.array(states), compression="gzip")

        grp.attrs["success"] = True
        grp.attrs["num_steps"] = len(actions_list)
        grp.attrs["instruction"] = instruction

        demo_idx += 1
        elapsed = time.time() - start_time
        eps = demo_idx / elapsed
        eta = (num_demos - demo_idx) / eps if eps > 0 else 0

        if demo_idx % 100 == 0 or demo_idx == num_demos:
            print(
                f"  [{demo_idx}/{num_demos}] "
                f"success={successes}/{attempts} ({successes/attempts*100:.0f}%) "
                f"speed={eps:.1f}/s ETA={eta/60:.1f}min"
            )

    # File-level metadata
    h5file.attrs["task"] = task
    h5file.attrs["instruction"] = instruction
    h5file.attrs["num_demos"] = num_demos
    h5file.attrs["control_mode"] = "pd_ee_delta_pose"
    h5file.attrs["obs_mode"] = "rgb"
    h5file.attrs["img_size"] = img_size
    h5file.attrs["max_episode_steps"] = max_steps
    h5file.attrs["control_freq_hz"] = 20
    h5file.attrs["action_dim"] = 7
    h5file.attrs["action_description"] = "delta_ee_pose: [dx,dy,dz,drx,dry,drz,gripper]"
    h5file.attrs["state_dim"] = 8
    h5file.attrs["state_description"] = "ee_pos(3) + ee_axis_angle(3) + gripper_qpos(2)"
    h5file.attrs["success_rate"] = successes / attempts if attempts > 0 else 0
    h5file.close()

    total_time = time.time() - start_time
    fsize = os.path.getsize(output_path) / 1024 / 1024 / 1024
    print(f"\nDone! {num_demos} demos in {total_time/60:.1f}min")
    print(f"Success rate: {successes}/{attempts} = {successes/attempts*100:.1f}%")
    print(f"Saved to: {output_path} ({fsize:.1f} GB)")

    env.close()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickCube-v1",
                        help="Task name or 'all' for all tasks")
    parser.add_argument("--num-demos", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str,
                        default="/home/bwang25/.maniskill/demos_fresh")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tasks = ALL_TASKS if args.task == "all" else [args.task]

    for task in tasks:
        ckpt_path = f"/home/bwang25/.maniskill/demos/{task}/rl/ppo_pd_ee_delta_pose_ckpt.pt"
        if not os.path.exists(ckpt_path):
            print(f"WARNING: No checkpoint for {task}, skipping")
            continue
        output_path = os.path.join(args.output_dir, task, f"demos_{args.num_demos}.h5")
        generate_demos(
            task=task,
            num_demos=args.num_demos,
            ckpt_path=ckpt_path,
            output_path=output_path,
            max_steps=args.max_steps,
            img_size=args.img_size,
            device=args.device,
        )
        print()
