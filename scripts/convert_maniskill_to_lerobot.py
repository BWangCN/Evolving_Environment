"""
Convert ManiSkill fresh demo h5 files to LeRobot v2.1 format for π0.5 fine-tuning.

Input:  ~/.maniskill/demos_fresh/<task>/demos_2000.h5
Output: LeRobot dataset at $HF_LEROBOT_HOME/<repo_name>/

The output matches openpi's expected format (same as LIBERO conversion):
  - image: (256, 256, 3) uint8 — third-person camera
  - wrist_image: (256, 256, 3) uint8 — duplicated from image (no wrist cam in ManiSkill default)
  - state: (8,) float32 — [ee_pos(3), ee_axis_angle(3), gripper_qpos(2)]
  - actions: (7,) float32 — [dx, dy, dz, drx, dry, drz, gripper]
  - task: str — language instruction

Usage (run from openpi directory with openpi's .venv):
    cd /home/bwang25/Desktop/Manipulation/openpi
    .venv/bin/python /home/bwang25/Desktop/Manipulation/Evolving_Environment/scripts/convert_maniskill_to_lerobot.py

    # Or for a specific task:
    .venv/bin/python .../convert_maniskill_to_lerobot.py --tasks PickCube-v1

    # With custom demo count (for ablation):
    .venv/bin/python .../convert_maniskill_to_lerobot.py --max-episodes 200
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

DEMO_DIR = Path("/home/bwang25/.maniskill/demos_fresh")

ALL_TASKS = ["PickCube-v1", "StackCube-v1", "PullCube-v1", "LiftPegUpright-v1"]

REPO_NAME = "bwang25/maniskill_pi05"  # Local dataset name


def convert(
    tasks: list[str],
    repo_name: str = REPO_NAME,
    max_episodes: int | None = None,
    demo_filename: str = "demos_2000.h5",
    demo_dir: Path | None = None,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset matching LIBERO format exactly
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=20,  # ManiSkill control frequency
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_episodes = 0
    total_frames = 0

    for task in tasks:
        base_dir = demo_dir if demo_dir is not None else DEMO_DIR
        h5_path = base_dir / task / demo_filename
        if not h5_path.exists():
            print(f"WARNING: {h5_path} not found, skipping {task}")
            continue

        print(f"\nConverting {task}...")
        f = h5py.File(h5_path, "r")

        # Count available episodes
        ep_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        n_episodes = len(ep_keys)
        if max_episodes is not None:
            n_episodes = min(n_episodes, max_episodes)
        print(f"  Episodes: {n_episodes} (of {len(ep_keys)} available)")

        for ep_idx in range(n_episodes):
            ep = f[f"traj_{ep_idx}"]
            rgb = ep["rgb"][:]          # (T+1, 256, 256, 3) uint8
            actions = ep["actions"][:]  # (T, 7) float32
            state = ep["state"][:]      # (T+1, 8) float32
            instruction = ep.attrs.get("instruction", f"complete the {task} task")
            n_steps = actions.shape[0]

            # Add each timestep as a frame
            # LeRobot convention: frame t has observation at t and action at t
            for t in range(n_steps):
                dataset.add_frame(
                    {
                        "image": rgb[t],            # observation at time t
                        "wrist_image": rgb[t],      # duplicate (no wrist cam)
                        "state": state[t],           # state at time t
                        "actions": actions[t],       # action taken at time t
                        "task": instruction,
                    }
                )

            dataset.save_episode()
            total_episodes += 1
            total_frames += n_steps

            if (ep_idx + 1) % 200 == 0:
                print(f"  [{ep_idx + 1}/{n_episodes}] episodes converted")

        f.close()
        print(f"  Done: {n_episodes} episodes, {total_frames} total frames so far")

    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_path}")
    print(f"  Size: {sum(p.stat().st_size for p in output_path.rglob('*') if p.is_file()) / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        help="Tasks to convert (default: all)")
    parser.add_argument("--repo-name", type=str, default=REPO_NAME,
                        help="LeRobot dataset repo name")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max episodes per task (for ablation, e.g., 10, 50, 200)")
    parser.add_argument("--demo-filename", type=str, default="demos_2000.h5",
                        help="Demo h5 filename within each task directory")
    parser.add_argument("--demo-dir", type=str, default=None,
                        help="Override demo directory (default: ~/.maniskill/demos_fresh)")
    args = parser.parse_args()

    convert(
        tasks=args.tasks,
        repo_name=args.repo_name,
        max_episodes=args.max_episodes,
        demo_filename=args.demo_filename,
        demo_dir=Path(args.demo_dir) if args.demo_dir else None,
    )
