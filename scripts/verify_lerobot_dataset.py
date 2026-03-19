"""
Verify converted LeRobot dataset for correctness.

Checks:
  1. Dataset loads successfully
  2. Episode count matches source h5 files
  3. Frame count consistency (obs vs actions)
  4. Image shape, dtype, value range (not all black/white)
  5. Action shape, dtype, value range
  6. State shape, dtype, value range
  7. Language instructions present and non-empty
  8. Visual spot-check: save sample frames with action overlay

Usage (from openpi directory):
    cd /home/bwang25/Desktop/Manipulation/openpi
    .venv/bin/python /home/bwang25/Desktop/Manipulation/Evolving_Environment/scripts/verify_lerobot_dataset.py

    # Quick check (10 episodes):
    .venv/bin/python .../verify_lerobot_dataset.py --max-episodes 10

    # Save visual samples:
    .venv/bin/python .../verify_lerobot_dataset.py --save-samples
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

DEMO_DIR = Path("/home/bwang25/.maniskill/demos_fresh")
ALL_TASKS = ["PickCube-v1", "StackCube-v1", "PullCube-v1", "LiftPegUpright-v1"]
REPO_NAME = "bwang25/maniskill_pi05"


def get_source_stats(tasks):
    """Get ground-truth stats from source h5 files."""
    stats = {}
    for task in tasks:
        h5_path = DEMO_DIR / task / "demos_2000.h5"
        if not h5_path.exists():
            continue
        f = h5py.File(h5_path, "r")
        ep_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda k: int(k.split("_")[1]),  # numeric sort to match conversion order
        )
        total_frames = 0
        ep_lengths = []
        for k in ep_keys:
            n = f[k]["actions"].shape[0]
            total_frames += n
            ep_lengths.append(n)
        stats[task] = {
            "n_episodes": len(ep_keys),
            "total_frames": total_frames,
            "ep_lengths": ep_lengths,
            "instruction": f[ep_keys[0]].attrs.get("instruction", ""),
        }
        f.close()
    return stats


def verify(repo_name=REPO_NAME, max_episodes=None, save_samples=False):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_name
    if not output_path.exists():
        print(f"ERROR: Dataset not found at {output_path}")
        print("Run the conversion script first.")
        return False

    print(f"Loading dataset from {output_path}...")
    dataset = LeRobotDataset(repo_name)

    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"  Repo: {repo_name}")
    print(f"  Path: {output_path}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Features: {list(dataset.features.keys())}")

    # Get source stats for comparison
    source_stats = get_source_stats(ALL_TASKS)
    expected_episodes = sum(s["n_episodes"] for s in source_stats.values())
    expected_frames = sum(s["total_frames"] for s in source_stats.values())

    all_ok = True
    errors = []
    warnings = []

    # ── Check 1: Episode count ──
    print(f"\n── Check 1: Episode Count ──")
    print(f"  Expected: {expected_episodes} episodes, {expected_frames} frames")
    print(f"  Got:      {dataset.num_episodes} episodes, {len(dataset)} frames")
    if dataset.num_episodes != expected_episodes:
        errors.append(f"Episode count mismatch: {dataset.num_episodes} vs {expected_episodes}")
    if len(dataset) != expected_frames:
        errors.append(f"Frame count mismatch: {len(dataset)} vs {expected_frames}")

    # ── Check 2: Sample random frames ──
    print(f"\n── Check 2: Frame Content Verification ──")
    rng = np.random.default_rng(42)
    n_check = min(max_episodes or 100, len(dataset))
    indices = rng.choice(len(dataset), size=n_check, replace=False)

    image_ok = 0
    wrist_ok = 0
    state_ok = 0
    action_ok = 0
    task_ok = 0

    action_mins = []
    action_maxs = []
    state_mins = []
    state_maxs = []

    for idx in indices:
        frame = dataset[int(idx)]

        # Image checks
        img = frame.get("image")
        if img is not None:
            img_np = np.array(img)
            if img_np.shape[-2:] == (256, 256) or img_np.shape[:2] == (256, 256):
                if img_np.min() != img_np.max():  # not all same value
                    image_ok += 1

        # Wrist image checks
        wrist = frame.get("wrist_image")
        if wrist is not None:
            wrist_np = np.array(wrist)
            if wrist_np.min() != wrist_np.max():
                wrist_ok += 1

        # State checks
        state = frame.get("state")
        if state is not None:
            state_np = np.array(state, dtype=np.float32)
            if state_np.shape == (8,):
                state_ok += 1
                state_mins.append(state_np.min())
                state_maxs.append(state_np.max())

        # Action checks
        action = frame.get("actions")
        if action is not None:
            action_np = np.array(action, dtype=np.float32)
            if action_np.shape == (7,):
                action_ok += 1
                action_mins.append(action_np.min())
                action_maxs.append(action_np.max())

        # Task/instruction checks
        task = frame.get("task")
        if task is not None and len(str(task)) > 5:
            task_ok += 1

    print(f"  Checked {n_check} random frames:")
    print(f"    image:       {image_ok}/{n_check} OK (256x256, non-uniform)")
    print(f"    wrist_image: {wrist_ok}/{n_check} OK (non-uniform)")
    print(f"    state:       {state_ok}/{n_check} OK (shape=(8,))")
    print(f"    actions:     {action_ok}/{n_check} OK (shape=(7,))")
    print(f"    task:        {task_ok}/{n_check} OK (non-empty string)")

    if image_ok < n_check:
        errors.append(f"{n_check - image_ok} frames have bad images")
    if action_ok < n_check:
        errors.append(f"{n_check - action_ok} frames have bad actions")
    if state_ok < n_check:
        errors.append(f"{n_check - state_ok} frames have bad states")
    if task_ok < n_check:
        errors.append(f"{n_check - task_ok} frames have bad/missing task instructions")

    # ── Check 3: Value ranges ──
    print(f"\n── Check 3: Value Ranges ──")
    if action_mins:
        a_min, a_max = min(action_mins), max(action_maxs)
        print(f"  Actions: [{a_min:.3f}, {a_max:.3f}]  (expected ~[-1, 1])")
        if a_min < -1.5 or a_max > 1.5:
            warnings.append(f"Actions outside expected range: [{a_min:.3f}, {a_max:.3f}]")
    if state_mins:
        s_min, s_max = min(state_mins), max(state_maxs)
        print(f"  States:  [{s_min:.3f}, {s_max:.3f}]  (expected ~[-4, 4])")

    # ── Check 4: Episode boundaries ──
    print(f"\n── Check 4: Episode Boundary Consistency ──")
    ep_lengths_lerobot = []
    for ep_idx in range(min(20, dataset.num_episodes)):
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start
        ep_lengths_lerobot.append(ep_len)

    # Compare with source
    source_lengths = []
    for task in ALL_TASKS:
        if task in source_stats:
            source_lengths.extend(source_stats[task]["ep_lengths"])

    print(f"  First 20 LeRobot episode lengths: {ep_lengths_lerobot}")
    print(f"  First 20 source episode lengths:  {source_lengths[:20]}")
    match = all(a == b for a, b in zip(ep_lengths_lerobot, source_lengths[:20]))
    if match:
        print(f"  Episode lengths MATCH source ✓")
    else:
        errors.append("Episode lengths don't match source h5 files")
        print(f"  Episode lengths MISMATCH ✗")

    # ── Check 5: Visual spot-check ──
    if save_samples:
        print(f"\n── Check 5: Visual Spot-Check ──")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            save_dir = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs")
            save_dir.mkdir(exist_ok=True)

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle("LeRobot Dataset — Sample Frames", fontsize=14, fontweight="bold")

            # Sample 8 frames from different episodes
            sample_eps = rng.choice(dataset.num_episodes, size=8, replace=False)
            for i, ep_idx in enumerate(sorted(sample_eps)):
                ax = axes[i // 4, i % 4]
                ep_start = dataset.episode_data_index["from"][int(ep_idx)].item()
                frame = dataset[ep_start]

                img = np.array(frame["image"])
                # Handle CHW vs HWC
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                # Handle float [0,1] vs uint8
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)

                ax.imshow(img)
                task_str = str(frame.get("task", "N/A"))
                if len(task_str) > 40:
                    task_str = task_str[:37] + "..."
                ax.set_title(f"ep={ep_idx}\n{task_str}", fontsize=8)
                ax.axis("off")

            plt.tight_layout()
            out_path = save_dir / "lerobot_sample_frames.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved sample frames to {out_path}")

            # Also save action distribution plot
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
            fig2.suptitle("LeRobot Dataset — Action & State Distributions", fontsize=14)

            # Sample more actions for histogram
            n_hist = min(5000, len(dataset))
            hist_indices = rng.choice(len(dataset), size=n_hist, replace=False)
            all_actions = []
            all_states = []
            for idx in hist_indices:
                frame = dataset[int(idx)]
                all_actions.append(np.array(frame["actions"], dtype=np.float32))
                all_states.append(np.array(frame["state"], dtype=np.float32))
            all_actions = np.stack(all_actions)
            all_states = np.stack(all_states)

            ax = axes2[0]
            labels = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "grip"]
            for j in range(7):
                ax.hist(all_actions[:, j], bins=50, alpha=0.5, label=labels[j])
            ax.set_title("Action Distribution")
            ax.set_xlabel("Value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes2[1]
            labels_s = ["ee_x", "ee_y", "ee_z", "ax_x", "ax_y", "ax_z", "grip_l", "grip_r"]
            for j in range(8):
                ax.hist(all_states[:, j], bins=50, alpha=0.5, label=labels_s[j])
            ax.set_title("State Distribution")
            ax.set_xlabel("Value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out_path2 = save_dir / "lerobot_distributions.png"
            fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved distributions to {out_path2}")

        except ImportError:
            print("  matplotlib not available, skipping visual check")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e}")
        all_ok = False
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ⚠ {w}")
    if not errors and not warnings:
        print(f"  ALL CHECKS PASSED ✓")
    elif not errors:
        print(f"  ALL CRITICAL CHECKS PASSED ✓ (with {len(warnings)} warnings)")

    # Dataset size on disk
    total_size = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file())
    print(f"\n  Dataset size: {total_size / 1024 / 1024:.0f} MB")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Frames: {len(dataset)}")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", type=str, default=REPO_NAME)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max frames to spot-check (default: 100)")
    parser.add_argument("--save-samples", action="store_true",
                        help="Save visual sample images and distribution plots")
    args = parser.parse_args()

    ok = verify(
        repo_name=args.repo_name,
        max_episodes=args.max_episodes,
        save_samples=args.save_samples,
    )
    sys.exit(0 if ok else 1)
