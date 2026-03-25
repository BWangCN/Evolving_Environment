"""Check norm stats and dataset for NaN values."""

import json
import os
import numpy as np

# Check norm stats
print("=== Checking norm stats ===")
ns_path = os.path.expanduser("~/openpi/assets/pi05_maniskill_lora/bwang25/maniskill_pi05_mp/norm_stats.json")
with open(ns_path) as f:
    d = json.load(f)
for k, v in d["norm_stats"].items():
    for stat, vals in v.items():
        for i, x in enumerate(vals):
            if x != x:
                print(f"  NaN found in {k}.{stat}[{i}]")
print("Norm stats check done")

# Check dataset
print("\n=== Checking dataset for NaN ===")
os.environ["HF_HOME"] = "/scratch/bwang25/hf_cache"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("bwang25/maniskill_pi05_mp")
nan_count = 0
for i in range(min(100, len(ds))):
    sample = ds[i]
    for k, v in sample.items():
        if hasattr(v, "numpy"):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            if np.any(np.isnan(v)):
                print(f"  NaN at sample {i}, key {k}")
                nan_count += 1
            if np.any(np.isinf(v)):
                print(f"  Inf at sample {i}, key {k}")
                nan_count += 1
print(f"Data check done ({nan_count} issues found)")
