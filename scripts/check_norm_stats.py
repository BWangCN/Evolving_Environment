"""Check norm stats and verify use_quantile_norm setting."""

import json
import os

# Check config
print("=== Config check ===")
from openpi.training.config import get_config
c = get_config("pi05_maniskill_lora")
d = c.data.create("assets", c.model)
print(f"use_quantile_norm: {d.use_quantile_norm}")
print(f"repo_id: {d.repo_id}")

# Check norm stats
print("\n=== Norm stats ===")
for root, dirs, files in os.walk("assets/pi05_maniskill_lora"):
    for f in files:
        if f == "norm_stats.json":
            path = os.path.join(root, f)
            print(f"Found: {path}")
            ns = json.load(open(path))
            for key in ["actions", "state"]:
                stats = ns["norm_stats"][key]
                print(f"\n  {key}:")
                print(f"    mean: {[round(x,4) for x in stats['mean']]}")
                print(f"    std:  {[round(x,4) for x in stats['std']]}")
                print(f"    q01:  {[round(x,4) for x in stats['q01']]}")
                print(f"    q99:  {[round(x,4) for x in stats['q99']]}")

print("\n=== Done ===")
