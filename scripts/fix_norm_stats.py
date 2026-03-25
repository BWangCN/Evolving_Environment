"""Fix norm stats: widen tiny quantile ranges to prevent extreme scaling.

The MP data has very small rotation deltas (dims 3-4: q01=-0.04, q99=0.04).
Quantile normalization maps this range to [-1, 1], creating 25x amplification.
Fix: enforce a minimum quantile range of 0.2 for all dimensions.
"""

if __name__ == "__main__":
    import json
    import os
    import numpy as np
    from pathlib import Path

    ns_path = Path(os.path.expanduser(
        "~/openpi/assets/pi05_maniskill_lora/bwang25/maniskill_pi05_mp/norm_stats.json"
    ))

    with open(ns_path) as f:
        d = json.load(f)

    MIN_RANGE = 0.2  # Minimum q99-q01 range

    for key in ["actions", "state"]:
        stats = d["norm_stats"][key]
        q01 = np.array(stats["q01"])
        q99 = np.array(stats["q99"])
        ranges = q99 - q01

        print(f"\n{key}:")
        for i, (lo, hi, r) in enumerate(zip(q01, q99, ranges)):
            status = "FIXED" if r < MIN_RANGE else "ok"
            if r < MIN_RANGE:
                # Widen symmetrically around the midpoint
                mid = (lo + hi) / 2
                q01[i] = mid - MIN_RANGE / 2
                q99[i] = mid + MIN_RANGE / 2
            print(f"  dim {i}: [{lo:.4f}, {hi:.4f}] range={r:.4f} → [{q01[i]:.4f}, {q99[i]:.4f}] {status}")

        stats["q01"] = q01.tolist()
        stats["q99"] = q99.tolist()

    # Save
    with open(ns_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"\nSaved fixed norm stats to {ns_path}")
