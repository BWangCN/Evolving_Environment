"""Test training with old RL dataset to isolate NaN cause."""

if __name__ == "__main__":
    import os
    os.environ["HF_HOME"] = "/scratch/bwang25/hf_cache"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["OPENPI_DATA_HOME"] = "/scratch/bwang25/openpi_cache"
    os.environ["SSL_CERT_FILE"] = ""

    import subprocess
    result = subprocess.run(
        ["python", "-c", "import certifi; print(certifi.where())"],
        capture_output=True, text=True
    )
    os.environ["SSL_CERT_FILE"] = result.stdout.strip()

    import dataclasses
    from openpi.training.config import get_config, LeRobotManiSkillDataConfig, DataConfig

    c = get_config("pi05_maniskill_lora")

    # Check if old RL dataset exists
    old_path = "/scratch/bwang25/hf_cache/lerobot/bwang25/maniskill_pi05/meta/info.json"
    if os.path.exists(old_path):
        print("Old RL dataset found. Swapping...")
        c = dataclasses.replace(c, data=LeRobotManiSkillDataConfig(
            repo_id="bwang25/maniskill_pi05",
            base_config=DataConfig(prompt_from_task=True),
        ))
    else:
        print(f"Old RL dataset NOT found at {old_path}")
        print("Cannot test. Exiting.")
        exit(1)

    # Override for quick test
    c = dataclasses.replace(c, batch_size=4, num_workers=0, wandb_enabled=False)

    print(f"repo_id: {c.data.repo_id}")
    d = c.data.create(c.assets_dirs, c.model)
    print(f"use_quantile_norm: {d.use_quantile_norm}")
    print(f"norm_stats present: {d.norm_stats is not None}")

    print("\nIf this config works, the issue is with MP data.")
    print("If this also NaNs, the issue is with the config/code.")
