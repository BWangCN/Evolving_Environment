"""Debug NaN in pi0.5 training — trace where NaN first appears."""

if __name__ == "__main__":
    import os
    os.environ["HF_HOME"] = "/scratch/bwang25/hf_cache"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["OPENPI_DATA_HOME"] = "/scratch/bwang25/openpi_cache"

    import numpy as np
    import jax
    import jax.numpy as jnp
    import dataclasses

    print("=== Step 1: Loading config ===")
    from openpi.training.config import get_config
    from openpi.training import data_loader as _data_loader

    config = get_config("pi05_maniskill_lora")
    config = dataclasses.replace(config, batch_size=4, num_workers=0)
    print(f"  use_quantile_norm in base_config: {config.data.base_config.use_quantile_norm}")

    print("\n=== Step 2: Loading one batch ===")
    loader = _data_loader.create_data_loader(config, shuffle=False)
    batch = next(iter(loader))

    print("\n=== Step 3: Checking batch for NaN ===")
    obs, actions = batch
    print(f"  Actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"  Actions has NaN: {jnp.any(jnp.isnan(actions))}")
    print(f"  Actions has Inf: {jnp.any(jnp.isinf(actions))}")
    print(f"  Actions min: {float(jnp.min(actions)):.4f}, max: {float(jnp.max(actions)):.4f}")
    print(f"  Actions sample [0,0,:7]: {actions[0, 0, :7]}")

    print(f"\n  State shape: {obs.state.shape}")
    print(f"  State has NaN: {jnp.any(jnp.isnan(obs.state))}")
    print(f"  State min: {float(jnp.min(obs.state)):.4f}, max: {float(jnp.max(obs.state)):.4f}")

    print(f"\n  Prompt shape: {obs.tokenized_prompt.shape}")
    print(f"  Prompt mask any True: {jnp.any(obs.tokenized_prompt_mask)}")

    for img_key, img_val in obs.images.items():
        has_nan = bool(jnp.any(jnp.isnan(img_val)))
        print(f"  images/{img_key}: shape={img_val.shape}, NaN={has_nan}, min={float(jnp.min(img_val)):.4f}, max={float(jnp.max(img_val)):.4f}")

    # Step 4: Check norm stats
    print("\n=== Step 4: Norm stats ===")
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"  use_quantile_norm: {data_config.use_quantile_norm}")
    if data_config.norm_stats:
        for key, ns in data_config.norm_stats.items():
            min_std = np.min(np.abs(ns.std))
            print(f"  {key}: min_std={min_std:.6f} {'DANGER' if min_std < 0.001 else 'OK'}")

    print("\n=== Done ===")
