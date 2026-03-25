#!/bin/bash
# Step 1: Replay MP trajectories with pd_ee_delta_pose conversion (no RGB, fast)
# Step 2: Re-render with RGB using generate_mp_rgb.py
# Step 3: Convert to LeRobot format

set -e
CONDA_BIN=/home/bwang25/miniconda3/envs/gaussian_grouping/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASKS=("PickCube-v1" "StackCube-v1")
COUNT=1000

echo "=== Step 1: Replay MP trajectories with pd_ee_delta_pose ==="
for task in "${TASKS[@]}"; do
    traj_path="$HOME/.maniskill/demos/$task/motionplanning/trajectory.h5"
    output="$HOME/.maniskill/demos/$task/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    if [ -f "$output" ] && [ $(stat -c%s "$output") -gt 1000000 ]; then
        echo "  $task: already converted, skipping"
    else
        echo "  $task: replaying $COUNT episodes..."
        $CONDA_BIN -m mani_skill.trajectory.replay_trajectory \
            --traj-path "$traj_path" \
            -c pd_ee_delta_pose \
            -o state \
            --save-traj \
            --count $COUNT
    fi
done

echo ""
echo "=== Step 2: Re-render with RGB ==="
$CONDA_BIN "$SCRIPT_DIR/generate_mp_rgb.py"

echo ""
echo "=== Step 3: Convert to LeRobot ==="
cd /home/bwang25/Desktop/Manipulation/openpi
.venv/bin/python "$SCRIPT_DIR/convert_maniskill_to_lerobot.py" --demo-dir "$HOME/.maniskill/demos_mp" --tasks PickCube-v1 StackCube-v1

echo ""
echo "=== Done! ==="
