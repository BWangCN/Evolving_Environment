#!/bin/bash
# ManiSkill dataset generation — uses RL demos already in pd_ee_delta_pose
# Replays with --use-env-states for 100% success rate + adds RGB observations
#
# Run: nohup bash scripts/generate_maniskill_dataset.sh > logs/dataset_generation.log 2>&1 &

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate gaussian_grouping

export CUDA_HOME=/usr/local/cuda

DEMODIR="/home/bwang25/.maniskill/demos"

# All use RL demos in pd_ee_delta_pose (100% replay success with --use-env-states)
TASK_LIST=(
    "PickCube-v1|$DEMODIR/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5|$DEMODIR/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.json"
    "StackCube-v1|$DEMODIR/StackCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5|$DEMODIR/StackCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.json"
    "PullCube-v1|$DEMODIR/PullCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5|$DEMODIR/PullCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.json"
    "LiftPegUpright-v1|$DEMODIR/LiftPegUpright-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5|$DEMODIR/LiftPegUpright-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.json"
)

TIERS=(10 50 200 500 1000)

echo "========================================="
echo "ManiSkill Dataset Generation (RL demos)"
echo "$(date)"
echo "Tasks: PickCube StackCube PullCube LiftPegUpright"
echo "Tiers: ${TIERS[@]}"
echo "========================================="

TOTAL_START=$SECONDS

for ENTRY in "${TASK_LIST[@]}"; do
    IFS='|' read -r TASK TRAJ_H5 TRAJ_JSON <<< "$ENTRY"

    echo ""
    echo "=== $TASK ==="

    AVAILABLE=$(python -c "
import h5py
f = h5py.File('$TRAJ_H5', 'r')
n = len([k for k in f.keys() if k.startswith('traj_')])
f.close()
print(n)
")
    echo "Available: $AVAILABLE episodes"

    for N in "${TIERS[@]}"; do
        USE_N=$N
        [ "$USE_N" -gt "$AVAILABLE" ] && USE_N=$AVAILABLE

        TIER_DIR="$DEMODIR/$TASK/tier_${USE_N}"

        # Skip if already rendered
        if ls "$TIER_DIR"/*.rgb.* 1>/dev/null 2>&1; then
            echo "  [$N] Already rendered, skipping"
            continue
        fi

        mkdir -p "$TIER_DIR"

        # Extract N episodes + copy json metadata
        python -c "
import h5py, json
src = h5py.File('$TRAJ_H5', 'r')
dst = h5py.File('$TIER_DIR/trajectory.h5', 'w')
for attr in src.attrs:
    dst.attrs[attr] = src.attrs[attr]
count = 0
for key in sorted(src.keys()):
    if key.startswith('traj_') and count < $USE_N:
        src.copy(key, dst, name=f'traj_{count}')
        count += 1
src.close()
dst.close()
with open('$TRAJ_JSON', 'r') as f:
    meta = json.load(f)
meta['num_episodes'] = count
if 'episodes' in meta:
    meta['episodes'] = meta['episodes'][:count]
with open('$TIER_DIR/trajectory.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'  [$N] Extracted {count} episodes')
"

        # Replay with RGB + env states (no control mode conversion needed)
        echo "  [$N] Rendering RGB... ($(date))"
        STEP_START=$SECONDS

        python -m mani_skill.trajectory.replay_trajectory \
            --traj-path "$TIER_DIR/trajectory.h5" \
            --save-traj \
            --obs-mode rgb \
            --use-env-states \
            --sim-backend gpu \
            --count "$USE_N" \
            2>&1 | tail -3

        ELAPSED=$(( SECONDS - STEP_START ))

        RENDERED=$(ls "$TIER_DIR"/*.rgb.*.h5 2>/dev/null | head -1)
        if [ -n "$RENDERED" ]; then
            SIZE=$(ls -lh "$RENDERED" | awk '{print $5}')
            NEPS=$(python -c "
import h5py
f = h5py.File('$RENDERED', 'r')
n = len([k for k in f.keys() if k.startswith('traj_')])
f.close()
print(n)
")
            echo "  [$N] Done in ${ELAPSED}s — $NEPS episodes, $SIZE"
        else
            echo "  [$N] FAILED in ${ELAPSED}s — no output"
        fi
    done
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))

echo ""
echo "========================================="
echo "COMPLETE — ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min)"
echo "$(date)"
echo "========================================="

echo ""
echo "Summary:"
for ENTRY in "${TASK_LIST[@]}"; do
    IFS='|' read -r TASK _ _ <<< "$ENTRY"
    echo "  $TASK:"
    for d in "$DEMODIR/$TASK"/tier_*/; do
        [ -d "$d" ] || continue
        RENDERED=$(ls "$d"*.rgb.*.h5 2>/dev/null | head -1)
        if [ -n "$RENDERED" ]; then
            SIZE=$(ls -lh "$RENDERED" | awk '{print $5}')
            NEPS=$(python -c "import h5py; f=h5py.File('$RENDERED','r'); print(len([k for k in f.keys() if k.startswith('traj_')])); f.close()")
            echo "    $(basename $d): $NEPS eps, $SIZE"
        fi
    done
done

echo ""
echo "Total disk:"
du -sh "$DEMODIR"
