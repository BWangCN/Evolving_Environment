"""Competence-Aware Adaptive Replay Scheduling (CARS).

Dynamically allocates replay budget across previous environments based on
per-environment competence scores. Environments where the model is degrading
get more replay; environments where the model is still competent get less.

Supports decoupled scheduling: separate competence scores for VLM (perception)
and action decoder (motor), with independent replay decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ReplaySchedule:
    """Replay allocation for one training step."""
    # Fraction of batch allocated to each environment's replay
    # env_id → fraction (sums to replay_fraction)
    allocations: dict[str, float]
    # Fraction of batch for new environment data
    new_data_fraction: float
    # Fraction of batch for replay (sum of allocations)
    replay_fraction: float
    # Which environments are skipped (competence above threshold)
    skipped_envs: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"New data: {self.new_data_fraction:.1%}"]
        for env_id, frac in sorted(self.allocations.items()):
            lines.append(f"  Replay {env_id}: {frac:.1%}")
        if self.skipped_envs:
            lines.append(f"  Skipped (competent): {', '.join(self.skipped_envs)}")
        return "\n".join(lines)


class CARSScheduler:
    """Competence-Aware Adaptive Replay Scheduling.

    Usage:
        scheduler = CARSScheduler(competence_threshold=0.75, replay_fraction=0.3)

        # After evaluating model on old environments:
        scheduler.update_competence("E1", perception=0.80, action=0.65)
        scheduler.update_competence("E2", perception=0.70, action=0.72)

        # Get replay schedule for current training batch:
        schedule = scheduler.get_schedule()
        # → E1 gets more action replay (0.65 < 0.75)
        # → E2 gets more perception replay (0.70 < 0.75)
    """

    def __init__(
        self,
        competence_threshold: float = 0.75,
        replay_fraction: float = 0.30,
        min_replay_per_env: float = 0.02,
    ):
        """
        Args:
            competence_threshold: Environments with competence above this get no replay.
            replay_fraction: Total fraction of training batch allocated to replay.
            min_replay_per_env: Minimum replay fraction for any environment that needs replay.
        """
        self.threshold = competence_threshold
        self.replay_fraction = replay_fraction
        self.min_replay = min_replay_per_env

        # env_id → {"perception": score, "action": score}
        self._competence: dict[str, dict[str, float]] = {}

    def update_competence(
        self,
        env_id: str,
        perception: Optional[float] = None,
        action: Optional[float] = None,
        overall: Optional[float] = None,
    ):
        """Update competence scores for an environment.

        Args:
            env_id: Environment identifier.
            perception: VLM perception competence (0-1).
            action: Action decoder competence (0-1).
            overall: If provided, used for both perception and action.
        """
        if overall is not None:
            perception = overall
            action = overall

        if env_id not in self._competence:
            self._competence[env_id] = {"perception": 1.0, "action": 1.0}

        if perception is not None:
            self._competence[env_id]["perception"] = np.clip(perception, 0.0, 1.0)
        if action is not None:
            self._competence[env_id]["action"] = np.clip(action, 0.0, 1.0)

    def get_schedule(self) -> ReplaySchedule:
        """Compute replay allocation based on current competence scores.

        Allocation is proportional to the competence deficit:
            need_j = max(0, threshold - combined_score_j)
        Environments above threshold are skipped.
        """
        if not self._competence:
            return ReplaySchedule(
                allocations={},
                new_data_fraction=1.0,
                replay_fraction=0.0,
            )

        # Compute per-environment replay need
        needs: dict[str, float] = {}
        skipped: list[str] = []

        for env_id, scores in self._competence.items():
            # Combined competence: take the minimum of perception and action
            # (weakest link determines need)
            combined = min(scores["perception"], scores["action"])
            need = max(0.0, self.threshold - combined)

            if need <= 0:
                skipped.append(env_id)
            else:
                needs[env_id] = need

        if not needs:
            # All environments are competent, no replay needed
            return ReplaySchedule(
                allocations={},
                new_data_fraction=1.0,
                replay_fraction=0.0,
                skipped_envs=sorted(skipped),
            )

        # Normalize needs to sum to replay_fraction
        total_need = sum(needs.values())
        allocations = {}
        for env_id, need in needs.items():
            raw_frac = (need / total_need) * self.replay_fraction
            allocations[env_id] = max(raw_frac, self.min_replay)

        # Re-normalize if min_replay pushed total above replay_fraction
        total_alloc = sum(allocations.values())
        if total_alloc > self.replay_fraction:
            scale = self.replay_fraction / total_alloc
            allocations = {k: v * scale for k, v in allocations.items()}
            total_alloc = self.replay_fraction

        return ReplaySchedule(
            allocations=allocations,
            new_data_fraction=1.0 - total_alloc,
            replay_fraction=total_alloc,
            skipped_envs=sorted(skipped),
        )

    def get_decoupled_schedule(self) -> dict[str, ReplaySchedule]:
        """Get separate replay schedules for VLM and action decoder.

        Returns dict with keys "perception" and "action", each a ReplaySchedule.
        This implements the asymmetric treatment described in the pipeline doc:
        VLM and decoder have different forgetting patterns, so replay decisions
        are made independently.
        """
        result = {}

        for component in ["perception", "action"]:
            needs: dict[str, float] = {}
            skipped: list[str] = []

            for env_id, scores in self._competence.items():
                score = scores[component]
                need = max(0.0, self.threshold - score)
                if need <= 0:
                    skipped.append(env_id)
                else:
                    needs[env_id] = need

            if not needs:
                result[component] = ReplaySchedule(
                    allocations={},
                    new_data_fraction=1.0,
                    replay_fraction=0.0,
                    skipped_envs=sorted(skipped),
                )
                continue

            total_need = sum(needs.values())
            allocations = {}
            for env_id, need in needs.items():
                raw_frac = (need / total_need) * self.replay_fraction
                allocations[env_id] = max(raw_frac, self.min_replay)

            total_alloc = sum(allocations.values())
            if total_alloc > self.replay_fraction:
                scale = self.replay_fraction / total_alloc
                allocations = {k: v * scale for k, v in allocations.items()}
                total_alloc = self.replay_fraction

            result[component] = ReplaySchedule(
                allocations=allocations,
                new_data_fraction=1.0 - total_alloc,
                replay_fraction=total_alloc,
                skipped_envs=sorted(skipped),
            )

        return result

    @property
    def competence_scores(self) -> dict[str, dict[str, float]]:
        """Return a copy of all competence scores."""
        return {k: dict(v) for k, v in self._competence.items()}

    def reset(self):
        """Clear all competence scores."""
        self._competence.clear()
