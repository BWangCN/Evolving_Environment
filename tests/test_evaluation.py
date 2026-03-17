"""Tests for EvoHome-Bench metrics and CARS scheduling."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import EvoHomeBenchMetrics
from src.evaluation.cars import CARSScheduler


class TestMetrics:
    def _make_sample_matrix(self):
        """A realistic 5-env performance matrix with moderate forgetting."""
        return np.array([
            [0.85, 0.00, 0.00, 0.00, 0.00],
            [0.70, 0.80, 0.00, 0.00, 0.00],
            [0.65, 0.75, 0.82, 0.00, 0.00],
            [0.60, 0.72, 0.78, 0.77, 0.00],
            [0.55, 0.68, 0.74, 0.73, 0.81],
        ])

    def test_fts(self):
        P = self._make_sample_matrix()
        result = EvoHomeBenchMetrics().compute(P)
        np.testing.assert_allclose(result.fts, [0.85, 0.80, 0.82, 0.77, 0.81])
        assert 0.7 < result.fts_avg < 0.9

    def test_forgetting_rate(self):
        P = self._make_sample_matrix()
        result = EvoHomeBenchMetrics().compute(P)
        # FR[1][0] = P[0][0] - P[1][0] = 0.85 - 0.70 = 0.15
        assert abs(result.forgetting_matrix[1, 0] - 0.15) < 1e-6
        # FR[4][0] = P[0][0] - P[4][0] = 0.85 - 0.55 = 0.30
        assert abs(result.forgetting_matrix[4, 0] - 0.30) < 1e-6
        # Average forgetting should be positive (there is forgetting)
        assert result.fr_avg > 0

    def test_zic(self):
        P = self._make_sample_matrix()
        result_zic1 = EvoHomeBenchMetrics().compute(P, zic=1)
        result_zic0 = EvoHomeBenchMetrics().compute(P, zic=0)
        assert result_zic1.zic == 1
        assert result_zic0.zic == 0
        # ZIC=0 should zero out the EvoHome score
        assert result_zic0.evohome_score == 0.0
        assert result_zic1.evohome_score > 0

    def test_compute_efficiency(self):
        P = self._make_sample_matrix()
        gpu_hours = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
        result = EvoHomeBenchMetrics().compute(P, gpu_hours=gpu_hours)
        assert result.ce is not None
        # CE[0] = 2.0 / 0.85 ≈ 2.35
        assert abs(result.ce[0] - 2.0 / 0.85) < 0.01
        assert result.ce_avg > 0

    def test_evohome_score(self):
        P = self._make_sample_matrix()
        result = EvoHomeBenchMetrics().compute(P, zic=1)
        # Score = mean(FTS) × (1 - FR_avg) × ZIC
        expected = result.fts_avg * (1 - result.fr_avg) * 1
        assert abs(result.evohome_score - expected) < 1e-6

    def test_no_forgetting(self):
        """Perfect model that never forgets."""
        P = np.array([
            [0.90, 0.00, 0.00],
            [0.90, 0.88, 0.00],
            [0.90, 0.88, 0.85],
        ])
        result = EvoHomeBenchMetrics().compute(P)
        assert result.fr_avg == 0.0
        assert result.evohome_score == result.fts_avg

    def test_catastrophic_forgetting(self):
        """Model that completely forgets old environments."""
        P = np.array([
            [0.90, 0.00, 0.00],
            [0.00, 0.85, 0.00],
            [0.00, 0.00, 0.80],
        ])
        result = EvoHomeBenchMetrics().compute(P)
        # FR should be very high (forgot everything)
        assert result.fr_avg > 0.8
        # EvoHome score should be very low
        assert result.evohome_score < 0.2

    def test_single_environment(self):
        P = np.array([[0.75]])
        result = EvoHomeBenchMetrics().compute(P)
        assert result.fts_avg == 0.75
        assert result.fr_avg == 0.0
        assert result.evohome_score == 0.75

    def test_summary_string(self):
        P = self._make_sample_matrix()
        result = EvoHomeBenchMetrics().compute(P, zic=1)
        s = result.summary()
        assert "EvoHome" in s
        assert "FTS" in s
        assert "FR" in s

    def test_compare_methods(self):
        metrics = EvoHomeBenchMetrics()
        P_ours = self._make_sample_matrix()
        P_naive = np.array([
            [0.85, 0.00, 0.00, 0.00, 0.00],
            [0.30, 0.82, 0.00, 0.00, 0.00],
            [0.10, 0.25, 0.80, 0.00, 0.00],
            [0.05, 0.10, 0.20, 0.78, 0.00],
            [0.02, 0.05, 0.08, 0.15, 0.79],
        ])
        results = {
            "Ours (Full)": metrics.compute(P_ours, zic=1),
            "Naive Finetuning": metrics.compute(P_naive, zic=1),
        }
        table = metrics.compare_methods(results)
        assert "Ours (Full)" in table
        assert "Naive Finetuning" in table
        # Our method should rank higher
        lines = table.strip().split("\n")
        # First data line (after header and separator) should be our method
        data_lines = [l for l in lines if l.strip() and not l.startswith("=") and not l.startswith("-") and "Method" not in l]
        assert data_lines[0].strip().startswith("Ours")


class TestCARS:
    def test_no_environments(self):
        scheduler = CARSScheduler()
        schedule = scheduler.get_schedule()
        assert schedule.new_data_fraction == 1.0
        assert len(schedule.allocations) == 0

    def test_all_competent(self):
        """All environments above threshold → no replay needed."""
        scheduler = CARSScheduler(competence_threshold=0.75)
        scheduler.update_competence("E1", overall=0.90)
        scheduler.update_competence("E2", overall=0.85)
        schedule = scheduler.get_schedule()
        assert schedule.replay_fraction == 0.0
        assert len(schedule.allocations) == 0
        assert "E1" in schedule.skipped_envs
        assert "E2" in schedule.skipped_envs

    def test_one_degraded(self):
        """One environment below threshold → gets all replay budget."""
        scheduler = CARSScheduler(competence_threshold=0.75, replay_fraction=0.30)
        scheduler.update_competence("E1", overall=0.90)  # competent
        scheduler.update_competence("E2", overall=0.50)  # degraded
        schedule = scheduler.get_schedule()
        assert "E1" in schedule.skipped_envs
        assert "E2" in schedule.allocations
        assert abs(schedule.allocations["E2"] - 0.30) < 0.01  # gets full replay budget

    def test_proportional_allocation(self):
        """Two degraded environments → replay proportional to deficit."""
        scheduler = CARSScheduler(competence_threshold=0.80, replay_fraction=0.30)
        scheduler.update_competence("E1", overall=0.60)  # deficit = 0.20
        scheduler.update_competence("E2", overall=0.40)  # deficit = 0.40
        schedule = scheduler.get_schedule()
        # E2 has 2x the deficit of E1, so should get ~2x the replay
        assert schedule.allocations["E2"] > schedule.allocations["E1"]
        # Total should be ~0.30
        total = sum(schedule.allocations.values())
        assert abs(total - 0.30) < 0.05

    def test_new_data_fraction(self):
        scheduler = CARSScheduler(replay_fraction=0.30)
        scheduler.update_competence("E1", overall=0.50)
        schedule = scheduler.get_schedule()
        assert abs(schedule.new_data_fraction + schedule.replay_fraction - 1.0) < 1e-6

    def test_decoupled_schedule(self):
        """VLM and decoder can have different replay needs."""
        scheduler = CARSScheduler(competence_threshold=0.75, replay_fraction=0.30)
        scheduler.update_competence("E1", perception=0.90, action=0.50)  # action degraded
        scheduler.update_competence("E2", perception=0.40, action=0.85)  # perception degraded

        decoupled = scheduler.get_decoupled_schedule()

        # Perception schedule: E1 should be skipped, E2 needs replay
        assert "E1" in decoupled["perception"].skipped_envs
        assert "E2" in decoupled["perception"].allocations

        # Action schedule: E2 should be skipped, E1 needs replay
        assert "E2" in decoupled["action"].skipped_envs
        assert "E1" in decoupled["action"].allocations

    def test_update_competence(self):
        scheduler = CARSScheduler()
        scheduler.update_competence("E1", perception=0.80, action=0.60)
        scores = scheduler.competence_scores
        assert scores["E1"]["perception"] == 0.80
        assert scores["E1"]["action"] == 0.60

    def test_reset(self):
        scheduler = CARSScheduler()
        scheduler.update_competence("E1", overall=0.50)
        scheduler.reset()
        assert len(scheduler.competence_scores) == 0

    def test_clipping(self):
        scheduler = CARSScheduler()
        scheduler.update_competence("E1", overall=1.5)  # should clip to 1.0
        scheduler.update_competence("E2", overall=-0.3)  # should clip to 0.0
        scores = scheduler.competence_scores
        assert scores["E1"]["perception"] == 1.0
        assert scores["E2"]["action"] == 0.0

    def test_schedule_summary(self):
        scheduler = CARSScheduler(competence_threshold=0.75, replay_fraction=0.30)
        scheduler.update_competence("E1", overall=0.90)
        scheduler.update_competence("E2", overall=0.50)
        scheduler.update_competence("E3", overall=0.60)
        schedule = scheduler.get_schedule()
        s = schedule.summary()
        assert "New data" in s
        assert "E1" in s  # should appear in skipped


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
