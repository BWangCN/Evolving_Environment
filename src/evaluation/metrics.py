"""EvoHome-Bench metrics computation.

Input:  Performance matrix P (N×N upper-triangular) where P[i][j] = success rate
        on environment E_j after adapting through E_i.
        Only P[i][j] for j <= i is valid (can't test on environments not yet seen).

Output: FTS (Forward Transfer Score), FR (Forgetting Rate), ZIC, CE, EvoHome Score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Complete benchmark evaluation result."""
    n_envs: int
    performance_matrix: np.ndarray         # (N, N), P[i][j] valid for j <= i
    fts: np.ndarray                        # (N,) Forward Transfer Scores (diagonal)
    fts_avg: float                         # Mean FTS
    forgetting_matrix: np.ndarray          # (N, N), FR[i][j] = P[j][j] - P[i][j] for i > j
    fr_avg: float                          # Average forgetting rate
    zic: int                               # Zero-Interaction Compliance (0 or 1)
    ce: Optional[np.ndarray] = None        # (N,) Compute Efficiency per env
    ce_avg: Optional[float] = None
    evohome_score: float = 0.0             # Composite score

    def summary(self) -> str:
        lines = [
            f"=== EvoHome-Bench Results ({self.n_envs} environments) ===",
            f"FTS (per-env):    {np.array2string(self.fts, precision=3)}",
            f"FTS (avg):        {self.fts_avg:.4f}",
            f"FR  (avg):        {self.fr_avg:.4f}",
            f"ZIC:              {self.zic}",
        ]
        if self.ce is not None:
            lines.append(f"CE  (per-env):    {np.array2string(self.ce, precision=3)}")
            lines.append(f"CE  (avg):        {self.ce_avg:.4f}")
        lines.append(f"EvoHome Score:    {self.evohome_score:.4f}")
        lines.append("")
        lines.append("Performance Matrix P[i][j] (row=adapted through E_i, col=tested on E_j):")
        lines.append(np.array2string(self.performance_matrix, precision=3))
        lines.append("")
        lines.append("Forgetting Matrix FR[i][j] = P[j][j] - P[i][j] for i > j:")
        lines.append(np.array2string(self.forgetting_matrix, precision=3))
        return "\n".join(lines)


class EvoHomeBenchMetrics:
    """Compute all EvoHome-Bench metrics from a performance matrix.

    Usage:
        metrics = EvoHomeBenchMetrics()
        result = metrics.compute(P, zic=1)
        print(result.summary())
    """

    def compute(
        self,
        performance_matrix: np.ndarray,
        zic: int = 1,
        gpu_hours: Optional[np.ndarray] = None,
    ) -> BenchmarkResult:
        """Compute all metrics.

        Args:
            performance_matrix: (N, N) matrix. P[i][j] = success rate on E_j after
                adapting through E_i. Values where j > i should be NaN or 0 (not yet seen).
            zic: 1 if zero-interaction (our method), 0 if per-object demos needed.
            gpu_hours: (N,) GPU hours spent adapting to each environment.
                If provided, compute efficiency CE is calculated.

        Returns:
            BenchmarkResult with all metrics.
        """
        P = np.array(performance_matrix, dtype=float)
        N = P.shape[0]
        assert P.shape == (N, N), f"Expected square matrix, got {P.shape}"

        # --- Forward Transfer Score (diagonal) ---
        fts = np.diag(P)
        fts_avg = float(np.mean(fts))

        # --- Forgetting Rate ---
        forgetting = np.zeros((N, N))
        fr_values = []
        for i in range(N):
            for j in range(i):
                # FR[i][j] = performance right after E_j minus performance after E_i
                fr_ij = fts[j] - P[i, j]
                forgetting[i, j] = fr_ij
                fr_values.append(fr_ij)

        fr_avg = float(np.mean(fr_values)) if fr_values else 0.0

        # --- Compute Efficiency ---
        ce = None
        ce_avg = None
        if gpu_hours is not None:
            gpu_hours = np.array(gpu_hours, dtype=float)
            assert len(gpu_hours) == N
            # CE = GPU-hours / FTS (lower is better)
            ce = np.where(fts > 0, gpu_hours / fts, np.inf)
            ce_avg = float(np.mean(ce[np.isfinite(ce)])) if np.any(np.isfinite(ce)) else float('inf')

        # --- EvoHome Score ---
        evohome_score = float(fts_avg * max(0, 1 - fr_avg) * zic)

        return BenchmarkResult(
            n_envs=N,
            performance_matrix=P,
            fts=fts,
            fts_avg=fts_avg,
            forgetting_matrix=forgetting,
            fr_avg=fr_avg,
            zic=zic,
            ce=ce,
            ce_avg=ce_avg,
            evohome_score=evohome_score,
        )

    def compare_methods(
        self,
        results: dict[str, BenchmarkResult],
    ) -> str:
        """Generate a comparison table across multiple methods.

        Args:
            results: dict mapping method name → BenchmarkResult.

        Returns:
            Formatted comparison string.
        """
        lines = ["=== Method Comparison ===", ""]
        header = f"{'Method':<25} {'FTS↑':>8} {'FR↓':>8} {'ZIC':>5} {'Score↑':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for name, r in sorted(results.items(), key=lambda x: -x[1].evohome_score):
            lines.append(
                f"{name:<25} {r.fts_avg:>8.3f} {r.fr_avg:>8.3f} {r.zic:>5d} {r.evohome_score:>8.3f}"
            )

        return "\n".join(lines)
