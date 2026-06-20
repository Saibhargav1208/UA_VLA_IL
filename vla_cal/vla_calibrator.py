# UA-VLA-IL: VLACalibrator
# VLM-Adaptive Calibration for Language-Conditioned Imitation Learning Policies
#
# Replaces the fixed temperature T and fixed neighborhood threshold ω in the
# original paper with VLM-predicted values conditioned on the current
# visual observation and language instruction.
#
# Original paper (Bucher et al., IROS 2024):
#   T  = single scalar learned on 25 demos (same for every scene)
#   ω  = fixed threshold (hand-designed, same for every task)
#
# This extension:
#   T  = T_base + alpha  × complexity(obs)       ← adaptive per observation
#   ω  = omega_base × (1 - precision(task) + ε)  ← adaptive per task
#
# Both complexity and precision are predicted zero-shot by Qwen2-VL-2B.
# No additional training data required.
#
# Usage:
#   calibrator = VLACalibrator(qwen_port=12190)
#   T     = calibrator.predict_temperature(obs_rgb, task_instruction)
#   omega = calibrator.predict_omega(obs_rgb, task_instruction)
#
# Then plug T into TemperatureScaler and omega into ActionSelection
# (see uncertainty_quant_cliport and uncertainty_quant_peract patches).

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from vla_cal.qwen_vl_client import QwenVLClient

# ── Default hyperparameters ───────────────────────────────────────────────────
# These match the paper's values at complexity=0, precision=0 (neutral scene).
# Tunable via VLACalibrator constructor or config YAML.

# Temperature formula: T = T_BASE + ALPHA * complexity
T_BASE: float = float(os.environ.get("VLA_CAL_T_BASE", "1.0"))
ALPHA: float = float(os.environ.get("VLA_CAL_ALPHA", "1.5"))

# Omega formula: ω = OMEGA_BASE * (1 - precision + EPS)
# CLIPort kernel size (attn_tau / trans_tau): integer 1–11
OMEGA_BASE_CLIPORT: float = float(os.environ.get("VLA_CAL_OMEGA_BASE_CLIPORT", "7.0"))
# PerAct tau (neighborhood radius in voxel space): float
OMEGA_BASE_PERACT: float = float(os.environ.get("VLA_CAL_OMEGA_BASE_PERACT", "5.0"))
EPS: float = 0.05  # prevents omega from reaching 0

# Cache TTL: reuse the last VLM prediction for this many steps (saves compute)
CACHE_STEPS: int = int(os.environ.get("VLA_CAL_CACHE_STEPS", "5"))

# ─────────────────────────────────────────────────────────────────────────────


class VLACalibrator:
    """
    VLM-Adaptive Calibrator for uncertainty-aware imitation learning.

    Wraps QwenVLClient to predict per-step adaptive values for:
        - temperature T (controls calibration sharpness)
        - neighborhood threshold ω (controls action selection conservatism)

    Both are functions of the current RGB observation and task instruction.

    Args:
        qwen_port:    Port of the Qwen2-VL-2B server (default: 12190).
        model:        "cliport" or "peract" — sets the ω scale.
        t_base:       Base temperature (used when complexity = 0).
        alpha:        Sensitivity of T to visual complexity.
        omega_base:   Base neighborhood size (used when precision = 0).
        eps:          Minimum precision floor (prevents ω → 0).
        cache_steps:  Number of steps to reuse a cached VLM prediction.
        enabled:      If False, returns fixed T=t_base and ω=omega_base
                      (identical to the original paper's behavior).
    """

    def __init__(
        self,
        qwen_port: int = 12190,
        model: str = "cliport",
        t_base: float = T_BASE,
        alpha: float = ALPHA,
        omega_base: Optional[float] = None,
        eps: float = EPS,
        cache_steps: int = CACHE_STEPS,
        enabled: bool = True,
    ) -> None:
        self._client = QwenVLClient(port=qwen_port)
        self._model = model
        self._t_base = t_base
        self._alpha = alpha
        self._eps = eps
        self._cache_steps = cache_steps
        self.enabled = enabled

        if omega_base is None:
            self._omega_base = (
                OMEGA_BASE_CLIPORT if model == "cliport" else OMEGA_BASE_PERACT
            )
        else:
            self._omega_base = omega_base

        # Cache state
        self._cached_complexity: Optional[float] = None
        self._cached_precision: Optional[float] = None
        self._steps_since_update: int = cache_steps + 1  # force update on first call

        print(
            f"[VLACalibrator] Initialized. model={model}, "
            f"T=T_base({t_base}) + alpha({alpha})*complexity, "
            f"ω=omega_base({self._omega_base})*(1-precision+{eps}), "
            f"cache_steps={cache_steps}, enabled={enabled}"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _should_refresh(self) -> bool:
        return self._steps_since_update >= self._cache_steps

    def _refresh(self, obs_rgb: np.ndarray, task: str) -> None:
        """Query Qwen2-VL for both scores and cache them."""
        self._cached_complexity = self._client.estimate_complexity(obs_rgb)
        self._cached_precision = self._client.estimate_precision(obs_rgb, task)
        self._steps_since_update = 0
        print(
            f"[VLACalibrator] VLM update: "
            f"complexity={self._cached_complexity:.3f}, "
            f"precision={self._cached_precision:.3f}"
        )

    def _get_scores(
        self, obs_rgb: np.ndarray, task: str
    ) -> Tuple[float, float]:
        """Return (complexity, precision), refreshing cache if needed."""
        if not self.enabled:
            return 0.0, 0.0  # neutral → original paper behavior

        if self._should_refresh():
            self._refresh(obs_rgb, task)
        self._steps_since_update += 1

        return self._cached_complexity, self._cached_precision  # type: ignore

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_temperature(self, obs_rgb: np.ndarray, task: str) -> float:
        """
        Predict adaptive temperature T for this observation.

        Formula: T = T_base + alpha × complexity(obs)

        High complexity → larger T → more aggressive logit smoothing →
        model less likely to pick an isolated spike caused by a distractor.

        Args:
            obs_rgb: Current RGB observation, shape (H, W, 3), dtype uint8.
            task:    Language instruction string.

        Returns:
            float — temperature T ≥ T_base.
        """
        complexity, _ = self._get_scores(obs_rgb, task)
        T = self._t_base + self._alpha * complexity
        print(f"[VLACalibrator] T={T:.3f} (base={self._t_base}, α×c={self._alpha * complexity:.3f})")
        return float(T)

    def predict_omega(self, obs_rgb: np.ndarray, task: str) -> float:
        """
        Predict adaptive neighborhood threshold ω for this task.

        Formula: ω = omega_base × (1 - precision + eps)

        High precision required → smaller ω → tighter neighborhood →
        action selection is more precise, not artificially smoothed.

        Low precision required (approximate ok) → larger ω → wider
        neighborhood → more conservative, robust action selection.

        For CLIPort: ω maps to kernel size (attn_tau / trans_tau).
                     Returned value is rounded to nearest odd integer in [1, 11].
        For PerAct:  ω maps to tau (voxel neighborhood radius, float).

        Args:
            obs_rgb: Current RGB observation, shape (H, W, 3), dtype uint8.
            task:    Language instruction string.

        Returns:
            float — omega ω > 0.
        """
        _, precision = self._get_scores(obs_rgb, task)
        omega = self._omega_base * (1.0 - precision + self._eps)

        if self._model == "cliport":
            # CLIPort uses kernel size (must be odd integer ≥ 1)
            omega_int = max(1, int(round(omega)))
            if omega_int % 2 == 0:
                omega_int += 1  # enforce odd kernel size
            omega = float(omega_int)

        print(
            f"[VLACalibrator] ω={omega:.3f} "
            f"(base={self._omega_base}, p={precision:.3f})"
        )
        return omega

    def predict_both(
        self, obs_rgb: np.ndarray, task: str
    ) -> Tuple[float, float]:
        """
        Predict both T and ω in a single VLM query (uses cache).

        Returns:
            (T, omega)
        """
        complexity, precision = self._get_scores(obs_rgb, task)

        T = self._t_base + self._alpha * complexity
        omega = self._omega_base * (1.0 - precision + self._eps)

        if self._model == "cliport":
            omega_int = max(1, int(round(omega)))
            if omega_int % 2 == 0:
                omega_int += 1
            omega = float(omega_int)

        return float(T), omega

    def reset(self) -> None:
        """Force cache refresh on next step (call at episode start)."""
        self._steps_since_update = self._cache_steps + 1
        self._cached_complexity = None
        self._cached_precision = None

    @property
    def last_complexity(self) -> Optional[float]:
        return self._cached_complexity

    @property
    def last_precision(self) -> Optional[float]:
        return self._cached_precision
