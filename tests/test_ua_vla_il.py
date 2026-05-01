# UA-VLA-IL Tests
# Tests for VLACalibrator and QwenVLClient.
# All Qwen model calls are mocked — no GPU or server required.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _rgb(h=64, w=64):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for _extract_float in qwen_vl_client
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractFloat:
    def _f(self, text):
        from vla_cal.qwen_vl_client import _extract_float
        return _extract_float(text)

    def test_plain_decimal(self):
        assert self._f("0.73") == pytest.approx(0.73)

    def test_in_sentence(self):
        assert self._f("The complexity score is 0.85 out of 1.") == pytest.approx(0.85)

    def test_clamp_above_one(self):
        assert self._f("1.8") == pytest.approx(1.0)

    def test_clamp_below_zero(self):
        assert self._f("-0.5") == pytest.approx(0.0)

    def test_no_number_returns_neutral(self):
        assert self._f("I cannot determine") == pytest.approx(0.5)

    def test_empty_string(self):
        assert self._f("") == pytest.approx(0.5)

    def test_integer_one(self):
        assert self._f("1") == pytest.approx(1.0)

    def test_integer_zero(self):
        assert self._f("0") == pytest.approx(0.0)

    def test_always_in_range(self):
        from vla_cal.qwen_vl_client import _extract_float
        for text in ["0.0", "1.0", "2.5", "-1", "", "yes", "NaN", "0.5 and 0.9"]:
            val = _extract_float(text)
            assert 0.0 <= val <= 1.0, f"Out of range for: {repr(text)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for VLACalibrator (QwenVLClient mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestVLACalibratorTemperature:
    """Tests for predict_temperature()."""

    def _make(self, complexity=0.5, precision=0.5, **kwargs):
        from vla_cal.vla_calibrator import VLACalibrator
        cal = VLACalibrator.__new__(VLACalibrator)
        cal._t_base = kwargs.get("t_base", 1.0)
        cal._alpha = kwargs.get("alpha", 1.5)
        cal._omega_base = kwargs.get("omega_base", 7.0)
        cal._eps = kwargs.get("eps", 0.05)
        cal._cache_steps = 1
        cal._steps_since_update = 999
        cal._cached_complexity = None
        cal._cached_precision = None
        cal._model = kwargs.get("model", "cliport")
        cal.enabled = True
        # Mock client
        client = MagicMock()
        client.estimate_complexity.return_value = complexity
        client.estimate_precision.return_value = precision
        cal._client = client
        return cal

    def test_temperature_formula(self):
        cal = self._make(complexity=0.6, t_base=1.0, alpha=1.5)
        T = cal.predict_temperature(_rgb(), "stack the block")
        assert T == pytest.approx(1.0 + 1.5 * 0.6)

    def test_high_complexity_raises_temperature(self):
        cal_low = self._make(complexity=0.1)
        cal_high = self._make(complexity=0.9)
        T_low = cal_low.predict_temperature(_rgb(), "task")
        T_high = cal_high.predict_temperature(_rgb(), "task")
        assert T_high > T_low

    def test_zero_complexity_returns_base(self):
        cal = self._make(complexity=0.0, t_base=1.2, alpha=2.0)
        T = cal.predict_temperature(_rgb(), "task")
        assert T == pytest.approx(1.2)

    def test_temperature_always_at_least_t_base(self):
        cal = self._make(complexity=0.0, t_base=1.0, alpha=1.5)
        T = cal.predict_temperature(_rgb(), "task")
        assert T >= cal._t_base

    def test_disabled_returns_base_temperature(self):
        from vla_cal.vla_calibrator import VLACalibrator
        cal = VLACalibrator.__new__(VLACalibrator)
        cal._t_base = 1.3
        cal._alpha = 2.0
        cal._omega_base = 7.0
        cal._eps = 0.05
        cal._cache_steps = 5
        cal._steps_since_update = 999
        cal._cached_complexity = None
        cal._cached_precision = None
        cal._model = "cliport"
        cal.enabled = False
        cal._client = MagicMock()
        T = cal.predict_temperature(_rgb(), "task")
        # disabled: complexity=0 → T = t_base
        assert T == pytest.approx(1.3)


class TestVLACalibratorOmega:
    """Tests for predict_omega()."""

    def _make(self, complexity=0.5, precision=0.5, model="cliport", omega_base=7.0):
        from vla_cal.vla_calibrator import VLACalibrator
        cal = VLACalibrator.__new__(VLACalibrator)
        cal._t_base = 1.0
        cal._alpha = 1.5
        cal._omega_base = omega_base
        cal._eps = 0.05
        cal._cache_steps = 1
        cal._steps_since_update = 999
        cal._cached_complexity = None
        cal._cached_precision = None
        cal._model = model
        cal.enabled = True
        client = MagicMock()
        client.estimate_complexity.return_value = complexity
        client.estimate_precision.return_value = precision
        cal._client = client
        return cal

    def test_omega_formula_cliport(self):
        # precision=0 → ω = omega_base * (1 - 0 + 0.05) = omega_base * 1.05
        cal = self._make(precision=0.0, omega_base=7.0)
        omega = cal.predict_omega(_rgb(), "put block in box")
        # Result is rounded to odd int
        expected_raw = 7.0 * (1.0 - 0.0 + 0.05)
        expected_int = int(round(expected_raw))
        if expected_int % 2 == 0:
            expected_int += 1
        assert omega == float(expected_int)

    def test_high_precision_reduces_omega(self):
        cal_low = self._make(precision=0.1)
        cal_high = self._make(precision=0.9)
        omega_low = cal_low.predict_omega(_rgb(), "task")
        omega_high = cal_high.predict_omega(_rgb(), "task")
        # Lower precision requirement → larger omega (wider neighborhood)
        assert omega_low > omega_high

    def test_cliport_omega_is_odd_integer(self):
        for precision in [0.0, 0.3, 0.5, 0.8, 1.0]:
            cal = self._make(precision=precision, model="cliport")
            omega = cal.predict_omega(_rgb(), "task")
            assert omega == int(omega), "CLIPort omega must be integer"
            assert int(omega) % 2 == 1, f"CLIPort omega must be odd, got {omega}"

    def test_cliport_omega_at_least_one(self):
        # Even with max precision, omega must be >= 1
        cal = self._make(precision=1.0, omega_base=1.0)
        omega = cal.predict_omega(_rgb(), "insert peg into hole precisely")
        assert omega >= 1.0

    def test_peract_omega_is_float(self):
        cal = self._make(precision=0.5, model="peract", omega_base=5.0)
        omega = cal.predict_omega(_rgb(), "task")
        assert isinstance(omega, float)
        assert omega > 0.0


class TestVLACalibratorCache:
    """Tests for caching behavior."""

    def _make(self, cache_steps=3):
        from vla_cal.vla_calibrator import VLACalibrator
        cal = VLACalibrator.__new__(VLACalibrator)
        cal._t_base = 1.0
        cal._alpha = 1.0
        cal._omega_base = 7.0
        cal._eps = 0.05
        cal._cache_steps = cache_steps
        cal._steps_since_update = cache_steps + 1  # force first refresh
        cal._cached_complexity = None
        cal._cached_precision = None
        cal._model = "cliport"
        cal.enabled = True
        client = MagicMock()
        client.estimate_complexity.return_value = 0.5
        client.estimate_precision.return_value = 0.5
        cal._client = client
        return cal

    def test_first_call_queries_client(self):
        cal = self._make(cache_steps=3)
        cal.predict_temperature(_rgb(), "task")
        cal._client.estimate_complexity.assert_called_once()

    def test_second_call_uses_cache(self):
        cal = self._make(cache_steps=3)
        cal.predict_temperature(_rgb(), "task")  # refreshes cache
        cal.predict_temperature(_rgb(), "task")  # should use cache
        # Client should only be called once
        assert cal._client.estimate_complexity.call_count == 1

    def test_cache_expires_after_n_steps(self):
        cal = self._make(cache_steps=2)
        cal.predict_temperature(_rgb(), "task")  # step 0: refresh
        cal.predict_temperature(_rgb(), "task")  # step 1: cache
        cal.predict_temperature(_rgb(), "task")  # step 2: expires → refresh
        assert cal._client.estimate_complexity.call_count == 2

    def test_reset_forces_refresh(self):
        cal = self._make(cache_steps=100)
        cal.predict_temperature(_rgb(), "task")  # refresh
        cal.reset()
        cal.predict_temperature(_rgb(), "task")  # should refresh again
        assert cal._client.estimate_complexity.call_count == 2


class TestVLACalibratorPredictBoth:
    """Tests for predict_both() — single VLM call for T and omega."""

    def _make(self, complexity=0.6, precision=0.4):
        from vla_cal.vla_calibrator import VLACalibrator
        cal = VLACalibrator.__new__(VLACalibrator)
        cal._t_base = 1.0
        cal._alpha = 1.5
        cal._omega_base = 7.0
        cal._eps = 0.05
        cal._cache_steps = 1
        cal._steps_since_update = 999
        cal._cached_complexity = None
        cal._cached_precision = None
        cal._model = "cliport"
        cal.enabled = True
        client = MagicMock()
        client.estimate_complexity.return_value = complexity
        client.estimate_precision.return_value = precision
        cal._client = client
        return cal

    def test_returns_tuple_of_two(self):
        cal = self._make()
        result = cal.predict_both(_rgb(), "task")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_T_matches_predict_temperature(self):
        cal = self._make(complexity=0.6, precision=0.4)
        T, _ = cal.predict_both(_rgb(), "task")
        assert T == pytest.approx(1.0 + 1.5 * 0.6)

    def test_only_one_client_call_per_both(self):
        cal = self._make()
        cal.predict_both(_rgb(), "task")
        # Only one refresh needed even though both T and omega are computed
        assert cal._client.estimate_complexity.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Integration: test that VLACalibrator correctly overrides TemperatureScaler
# ─────────────────────────────────────────────────────────────────────────────

class TestTemperatureScalerSetTemperature:
    """
    Tests for the set_temperature() method added to PerAct's TemperatureScaler.
    """

    def test_set_temperature_updates_value(self):
        try:
            import torch
            from uncertainty_quant_peract.uncertainty_module.src.temperature_scaling.temperature_scaling import TemperatureScaler
        except ImportError:
            pytest.skip("PerAct module not installed")

        # Minimal TemperatureScaler without full PerAct deps
        scaler = TemperatureScaler.__new__(TemperatureScaler)
        import torch
        scaler.temperature = torch.nn.Parameter(torch.ones(1))
        scaler.use_hard_temp = False

        scaler.set_temperature(2.5)
        assert float(scaler.temperature.data) == pytest.approx(2.5)

    def test_set_temperature_preserves_parameter_type(self):
        try:
            import torch
            from uncertainty_quant_peract.uncertainty_module.src.temperature_scaling.temperature_scaling import TemperatureScaler
        except ImportError:
            pytest.skip("PerAct module not installed")

        scaler = TemperatureScaler.__new__(TemperatureScaler)
        import torch
        scaler.temperature = torch.nn.Parameter(torch.ones(1))
        scaler.use_hard_temp = False

        scaler.set_temperature(1.8)
        assert isinstance(scaler.temperature, torch.nn.Parameter)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end logic: does adaptive T actually change action selection behavior?
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveTEffect:
    """
    Simulate the core claim of UA-VLA-IL:
    A complex cluttered scene → higher T → more smoothed logits →
    less likely to pick an isolated distractor spike.
    """

    def test_higher_T_reduces_confidence_spike(self):
        """
        Given logits with a sharp spike (distractor) and a broad peak (true target),
        higher T should reduce the relative confidence of the sharp spike
        more than the broad peak.
        """
        import torch
        import torch.nn.functional as F

        # Simulate logits: sharp spike at index 5, broad peak at indices 10-15
        logits = torch.zeros(20)
        logits[5] = 10.0   # sharp spike → distractor in cluttered scene
        logits[12] = 6.0   # part of broad peak → true target

        T_low = 1.0   # original paper (fixed)
        T_high = 2.5  # UA-VLA-IL (high complexity scene)

        probs_low = F.softmax(logits / T_low, dim=0)
        probs_high = F.softmax(logits / T_high, dim=0)

        # At high T, the spike at index 5 should lose its dominance
        spike_ratio_low = probs_low[5] / probs_low[12]
        spike_ratio_high = probs_high[5] / probs_high[12]

        assert spike_ratio_high < spike_ratio_low, (
            "Higher T must reduce the relative advantage of the spike. "
            f"ratio_low={spike_ratio_low:.3f}, ratio_high={spike_ratio_high:.3f}"
        )

    def test_lower_omega_preserves_sharp_peak(self):
        """
        For a high-precision task, smaller ω means the neighborhood
        aggregation stays tight — it doesn't blur away a sharp correct peak.
        """
        import torch
        import torch.nn.functional as F

        # 1D confidence heatmap with a sharp correct peak at position 10
        heatmap = torch.zeros(20)
        heatmap[10] = 1.0  # sharp true peak

        def neighborhood_sum(heatmap, omega):
            """Sum confidence scores within ±omega of each position."""
            scores = torch.zeros_like(heatmap)
            for i in range(len(heatmap)):
                lo = max(0, i - omega)
                hi = min(len(heatmap), i + omega + 1)
                scores[i] = heatmap[lo:hi].sum()
            return scores

        omega_small = 1   # high precision task
        omega_large = 5   # low precision task

        scores_small = neighborhood_sum(heatmap, omega_small)
        scores_large = neighborhood_sum(heatmap, omega_large)

        # With small omega, peak at 10 gets score = heatmap[10] = 1.0
        # With large omega, peak at 10 gets higher score but so do neighbors
        # The key: with small omega, position 10 is MORE dominant relative to neighbors
        dominance_small = scores_small[10] / scores_small.sum()
        dominance_large = scores_large[10] / scores_large.sum()

        assert dominance_small > dominance_large, (
            "Smaller omega must preserve the dominance of a sharp correct peak. "
            f"dominance_small={dominance_small:.3f}, dominance_large={dominance_large:.3f}"
        )
