# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""The published temporal contract must hold on the real history runner path."""

import unittest

import numpy as np

from gamematch.data.history import HistoryData
from gamematch.features.timing import assert_strictly_earlier, recover_availability

try:
    from gamematch.history_runner import audit_history_timing

    TORCH_RUNNER_AVAILABLE = True
except ImportError:  # base installation without the neural extra
    TORCH_RUNNER_AVAILABLE = False


def make_history_data():
    rows, players, steps = 4, 2, 3
    history = np.zeros((rows, 2, players, steps, 5), dtype=np.float64)
    static = np.zeros((rows, 2, players, 13), dtype=np.float64)
    valid = np.zeros((rows, 2, players, steps), dtype=bool)
    # One observed step per player, one day before its prediction time.
    valid[:, :, :, -1] = True
    history[:, :, :, -1, 3] = np.log1p(1.0)
    time = np.array([100.0, 200.0, 300.0, 400.0])
    return HistoryData(
        history,
        static,
        valid,
        np.array([0, 1, 2, 1], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([f"row-{i}" for i in range(rows)]),
        time,
        time + 10.0,
    )


class FeedbackTimingContractTests(unittest.TestCase):
    """The leakage rule itself must be testable without PyTorch."""

    def test_inversion_recovers_one_day_earlier_feedback(self):
        data = make_history_data()
        available, time = recover_availability(data.history, data.time)
        self.assertEqual(available.shape, data.valid.shape)
        self.assertTrue(np.allclose(available[..., -1], time[:, None, None] - 86400.0))
        self.assertEqual(assert_strictly_earlier(available, time, data.valid), 16)

    def test_feedback_tied_to_prediction_time_is_rejected(self):
        data = make_history_data()
        data.history[0, 0, 0, -1, 3] = 0.0
        available, time = recover_availability(data.history, data.time)
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            assert_strictly_earlier(available, time, data.valid)

    def test_future_feedback_is_rejected(self):
        data = make_history_data()
        data.history[1, 1, 0, -1, 3] = np.log1p(-0.5)
        available, time = recover_availability(data.history, data.time)
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            assert_strictly_earlier(available, time, data.valid)

    def test_masked_padding_is_not_treated_as_evidence(self):
        data = make_history_data()
        data.history[:, :, :, :-1, 3] = 0.0  # unobserved positions equal prediction time
        available, time = recover_availability(data.history, data.time)
        self.assertEqual(assert_strictly_earlier(available, time, data.valid), 16)

    def test_nonfinite_and_misaligned_inputs_rejected(self):
        data = make_history_data()
        data.history[0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            recover_availability(data.history, data.time)
        clean = make_history_data()
        available, time = recover_availability(clean.history, clean.time)
        with self.assertRaises(ValueError):
            assert_strictly_earlier(available, time[:-1], clean.valid)


@unittest.skipUnless(TORCH_RUNNER_AVAILABLE, "requires the neural extra")
class HistoryTimingAuditTests(unittest.TestCase):
    def test_clean_history_passes_the_full_contract(self):
        report = audit_history_timing(make_history_data())
        self.assertTrue(report["temporal_evidence_checked"])
        self.assertEqual(report["rows_audited_for_leakage"], 4)
        self.assertEqual(report["observed_steps"], 4 * 2 * 2)
        self.assertEqual(report["classes"], ["loss", "draw", "win"])

    def test_feedback_tied_to_prediction_time_is_rejected(self):
        data = make_history_data()
        data.history[0, 0, 0, -1, 3] = 0.0  # age zero -> available == time
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            audit_history_timing(data)

    def test_feedback_after_prediction_time_is_rejected(self):
        data = make_history_data()
        data.history[1, 1, 0, -1, 3] = np.log1p(-0.5)  # negative age -> future feedback
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            audit_history_timing(data)

    def test_nonfinite_features_rejected_at_the_boundary(self):
        data = make_history_data()
        data.static[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            audit_history_timing(data)

    def test_contract_sample_size_is_bounded(self):
        data = make_history_data()
        report = audit_history_timing(data, max_contract_rows=2)
        self.assertEqual(report["rows_checked_against_full_contract"], 2)
        self.assertEqual(report["rows_audited_for_leakage"], 4)


if __name__ == "__main__":
    unittest.main()
