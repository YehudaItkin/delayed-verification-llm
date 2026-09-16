"""Checks for the remark 'non-negative error and absorption'.

With no forcing, the projected tanh recurrence converges monotonically to zero.
With positive forcing, zero need not be absorbing and nonnegative cycles exist.
These checks use amplitude and repeated delay histories, not sign changes alone.
Run: python3 paper/verify_absorbing.py
"""
import unittest

import numpy as np


def simulate(alpha, delay, forcing=0.0, project=True, steps=4000,
             history=None, sigma=np.tanh):
    """Return initial history followed by the simulated states."""
    states = [0.6] * (delay + 1) if history is None else list(history)
    if len(states) != delay + 1:
        raise ValueError("History must have delay + 1 entries")
    for _ in range(steps):
        value = states[-1] - alpha * sigma(states[-1 - delay]) + forcing
        states.append(max(0.0, value) if project else value)
    return np.asarray(states)


class ProjectedRecurrenceChecks(unittest.TestCase):
    def test_unforced_tanh_is_monotone_and_converges(self):
        for delay in (0, 1, 2, 6):
            for alpha in (0.01, 0.5, 1.5, 2.5):
                with self.subTest(delay=delay, alpha=alpha):
                    states = simulate(alpha, delay)
                    self.assertTrue(np.all(states >= 0.0))
                    self.assertTrue(np.all(np.diff(states) <= 0.0))
                    self.assertLess(states[-1], 1e-6)

    def test_zero_absorbs_without_forcing_even_with_positive_past(self):
        states = simulate(0.5, 2, history=[0.8, 0.4, 0.0])
        np.testing.assert_array_equal(states[2:], 0.0)

    def test_positive_forcing_leaves_zero(self):
        states = simulate(1.0, 1, forcing=0.2, history=[0.0, 0.0], steps=1)
        self.assertEqual(states[-1], 0.2)

    def test_exact_nonnegative_period_six_orbit(self):
        states = simulate(1, 1, forcing=1, history=[0, 0], steps=18,
                          sigma=lambda p: max(-2, min(p, 2)))
        expected = [0, 0, 1, 2, 2, 1] * 3 + [0, 0]
        np.testing.assert_array_equal(states, expected)
        np.testing.assert_array_equal(states[6:], states[:-6])
        self.assertEqual(np.ptp(states), 2)

    def test_forced_tanh_can_cycle_without_crossing_zero(self):
        states = simulate(1.5, 1, forcing=0.2)
        tail = states[-200:]
        self.assertTrue(np.all(tail >= 0.0))
        self.assertGreater(np.ptp(tail), 0.3)
        np.testing.assert_allclose(tail[5:], tail[:-5], rtol=0, atol=1e-12)

    def test_signed_unforced_reference_oscillates(self):
        tail = simulate(0.5, 6, project=False)[-200:]
        self.assertLess(tail.min(), -0.01)
        self.assertGreater(tail.max(), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
