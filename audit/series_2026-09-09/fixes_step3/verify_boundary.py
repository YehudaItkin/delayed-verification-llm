"""Independent root/phase checks against the boundary implementation in paper/demo.py.

Only function definitions are loaded, avoiding the demo's simulation and figure writes.
Run from any directory: python3 path/to/verify_boundary.py
"""
import ast
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
tree = ast.parse((ROOT / "paper/demo.py").read_text())
definitions = ast.Module(
    body=[node for node in tree.body
          if isinstance(node, ast.FunctionDef) and node.name in {"chebU", "beta_c"}],
    type_ignores=[],
)
namespace = {"np": np}
exec(compile(definitions, "paper/demo.py", "exec"), namespace)
boundary = namespace["beta_c"]


def roots(a, delay, beta):
    coefficients = np.zeros(delay + 2)
    coefficients[0], coefficients[1], coefficients[-1] = 1, -a, beta
    return np.roots(coefficients)


class BoundaryChecks(unittest.TestCase):
    def test_first_pair_and_outward_crossing(self):
        for a in (0.05, 0.25, 0.5, 0.9, 0.99):
            for delay in range(1, 33):
                with self.subTest(a=a, delay=delay):
                    beta = boundary(a, delay)
                    self.assertLess(max(abs(roots(a, delay, beta * (1 - 1e-6)))), 1)
                    at = roots(a, delay, beta)
                    self.assertAlmostEqual(max(abs(at)), 1, places=10)
                    self.assertEqual(np.count_nonzero(abs(abs(at) - 1) < 1e-8), 2)
                    above = roots(a, delay, beta * (1 + 1e-6))
                    self.assertEqual(np.count_nonzero(abs(above) > 1 + 1e-10), 2)

    def test_phase_equation_independently_matches_chebyshev(self):
        for a in (0.05, 0.5, 0.99):
            for delay in (1, 2, 8, 32, 64):
                lo, hi = 0.0, np.pi
                for _ in range(80):
                    theta = (lo + hi) / 2
                    if delay * theta + np.angle(np.exp(1j * theta) - a) < np.pi:
                        lo = theta
                    else:
                        hi = theta
                theta = (lo + hi) / 2
                self.assertGreater(theta, np.pi / (2 * delay + 1))
                self.assertLess(theta, np.pi / (delay + 1))
                self.assertAlmostEqual(abs(np.exp(1j * theta) - a), boundary(a, delay), places=10)

    def test_monotonicity_and_exceptions(self):
        aa = np.linspace(0, 1, 41)
        np.testing.assert_array_equal([boundary(a, 1) for a in aa], 1)
        for delay in range(2, 33):
            self.assertTrue(np.all(np.diff([boundary(a, delay) for a in aa]) < 0))
        for a in (0.05, 0.5, 0.99, 1):
            self.assertTrue(np.all(np.diff([boundary(a, d) for d in range(1, 65)]) < 0))
        np.testing.assert_array_equal([boundary(0, d) for d in range(1, 65)], 1)

    def test_exact_special_cases_and_limit(self):
        for a in (0, 0.25, 0.5, 0.99, 1):
            self.assertAlmostEqual(boundary(a, 2), (np.sqrt(a*a + 4) - a)/2, places=12)
        for delay in range(1, 65):
            self.assertAlmostEqual(boundary(1, delay), 2*np.sin(np.pi/(4*delay + 2)), places=10)
        for delay in range(1, 13):
            np.testing.assert_allclose(abs(roots(0, delay, 1)), 1, atol=1e-12)

    def test_negative_a_counterexample(self):
        at = roots(-0.5, 2, 0.5)
        np.testing.assert_allclose(sorted(abs(at)), [1/np.sqrt(2), 1/np.sqrt(2), 1], atol=1e-12)
        self.assertLess(min(abs(at + 1)), 1e-12)
        self.assertLess(max(abs(roots(-0.5, 2, 0.49))), 1)
        self.assertGreater(max(abs(roots(-0.5, 2, 0.51))), 1)

    def test_unsupported_parameters_rejected(self):
        for a in (-0.5, -0.01, 1.01, np.nan, np.inf):
            for delay in (1, 2):
                with self.assertRaises(ValueError):
                    boundary(a, delay)
        for delay in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                boundary(0.5, delay)


if __name__ == "__main__":
    unittest.main(verbosity=2)
