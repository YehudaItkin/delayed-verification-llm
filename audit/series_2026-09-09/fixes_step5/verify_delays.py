"""Independent characteristic-root checks for the corrected two-delay section."""
import unittest
import numpy as np


def coefficients(d, delta, p, q):
    m = max(d, delta)
    c = np.zeros(m + 2)
    c[0], c[1] = 1, -1
    c[1 + d] += p
    c[1 + delta] += q
    return c


def radius(d, delta, p, q):
    return float(max(abs(np.roots(coefficients(d, delta, p, q)))))


class DelayChecks(unittest.TestCase):
    def test_fixed_total_delay_counterexample(self):
        p, q = 1/10, 98/125
        self.assertAlmostEqual(radius(1, 1, p, q), np.sqrt(221/250), places=12)
        self.assertAlmostEqual(radius(0, 2, p, q), np.sqrt(28/25), places=12)
        self.assertLess(radius(1, 1, p, q), 1)
        self.assertGreater(radius(0, 2, p, q), 1)

    def test_regular_locus_satisfies_characteristic_equation(self):
        for d in range(6):
            for delta in range(1, 7):
                if d == delta:
                    continue
                for theta in np.linspace(0.05, np.pi-0.05, 41):
                    denominator = np.sin((delta-d)*theta)
                    if abs(denominator) < 0.02:
                        continue
                    p = (np.sin(delta*theta)-np.sin((delta+1)*theta))/denominator
                    q = (np.sin((d+1)*theta)-np.sin(d*theta))/denominator
                    c = coefficients(d, delta, p, q)
                    self.assertLess(abs(np.polyval(c, np.exp(1j*theta)))/(1+abs(p)+abs(q)), 1e-12)

    def test_singular_line_is_a_stability_boundary(self):
        roots = np.roots(coefficients(1, 4, 1.1, 0.1))
        self.assertEqual(np.count_nonzero(abs(abs(roots)-1) < 1e-10), 2)
        self.assertEqual(np.count_nonzero(abs(roots) < 1-1e-8), 3)
        self.assertLess(radius(1, 4, 1.1-1e-5, 0.1), 1)
        self.assertGreater(radius(1, 4, 1.1+1e-5, 0.1), 1)

    def test_equal_delay_combined_gain_ceiling(self):
        for delay in range(1, 33):
            critical = 2*np.sin(np.pi/(4*delay+2))
            for share in (0.1, 0.5, 0.9):
                with self.subTest(delay=delay, share=share):
                    self.assertLess(radius(delay, delay, share*critical*0.999, (1-share)*critical*0.999), 1)
                    self.assertGreater(radius(delay, delay, share*critical*1.001, (1-share)*critical*1.001), 1)

    def test_equal_delay_network_binds_at_largest_eigenvalue(self):
        eta, delay = 0.2, 2
        mus = (0.2, 0.8, 1.4)
        critical = 2*np.sin(np.pi/(4*delay+2))
        kappa = critical/eta-max(mus)
        below = [radius(delay, delay, eta*mu, eta*(kappa-1e-4)) for mu in mus]
        above = [radius(delay, delay, eta*mu, eta*(kappa+1e-4)) for mu in mus]
        self.assertTrue(all(r < 1 for r in below))
        self.assertTrue(all(r < 1 for r in above[:-1]))
        self.assertGreater(above[-1], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
