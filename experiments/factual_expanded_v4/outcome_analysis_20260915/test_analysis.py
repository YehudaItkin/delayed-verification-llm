import unittest
import numpy as np
from analyze import features, METRICS, cluster_components, resampled_means, error_frequency_bounds

def state(label):
    return [{'label':label} for _ in range(3)]

class OutcomeTests(unittest.TestCase):
    def test_constant_abstention_has_high_rate_without_oscillation(self):
        f=features([state('abstention')]*24)
        self.assertEqual(f[METRICS.index('abstention.mean')],1)
        self.assertEqual(f[METRICS.index('abstention.sd')],0)
        self.assertEqual(f[METRICS.index('contradicts_reference.mean')],0)

    def test_switching_outcomes_has_half_amplitude(self):
        f=features([state('abstention')]*12+[state('matches_reference')]*12)
        for label in ('abstention','matches_reference'):
            self.assertAlmostEqual(f[METRICS.index(label+'.mean')],.5)
            self.assertAlmostEqual(f[METRICS.index(label+'.sd')],.5)

    def test_cluster_resampling_preserves_question_weights_and_whole_clusters(self):
        keys,totals,sizes=cluster_components(np.array([[0.],[0.],[0.],[1.]]),['a','a','a','b'])
        self.assertEqual(keys,['a','b'])
        got=resampled_means(totals,sizes,np.array([[0,1],[0,0],[1,1]]))
        np.testing.assert_allclose(got[:,0],[.25,0,1])

    def test_paired_identical_conditions_and_unknown_bounds(self):
        u=features([state('unresolved')]*24)[None,:]
        a=features([state('abstention')]*24)[None,:]
        np.testing.assert_array_equal(a-a,np.zeros_like(a))
        np.testing.assert_allclose(error_frequency_bounds(a,a),[[0,0]])
        np.testing.assert_allclose(error_frequency_bounds(u,a),[[-1,0]])
        np.testing.assert_allclose(error_frequency_bounds(a,u),[[0,1]])

    def test_unexpected_labels_fail_instead_of_becoming_zero(self):
        with self.assertRaises(AssertionError): features([state('new_label')]*24)

if __name__=='__main__':
    unittest.main()
