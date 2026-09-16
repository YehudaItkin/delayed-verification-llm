import itertools
import math
import unittest
from engine import trajectory, parse, amplitude_bounds


class EngineTests(unittest.TestCase):
    def test_actual_delay_and_synchronous_updates(self):
        item = {'q': 'synthetic Q', 'wrong': 'false', 'evidence_excerpt': 'reference'}
        start = [{'status': 'valid', 'answer': f'initial-{i}'} for i in range(3)]
        calls = []
        def call(identity, messages, temperature):
            calls.append((identity, messages))
            return {'status': 'valid', 'answer': f"{identity['role']}-{identity['step']}-{identity['agent']}"}
        row = trajectory(item, start, 5, True, 8, call)
        self.assertEqual(row['history_indices'], [0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(len(calls), 42)
        first_agent = next(m[1]['content'] for i, m in calls if i == {'role': 'agent', 'step': 1, 'agent': 1})
        self.assertIn('initial-0', first_agent)
        self.assertNotIn('agent-1-0', first_agent)
        last_verifier = next(m[1]['content'] for i, m in calls if i == {'role': 'verifier', 'step': 7, 'agent': 0})
        self.assertIn('agent-1-0', last_verifier)
        self.assertEqual(start[0]['answer'], 'initial-0')

    def test_missing_stays_explicit_and_verifier_off_has_no_calls(self):
        seen = []
        def call(i, messages, t):
            seen.append((i, messages))
            return {'status': 'abstention', 'answer': None}
        row = trajectory({'q': 'Q', 'wrong': 'W', 'evidence_excerpt': 'E'},
                         [{'status': 'invalid_json', 'answer': None}]*3, 0, False, 3, call)
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(i['role'] == 'agent' for i, _ in seen))
        self.assertIn('"status": "invalid_json", "answer": null', seen[0][1][1]['content'])
        self.assertEqual(row['states'][1][0], {'status': 'abstention', 'answer': None})

    def test_parser_never_repairs_missing_or_truncated_outputs(self):
        self.assertEqual(parse('{"status":"answer","answer":"null"}', 'stop')['status'], 'invalid_placeholder')
        self.assertEqual(parse('{"status":"answer","answer":"I do not know"}', 'stop'), {'status': 'abstention', 'answer': None})
        self.assertEqual(parse('{"status":"answer","answer":"Paris"}', 'length')['status'], 'truncated')
        self.assertEqual(parse('{"status":"answer","answer":"A","answer":"B"}', 'stop')['status'], 'invalid_json')

    def test_interval_matches_exhaustive_binary_completions(self):
        rows = [[0, None, 1], [None, 1, None], [0, 0, None]]
        values = []
        for bits in itertools.product([0, 1], repeat=4):
            it = iter(bits)
            means = [sum(next(it) if x is None else x for x in r)/3 for r in rows]
            mean = sum(means)/3
            values.append(math.sqrt(sum((x-mean)**2 for x in means)/3))
        low, high = amplitude_bounds(rows)
        self.assertAlmostEqual(low, min(values)); self.assertAlmostEqual(high, max(values))
        self.assertEqual(amplitude_bounds([[0, 0, 0]]*24), [0., 0.])
        self.assertAlmostEqual(amplitude_bounds([[None]*3]*24)[1], .5)


if __name__ == '__main__': unittest.main()
