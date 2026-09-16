import json
import unittest
from engine import parse


class ContractTests(unittest.TestCase):
    def test_abstention_branch_has_no_content(self):
        self.assertEqual(parse('{"status":"abstention"}', 'stop'), {'status': 'abstention', 'answer': None})
        for content in ('{"status":"abstention","answer":"A"}', '{"status":"abstention","answer":null}',
                        '{"status":"answer","answer":null}', '{"answer":"A"}'):
            self.assertNotIn(parse(content, 'stop')['status'], ['valid', 'abstention'])

    def test_escaped_quotes_are_real_answer_content(self):
        answer = 'He said "yes"'
        self.assertEqual(parse(json.dumps({'status': 'answer', 'answer': answer}), 'stop'), {'status': 'valid', 'answer': answer})
        self.assertEqual(parse('{"status":"answer","answer":"null"}', 'stop')['status'], 'invalid_placeholder')


if __name__ == '__main__': unittest.main()
