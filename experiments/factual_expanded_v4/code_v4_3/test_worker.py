from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import worker
from engine import digest
from scorer import identity


class FakeResponse:
    def json(self):
        return {'done': True, 'done_reason': 'stop', 'message': {'content': '{"status":"answer","answer":"A"}'},
                'prompt_eval_count': 100, 'eval_count': 10}
    def raise_for_status(self): pass


class FakeSession:
    def __init__(self, fail_after=None): self.calls = []; self.fail_after = fail_after
    def post(self, api, json, timeout):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise ConnectionError('synthetic interruption')
        self.calls.append(json)
        return FakeResponse()


class WorkerTests(unittest.TestCase):
    def test_interrupted_transport_replays_saved_responses_and_shared_initials(self):
        item = {'id': 'synthetic', 'dataset': 'synthetic', 'q': 'Synthetic Q', 'gold': 'A',
                'wrong': 'B', 'evidence': 'Synthetic reference A', 'evidence_excerpt': 'Synthetic reference A'}
        item['catalog'] = {'version': 'reference-v1', 'identity': identity(item), 'item_sha256': digest(identity(item)),
                           'accepted': [{'text': 'A', 'basis': 'synthetic'}], 'rejected': [{'text': 'B', 'basis': 'synthetic'}]}
        spec = {'items': [item], 'T': 3, 'cells': [{'delay': 0, 'verifier_on': True}, {'delay': 5, 'verifier_on': True}]}
        config = {'api': 'synthetic', 'backend': 'ollama', 'model': 'synthetic', 'model_digest': 'synthetic'}
        with tempfile.TemporaryDirectory(prefix='synthetic-expanded-test-') as tmp:
            out = Path(tmp)
            failing = FakeSession(fail_after=8)
            with patch.object(worker, 'inspect_model', return_value={'model': {'digest': 'synthetic'}}), patch.object(worker.requests, 'Session', return_value=failing):
                with self.assertRaises(ConnectionError): worker.phase(config, spec, 'synthetic', out)
            saved = (out/'requests.jsonl').read_bytes()
            continuation = FakeSession()
            with patch.object(worker, 'inspect_model', return_value={'model': {'digest': 'synthetic'}}), patch.object(worker.requests, 'Session', return_value=continuation):
                result = worker.phase(config, spec, 'synthetic', out)
            self.assertEqual(result['requests'], 27)
            self.assertEqual(len(continuation.calls), 19)
            self.assertTrue((out/'requests.jsonl').read_bytes().startswith(saved))
            rows = worker.rows(out/'runs.jsonl')
            self.assertEqual(rows[0]['states'][0], rows[1]['states'][0])
            self.assertEqual(sum(r['identity']['role'] == 'initial' for r in worker.rows(out/'requests.jsonl')), 3)
            replay = FakeSession()
            with patch.object(worker, 'inspect_model', return_value={'model': {'digest': 'synthetic'}}), patch.object(worker.requests, 'Session', return_value=replay):
                worker.phase(config, spec, 'synthetic', out)
            self.assertEqual(replay.calls, [])


if __name__ == '__main__': unittest.main()
