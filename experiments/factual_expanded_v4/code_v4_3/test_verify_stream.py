from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import worker
from engine import digest
from scorer import identity
from test_worker import FakeSession
from verify_stream import verify


class StreamVerificationTests(unittest.TestCase):
    def test_complete_prompt_chain_and_corruption_detection(self):
        item = {'id': 'synthetic', 'dataset': 'synthetic', 'q': 'Synthetic Q', 'gold': 'A',
                'wrong': 'B', 'evidence': 'A', 'evidence_excerpt': 'A'}
        item['catalog'] = {'version': 'reference-v1', 'identity': identity(item), 'item_sha256': digest(identity(item)),
            'accepted': [{'text': 'A', 'basis': 'synthetic'}], 'rejected': [{'text': 'B', 'basis': 'synthetic'}]}
        spec = {'items': [item], 'T': 3, 'cells': [{'delay': 0, 'verifier_on': True}, {'delay': 5, 'verifier_on': True}]}
        config = {'api': 'synthetic', 'backend': 'ollama', 'model': 'synthetic', 'model_digest': 'synthetic'}
        with tempfile.TemporaryDirectory(prefix='synthetic-expanded-verify-') as tmp:
            base = Path(tmp); (base/'code').mkdir()
            for name in ('worker.py', 'engine.py', 'contract.py', 'scorer.py', 'model_api.py', 'PROTOCOL_RU.md'):
                shutil.copyfile(Path(__file__).parent/name, base/'code'/name)
            out = base/'results/pilot'
            with patch.object(worker, 'inspect_model', return_value={'model': {'digest': 'synthetic'}}), patch.object(worker.requests, 'Session', return_value=FakeSession()):
                worker.phase(config, spec, 'pilot', out)
            self.assertEqual(verify(out)['requests'], 27)
            p = out/'requests.jsonl'; data = worker.rows(p)
            data[0]['request']['messages'][1]['content'] = 'tampered prompt'
            p.write_text(''.join(json.dumps(r)+'\n' for r in data))
            with self.assertRaises(AssertionError): verify(out)


if __name__ == '__main__': unittest.main()
