"""Post-hoc paired, cluster-resampled analysis of observed outcome labels."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
VERSIONS = {'v1': BASE/'semantic_review_20260915', 'v2': BASE/'semantic_review_v2_20260915'}
LABELS = ['matches_reference', 'contradicts_reference', 'abstention', 'unresolved', 'invalid_output']
METRICS = [f'{label}.{metric}' for label in LABELS for metric in ('mean', 'sd')]
REPS = 20000

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def features(states):
    assert len(states) == 24 and all(len(row) == 3 for row in states)
    assert all(a['label'] in LABELS for row in states for a in row)
    shares = np.array([[sum(a['label'] == label for a in row)/3 for label in LABELS] for row in states])
    assert np.allclose(shares.sum(axis=1), 1)
    result = np.stack([shares.mean(axis=0), shares.std(axis=0, ddof=0)], axis=1).ravel()
    assert np.isfinite(result).all()
    return result

def cluster_components(values, clusters):
    keys = sorted(set(clusters))
    totals = np.array([values[np.array([c == key for c in clusters])].sum(axis=0) for key in keys])
    sizes = np.array([clusters.count(key) for key in keys])
    return keys, totals, sizes

def resampled_means(totals, sizes, draws):
    return totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]

def bootstrap_difference(values, clusters, seed):
    keys, totals, sizes = cluster_components(values, clusters)
    rng = np.random.default_rng(seed)
    blocks = []
    # Batches avoid allocating reps*clusters*metrics all at once.
    for start in range(0, REPS, 500):
        draws = rng.integers(0, len(keys), size=(min(500, REPS-start), len(keys)))
        blocks.append(resampled_means(totals, sizes, draws))
    samples = np.concatenate(blocks)
    return np.quantile(samples, [.025, .975], axis=0), samples

def error_frequency_bounds(delay0, delay5):
    c, u = METRICS.index('contradicts_reference.mean'), METRICS.index('unresolved.mean')
    return np.stack([delay5[:, c]-delay0[:, c]-delay0[:, u],
                     delay5[:, c]+delay5[:, u]-delay0[:, c]], axis=1)

def main():
    results = HERE/'results'
    assert not results.exists(), 'refuse overwriting an analysis run'
    manifest_path = BASE/'qwen38/v4_3_snapshot/results/primary/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    items = {i['id']: i for i in manifest['specification']['items']}
    assert len(items) == 400
    inputs = {'PROTOCOL_RU.md': sha(HERE/'PROTOCOL_RU.md'), 'analyze.py': sha(Path(__file__)),
              str(manifest_path.relative_to(BASE)): sha(manifest_path)}
    summaries = {}; per_question = []; bootstrap_arrays = {}
    versions_arrays = {}
    for version, folder in VERSIONS.items():
        s = json.loads((folder/'results/summary.json').read_text())
        assert s['status'] == 'passed'
        assert sha(manifest_path) == s['source_files_sha256']['manifest.json']
        assert sha(folder/'review_frozen.json') == s['review_sha256']
        assert sha(folder/'results/runs.jsonl') == s['derived_files_sha256']['runs.jsonl']
        for name in ['review_frozen.json', 'results/summary.json', 'results/runs.jsonl']:
            inputs[str((folder/name).relative_to(BASE))] = sha(folder/name)
        rows = {}; observed_counts = collections.defaultdict(collections.Counter)
        with (folder/'results/runs.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                qid, ident = row['question_id'], row['identity']
                assert qid in items and ident['dataset'] == items[qid]['dataset']
                assert ident['q'] == items[qid]['q'] and ident['verifier_on'] is True
                key = (qid, ident['delay'])
                assert ident['delay'] in (0,5) and key not in rows
                rows[key] = features(row['labels'][24:])
                observed_counts[ident['dataset']].update(a['label'] for state in row['labels'][24:] for a in state)
        assert len(rows) == 800
        summaries[version] = {}
        for dataset, seed in [('psilo',20260915), ('tqa',20260916)]:
            qids = sorted(qid for qid,i in items.items() if i['dataset'] == dataset)
            assert len(qids) == 200
            clusters = [items[qid]['question_cluster'] for qid in qids]
            assert len(set(clusters)) == s['datasets'][dataset]['clusters']
            assert dict(observed_counts[dataset]) == s['tail_state_label_counts'][dataset]
            a = np.array([rows[qid,0] for qid in qids])
            b = np.array([rows[qid,5] for qid in qids])
            versions_arrays[version,dataset] = (qids,a,b)
            diff = b-a
            ci, samples = bootstrap_difference(diff, clusters, seed)
            bootstrap_arrays[f'{version}_{dataset}'] = samples
            bounds = error_frequency_bounds(a,b)
            report = {'questions':200, 'clusters':len(set(clusters)),
                      'cluster_size_max':max(collections.Counter(clusters).values()),
                      'bootstrap_seed':seed, 'metrics':{},
                      'erroneous_response_frequency_difference_outer_bounds':bounds.mean(axis=0).tolist()}
            for j, name in enumerate(METRICS):
                report['metrics'][name] = {'delay0':float(a[:,j].mean()), 'delay5':float(b[:,j].mean()),
                    'difference':float(diff[:,j].mean()), 'bootstrap_percentile_95':ci[:,j].tolist(),
                    'questions_increased':int((diff[:,j]>1e-12).sum()),
                    'questions_decreased':int((diff[:,j]<-1e-12).sum()),
                    'questions_equal':int((abs(diff[:,j])<=1e-12).sum())}
            # Rates averaged over the two conditions reproduce the saved tail counts.
            for label in LABELS:
                m = report['metrics'][label+'.mean']
                assert np.isclose((m['delay0']+m['delay5'])/2, observed_counts[dataset][label]/28800)
            summaries[version][dataset] = report
            for i,qid in enumerate(qids):
                per_question.append({'version':version, 'dataset':dataset, 'question_id':qid,
                    'question_cluster':clusters[i], 'delay0':dict(zip(METRICS,a[i].tolist())),
                    'delay5':dict(zip(METRICS,b[i].tolist())),
                    'difference':dict(zip(METRICS,diff[i].tolist())),
                    'erroneous_response_frequency_difference_outer_bounds':bounds[i].tolist()})
    for dataset in ('psilo','tqa'):
        q1,a1,b1 = versions_arrays['v1',dataset]
        q2,a2,b2 = versions_arrays['v2',dataset]
        assert q1 == q2
        for label in ('abstention','invalid_output'):
            for metric in ('mean','sd'):
                j=METRICS.index(label+'.'+metric)
                assert np.array_equal(a1[:,j], a2[:,j]) and np.array_equal(b1[:,j], b2[:,j])
                assert summaries['v1'][dataset]['metrics'][label+'.'+metric] == summaries['v2'][dataset]['metrics'][label+'.'+metric]
    results.mkdir()
    np.savez_compressed(results/'bootstrap_samples.npz', **bootstrap_arrays)
    (results/'pairs.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in per_question))
    report={'status':'passed', 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'numpy_version':np.__version__, 'bootstrap_replicates':REPS,
        'input_sha256':inputs, 'datasets':summaries,
        'validation':{'runs_per_version':800,'questions_per_version':400,'tail_states_per_version':57600,
                      'tail_counts_match_semantic_reports':True,'absence_metrics_identical_across_versions':True},
        'interpretation':'Post-hoc pointwise cluster-bootstrap sensitivity; not multiplicity-adjusted or confirmatory. '
          'Conditioned on frozen labels and observed generation trajectories. All questions retained; unknown, abstention, format separate. '
          'C is recognised erroneous-response share, not the original unidentified binary truth-error amplitude.'}
    report['derived_sha256']={n:sha(results/n) for n in ('pairs.jsonl','bootstrap_samples.npz')}
    (results/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for dataset in ('psilo','tqa'):
        print(dataset, json.dumps(summaries['v2'][dataset],ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
