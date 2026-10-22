"""Combine final_eval.py runs into the §9 results: accuracy per condition, tier and budget, and paired, clustered
confidence intervals (Miller 2024). CPU only; reads scores.jsonl and summary.json, never the output texts.

Score of a problem under a condition: the mean of its correct flags over all training seeds and samples. Accuracy:
the mean over scored problems. Clustered standard error: sqrt(sum over clusters of (sum over its problems of
(x - mean))^2) / n, with x the problem score (or, for a comparison, the difference of two conditions' problem
scores) and n the number of scored problems; clusters are the source problem in tier 1 and the problem in tiers 2
and 3. 95% interval: mean +- 1.96 SE. Conditions 2-4 must have the same training seeds; every (seed, problem,
sample) must appear exactly once.

usage: python final_stats.py OUT_JSON RUN_DIR [RUN_DIR ...]   (RUN_DIR = an --out directory of final_eval.py)
"""
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

Z = 1.959963984540054
FULL = '32768'
PAIRS = [(3, 2), (4, 2), (3, 4), (2, 1), (3, 1), (4, 1)]
SAMPLES = {'tier1': 8, 'tier2': 8, 'tier3': 4}


def load(run_dirs):
    """{tier: {condition: [records]}}, after checking completeness, hashes and coverage."""
    data = defaultdict(lambda: defaultdict(list))
    seen = set()
    for run in run_dirs:
        for d in sorted(Path(run).iterdir()):
            if not (d / 'summary.json').exists():
                continue
            s = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
            assert s.get('complete'), f'{d}: not complete'
            assert hashlib.sha256((d / 'scores.jsonl').read_bytes()).hexdigest() == s['scores_sha256'], f'{d}: scores changed'
            key = (s['tier'], s['condition'], s['train_seed'])
            assert key not in seen, f'{key} appears twice'
            seen.add(key)
            rows = [json.loads(x) for x in (d / 'scores.jsonl').read_text(encoding='utf-8').splitlines()]
            pairs = {(r['id'], r['sample']) for r in rows}
            ids = {r['id'] for r in rows}
            assert len(pairs) == len(rows) == len(ids) * SAMPLES[s['tier']], f'{d}: samples missing or repeated'
            data[s['tier']][s['condition']] += rows
    for tier, conds in data.items():
        seeds = {c: sorted({r['train_seed'] for r in rows}) for c, rows in conds.items()}
        trained = [seeds[c] for c in (2, 3, 4) if c in seeds]
        assert all(x == trained[0] for x in trained), f'{tier}: training seeds differ {seeds}'
        idsets = [{r['id'] for r in rows} for rows in conds.values()]
        assert all(x == idsets[0] for x in idsets), f'{tier}: problem sets differ'
    return data


def correct(r, budget):
    return r['correct'] if budget == FULL else r['prefix'][budget]['correct']


def problem_scores(rows, budget=FULL, subset=None):
    by = defaultdict(list)
    for r in rows:
        if r['scored'] and (subset is None or r['subset'] == subset):
            by[r['id']].append(correct(r, budget))
    return {i: sum(v) / len(v) for i, v in by.items()}


def clustered(values, cluster):
    """Mean of values (problem -> number) with its clustered standard error and 95% interval."""
    n = len(values)
    m = sum(values.values()) / n
    sums = defaultdict(float)
    for i, x in values.items():
        sums[cluster[i]] += x - m
    se = math.sqrt(sum(v * v for v in sums.values())) / n
    return {'mean': m, 'se': se, 'ci95': [m - Z * se, m + Z * se], 'problems': n, 'clusters': len(sums)}


def tier_report(conds):
    any_rows = next(iter(conds.values()))
    cluster = {r['id']: r['cluster'] for r in any_rows}
    subsets = sorted({r['subset'] for r in any_rows if r['scored']})
    budgets = sorted({b for r in any_rows for b in r['prefix']}, key=int) + [FULL]
    rep = {'conditions': {}, 'comparisons': {}}
    for c, rows in sorted(conds.items()):
        scored = [r for r in rows if r['scored']]
        tokens = [r['output_tokens'] for r in scored]
        seeds = sorted({r['train_seed'] for r in rows}, key=lambda x: (x is None, x))
        rep['conditions'][str(c)] = {
            'training_seeds': seeds,
            'accuracy': {b: clustered(problem_scores(rows, b), cluster) for b in budgets},
            'accuracy_per_seed': {str(s): sum(problem_scores([r for r in rows if r['train_seed'] == s]).values())
                                  / len(problem_scores([r for r in rows if r['train_seed'] == s])) for s in seeds},
            'accuracy_eval_common_only': statistics.mean(
                statistics.mean(v) for v in _group(scored, lambda r: r['correct_eval_common']).values()),
            'truncated': {**{b: sum(r['output_tokens'] > int(b) for r in scored) / len(scored) for b in budgets[:-1]},
                          FULL: sum(r['finish_reason'] == 'length' for r in scored) / len(scored)},
            'mean_output_tokens': statistics.mean(tokens), 'median_output_tokens': statistics.median(tokens),
            'think_closed': (sum(r['think_closed'] for r in scored) / len(scored)) if c != 1 else None,
            'by_subset': {s: clustered(problem_scores(rows, FULL, s), cluster) for s in subsets},
            'excluded': {i: statistics.mean(r['correct'] for r in rows if r['id'] == i)
                         for i in sorted({r['id'] for r in rows if not r['scored']})}}
    for a, b in PAIRS:
        if a in conds and b in conds:
            res = {}
            for bud in budgets:
                sa, sb = problem_scores(conds[a], bud), problem_scores(conds[b], bud)
                res[bud] = clustered({i: sa[i] - sb[i] for i in sa}, cluster)
            rep['comparisons'][f'{a}-{b}'] = res
    return rep


def _group(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r['id']].append(key(r))
    return by


def table(report):
    lines = []
    for tier, rep in report['tiers'].items():
        lines.append(f'{tier}  accuracy at 32k [95% CI]  (4k / 8k / 16k)')
        for c, x in rep['conditions'].items():
            acc = x['accuracy']
            pre = ' / '.join(f"{acc[b]['mean']:.3f}" for b in acc if b != FULL)
            lines.append(f"  condition {c}: {acc[FULL]['mean']:.3f} [{acc[FULL]['ci95'][0]:.3f}, {acc[FULL]['ci95'][1]:.3f}]"
                         f"  ({pre})  truncated {x['truncated'][FULL]:.3f}  mean tokens {x['mean_output_tokens']:.0f}")
        for k, v in rep['comparisons'].items():
            lines.append(f"  {k}: {v[FULL]['mean']:+.3f} [{v[FULL]['ci95'][0]:+.3f}, {v[FULL]['ci95'][1]:+.3f}]")
    return '\n'.join(lines)


def main(out, run_dirs):
    data = load(run_dirs)
    report = {'method': __doc__.split('usage:')[0].strip(), 'runs': [str(Path(r)) for r in run_dirs],
              'tiers': {t: tier_report(data[t]) for t in sorted(data)}}
    Path(out).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(table(report))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
