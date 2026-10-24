"""Offline estimate of a loop early-stop rule on the final outputs (exploratory, chosen after the results; DESIGN.md §12).

Rule, applied to each saved output as if generation had been checked every 1,000 characters: at check point L, let
the last 300 characters of text[:L] occur earlier (within 20,000 characters, overlap allowed) at the nearest
distance p; if the text up to L repeats with period p over the last max(2p, 1,500) characters, generation stops at L.
The rule was chosen after the final results were known, so the estimate is exploratory. Nothing is generated: the
stopped output is text[:L], graded under the §9 rule (a think block that is not closed is wrong). Token positions are
character positions times the output's own tokens per character.

Inputs: RUNS_DIR/c{2,3,4}_s{1,2,3}/tier{1,2,3}/{summary.json,scores.jsonl,outputs.jsonl.gz} of the final run (outputs
checked against summary.json). Output: OUT/early_stop_estimate.json with, per condition and tier, the accuracy
(mean over problems of the mean over seeds and samples) and the generated tokens with and without the rule, and how
many outputs it stops (truncated or finished, correct or not).

usage: python -I early_stop_estimate.py RUNS_DIR OUT   (needs final_eval.py and the frozen grader next to it)
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_eval  # noqa: E402  (§9 scoring rule)

CHECK_EVERY, PROBE, MAX_P, MIN_LOOP = 1000, 300, 20000, 1500


def loops(s, probe=PROBE, max_p=MAX_P):
    """True if the end of s repeats with its smallest period p over at least max(2p, MIN_LOOP) characters."""
    n = len(s)
    if n < 2 * probe:
        return False
    j = s.rfind(s[-probe:], max(0, n - probe - max_p), n - 1)
    if j < 0:
        return False
    p = n - probe - j
    k = max(2 * p, MIN_LOOP)
    return k <= n - p and s[n - k:] == s[n - k - p:n - p]


def stop_point(s):
    """First check point at which the rule fires, or None."""
    for L in range(CHECK_EVERY, len(s), CHECK_EVERY):
        if loops(s[:L]):
            return L
    return None


def problem_mean(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r['id'], []).append(key(r))
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def main(runs_dir, out):
    runs_dir, out = Path(runs_dir), Path(out)
    res = {'rule': {'check_every_chars': CHECK_EVERY, 'probe_chars': PROBE, 'max_period_chars': MAX_P,
                    'min_loop_chars': MIN_LOOP, 'condition': 'repeats over max(2p, 1500) chars'},
           'note': 'exploratory; rule chosen after the final results; offline, no generation', 'cells': {}}
    for c in (2, 3, 4):
        for tier in ('tier1', 'tier2', 'tier3'):
            rows = []
            for s in (1, 2, 3):
                d = runs_dir / f'c{c}_s{s}' / tier
                summ = json.loads((d / 'summary.json').read_text())
                assert hashlib.sha256((d / 'outputs.jsonl.gz').read_bytes()).hexdigest() == summ['outputs_sha256'], d
                with gzip.open(d / 'outputs.jsonl.gz', 'rt', encoding='utf-8') as f:
                    texts = {(o['id'], o['sample']): o['text'] for o in map(json.loads, f)}
                for r in map(json.loads, (d / 'scores.jsonl').read_text().splitlines()):
                    if not r['scored']:
                        continue
                    t = texts[(r['id'], r['sample'])]
                    L = stop_point(t)
                    row = {'id': r['id'], 'seed': s, 'tokens': r['output_tokens'], 'correct': r['correct'],
                           'truncated': r['finish_reason'] == 'length', 'stopped': L is not None,
                           'tokens_after': r['output_tokens'], 'correct_after': r['correct']}
                    if L is not None:
                        row['tokens_after'] = round(L * r['output_tokens'] / max(1, len(t)))
                        row['correct_after'] = final_eval.score(t[:L], r['gold'], True)['correct']
                    rows.append(row)
            st = [r for r in rows if r['stopped']]
            tok, tok_after = sum(r['tokens'] for r in rows), sum(r['tokens_after'] for r in rows)
            res['cells'][f'c{c}/{tier}'] = {
                'outputs': len(rows), 'truncated': sum(r['truncated'] for r in rows),
                'stopped': len(st), 'stopped_truncated': sum(r['truncated'] for r in st),
                'stopped_finished': sum(not r['truncated'] for r in st),
                'stopped_was_correct': sum(r['correct'] for r in st),
                'stopped_correct_after': sum(r['correct_after'] for r in st),
                'accuracy': problem_mean(rows, lambda r: r['correct']),
                'accuracy_after': problem_mean(rows, lambda r: r['correct_after']),
                'generated_tokens': tok, 'generated_tokens_after': tok_after, 'saved_share': 1 - tok_after / tok,
                'tokens_in_truncated_outputs': sum(r['tokens'] for r in rows if r['truncated']),
                'median_stop_token': sorted(r['tokens_after'] for r in st)[len(st) // 2] if st else None}
            print(f'c{c}/{tier}', json.dumps({k: res['cells'][f'c{c}/{tier}'][k] for k in
                                              ('outputs', 'stopped', 'stopped_was_correct', 'accuracy', 'accuracy_after',
                                               'saved_share')}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'early_stop_estimate.json').write_text(json.dumps(res, indent=1))
    print('EARLY_STOP_DONE', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
