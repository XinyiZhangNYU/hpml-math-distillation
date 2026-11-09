"""Extract the outputs shown in the side-by-side demo of the four conditions (DESIGN.md §13). Nothing is generated.

Selection, fixed before any output is read: in each tier, the scored problem ids are sorted and
random.Random(20261109 + tier number).sample() draws 4 (tier 1), 4 (tier 2) and 2 (tier 3) of them. For each drawn
problem the demo shows sample 0 of each condition: condition 1 from its one run (evals/c1), conditions 2-4 from
training seed 1 (evals/c{2,3,4}_s1). The request seed of an item and sample is the same for every model (§9), so the
four outputs answer the same prompt draw. Next to each output: its tokens, finish reason, closed think block and
grade as recorded in scores.jsonl; for conditions 2-4 the stop point of the §12 loop rule
(early_stop_estimate.stop_point) and the §9 grade of the stopped prefix; and the problem's accuracy over all seeds
and samples of each condition.

Inputs: EVALS/{c1,c2_s1..c4_s3}/tier{1,2,3}/{summary.json,scores.jsonl,outputs.jsonl.gz}, outputs checked against
the SHA-256 in summary.json. Output: OUT/demo_data.json (texts included; problem statements are joined later from the
frozen test files). CPU only; no model, no API.

usage: python -I demo_extract.py EVALS OUT   (needs early_stop_estimate.py, final_eval.py and the frozen grader next to it)
"""
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import early_stop_estimate  # noqa: E402  (§12 rule)
import final_eval  # noqa: E402  (§9 scoring rule)

DRAWS = {'tier1': 4, 'tier2': 4, 'tier3': 2}
SELECT_SEED = 20261109
SEED1_RUN = {1: 'c1', 2: 'c2_s1', 3: 'c3_s1', 4: 'c4_s1'}
ALL_RUNS = {1: ['c1'], **{c: [f'c{c}_s{s}' for s in (1, 2, 3)] for c in (2, 3, 4)}}


def load(d):
    summ = json.loads((d / 'summary.json').read_text())
    digest = hashlib.sha256((d / 'outputs.jsonl.gz').read_bytes()).hexdigest()
    assert digest == summ['outputs_sha256'], d
    scores = [json.loads(x) for x in (d / 'scores.jsonl').read_text().splitlines()]
    return summ, scores, digest


def main(evals, out):
    evals, out = Path(evals), Path(out)
    res = {'selection': {'seed': f'{SELECT_SEED} + tier number', 'draws': DRAWS, 'sample': 0,
                         'runs': SEED1_RUN, 'rule': 'random.Random(seed).sample(sorted scored ids, k)'},
           'loop_rule': {'check_every_chars': early_stop_estimate.CHECK_EVERY, 'probe_chars': early_stop_estimate.PROBE,
                         'max_period_chars': early_stop_estimate.MAX_P, 'min_loop_chars': early_stop_estimate.MIN_LOOP},
           'sources': {}, 'problems': []}
    for tier, k in DRAWS.items():
        tnum = int(tier[-1])
        acc, scores_by_run = {}, {}
        for c, runs in ALL_RUNS.items():
            by = {}
            for run in runs:
                summ, scores, digest = load(evals / run / tier)
                res['sources'][f'{run}/{tier}'] = digest
                scores_by_run[run] = scores
                for r in scores:
                    if r['scored']:
                        by.setdefault(r['id'], []).append(r['correct'])
            acc[c] = {i: (sum(v) / len(v), len(v)) for i, v in by.items()}
        ids = sorted({r['id'] for r in scores_by_run['c1'] if r['scored']})
        drawn = random.Random(SELECT_SEED + tnum).sample(ids, k)
        print(tier, 'DRAWN', drawn, flush=True)
        for pid in drawn:
            prob = {'tier': tier, 'id': pid, 'outputs': {},
                    'accuracy_all_samples': {str(c): {'mean': round(acc[c][pid][0], 4), 'n': acc[c][pid][1]} for c in acc}}
            for c, run in SEED1_RUN.items():
                r = next(x for x in scores_by_run[run] if x['id'] == pid and x['sample'] == 0)
                prob['gold'] = r['gold']
                with gzip.open(evals / run / tier / 'outputs.jsonl.gz', 'rt', encoding='utf-8') as f:
                    text = next(o['text'] for o in map(json.loads, f) if o['id'] == pid and o['sample'] == 0)
                o = {'run': run, 'tokens': r['output_tokens'], 'finish_reason': r['finish_reason'],
                     'think_closed': r.get('think_closed'), 'correct': r['correct'], 'chars': len(text), 'text': text,
                     'loop_stop': None}
                if c > 1:
                    L = early_stop_estimate.stop_point(text)
                    if L is not None:
                        o['loop_stop'] = {'chars': L, 'tokens_est': round(L * r['output_tokens'] / max(1, len(text))),
                                          'correct_after': final_eval.score(text[:L], r['gold'], True)['correct']}
                prob['outputs'][str(c)] = o
                print(' ', tier, pid, f'c{c}', o['tokens'], o['finish_reason'], o['correct'],
                      o['loop_stop'] and o['loop_stop']['tokens_est'], flush=True)
            res['problems'].append(prob)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'demo_data.json').write_text(json.dumps(res, ensure_ascii=False))
    print('DEMO_EXTRACT_DONE', hashlib.sha256((out / 'demo_data.json').read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
