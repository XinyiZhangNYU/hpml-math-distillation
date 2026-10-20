"""Teacher generation, step 1 (free, offline): fix the 2,000 training problems and their batch order.

Usage: python -I select_train2000.py PROJECT_ROOT OUT_DIR

- Pool: the frozen train-eligible pool, 6,356 problems (test_freeze/stage2_run1/train_eligible_pool.jsonl).
- Multiple-choice problems are removed before generation, by the problem text only.
- Draw: 2,000 problems by simple random sampling, so the levels and subjects follow the pool's
  natural distribution. random.Random(SEED).sample over the ids sorted as strings; the draw order is also the batch order.
- Batches: b0 = the first 100 (pilot), b1-b4 = the next 4 x 475. The pilot is part of the 2,000, so its traces are kept.
Writes OUT_DIR/train2000.jsonl and OUT_DIR/train2000_selection.json. No API call, no network.
"""
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

POOL = ('validation/next_tests/test_freeze/stage2_run1/train_eligible_pool.jsonl',
        'd451355f171602e0ca55b48f4090f7829d99585df8d73bf65ec85b0011ea4a54')
SEED = 20261019
N = 2000
BATCHES = [('b0', 100), ('b1', 475), ('b2', 475), ('b3', 475), ('b4', 475)]


def is_mc(problem):  # option labels A and B as (A)/(B), A./B. or A)/B)
    paren = lambda L: r'\(\s*' + L + r'\s*\)'
    return bool((re.search(paren('A'), problem) and re.search(paren('B'), problem))
                or (re.search(r'(?m)(^|\s)A\.\s', problem) and re.search(r'(?m)(^|\s)B\.\s', problem))
                or (re.search(r'(?m)(^|\s)A\)\s', problem) and re.search(r'(?m)(^|\s)B\)\s', problem)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(project, out_dir):
    data = (project / POOL[0]).read_bytes()
    assert sha(data) == POOL[1], 'pool hash mismatch'
    pool = [json.loads(x) for x in data.decode('utf-8').splitlines() if x.strip()]
    assert len(pool) == len({r['id'] for r in pool}) == 6356
    mc = sorted(r['id'] for r in pool if is_mc(r['problem']))
    eligible = sorted((r for r in pool if not is_mc(r['problem'])), key=lambda r: r['id'])
    drawn = random.Random(SEED).sample(eligible, N)
    plan = [b for b, n in BATCHES for _ in range(n)]
    assert len(plan) == N
    rows = [{**r, 'order': i, 'batch': plan[i]} for i, r in enumerate(drawn)]

    out_dir.mkdir(parents=True, exist_ok=True)
    assert not (out_dir / 'train2000.jsonl').exists(), 'Output exists: inspect it instead of overwriting.'
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
    (out_dir / 'train2000.jsonl').write_bytes(body)

    def dist(sel, key):
        return dict(sorted(collections.Counter(str(r[key]) for r in sel).items()))
    record = {
        'pool': {'path': POOL[0], 'sha256': POOL[1], 'problems': len(pool)},
        'mc_rule': 'is_mc(): option labels A and B written as (A)/(B), A./B. or A)/B)',
        'mc_excluded': len(mc), 'mc_excluded_by_level': dist([r for r in pool if r['id'] in set(mc)], 'level'),
        'mc_excluded_ids': mc,
        'eligible_after_mc': len(eligible), 'seed': SEED, 'drawn': N,
        'method': 'random.Random(SEED).sample(eligible sorted by id, N); draw order = batch order',
        'batches': {b: n for b, n in BATCHES},
        'by_level': {'eligible': dist(eligible, 'level'), 'drawn': dist(rows, 'level'),
                     **{b: dist([r for r in rows if r['batch'] == b], 'level') for b, _ in BATCHES}},
        'by_subject': {'eligible': dist(eligible, 'subject'), 'drawn': dist(rows, 'subject')},
        'files': {'train2000.jsonl': sha(body)}}
    (out_dir / 'train2000_selection.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    print('SELECTED', json.dumps({k: record[k] for k in ('mc_excluded', 'eligible_after_mc', 'drawn', 'files')}))
    print('BY_LEVEL', json.dumps(record['by_level']))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
