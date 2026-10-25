"""Post-hoc look at the tier-2 outputs of conditions 3 and 4 that hit the 32k budget (not pre-registered).

Reads hpml_final/evals/c{3,4}_s{1,2,3}/tier2/{summary.json,scores.jsonl,outputs.jsonl.gz} (outputs checked
against the SHA-256 in summary.json) and writes hpml_final/analysis/trunc_tier2/{report.json,examples.txt}.
Per output: exact periodic tail (the end of the text repeats with a fixed period at least 3 times), zlib ratio
and duplicate-line share of the last 16,000 characters, and every \\boxed{} answer graded with math-verify as in
the frozen eval_common.grade. Character positions are turned into tokens with the output's own tokens/char.
CPU only; no model, no API.
"""
import gzip, hashlib, json, random, re, statistics, zlib
from pathlib import Path
from math_verify import parse, verify, LatexExtractionConfig

S = Path('/content/drive/MyDrive/hpml_final')
OUT = S / 'analysis/trunc_tier2'
RUNS = [f'c{c}_s{s}' for c in (3, 4) for s in (1, 2, 3)]
TAIL = 16000


def boxed_all(text):
    """(position, content) of every \\boxed{}/\\fbox{} in order; same brace matching as eval_common.boxed."""
    found = []
    for m in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        depth = 1
        for i in range(m.end(), len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    found.append((m.start(), text[m.end():i]))
                    break
    return found


def parse_box(answer):
    return parse('$' + answer + '$', extraction_config=[LatexExtractionConfig()])


def periodic_tail(s, probe=300, max_p=20000):
    """Smallest period p at which the end of s repeats, and how many chars from the end follow it (k)."""
    n = len(s)
    if n < 2 * probe:
        return 0, 0
    j = s.rfind(s[-probe:], max(0, n - probe - max_p), n - 1)  # nearest earlier start, overlap allowed
    if j < 0:
        return 0, 0
    p = n - probe - j
    k = 0
    while k < n - p and s[n - 1 - k] == s[n - 1 - k - p]:
        k += 1
    return p, k


def q(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(xs)
    return {str(p): round(xs[min(len(xs) - 1, int(p * len(xs)))], 3) for p in ps} if xs else {}


rows, texts, cache = [], {}, {}
for run in RUNS:
    d = S / 'evals' / run / 'tier2'
    summ = json.loads((d / 'summary.json').read_text())
    assert hashlib.sha256((d / 'outputs.jsonl.gz').read_bytes()).hexdigest() == summ['outputs_sha256'], run
    scores = [json.loads(x) for x in (d / 'scores.jsonl').read_text().splitlines()]
    with gzip.open(d / 'outputs.jsonl.gz', 'rt', encoding='utf-8') as f:
        outs = {(o['id'], o['sample']): o['text'] for o in map(json.loads, f)}
    for r in scores:
        if not r['scored']:
            continue
        s = outs[(r['id'], r['sample'])]
        tpc = r['output_tokens'] / max(1, len(s))
        p, k = periodic_tail(s)
        loop = k >= max(2 * p, 1500)
        tail = s[-TAIL:].encode()
        lines = [x.strip() for x in s[-TAIL:].split('\n') if x.strip()]
        key = ('$' + r['gold'] + '$')
        if key not in cache:
            cache[key] = parse_box(r['gold'])
        gold = cache[key]
        first_ok, n_box, seen = None, 0, {}
        for pos, ans in boxed_all(s):
            n_box += 1
            if ans not in seen:
                try:
                    pred = parse_box(ans)
                    seen[ans] = bool(pred and verify(gold, pred))
                except Exception:
                    seen[ans] = False
            if seen[ans] and first_ok is None:
                first_ok = pos
        row = {'run': run, 'cond': int(run[1]), 'id': r['id'], 'sample': r['sample'], 'tokens': r['output_tokens'],
               'truncated': r['finish_reason'] == 'length', 'think_closed': r['think_closed'], 'correct': r['correct'],
               'last_box_correct': r['correct_eval_common'], 'chars': len(s),
               'loop': loop, 'loop_period_tokens': round(p * tpc) if loop else None,
               'loop_start_tokens': round((len(s) - k - p) * tpc) if loop else None,
               'zlib_tail': round(len(zlib.compress(tail, 9)) / max(1, len(tail)), 4),
               'dup_lines_tail': round(1 - len(set(lines)) / len(lines), 4) if lines else 0.0,
               'n_boxed': n_box, 'any_box_correct': first_ok is not None,
               'first_correct_box_tokens': round(first_ok * tpc) if first_ok is not None else None}
        rows.append(row)
        if row['truncated']:
            texts[(run, r['id'], r['sample'])] = s

# Near-loop threshold from finished long outputs: below their 5th percentile of zlib ratio.
fin_long = [r for r in rows if not r['truncated'] and r['tokens'] > 8000]
z_cut = sorted(r['zlib_tail'] for r in fin_long)[max(0, int(0.05 * len(fin_long)) - 1)] if fin_long else 0.2
for r in rows:
    if r['truncated']:
        r['kind'] = 'exact_loop' if r['loop'] else ('near_repetitive' if r['zlib_tail'] < z_cut else 'no_repetition_found')

report = {'note': 'post-hoc analysis, not pre-registered; tier 2 (AIME/HMMT 2026), 32k budget, conditions 3 and 4, 3 seeds',
          'near_loop_rule': f'truncated, no exact loop, zlib ratio of last {TAIL} chars < {z_cut} '
                            '(5th percentile of finished outputs longer than 8k tokens)',
          'by_condition': {}}
for c in (3, 4):
    rs = [r for r in rows if r['cond'] == c]
    tr = [r for r in rs if r['truncated']]
    fl = [r for r in rs if not r['truncated'] and r['tokens'] > 8000]
    by_prob = {}
    for r in rs:
        by_prob.setdefault(r['id'], []).append(r)
    solved = {i for i, v in by_prob.items() if any(x['correct'] for x in v)}
    kinds = {k: sum(r['kind'] == k for r in tr) for k in ('exact_loop', 'near_repetitive', 'no_repetition_found')}
    report['by_condition'][f'c{c}'] = {
        'outputs': len(rs), 'truncated': len(tr), 'truncated_share': round(len(tr) / len(rs), 3),
        'truncated_kinds': kinds,
        'truncated_with_correct_box_somewhere': sum(r['any_box_correct'] for r in tr),
        'truncated_last_box_correct': sum(r['last_box_correct'] for r in tr),
        'truncated_think_closed': sum(r['think_closed'] for r in tr),
        'correct_box_by_kind': {k: sum(r['any_box_correct'] for r in tr if r['kind'] == k) for k in kinds},
        'truncated_on_problems_never_solved': sum(r['id'] not in solved for r in tr),
        'problems': len(by_prob), 'problems_solved_at_least_once': len(solved),
        'loop_start_tokens_quantiles': q([r['loop_start_tokens'] for r in tr if r['loop']]),
        'loop_period_tokens_quantiles': q([r['loop_period_tokens'] for r in tr if r['loop']]),
        'first_correct_box_tokens_quantiles_truncated': q([r['first_correct_box_tokens'] for r in tr if r['any_box_correct']]),
        'zlib_tail_quantiles': {'truncated': q([r['zlib_tail'] for r in tr]), 'finished_over_8k': q([r['zlib_tail'] for r in fl])},
        'dup_lines_tail_quantiles': {'truncated': q([r['dup_lines_tail'] for r in tr]), 'finished_over_8k': q([r['dup_lines_tail'] for r in fl])},
        'finished_over_8k': len(fl), 'finished_over_8k_exact_loop': sum(r['loop'] for r in fl),
        'finished_over_8k_accuracy': round(statistics.mean(r['correct'] for r in fl), 3) if fl else None,
        'by_seed_truncated_kinds': {run: {k: sum(r['kind'] == k for r in tr if r['run'] == run) for k in kinds}
                                    for run in RUNS if run.startswith(f'c{c}')}}
report['rows'] = rows

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'report.json').write_text(json.dumps(report, indent=1))
rng = random.Random(0)
ex = []
for c in (3, 4):
    for kind in ('exact_loop', 'near_repetitive', 'no_repetition_found'):
        pool = [r for r in rows if r['cond'] == c and r.get('kind') == kind]
        for r in rng.sample(pool, min(3, len(pool))):
            s = texts[(r['run'], r['id'], r['sample'])]
            meta = {k: r[k] for k in ('run', 'id', 'sample', 'tokens', 'kind', 'loop_period_tokens', 'loop_start_tokens',
                                      'zlib_tail', 'any_box_correct', 'first_correct_box_tokens')}
            ex.append('=' * 30 + '\n' + json.dumps(meta) + '\n--- middle 1200 chars ---\n' + s[len(s) // 2: len(s) // 2 + 1200]
                      + '\n--- last 1500 chars ---\n' + s[-1500:] + '\n')
(OUT / 'examples.txt').write_text('\n'.join(ex), encoding='utf-8')
print(json.dumps({k: {x: v[x] for x in ('outputs', 'truncated', 'truncated_kinds', 'truncated_with_correct_box_somewhere',
                                         'truncated_on_problems_never_solved')} for k, v in report['by_condition'].items()}))
print('ZCUT', z_cut, 'DONE')
