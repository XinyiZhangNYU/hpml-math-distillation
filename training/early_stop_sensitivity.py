"""Sensitivity and seed split of the loop early-stop rule (exploratory; DESIGN.md §12.1). Offline, CPU only.

The rule, the §9 grading of the stopped prefix and the token estimate are those of early_stop_estimate.py (§12); only
its four parameters vary: probe 150/300/600, minimum loop 750/1,500/3,000, check every 500/1,000/2,000 and maximum
period 10,000/20,000/40,000 characters, all 81 combinations. For each output the first firing check point of every
setting is found in one pass over the multiples of 500 characters (the rule at check point L only reads text[:L]),
and the §12 setting is also checked output by output against early_stop_estimate.stop_point().

First, the §12 setting must reproduce early_stop_estimate.json in every cell; if it does not, only the reproduction
check is written and the script exits with an error. Then, per setting: per cell (condition / tier, seeds pooled and
per seed) and per condition (tiers pooled by tokens, seeds pooled and per seed), the outputs stopped, stops of
finished outputs, grade changes (correct to wrong and wrong to correct apart), the share of generated tokens saved and
the share of truncated outputs stopped; then the sensitivity reading and the seed split fixed in §12.1.

Inputs: RUNS_DIR/c{2,3,4}_s{1,2,3}/tier{1,2,3}/{summary.json,scores.jsonl,outputs.jsonl.gz} of the final run (outputs
checked against summary.json) and ESTIMATE, the early_stop_estimate.json of §12. Output: OUT/early_stop_sensitivity.json.

usage: python -I early_stop_sensitivity.py RUNS_DIR ESTIMATE OUT   (needs early_stop_estimate.py, final_eval.py and
       the frozen grader next to it)
"""
import gzip
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import early_stop_estimate as es  # noqa: E402  (§12 rule, used for the output-by-output check)
import final_eval  # noqa: E402  (§9 scoring rule)

PROBES, MIN_LOOPS, CHECKS, MAX_PS = (150, 300, 600), (750, 1500, 3000), (500, 1000, 2000), (10000, 20000, 40000)
NAMES = ('probe', 'min_loop', 'check', 'max_period')
SETTINGS = [(pr, ml, ck, mp) for pr in PROBES for ml in MIN_LOOPS for ck in CHECKS for mp in MAX_PS]
BASE = (es.PROBE, es.MIN_LOOP, es.CHECK_EVERY, es.MAX_P)
assert BASE == (300, 1500, 1000, 20000) and BASE in SETTINGS
CONDS, TIERS, SEEDS = (2, 3, 4), ('tier1', 'tier2', 'tier3'), (1, 2, 3)
TOL_PP = 0.05


def label(s):
    return 'probe{}_minloop{}_check{}_maxp{}'.format(*s)


def stop_points(t):
    """First check point at which each setting (in SETTINGS order) fires, or None; same test as es.loops(t[:L])."""
    first = {}
    for pr in PROBES:
        for mp in MAX_PS:
            found = {ml: {} for ml in MIN_LOOPS}
            for L in range(500, len(t), 500):
                if L < 2 * pr:
                    continue
                j = t.rfind(t[L - pr:L], max(0, L - pr - mp), L - 1)
                if j < 0:
                    continue
                p = L - pr - j
                for ml in MIN_LOOPS:
                    f = found[ml]
                    if 2000 in f:
                        continue
                    k = max(2 * p, ml)
                    if k <= L - p and t[L - k:L] == t[L - k - p:L - p]:
                        for ck in CHECKS:
                            if ck not in f and L % ck == 0:
                                f[ck] = L
                if all(2000 in f for f in found.values()):
                    break
            for ml in MIN_LOOPS:
                for ck in CHECKS:
                    first[(pr, ml, ck, mp)] = found[ml].get(ck)
    return [first[s] for s in SETTINGS]


def work(item):
    """One output: per setting (stop point, tokens after, correct after); the §12 setting checked against es."""
    key, t, gold, tokens, correct = item
    pts = stop_points(t)
    base_old = es.stop_point(t)
    graded = {}
    out = []
    for L in pts:
        if L is None:
            out.append((None, tokens, correct))
            continue
        if L not in graded:
            graded[L] = final_eval.score(t[:L], gold, True)['correct']
        out.append((L, round(L * tokens / max(1, len(t))), graded[L]))
    return key, out, pts[SETTINGS.index(BASE)] == base_old


def problem_mean(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r['id'], []).append(key(r))
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def counts(rows, i):
    """Counts of setting i over rows."""
    c = dict(outputs=len(rows), truncated=0, stopped=0, stopped_truncated=0, stopped_finished=0, correct_to_wrong=0,
             wrong_to_correct=0, generated_tokens=0, generated_tokens_after=0)
    for r in rows:
        L, tok_after, corr_after = r['by'][i]
        c['truncated'] += r['truncated']
        c['generated_tokens'] += r['tokens']
        c['generated_tokens_after'] += tok_after
        if L is not None:
            c['stopped'] += 1
            c['stopped_truncated'] += r['truncated']
            c['stopped_finished'] += not r['truncated']
            c['correct_to_wrong'] += r['correct'] and not corr_after
            c['wrong_to_correct'] += corr_after and not r['correct']
    return finish(c)


def add(a, b):
    return finish({k: a[k] + b[k] for k in a if k not in ('saved_share', 'truncated_stopped_share')})


def finish(c):
    c['saved_share'] = 1 - c['generated_tokens_after'] / c['generated_tokens']
    c['truncated_stopped_share'] = c['stopped_truncated'] / c['truncated'] if c['truncated'] else None
    return c


def pooled(cs):
    out = cs[0]
    for c in cs[1:]:
        out = add(out, c)
    return out


def compare_share(new, ref):
    return {'share': new, 'reference_share': ref, 'difference_pp': 100 * (new - ref),
            'relative_change': (new - ref) / ref if ref else None, 'within_5pp': abs(new - ref) <= TOL_PP}


def main(runs_dir, estimate, out):
    t0 = time.time()
    runs_dir, out = Path(runs_dir), Path(out)
    est = json.loads(Path(estimate).read_text())
    assert (est['rule']['check_every_chars'], est['rule']['probe_chars'], est['rule']['max_period_chars'],
            est['rule']['min_loop_chars']) == (BASE[2], BASE[0], BASE[3], BASE[1]), est['rule']
    rows = {}  # cell -> list of row dicts
    for c in CONDS:
        for tier in TIERS:
            cell, items, meta = f'c{c}/{tier}', [], {}
            for s in SEEDS:
                d = runs_dir / f'c{c}_s{s}' / tier
                summ = json.loads((d / 'summary.json').read_text())
                assert hashlib.sha256((d / 'outputs.jsonl.gz').read_bytes()).hexdigest() == summ['outputs_sha256'], d
                with gzip.open(d / 'outputs.jsonl.gz', 'rt', encoding='utf-8') as f:
                    texts = {(o['id'], o['sample']): o['text'] for o in map(json.loads, f)}
                for r in map(json.loads, (d / 'scores.jsonl').read_text().splitlines()):
                    if not r['scored']:
                        continue
                    key = (s, r['id'], r['sample'])
                    meta[key] = {'id': r['id'], 'seed': s, 'tokens': r['output_tokens'], 'correct': r['correct'],
                                 'truncated': r['finish_reason'] == 'length'}
                    items.append((key, texts[(r['id'], r['sample'])], r['gold'], r['output_tokens'], r['correct']))
            mismatch = 0
            with Pool(os.cpu_count()) as pool:
                for key, by, same in pool.imap_unordered(work, items, chunksize=4):
                    meta[key]['by'] = by
                    mismatch += not same
            assert mismatch == 0, f'{cell}: {mismatch} outputs where the §12 setting differs from es.stop_point'
            rows[cell] = list(meta.values())
            print(cell, len(rows[cell]), 'outputs', f'{time.time() - t0:.0f}s', flush=True)

    res = {'note': 'exploratory (DESIGN.md §12.1); offline, no generation; tokens estimated from characters',
           'grid': dict(zip(NAMES, (PROBES, MIN_LOOPS, CHECKS, MAX_PS))), 'base': dict(zip(NAMES, BASE)),
           'settings': [label(s) for s in SETTINGS],
           'environment': {'python': platform.python_version(), 'cpus': os.cpu_count(),
                           'math-verify': metadata.version('math-verify'),
                           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'estimate_sha256': hashlib.sha256(Path(estimate).read_bytes()).hexdigest()}}

    # reproduction check of the §12 setting
    bi, rep, ok = SETTINGS.index(BASE), {}, True
    for cell, rs in rows.items():
        c = counts(rs, bi)
        ours = {'outputs': c['outputs'], 'truncated': c['truncated'], 'stopped': c['stopped'],
                'stopped_truncated': c['stopped_truncated'], 'stopped_finished': c['stopped_finished'],
                'generated_tokens': c['generated_tokens'], 'generated_tokens_after': c['generated_tokens_after'],
                'accuracy': problem_mean(rs, lambda r: r['correct']),
                'accuracy_after': problem_mean(rs, lambda r: r['by'][bi][2])}
        ref = est['cells'][cell]
        diff = {k: [v, ref[k]] for k, v in ours.items()
                if (abs(v - ref[k]) > 1e-12 if isinstance(v, float) else v != ref[k])}
        rep[cell] = {'match': not diff, 'differences': diff}
        ok &= not diff
    res['reproduction'] = {'pass': ok, 'cells': rep}
    print('REPRODUCTION', 'pass' if ok else 'FAIL', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'early_stop_sensitivity.json'
    if not ok:
        dest.write_text(json.dumps(res, indent=1))
        sys.exit('The §12 setting does not reproduce early_stop_estimate.json; nothing else is reported.')

    # per setting: cells (pooled and per seed), conditions (pooled and per seed)
    per = {}
    for i, s in enumerate(SETTINGS):
        cells = {}
        for cell, rs in rows.items():
            cells[cell] = counts(rs, i)
            cells[cell]['seeds'] = {str(sd): counts([r for r in rs if r['seed'] == sd], i) for sd in SEEDS}
        conds = {}
        for c in CONDS:
            names = [f'c{c}/{t}' for t in TIERS]
            conds[f'c{c}'] = pooled([{k: v for k, v in cells[n].items() if k != 'seeds'} for n in names])
            conds[f'c{c}']['seeds'] = {str(sd): pooled([cells[n]['seeds'][str(sd)] for n in names]) for sd in SEEDS}
        per[label(s)] = {'setting': dict(zip(NAMES, s)), 'cells': cells, 'conditions': conds}
    res['results'] = per

    # sensitivity reading: the 8 one-step neighbours of the §12 setting
    base = per[label(BASE)]
    neigh, breaking = {}, []
    for pi, values in enumerate((PROBES, MIN_LOOPS, CHECKS, MAX_PS)):
        for v in (values[0], values[2]):
            s = list(BASE)
            s[pi] = v
            r = per[label(tuple(s))]
            g = sum(r['conditions'][f'c{c}'][k] for c in CONDS for k in ('correct_to_wrong', 'wrong_to_correct'))
            fin = sum(r['conditions'][f'c{c}']['stopped_finished'] for c in CONDS)
            shares = {f'c{c}': compare_share(r['conditions'][f'c{c}']['saved_share'],
                                             base['conditions'][f'c{c}']['saved_share']) for c in CONDS}
            reasons = (['grade change'] if g else []) + (['stop of a finished output'] if fin else []) + \
                [f'{c} share' for c, x in shares.items() if not x['within_5pp']]
            name = f'{NAMES[pi]}={v}'
            neigh[name] = {'setting': label(tuple(s)), 'grade_changes': g, 'stopped_finished': fin,
                           'saved_share_vs_base': shares, 'holds': not reasons, 'breaks_on': reasons}
            if reasons:
                breaking.append(name)
    res['sensitivity'] = {'stable': not breaking, 'neighbours': neigh, 'breaking_neighbours': breaking,
                          'breaking_parameters': sorted({b.split('=')[0] for b in breaking})}
    print('SENSITIVITY', json.dumps({'stable': not breaking, 'breaking': breaking}), flush=True)

    # seed split: choose on seed 1, report on seeds 2 and 3
    def seed_view(r, sd):
        cs = {f'c{c}': r['conditions'][f'c{c}']['seeds'][str(sd)] for c in CONDS}
        g = sum(x['correct_to_wrong'] + x['wrong_to_correct'] for x in cs.values())
        fin = sum(x['stopped_finished'] for x in cs.values())
        tok = sum(x['generated_tokens'] for x in cs.values())
        saved = tok - sum(x['generated_tokens_after'] for x in cs.values())
        return cs, g, fin, saved, tok

    cand = []
    for s in SETTINGS:
        _, g, fin, saved, tok = seed_view(per[label(s)], 1)
        if g == 0 and fin == 0:
            cand.append(((-saved, -s[1], -s[0], -s[2], s[3]), s, saved / tok))
    cand.sort()
    split = {'eligible_on_seed1': len(cand)}
    if cand:
        chosen = cand[0][1]
        rc, rb = per[label(chosen)], base
        cs1, *_ = seed_view(rc, 1)
        split.update({'chosen': label(chosen), 'chosen_setting': dict(zip(NAMES, chosen)),
                      'chosen_seed1_saved_share_all': cand[0][2],
                      'base_seed1_saved_share_all': seed_view(rb, 1)[3] / seed_view(rb, 1)[4],
                      'seed1_saved_share': {c: x['saved_share'] for c, x in cs1.items()}, 'seeds': {}})
        carries = True
        for sd in (2, 3):
            cs, g, fin, _, _ = seed_view(rc, sd)
            bcs, bg, bfin, _, _ = seed_view(rb, sd)
            shares = {c: compare_share(cs[c]['saved_share'], cs1[c]['saved_share']) for c in cs}
            holds = g == 0 and fin == 0 and all(x['within_5pp'] for x in shares.values())
            carries &= holds
            split['seeds'][str(sd)] = {
                'chosen': {'grade_changes': g, 'stopped_finished': fin,
                           'saved_share': {c: x['saved_share'] for c, x in cs.items()}},
                'base': {'grade_changes': bg, 'stopped_finished': bfin,
                         'saved_share': {c: x['saved_share'] for c, x in bcs.items()}},
                'chosen_vs_seed1': shares, 'holds': holds}
        split['carries_over'] = carries
    else:
        split['carries_over'] = None
    res['seed_split'] = split
    print('SEED_SPLIT', json.dumps({k: split.get(k) for k in ('eligible_on_seed1', 'chosen', 'carries_over')}),
          flush=True)
    res['wall_seconds'] = time.time() - t0
    tmp = dest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(dest)
    print('SENSITIVITY_DONE', f'{res["wall_seconds"]:.0f}s', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
