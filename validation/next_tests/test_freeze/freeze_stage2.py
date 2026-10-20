"""Test-set freeze, stage 2: verify the 90 tier-1 variants and deduplicate them. No API call, no model.

  1. Runs every variant's independent check (variants.py) and requires it to equal the recorded `value`, and requires
     the LaTeX `answer` to denote the same value. Wording variants must also equal the source's reference answer.
  2. Requires exactly one wording / numbers / condition variant for each of the 30 primary sources (no reserve used).
  3. Deduplicates with the stage-1 rule (select_length.trigrams, Jaccard >= 0.5 or identical normalized text):
     training-pool problems matching any variant are removed from the training-eligible pool; 0.3 <= J < 0.5 pairs
     are listed for review. Overlaps with development / MATH-Perturb problems, the other test tiers, the reserve
     sources and between variant families are reported.
  4. Writes tier1_variants.jsonl, the final train_eligible_pool.jsonl, stage2_dedup.json and freeze_manifest.json
     (SHA-256 of every frozen file, stage-1 tier files included by reference).

Usage (run from outside the stage-1 download directory; sympy comes from the user's own site-packages):
    python -I freeze_stage2.py PROJECT_ROOT STAGE1_FILES_DIR OUT_DIR
"""
import hashlib
import importlib.util
import io
import json
import re
import site
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

site.addsitedir(site.getusersitepackages())  # -I drops the user site; sympy is installed there, nothing is downloaded
import sympy as sp  # noqa: E402

HERE = 'validation/next_tests/test_freeze/'
SELECT_LENGTH = 'validation/next_tests/length_tests/select_length.py'
STAGE1_ZIP = (HERE + 'stage1_run1/test_freeze_stage1.zip',
              '87055cbc387afdfd92165017c5a341ba9c8fb6f29f1cb99131be4a7bdd773b71')
JACCARD, REVIEW = 0.5, 0.3
TYPES = ('wording', 'numbers', 'condition')
a_, b_ = sp.symbols('a b', positive=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_module(path, digest, name):
    data = path.read_bytes()
    assert digest is None or sha(data) == digest, ('hash mismatch', str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, sha(data)


def latex_value(s):
    """The value a final-answer LaTeX string denotes, for the answer forms used here (numbers, fractions, pi, square
    roots, degrees, text, base-b numerals, unions of open intervals, expressions in a and b)."""
    s = s.strip()
    m = re.fullmatch(r'\\text\{([A-Za-z]+)\}', s)
    if m:
        return m.group(1)
    if re.fullmatch(r'\d+_\d+', s):
        return s
    if '\\cup' in s:
        parts = []
        for p in s.split('\\cup'):
            m = re.fullmatch(r'\((.+),(.+)\)', p.strip())
            lo, hi = (latex_value(t) for t in m.groups())
            parts.append(sp.Interval.open(lo, hi))
        return sp.Union(*parts)
    s = s.replace('^\\circ', '').replace('\\infty', 'oo').replace('\\pi', 'pi')
    for _ in range(4):
        s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', s)
        s = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', s)
    assert '\\' not in s and '{' not in s, s
    s = re.sub(r'(\d)\s*([a-z(])', r'\1*\2', s)  # 2a, 4pi, 2sqrt(11), 2(…)
    s = re.sub(r'\b([ab])\s*([ab])\b', r'\1*\2', s)  # ab
    return sp.sympify(s, locals={'a': a_, 'b': b_, 'pi': sp.pi, 'oo': sp.oo})


def recorded_value(s):
    if re.fullmatch(r'[A-Za-z]+|\d+_\d+', s):
        return s
    return sp.sympify(s, locals={'a': a_, 'b': b_, 'oo': sp.oo, 'Union': sp.Union, 'Interval': sp.Interval})


def same(u, w):
    if isinstance(u, str) or isinstance(w, str):
        return str(u) == str(w)
    if isinstance(u, sp.Set) or isinstance(w, sp.Set):
        return u == w
    return sp.simplify(sp.nsimplify(u) - sp.nsimplify(w)) == 0


def main(project, stage1, out):
    project, stage1, out = Path(project), Path(stage1), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sl, _ = load_module(project / SELECT_LENGTH, None, 'select_length')
    variants_mod, variants_sha = load_module(project / HERE / 'variants.py', None, 'variants')

    # Stage-1 files: every one must match the stage-1 manifest, which must match the downloaded zip.
    zdata = sl.read_checked(project / STAGE1_ZIP[0], STAGE1_ZIP[1])
    manifest1 = json.loads((stage1 / 'freeze_manifest.json').read_bytes())
    with zipfile.ZipFile(io.BytesIO(zdata)) as z:
        zipped = {Path(n).name: z.read(n) for n in z.namelist() if not n.endswith('/')}
    assert zipped['freeze_manifest.json'] == (stage1 / 'freeze_manifest.json').read_bytes()
    stage1_files = {}
    for name, meta in manifest1['files'].items():
        data = (stage1 / name).read_bytes()
        assert sha(data) == meta['sha256'] == sha(zipped[name]), name
        stage1_files[name] = data
    sources = sl.jsonl(stage1_files['tier1_variant_sources.jsonl'])
    primary = {r['id']: r for r in sources if r['role'] == 'primary'}
    assert len(primary) == 30

    # 1-2. Answers.
    symbolic = variants_mod.symbolic_checks()
    rows, problems = [], []
    seen = {}
    for k, var in enumerate(variants_mod.VARIANTS):
        fam, kind = var['family'], var['type']
        assert fam in primary and kind in TYPES and (fam, kind) not in seen, (fam, kind)
        seen[(fam, kind)] = k
        got = var['check']()
        if isinstance(got, str) and got in symbolic:
            got = symbolic[got]
        assert got is not None, (fam, kind, 'check failed')
        if isinstance(got, Fraction):
            got = sp.Rational(got.numerator, got.denominator)
        rec = recorded_value(var['value'])
        tex = latex_value(var['answer'])
        if not (same(got, rec) and same(tex, rec)):
            problems.append({'family': fam, 'type': kind, 'check': str(got), 'value': var['value'],
                             'answer': var['answer']})
        if kind == 'wording' and not same(latex_value(primary[fam]['gold']), rec):
            problems.append({'family': fam, 'type': kind, 'error': 'wording variant changes the answer',
                             'source_gold': primary[fam]['gold']})
        if kind == 'wording' and ' '.join(var['problem'].split()) == ' '.join(primary[fam]['problem'].split()):
            problems.append({'family': fam, 'type': kind, 'error': 'identical to the source'})
        rows.append({'id': f'variant/{fam}/{kind}', 'family': fam, 'type': kind, 'level': primary[fam]['level'],
                     'subject': primary[fam]['subject'], 'problem': var['problem'], 'answer': var['answer'],
                     'source_answer': primary[fam]['gold'],
                     'answer_changed': kind != 'wording',
                     'problem_sha256': sl.sha(' '.join(var['problem'].split()).encode())})
    assert len(seen) == 90 and set(f for f, _ in seen) == set(primary)
    if problems:
        print(json.dumps(problems, indent=1, ensure_ascii=False))
        raise SystemExit('answer checks failed')

    # 3. Deduplication.
    pool = sl.jsonl(stage1_files['train_eligible_pool.jsonl'])
    grams = [sl.trigrams(r['problem']) for r in rows]
    pool_grams = [sl.trigrams(r['problem']) for r in pool]
    pool_norm = {r['id']: ' '.join(r['problem'].split()) for r in pool}
    removed, review = {}, []
    for r, g in zip(rows, grams):
        norm = ' '.join(r['problem'].split())
        best = 0.0
        for p, pg in zip(pool, pool_grams):
            j = sl.jaccard(g, pg)
            best = max(best, j)
            if j >= JACCARD or pool_norm[p['id']] == norm:
                removed.setdefault(p['id'], []).append([r['id'], round(j, 4)])
            elif j >= REVIEW:
                review.append([r['id'], p['id'], round(j, 4)])
        r['max_pool_jaccard'] = round(best, 4)
    final_pool = [p for p in pool if p['id'] not in removed]

    refs = []
    dev = {}
    for rel, digest in sl.DEV_FILES.items():
        for d in sl.jsonl(sl.read_checked(project / rel, digest)):
            dev[d['id']] = d['problem']
    assert len(dev) == 150
    refs += [('dev:' + i, p) for i, p in sorted(dev.items())]
    for name, digest in sl.PERTURB.items():
        refs += [('perturb:' + name.split('/')[-1] + '#' + str(d['problem_id']), d['problem'])
                 for d in sl.jsonl(sl.read_checked(project / name, digest))]
    for name in ('tier2_aime2026.jsonl', 'tier2_hmmt_feb2026.jsonl', 'tier3_math500.jsonl'):
        refs += [(d['id'], d['problem']) for d in sl.jsonl(stage1_files[name])]
    refs += [('source:' + d['id'], d['problem']) for d in sources if d['role'] == 'reserve']
    refs += [('source:' + i, d['problem']) for i, d in primary.items()]
    ref_grams = [(name, sl.trigrams(p)) for name, p in refs]
    other = []
    for r, g in zip(rows, grams):
        own = 'source:' + r['family']
        r['source_jaccard'] = round(sl.jaccard(g, sl.trigrams(primary[r['family']]['problem'])), 4)
        for name, rg in ref_grams:
            if name == own:
                continue
            j = sl.jaccard(g, rg)
            if j >= REVIEW:
                other.append([r['id'], name, round(j, 4)])
    cross = []
    for i in range(len(rows)):
        for k in range(i + 1, len(rows)):
            if rows[i]['family'] != rows[k]['family']:
                j = sl.jaccard(grams[i], grams[k])
                if j >= REVIEW:
                    cross.append([rows[i]['id'], rows[k]['id'], round(j, 4)])

    # 4. Outputs.
    def write_jsonl(name, items):
        data = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items).encode('utf-8')
        (out / name).write_bytes(data)
        return {'sha256': sha(data), 'rows': len(items)}

    files = {'tier1_variants.jsonl': write_jsonl('tier1_variants.jsonl', rows),
             'train_eligible_pool.jsonl': write_jsonl('train_eligible_pool.jsonl', final_pool)}
    dedup = {'rule': f'remove a training-pool problem if its normalized text equals, or its select_length.trigrams '
                     f'Jaccard is >= {JACCARD} with, any tier-1 variant; {REVIEW} <= J < {JACCARD} listed only',
             'removed_for_variants': [{'pool_id': k, 'matches': v} for k, v in sorted(removed.items())],
             'review_pairs': review,
             'variant_vs_dev_perturb_tiers_sources': other,
             'cross_family_variant_pairs': cross}
    data = json.dumps(dedup, indent=1, ensure_ascii=False).encode('utf-8')
    (out / 'stage2_dedup.json').write_bytes(data)
    files['stage2_dedup.json'] = {'sha256': sha(data)}
    frozen = {name: {'sha256': meta['sha256'], 'rows': meta.get('rows'), 'path': HERE + 'stage1_run1/files/' + name}
              for name, meta in manifest1['files'].items()
              if name.startswith('tier') or name == 'train_test_dedup.json'}
    manifest = {
        'stage': 2, 'date': '2026-10-19',
        'variants_py_sha256': variants_sha,
        'stage1_manifest_sha256': sha(stage1_files.get('freeze_manifest.json', (stage1 / 'freeze_manifest.json').read_bytes())),
        'stage1_zip_sha256': STAGE1_ZIP[1],
        'tiers': {'tier1_variants': 'tier1_variants.jsonl (this directory)',
                  'tier2': ['tier2_aime2026.jsonl', 'tier2_hmmt_feb2026.jsonl'],
                  'tier3': 'tier3_math500.jsonl'},
        'counts': {'variants': len(rows), 'families': len(primary), 'reserve_used': 0,
                   'pool_before': len(pool), 'removed_for_variants': len(removed), 'pool_after': len(final_pool),
                   'review_pairs': len(review), 'other_overlaps_ge_0.3': len(other),
                   'other_overlaps_ge_0.5': sum(1 for x in other if x[2] >= JACCARD),
                   'cross_family_pairs_ge_0.3': len(cross)},
        'answer_checks': 'every variant: independent check == recorded value == value of the LaTeX answer; '
                         'wording variants also == source reference answer. Not yet parsed by math-verify '
                         '(needs Linux).',
        'files': files, 'stage1_files': frozen}
    (out / 'freeze_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
