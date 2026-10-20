"""Test-set freeze, stage 1: three test tiers, deduplicated against training data and frozen before training. No API call, no model. Free CPU work.

What it does:
  1. Rebuilds the MATH train pool exactly as select_length.py / eval_common.prepare_data() did and checks the
     recorded exclusion counts and the 150 development problems.
  2. Loads the test tiers at pinned revisions:
       - tier 2: MathArena AIME 2026 (30) and HMMT February 2026 (33);
       - tier 3: HuggingFaceH4/MATH-500 (500);
       - tier 1 sources: the MATH test split (same hendrycks revision as the pool). Variants are written in stage 2.
  3. Picks the tier-1 source problems before any model has seen them: within each level, ascending
     sha256(f"{SEED}:variant-source:{problem_sha256}"), skipping diagram problems, unparseable references,
     MATH-500 members and near-duplicates of MATH-500. The first 6 per level are "primary", the next 4 "reserve".
     A primary is replaced by the next reserve of the same level only if it cannot carry three verifiable
     variants; the reason is recorded in stage 2. Test outputs never decide replacements.
  4. Deduplicates against training data: any pool problem whose normalized text equals, or whose word-trigram
     Jaccard is >= 0.5 with, any test item (all tiers, primary and reserve sources) is removed from the
     training-eligible pool. Development ids and the earlier near-duplicates of development / MATH-Perturb
     problems are removed as in select_length.py. Pairs with 0.3 <= Jaccard < 0.5 are listed for review only.
  5. Writes the frozen tier files, the training-eligible pool and freeze_manifest.json with SHA-256 of every file.

Usage (Linux / Colab; math-verify needs signal.alarm and the main thread):
    python3 -I freeze_tests.py PROJECT_ROOT OUT_DIR
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SEED = 91432026
JACCARD = 0.5
REVIEW = 0.3
PRIMARY, RESERVE = 6, 4
SELECT_LENGTH = 'validation/next_tests/length_tests/select_length.py'
TIERS = {  # name: (repo, revision, split, expected rows)
    'aime2026': ('MathArena/aime_2026', 'd61f486805306e6e6b5913504c07763a99e1b112', 'train', 30),
    'hmmt_feb2026': ('MathArena/hmmt_feb_2026', '589a53fe152febc0f4f99f3e561007bf8da8e664', 'train', 33),
    'math500': ('HuggingFaceH4/MATH-500', '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be', 'test', 500)}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Index:
    """Word-trigram inverted index over pool problems, so each query only scores pool items sharing a trigram."""

    def __init__(self, sl, rows):
        self.sl, self.rows = sl, rows
        self.grams = [sl.trigrams(r['problem']) for r in rows]
        self.by_gram = {}
        for i, g in enumerate(self.grams):
            for t in g:
                self.by_gram.setdefault(t, []).append(i)
        self.by_sha = {}
        for i, r in enumerate(rows):
            self.by_sha.setdefault(r['problem_sha256'], []).append(i)

    def matches(self, text, floor):
        q = self.sl.trigrams(text)
        shared = {}
        for t in q:
            for i in self.by_gram.get(t, ()):
                shared[i] = shared.get(i, 0) + 1
        out = []
        for i, n in shared.items():
            j = n / (len(q) + len(self.grams[i]) - n)
            if j >= floor:
                out.append((round(j, 4), i))
        exact = self.by_sha.get(self.sl.sha(' '.join(text.split()).encode()), [])
        return sorted(out, reverse=True), exact


def load_tier(name):
    from datasets import load_dataset
    repo, rev, split, n = TIERS[name]
    rows = list(load_dataset(repo, split=split, revision=rev))
    assert len(rows) == n, (name, len(rows))
    return rows


def main(project, out_dir):
    import signal
    import threading
    from importlib import metadata
    assert hasattr(signal, 'alarm') and threading.current_thread() is threading.main_thread(), \
        'Run under Linux / Colab in the main thread (math-verify uses signal.alarm).'
    versions = {n: metadata.version(n) for n in ('math-verify', 'datasets', 'huggingface_hub')}
    assert versions['math-verify'] == '0.9.0', versions
    assert not out_dir.exists(), 'Output folder exists: inspect it instead of overwriting.'

    sl = load_module(project / SELECT_LENGTH, 'select_length')
    sl.read_checked(project / sl.EVAL_COMMON[0], sl.EVAL_COMMON[1])
    sys.path.insert(0, str(project / 'validation'))
    from eval_common import parse_answer

    def parses(answer):
        return bool(parse_answer('\\boxed{' + str(answer) + '}')[1])

    # Development problems and MATH-Perturb, as in select_length.py.
    dev = {}
    for rel, digest in sl.DEV_FILES.items():
        for r in sl.jsonl(sl.read_checked(project / rel, digest)):
            assert dev.setdefault(r['id'], r)['problem_sha256'] == r['problem_sha256'], r['id']
    assert len(dev) == 150, len(dev)
    perturb = []
    for name, digest in sl.PERTURB.items():
        perturb += [(name.split('/')[-1] + '#' + str(r['problem_id']), r['problem'])
                    for r in sl.jsonl(sl.read_checked(project / name, digest))]

    # 1. Training pool.
    pool, excluded = sl.build_pool(parse_answer)
    assert excluded == sl.EXPECTED_EXCLUDED, ('pool differs from the original prepare_data run', excluded)
    by_id = {r['id']: r for r in pool}
    for i, d in dev.items():
        assert i in by_id and by_id[i]['problem_sha256'] == d['problem_sha256'] and by_id[i]['gold'] == d['gold'], i
    index = Index(sl, pool)

    # 2. Tiers 2 and 3.
    tests = {}
    aime = load_tier('aime2026')
    tests['aime2026'] = [{'id': f'aime2026/{r["problem_idx"]}', 'problem': r['problem'], 'answer': str(r['answer'])}
                         for r in aime]
    hmmt = load_tier('hmmt_feb2026')
    tests['hmmt_feb2026'] = [{'id': f'hmmt_feb2026/{r["problem_idx"]}', 'problem': r['problem'],
                              'answer': str(r['answer']), 'problem_type': list(r['problem_type'] or [])} for r in hmmt]
    m500 = load_tier('math500')
    tests['math500'] = [{'id': 'math500/' + r['unique_id'], 'problem': r['problem'], 'answer': r['answer'],
                         'subject': r['subject'], 'level': int(r['level'])} for r in m500]
    for name, rows in tests.items():
        assert len({r['id'] for r in rows}) == len(rows), name
        for r in rows:
            r['problem_sha256'] = sl.sha(' '.join(r['problem'].split()).encode())
            r['answer_parseable'] = parses(r['answer'])
    m500_sha = {r['problem_sha256'] for r in tests['math500']}

    # 3. Tier-1 sources from the MATH test split, chosen by seed order only.
    from datasets import load_dataset
    m500_index = Index(sl, tests['math500'])
    candidates = {level: [] for level in range(1, 6)}
    skipped = {'diagram': 0, 'reference_parse': 0, 'missing_level': 0, 'math500_exact': 0, 'math500_near': 0}
    for subject in sl.SUBJECTS:
        for i, row in enumerate(load_dataset(sl.DATASET, subject, split='test', revision=sl.REVISION)):
            problem = row['problem']
            fp = sl.sha(' '.join(problem.split()).encode())
            if '[asy]' in problem or '\\includegraphics' in problem:
                skipped['diagram'] += 1
                continue
            gold, parsed = parse_answer(row['solution'])
            if not parsed:
                skipped['reference_parse'] += 1
                continue
            level = re.search(r'\d+', str(row['level']))
            if not level or int(level.group()) not in range(1, 6):
                skipped['missing_level'] += 1
                continue
            if fp in m500_sha:
                skipped['math500_exact'] += 1
                continue
            if m500_index.matches(problem, JACCARD)[0]:
                skipped['math500_near'] += 1
                continue
            candidates[int(level.group())].append({'id': f'{subject}/test/{i}', 'subject': subject,
                                                   'level': int(level.group()), 'problem': problem, 'gold': gold,
                                                   'solution': row['solution'], 'problem_sha256': fp})
    key = lambda r: sl.sha(f'{SEED}:variant-source:{r["problem_sha256"]}'.encode())
    sources = []
    for level in range(1, 6):
        ordered = sorted(candidates[level], key=key)[:PRIMARY + RESERVE]
        assert len(ordered) == PRIMARY + RESERVE, level
        sources += [{**r, 'role': 'primary' if k < PRIMARY else 'reserve', 'order': k} for k, r in enumerate(ordered)]
    tests['variant_sources'] = sources

    # 4. Deduplicate training data against every test item.
    removed, review, overlaps = {}, [], []
    for name, rows in tests.items():
        for r in rows:
            near, exact = index.matches(r['problem'], REVIEW)
            for i in exact:
                removed.setdefault(pool[i]['id'], []).append({'test': r['id'], 'match': 'exact'})
            for j, i in near:
                if j >= JACCARD:
                    removed.setdefault(pool[i]['id'], []).append({'test': r['id'], 'match': 'jaccard', 'jaccard': j})
                else:
                    review.append({'test': r['id'], 'pool': pool[i]['id'], 'jaccard': j})
            r['max_pool_jaccard'] = near[0][0] if near else 0.0
    dev_refs = [('dev:' + i, sl.trigrams(d['problem'])) for i, d in sorted(dev.items())]
    dev_refs += [('perturb:' + k, sl.trigrams(p)) for k, p in perturb]
    for name, rows in tests.items():  # report only: development problems are never trained on
        for r in rows:
            g = sl.trigrams(r['problem'])
            for k, h in dev_refs:
                j = sl.jaccard(g, h)
                if j >= JACCARD:
                    overlaps.append({'test': r['id'], 'reference': k, 'jaccard': round(j, 4)})
    earlier_near = {r['id'] for r, g in zip(pool, index.grams) if r['id'] not in dev
                    and max((sl.jaccard(g, h) for _, h in dev_refs), default=0.0) >= JACCARD}
    assert len(earlier_near) == 204, len(earlier_near)  # length_selection.json near_duplicates_excluded
    eligible = [r for r in pool if r['id'] not in dev and r['id'] not in earlier_near and r['id'] not in removed]

    # 5. Write.
    out_dir.mkdir(parents=True)
    files = {}

    def dump(name, rows):
        body = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows).encode('utf-8')
        (out_dir / name).write_bytes(body)
        files[name] = {'rows': len(rows), 'sha256': sl.sha(body)}

    dump('tier2_aime2026.jsonl', tests['aime2026'])
    dump('tier2_hmmt_feb2026.jsonl', tests['hmmt_feb2026'])
    dump('tier3_math500.jsonl', tests['math500'])
    dump('tier1_variant_sources.jsonl', tests['variant_sources'])
    dump('train_eligible_pool.jsonl', eligible)
    dedup = {'rule': f'remove pool problems with identical normalized text or word-trigram Jaccard >= {JACCARD} '
                     'to any test item; list 0.3 <= Jaccard < 0.5 pairs for review only',
             'removed_for_tests': [{'id': k, 'matches': v} for k, v in sorted(removed.items())],
             'review_pairs': sorted(review, key=lambda x: -x['jaccard']),
             'test_vs_development_overlaps': overlaps}
    body = json.dumps(dedup, indent=1, ensure_ascii=False).encode('utf-8')
    (out_dir / 'train_test_dedup.json').write_bytes(body)
    files['train_test_dedup.json'] = {'sha256': sl.sha(body)}
    manifest = {
        'stage': 1, 'seed': SEED, 'versions': versions,
        'pool': {'dataset': sl.DATASET, 'revision': sl.REVISION, 'split': 'train', 'size': len(pool),
                 'exclusions_reproduced': excluded, 'development_ids': len(dev),
                 'near_dev_or_perturb': len(earlier_near), 'removed_for_tests': len(removed),
                 'eligible': len(eligible)},
        'tiers': {k: {'repo': v[0], 'revision': v[1], 'split': v[2], 'rows': v[3],
                      'answer_unparseable': [r['id'] for r in tests[k] if not r['answer_parseable']]}
                  for k, v in TIERS.items()},
        'variant_sources': {'dataset': sl.DATASET, 'revision': sl.REVISION, 'split': 'test',
                            'skipped': skipped, 'candidates_per_level': {str(k): len(v) for k, v in candidates.items()},
                            'order': 'ascending sha256(f"{SEED}:variant-source:{problem_sha256}") within each level',
                            'primary_per_level': PRIMARY, 'reserve_per_level': RESERVE},
        'files': files,
        'note': 'Frozen before any model sees the test items. Stage 2 adds the variants, checks them against '
                'train_eligible_pool.jsonl with the same rule and writes the final manifest.'}
    (out_dir / 'freeze_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    print('FREEZE_STAGE1_DONE', json.dumps(manifest['pool']), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
