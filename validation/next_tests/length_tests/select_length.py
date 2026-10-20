"""Problem selection for the teacher length test. No API call, no model.

Rebuilds the MATH train pool exactly as validation/eval_common.py prepare_data() did (same dataset revision,
same order, same duplicate / diagram / reference-parse / level exclusions), checks that every development problem
is reproduced with the same problem_sha256 and gold, then removes from the pool:
  - the three development sets: validation_100, teacher_candidates_50, teacher_math_50 (150 unique ids);
  - near-duplicates (word-trigram Jaccard >= 0.5) of any development problem or any MATH-Perturb hard/simple problem.
From what remains it takes 40 problems per level, ordered by sha256(f"{SEED}:{problem_sha256}") (set "pool"),
and adds the 20 thinking-pilot problems from teacher_math_50 (set "paired"), so the same 20 can be compared
with and without "concisely" in the prompt.

Usage (Linux / Colab; math-verify needs signal.alarm and the main thread):
    python3 -I select_length.py PROJECT_ROOT OUT_DIR
Writes OUT_DIR/length_problems.jsonl and OUT_DIR/length_selection.json.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

DATASET = 'EleutherAI/hendrycks_math'
REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'
SUBJECTS = ['algebra', 'counting_and_probability', 'geometry',
            'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
SEED = 91432026
PER_LEVEL = 40
JACCARD = 0.5
# Exclusion counts recorded by the original prepare_data() run (validation/evidence/data_manifest.json).
EXPECTED_EXCLUDED = {'diagram': 707, 'reference_parse': 5, 'duplicate': 1, 'missing_level': 2}
DEV_FILES = {'validation/evidence/validation_100.jsonl':
             '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0',
             'validation/evidence/teacher_candidates_50.jsonl':
             '89a3fa804de3510c5d65b12a5a36687b493eaae2dc1db1b85e3510e360b3269b',
             'validation/next_tests/teacher_math50/teacher_math_50.jsonl':
             'a1a423640fe001862b74153d32acbda219993d1c418796d3208059a7f9bdc930'}
EVAL_COMMON = ('validation/eval_common.py', '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76')
PERTURB = {'validation/next_tests/perturb_materials/math_perturb_hard.jsonl':
           '418ca51e02f3b639a919dda72b5201ce35ac6c6d7d31c3231735beb4ea012dd7',
           'validation/next_tests/perturb_materials/math_perturb_simple.jsonl':
           '2130821d4235dc1f5d962bee7f84a096651bf98fcb17b6dcfd3d0a48875ba5bb'}
# The 20 problems of the thinking-mode pilot, 4 per level.
PAIRED = ('algebra/train/448', 'intermediate_algebra/train/231', 'algebra/train/273', 'prealgebra/train/1138',
          'algebra/train/302', 'precalculus/train/642', 'algebra/train/1108', 'algebra/train/1156',
          'precalculus/train/659', 'counting_and_probability/train/157', 'prealgebra/train/252', 'algebra/train/82',
          'counting_and_probability/train/314', 'intermediate_algebra/train/996', 'prealgebra/train/1010',
          'intermediate_algebra/train/420',
          'number_theory/train/475', 'prealgebra/train/154', 'intermediate_algebra/train/782', 'number_theory/train/255')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_checked(path, digest):
    data = path.read_bytes()
    assert digest is None or sha(data) == digest, ('hash mismatch', str(path))
    return data


def jsonl(data):
    return [json.loads(x) for x in data.decode('utf-8').splitlines() if x.strip()]


def trigrams(text):
    words = re.findall(r'\w+|[^\w\s]', text.lower())
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def build_pool(parse_answer):
    """Same loop as eval_common.prepare_data(), without the shuffle and split."""
    from datasets import load_dataset
    pool = []
    excluded = {'diagram': 0, 'reference_parse': 0, 'duplicate': 0, 'missing_level': 0}
    seen = set()
    for subject in SUBJECTS:
        dataset = load_dataset(DATASET, subject, split='train', revision=REVISION)
        for i, row in enumerate(dataset):
            problem = row['problem']
            fingerprint = sha(' '.join(problem.split()).encode())
            if fingerprint in seen:
                excluded['duplicate'] += 1
                continue
            seen.add(fingerprint)
            if '[asy]' in problem or '\\includegraphics' in problem:
                excluded['diagram'] += 1
                continue
            gold, parsed = parse_answer(row['solution'])
            if not parsed:
                excluded['reference_parse'] += 1
                continue
            level_match = re.search(r'\d+', str(row['level']))
            if not level_match or int(level_match.group()) not in range(1, 6):
                excluded['missing_level'] += 1
                continue
            pool.append({'id': subject + '/train/' + str(i), 'subject': subject,
                         'level': int(level_match.group()), 'problem': problem, 'gold': gold,
                         'problem_sha256': fingerprint})
        print('DATA_SUBJECT_READY', subject, flush=True)
    return pool, excluded


def main(project, out_dir):
    import signal
    import threading
    from importlib import metadata
    assert hasattr(signal, 'alarm') and threading.current_thread() is threading.main_thread(), \
        'Run under Linux / Colab in the main thread (math-verify uses signal.alarm).'
    versions = {n: metadata.version(n) for n in ('math-verify', 'datasets', 'huggingface_hub')}
    assert versions['math-verify'] == '0.9.0', versions
    out_dir.mkdir(parents=True, exist_ok=True)
    assert not (out_dir / 'length_problems.jsonl').exists(), 'Output exists: inspect it instead of overwriting.'

    # eval_common.py must be the copy used for the original split (identical to validation/evidence/eval_common.py).
    read_checked(project / EVAL_COMMON[0], EVAL_COMMON[1])
    sys.path.insert(0, str(project / 'validation'))
    from eval_common import parse_answer

    dev = {}
    for rel, digest in DEV_FILES.items():
        for r in jsonl(read_checked(project / rel, digest)):
            assert dev.setdefault(r['id'], r)['problem_sha256'] == r['problem_sha256'], r['id']
    assert len(dev) == 150, len(dev)  # validation_100 and teacher_candidates_50 are disjoint; teacher_math_50 is a subset of validation_100
    math50 = {r['id']: r for r in jsonl((project / 'validation/next_tests/teacher_math50/teacher_math_50.jsonl').read_bytes())}
    assert set(PAIRED) <= set(math50)

    perturb = []
    for name, digest in PERTURB.items():
        rows = jsonl(read_checked(project / name, digest))
        assert len(rows) == 279, name
        perturb += [(name.split('/')[-1] + '#' + str(r['problem_id']), r['problem']) for r in rows]

    pool, excluded = build_pool(parse_answer)
    assert excluded == EXPECTED_EXCLUDED, ('pool differs from the original prepare_data run', excluded)
    by_id = {r['id']: r for r in pool}
    for i, d in dev.items():  # every development problem must be reproduced exactly
        assert i in by_id and by_id[i]['problem_sha256'] == d['problem_sha256'] and by_id[i]['gold'] == d['gold'], i

    refs = [('dev:' + i, trigrams(d['problem'])) for i, d in sorted(dev.items())]
    refs += [('perturb:' + k, trigrams(p)) for k, p in perturb]
    near, candidates = [], {level: [] for level in range(1, 6)}
    for r in pool:
        if r['id'] in dev:
            continue
        grams = trigrams(r['problem'])
        best = max(((jaccard(grams, g), k) for k, g in refs), default=(0.0, None))
        if best[0] >= JACCARD:
            near.append({'id': r['id'], 'level': r['level'], 'matched': best[1], 'jaccard': round(best[0], 4)})
            continue
        candidates[r['level']].append(r)

    key = lambda r: sha(f'{SEED}:{r["problem_sha256"]}'.encode())
    chosen = []
    for level in range(1, 6):
        picked = sorted(candidates[level], key=key)[:PER_LEVEL]
        assert len(picked) == PER_LEVEL, level
        chosen += [{**r, 'set': 'pool'} for r in picked]
    for i in PAIRED:
        m = math50[i]
        chosen.append({k: m[k] for k in ('id', 'subject', 'level', 'problem', 'gold', 'problem_sha256')} | {'set': 'paired'})
    assert len({r['id'] for r in chosen}) == len(chosen) == 5 * PER_LEVEL + len(PAIRED)

    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in chosen).encode('utf-8')
    (out_dir / 'length_problems.jsonl').write_bytes(body)
    record = {
        'dataset': DATASET, 'dataset_revision': REVISION, 'source_split': 'train', 'seed': SEED,
        'versions': versions, 'pool_exclusions_reproduced': excluded, 'pool_size': len(pool),
        'development_ids_excluded': len(dev), 'development_reproduced': True,
        'near_duplicate_rule': f'word-trigram Jaccard >= {JACCARD} against the 150 development problems and the '
                               f'{len(perturb)} MATH-Perturb hard+simple problems (commit df4840f6, training not permitted)',
        'near_duplicates_excluded': len(near), 'near_duplicates': near,
        'candidates_per_level': {str(k): len(v) for k, v in candidates.items()},
        'per_level': PER_LEVEL, 'order': 'ascending sha256(f"{SEED}:{problem_sha256}") within each level',
        'paired_ids': list(PAIRED),
        'counts': {s: {str(l): sum(r['set'] == s and r['level'] == l for r in chosen) for l in range(1, 6)}
                   for s in ('pool', 'paired')},
        'files': {'length_problems.jsonl': sha(body)},
        'scope': 'Measurement of teacher trace length only. Pool problems stay eligible for the training pool; '
                 'paired problems are development problems and are never used for training.'}
    (out_dir / 'length_selection.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print('LENGTH_SELECTION_DONE', len(chosen), 'near_duplicates', len(near), 'sha256', sha(body), flush=True)
    return record


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
