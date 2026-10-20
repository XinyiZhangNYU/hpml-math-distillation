"""Validation-only protocol; none of the final MATH-500 questions are used."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path('/content/hpml_validation')
MODEL = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
SEED = 91432026
PROMPT = 'Solve the following mathematical problem. Explain the necessary steps concisely and put the final answer in \\boxed{}.\n\n'
PARAMS = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
              repetition_penalty=1.0, max_tokens=4096)


def write_json(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(p)


def boxed(text):
    found = None
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    found = text[start:i]
                    break
    return found


def parse_answer(text):
    from math_verify import parse, LatexExtractionConfig
    answer = boxed(text.split('</think>')[-1])
    if answer is None:
        return None, []
    parsed = parse('$' + answer + '$', extraction_config=[LatexExtractionConfig()])
    return answer, parsed


def grade(text, gold):
    from math_verify import verify
    answer, pred = parse_answer(text)
    _, expected = parse_answer('\\boxed{' + gold + '}')
    if not expected:
        raise ValueError('Unparseable reference: ' + gold)
    try:
        correct = bool(pred and verify(expected, pred))
    except Exception:
        correct = False
    return {'extracted_answer': answer, 'parseable': bool(pred), 'correct': correct}


def parser_checks():
    cases = [
        (r'\boxed{\frac{1}{2}}', r'\frac{2}{4}', True),
        (r'\boxed{\sqrt{4}}', '2', True),
        (r'\boxed{7} then \boxed{8}', '8', True),
        (r'<think>\boxed{99}</think> Final: \boxed{3}', '3', True),
        (r'\boxed{4}', '3', False),
        ('No final answer', '3', False),
    ]
    rows = []
    for prediction, gold, expected in cases:
        result = grade(prediction, gold)
        assert result['correct'] == expected, (prediction, gold, result)
        rows.append({'prediction': prediction, 'gold': gold, 'expected': expected, **result})
    write_json('parser_checks.json', rows)
    return rows


def prepare_data():
    import random
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset_id = 'EleutherAI/hendrycks_math'
    revision = HfApi().dataset_info(dataset_id).sha
    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    pools = {level: [] for level in range(1, 6)}
    excluded = {'diagram': 0, 'reference_parse': 0, 'duplicate': 0, 'missing_level': 0}
    seen = set()
    for subject in subjects:
        dataset = load_dataset(dataset_id, subject, split='train', revision=revision)
        for i, row in enumerate(dataset):
            problem = row['problem']
            fingerprint = hashlib.sha256(' '.join(problem.split()).encode()).hexdigest()
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
            if not level_match or int(level_match.group()) not in pools:
                excluded['missing_level'] += 1
                continue
            level = int(level_match.group())
            pools[level].append({'id': subject + '/train/' + str(i), 'subject': subject,
                                 'level': level, 'problem': problem, 'gold': gold,
                                 'human_solution': row['solution'], 'problem_sha256': fingerprint})
        print('DATA_SUBJECT_READY', subject, flush=True)
    rng = random.Random(SEED)
    validation, teacher = [], []
    for level in range(1, 6):
        rng.shuffle(pools[level])
        assert len(pools[level]) >= 30
        validation += pools[level][:20]
        teacher += pools[level][20:30]
    rng.shuffle(validation)
    rng.shuffle(teacher)
    assert len({x['problem_sha256'] for x in validation + teacher}) == 150
    for name, rows in [('validation_100', validation), ('teacher_candidates_50', teacher)]:
        (ROOT / (name + '.jsonl')).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    write_json('data_manifest.json', {
        'dataset': dataset_id, 'dataset_revision': revision, 'source_split': 'train',
        'validation_count': len(validation), 'per_level': 20, 'teacher_candidate_count': len(teacher),
        'seed': SEED, 'exclusions': excluded, 'final_test_evaluated': False,
        'loader_note': 'Dataset library may materialize its other splits in cache; selection and all measurements use train only. No MATH-500 or MATH test questions are read for this study.',
        'future_training_rule': 'Exclude all 100 validation question IDs and exact/near duplicates from final training.',
        'scope': 'Stratified development validation; scores are NOT MATH-500 test scores.',
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [ROOT/'validation_100.jsonl', ROOT/'teacher_candidates_50.jsonl']}})
    return validation


def load_rows():
    return [json.loads(x) for x in (ROOT/'validation_100.jsonl').read_text().splitlines()]


def summarize(rows, wall_seconds):
    import statistics
    n = len(rows)
    successes = sum(x['correct'] for x in rows)
    accuracy = successes / n if n else 0.0
    z = 1.959963984540054
    denom = 1 + z*z/n if n else 1
    centre = (accuracy + z*z/(2*n))/denom if n else 0
    half = z*math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/denom if n else 0
    tokens = sum(x['output_tokens'] for x in rows)
    return {'n': n, 'correct': successes, 'accuracy': accuracy,
            'wilson_95_ci': [max(0, centre-half), min(1, centre+half)],
            'mean_output_tokens': tokens/n if n else None,
            'median_output_tokens': statistics.median(x['output_tokens'] for x in rows) if n else None,
            'total_output_tokens': tokens, 'wall_seconds': wall_seconds,
            'aggregate_tokens_per_second': tokens/wall_seconds if wall_seconds else None,
            'truncated': sum(x.get('finish_reason') == 'length' for x in rows),
            'unparseable_outputs': sum(not x['parseable'] for x in rows),
            'total_output_tokens_per_correct': tokens/successes if successes else None,
            'total_wall_seconds_per_correct': wall_seconds/successes if successes else None,
            'by_level': {str(level): {'n': sum(x['level'] == level for x in rows),
                         'correct': sum(x['correct'] for x in rows if x['level'] == level)}
                         for level in range(1,6)}}
