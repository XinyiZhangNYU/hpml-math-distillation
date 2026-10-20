"""Read-only check: can the project grader (eval_common.parse_answer / grade, math-verify)
parse the 90 tier-1 variant reference answers, accept them, reject a wrong value,
and accept common equivalent forms?  Spec: stage2_run1/math_verify_check_prompt.md.
usage: python math_verify_check.py EVAL_COMMON TIER1_JSONL OUT_JSON   (Linux only)"""
import hashlib, importlib.metadata as md, importlib.util, json, platform, re, sys

EXPECTED = {
    'eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'tier1_variants.jsonl': 'c5e1eb3cea9ddcabdadd93952951cc1673968083c113662926c8c2245e16331b',
}
EQUIV = [
    (r'-\frac{9}{7}', '-9/7'), (r'\frac{1}{9}', '1/9'), (r'\frac{2}{5}', '0.4'),
    (r'-\frac{1}{23}', '-1/23'), (r'4\pi', r'4 \pi'),
    (r'\frac{4\pi}{3}', r'\dfrac{4\pi}{3}'), (r'\frac{4\pi}{3}', r'4\pi/3'),
    (r'\text{Friday}', 'Friday'), (r'\text{Thursday}', r'\textbf{Thursday}'),
    (r'60^\circ', '60'), (r'60^\circ', r'60^{\circ}'), (r'\frac{225}{4}', '56.25'),
    (r'\text{ACF}', 'ACF'), (r'\frac{4}{3}', '4/3'), (r'\frac{1}{2}', '0.5'),
    ('1202_3', '1202_{3}'), (r'\sqrt{6}', r'\sqrt6'),
    (r'\frac{\sqrt{11}}{2}', r'\frac{1}{2}\sqrt{11}'), (r'\frac{3}{2}', '1.5'),
    (r'2\sqrt{11}', r'\sqrt{44}'),
    (r'(-\infty,-4)\cup(-4,3)\cup(3,\infty)', r'(-\infty, -4) \cup (-4, 3) \cup (3, \infty)'),
    (r'-\frac{15}{2}', '-7.5'), (r'\frac{2a + b}{2}', r'a + \frac{b}{2}'),
    (r'\frac{2ab}{a + b}', r'\frac{2ab}{b + a}'),
]
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def wrong_value(a):
    """Wrong answer per the spec's rules; returns (value, rule)."""
    if re.fullmatch(r'-?\d+', a):
        return str(int(a) + 1), 'integer+1'
    m = re.fullmatch(r'(\d+)\^\\circ', a)
    if m:
        return str(int(m.group(1)) + 1) + r'^\circ', 'degrees+1'
    m = re.fullmatch(r'\\text\{(\w+)\}', a)
    if m and m.group(1) in WEEKDAYS:
        return r'\text{' + ('Tuesday' if m.group(1) == 'Monday' else 'Monday') + '}', 'other weekday'
    if m:
        return r'\text{' + ('ABD' if m.group(1) == 'ABC' else 'ABC') + '}', 'other letters'
    m = re.fullmatch(r'(\d+)_(\d+)', a)
    if m:
        return m.group(1)[:-1] + str(int(m.group(1)[-1]) + 1) + '_' + m.group(2), 'base: last digit+1'
    if r'\cup' in a:
        m = re.search(r'(?<![\\\w])-?\d+', a)
        return a[:m.start()] + str(int(m.group()) + 1) + a[m.end():], 'interval: first finite endpoint+1'
    return a + '+1', 'expression+1'


def main(eval_path, rows_path, out_path):
    hashes = {'eval_common.py': sha(eval_path), 'tier1_variants.jsonl': sha(rows_path)}
    if hashes != EXPECTED:
        sys.exit('hash mismatch: ' + json.dumps(hashes))
    spec = importlib.util.spec_from_file_location('eval_common', eval_path)
    ec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec)
    rows = [json.loads(l) for l in open(rows_path, encoding='utf-8')]
    assert len(rows) == 90
    results = []
    for r in rows:
        a = r['answer']
        b = '\\boxed{' + a + '}'
        extracted, parsed = ec.parse_answer(b)
        w, rule = wrong_value(a)
        try:
            self_ok = ec.grade(b, a)['correct']
            wrong_ok = ec.grade('\\boxed{' + w + '}', a)['correct']
            err = None
        except Exception as e:
            self_ok = wrong_ok = None
            err = repr(e)
        results.append({
            'id': r['id'], 'answer': a,
            'parse_ok': bool(parsed), 'parsed': [str(p) for p in parsed],
            'self_correct': self_ok, 'wrong_value': w, 'wrong_rule': rule,
            'wrong_graded_correct': wrong_ok, 'error': err,
            'pass': bool(parsed) and self_ok is True and wrong_ok is False,
        })
    equiv = []
    for gold, alt in EQUIV:
        try:
            g = ec.grade('\\boxed{' + alt + '}', gold)
            equiv.append({'gold': gold, 'alt': alt, 'correct': g['correct'], 'parseable': g['parseable']})
        except Exception as e:
            equiv.append({'gold': gold, 'alt': alt, 'correct': None, 'error': repr(e)})
    versions = {p: md.version(p) for p in ('math-verify', 'latex2sympy2_extended', 'sympy')}
    versions['python'] = platform.python_version()
    versions['platform'] = platform.platform()
    n_pass = sum(x['pass'] for x in results)
    n_eq = sum(x['correct'] is True for x in equiv)
    failed = [x for x in results if not x['pass']]
    eq_bad = [x for x in equiv if x['correct'] is not True]
    summary = (f"checks 1-3: {n_pass}/90 pass; equivalent forms: {n_eq}/{len(equiv)} accepted"
               + ('' if not eq_bad else '; rejected: ' + ', '.join(f"{x['alt']} vs {x['gold']}" for x in eq_bad)))
    report = {'versions': versions, 'input_sha256': hashes, 'summary': summary,
              'checks_1_3_failed': failed, 'checks_1_3': results, 'check_4_equivalents': equiv}
    json.dump(report, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(versions))
    print(summary)
    for x in failed:
        print('FAIL', x['id'], x['answer'], x['parsed'], x['self_correct'], x['wrong_value'], x['wrong_graded_correct'], x['error'])


if __name__ == '__main__':
    main(*sys.argv[1:4])
