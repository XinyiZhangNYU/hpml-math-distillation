"""Offline dry run of teacher_gen.py with a fake transport, grader and counter. No API call, no network, no key.

Usage: python -I dry_run_teacher_gen.py TEACHER_GEN_DIR WORK_DIR   (WORK_DIR must not exist)

Each problem of batch b0 gets a scripted behaviour from its order number (order % 10):
  0 wrong, then right      -> kept on attempt 2      5 no box, then right       -> kept on attempt 2
  1 wrong twice            -> dropped                6 request error once, then right -> kept on attempt 1 (resent)
  2 truncated, then right  -> kept on attempt 2      7 empty reasoning twice    -> dropped
  3 over 32k twice         -> dropped                8, 9 right                 -> kept on attempt 1
  4 right
Call 1 crashes (KeyboardInterrupt) after 30 requests; call 2 stops at the error limit (set to 3); call 3 finishes.
A crashed request never returned, so the fake counts only answered requests as attempts.
Then the ledger is checked: no paid response repeated, attempt 2 only after a failed attempt 1, outcomes as scripted,
costs add up. Last, batch b1 with a budget below one reservation must send nothing.
"""
import importlib.util
import json
import shutil
import sys
import threading
from pathlib import Path

src, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
assert not work.exists(), 'WORK_DIR exists'
work.mkdir(parents=True)
for name in ('teacher_gen.py', 'train2000.jsonl'):
    shutil.copyfile(src / name, work / name)
spec = importlib.util.spec_from_file_location('teacher_gen', work / 'teacher_gen.py')
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)

KEY = 'DRYRUN_FAKE_KEY_NOT_REAL'
problems = [json.loads(x) for x in (work / 'train2000.jsonl').read_text(encoding='utf-8').splitlines()]
order = {p['id']: p['order'] for p in problems}
gold = {p['id']: p['gold'] for p in problems}
problem_of = {tg.PROMPT + p['problem']: p['id'] for p in problems}
lock = threading.Lock()
seen = {'calls': 0, 'per_id': {}, 'crash_after': None, 'errored_once': set()}


def fake_post(body):
    pid = problem_of[body['messages'][0]['content']]
    assert body['model'] == 'deepseek-flash' and 'thinking' not in body and body['max_tokens'] == 65536
    with lock:
        seen['calls'] += 1
        if seen['crash_after'] is not None and seen['calls'] > seen['crash_after']:
            raise KeyboardInterrupt('simulated crash')
        k = order[pid] % 10
        if k == 6 and pid not in seen['errored_once']:
            seen['errored_once'].add(pid)
            raise TimeoutError('simulated timeout')
        n = seen['per_id'][pid] = seen['per_id'].get(pid, 0) + 1  # answered responses only
    attempt_two = n >= 2
    answer, reasoning, finish = '\\boxed{' + gold[pid] + '}', 'short reasoning', 'stop'
    if k in (0, 1) and (not attempt_two or k == 1):
        answer = '\\boxed{WRONG}'
    if k == 2 and not attempt_two:
        finish, answer = 'length', ''
    if k == 3:
        reasoning = 'x' * 40000
    if k == 5 and not attempt_two:
        answer = 'no box here'
    if k == 7:
        reasoning = ''
    return 200, {'id': 'fake-%s-%d' % (pid, n), 'model': 'deepseek-flash', 'system_fingerprint': 'fake',
                 'choices': [{'finish_reason': finish, 'message': {'content': answer, 'reasoning_content': reasoning}}],
                 'usage': {'prompt_tokens': 100, 'completion_tokens': 1000 + len(reasoning),
                           'completion_tokens_details': {'reasoning_tokens': len(reasoning)}}}


def fake_grade(text, g):
    import re
    m = re.findall(r'\\boxed\{(.*)\}', text)
    return {'extracted_answer': m[-1] if m else None, 'has_boxed': bool(m), 'correct': bool(m) and m[-1] == g}


def fake_count(prompt, reasoning, answer):
    return len(prompt) + len(reasoning) + len(answer)  # 40,000 characters of reasoning -> over 32,768


kw = dict(grade_fn=fake_grade, count_fn=fake_count, transport=fake_post, workers=4)
b0 = [p for p in problems if p['batch'] == 'b0']

# call 1: crash after 30 responses
seen['crash_after'] = 30
try:
    tg.run(KEY, work, 'b0', 3.0, **kw)
    raise AssertionError('call 1 should have crashed')
except KeyboardInterrupt:
    pass
raw1 = [json.loads(x) for x in (work / 'raw_responses.jsonl').read_text(encoding='utf-8').splitlines()]
assert 25 <= len(raw1) <= 31, len(raw1)
print('CALL1 crashed after', len(raw1), 'saved responses')

# call 2: no crash, but the error limit is set to 3 so it stops early
seen['crash_after'] = None
tg.MAX_ERRORS = 3
s2 = tg.run(KEY, work, 'b0', 3.0, **kw)
calls = [json.loads(x) for x in (work / 'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert calls[-1]['stopped'] == 'error limit', calls[-1]
print('CALL2 stopped:', calls[-1]['stopped'], 'pending', s2['by_batch']['b0']['pending'])

# call 3: normal limit, finishes the batch
tg.MAX_ERRORS = 5
s3 = tg.run(KEY, work, 'b0', 3.0, **kw)
b = s3['by_batch']['b0']
print('CALL3', json.dumps({k: b[k] for k in ('problems', 'kept', 'kept_on_attempt_2', 'dropped', 'pending', 'request_errors',
                                              'attempt_1_failure_reasons', 'attempt_2_failure_reasons', 'priced_at_peak_usd')}))

raw = [json.loads(x) for x in (work / 'raw_responses.jsonl').read_text(encoding='utf-8').splitlines()]
grades = [json.loads(x) for x in (work / 'grades.jsonl').read_text(encoding='utf-8').splitlines()]
status = {s['id']: s for s in map(json.loads, (work / 'problem_status.jsonl').read_text(encoding='utf-8').splitlines())}
ok = [r for r in raw if not r['error']]
keys = [(r['id'], r['attempt']) for r in ok]
assert len(keys) == len(set(keys)), 'a paid response was repeated'
assert len(grades) == len({(g['id'], g['attempt']) for g in grades}) == len(ok), 'every response graded exactly once'
first_ok = {g['id']: g['admitted'] for g in grades if g['attempt'] == 1}
assert all(first_ok[r['id']] is False for r in ok if r['attempt'] == 2), 'attempt 2 only after a failed attempt 1'
expect = {0: ('kept', 2), 1: ('dropped', None), 2: ('kept', 2), 3: ('dropped', None), 4: ('kept', 1),
          5: ('kept', 2), 6: ('kept', 1), 7: ('dropped', None), 8: ('kept', 1), 9: ('kept', 1)}
for p in b0:
    st, att = expect[p['order'] % 10]
    s = status[p['id']]
    assert s['status'] == st and s.get('kept_attempt') == att, (p['id'], p['order'], s)
assert all(status[p['id']]['status'] == 'pending' for p in problems if p['batch'] != 'b0')
final = [json.loads(x) for x in (work / 'final_traces.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(final) == b['kept'] == 70 and b['dropped'] == 30 and b['pending'] == 0
assert all(f['content'].endswith('{' + f['gold'] + '}') for f in final)
dropped = {s['id']: s for s in status.values() if s['status'] == 'dropped'}
assert {tuple(s['reasons_attempt_2']) for s in dropped.values()} == {('wrong',), ('over_32k',), ('empty_reasoning',)}
errors = [r for r in raw if r['error']]
assert len(errors) == sum(p['order'] % 10 == 6 for p in b0) and all(r['priced_at_peak_usd'] > .07 for r in errors)
total = sum(r['priced_at_peak_usd'] for r in raw)
assert abs(total - s3['all']['priced_at_peak_usd']) < 1e-6
calls = [json.loads(x) for x in (work / 'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert abs(calls[-1]['ledger_spent_after_usd'] - total) < 1e-6, (calls[-1], total)
print('LEDGER', len(raw), 'responses (', len(errors), 'request errors ), priced', round(total, 4), 'USD; calls so far', len(calls))

# budget guard: batch b1 with a budget below one worst-case reservation sends nothing
before = seen['calls']
s4 = tg.run(KEY, work, 'b1', 0.05, **kw)
calls = [json.loads(x) for x in (work / 'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert seen['calls'] == before and calls[-1]['stopped'] == 'budget' and calls[-1]['sent'] == 0
# total cap: pretend the ledger is near USD 10 by lowering the cap
tg.TOTAL_CAP = total + 0.05
s5 = tg.run(KEY, work, 'b1', 5.0, **kw)
calls = [json.loads(x) for x in (work / 'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert seen['calls'] == before and calls[-1]['stopped'] == 'budget' and calls[-1]['sent'] == 0
print('BUDGET guard: b1 sent nothing under a USD 0.05 batch budget and under a total cap just above the ledger')
for p in (q for q in work.iterdir() if q.is_file()):
    assert KEY not in p.read_text(encoding='utf-8', errors='replace'), p.name
print('DRY_RUN_PASSED')
