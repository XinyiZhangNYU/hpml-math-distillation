"""Teacher generation: deepseek-flash thinking traces for the 2,000 training problems.

Request settings: the prompt below (without "concisely"), no thinking field (API default thinking; effort not sent, default high), temperature 0.6 (no effect in
thinking mode), max_tokens 65536, timeout 1800 s, stream false. One trace per problem.

Admission: finish_reason stop, non-empty reasoning_content,
a \\boxed in content, math_grade correct against gold (base_probe_worker.py, math-verify 0.9.0, last box), and the Qwen
SFT sequence at most 32,768 tokens. A problem whose attempt 1 fails is resampled once with the same request (attempt 2);
if attempt 2 fails too, the problem is dropped from all groups. Nothing is truncated. Failed attempts stay in the ledger.

Ledger and resume (run() can be called again on the same root and batch after a crash or a stop):
- raw_responses.jsonl, append-only: each response or request error is appended and fsynced before any grading.
- grades.jsonl, append-only: one line per graded response, written in the main thread (math-verify uses signal.alarm).
- Whatever is already answered is skipped. A request that ended in an error is sent once more in the same call, and
  again on the next call; a resend is not an attempt. In-flight requests lost in a crash are not in the ledger and are sent again.
- Budget: each request reserves its worst case (request bytes + 512 input tokens, 65536 output tokens, peak rates)
  before it is sent; after the response the reservation is replaced by the priced usage. Requests that end in an
  error keep the full reservation. Two caps, both checked before every request:
  the batch budget passed to run() (all responses of this batch in the ledger count), and TOTAL_CAP over the whole ledger.
- After MAX_ERRORS request errors in one call, no new requests are sent.
Derived files (rewritten after every call): final_traces.jsonl (one admitted trace per problem), problem_status.jsonl,
summary.json, manifest.json. The key is passed in memory, never written; every written file is checked for it at the end.
"""
import collections
import concurrent.futures
import datetime
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-flash'
IN_RATE, OUT_RATE = .30, 1.20  # USD per million tokens at peak (DeepSeek pricing page, checked 2026-10-18T01:22Z)
TOTAL_CAP = 10.00  # USD, hard cap for the 2,000 problems
MAX_TOKENS = 65536
QWEN_LIMIT = 32768
WORKERS = 8
MAX_ERRORS = 5
PROMPT = 'Solve the following mathematical problem. Explain the necessary steps and put the final answer in \\boxed{}.\n\n'
TOKENIZER = 'Qwen/Qwen3.5-9B-Base'
TOKENIZER_REVISION = '68c46c4b3498877f3ef123c856ecfde50c39f404'
SCRIPT_VERSION = 1


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def layout(prompt, reasoning, answer):  # same SFT layout as length_tests/count_qwen_tokens.py
    head = '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    return head, head + '<think>\n' + reasoning.strip() + '\n</think>\n\n' + answer.strip() + '<|im_end|>\n'


def qwen_counter():
    """Count the full SFT sequence with the student's tokenizer (tokenizer files only)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)

    def count(prompt, reasoning, answer):
        head, full = layout(prompt, reasoning, answer)
        head_ids = tok(head, add_special_tokens=False)['input_ids']
        ids = tok(full, add_special_tokens=False)['input_ids']
        assert ids[:len(head_ids)] == head_ids, 'prompt boundary moved'
        return len(ids)
    return count


def urllib_transport(api_key):
    def post(body):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method='POST',
                                     headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    return post


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def admission(rec, row, grade_fn, count_fn):
    g = {'id': rec['id'], 'attempt': rec['attempt'], 'response_seq': rec['seq'], 'graded_at_utc': utc()}
    reasons = []
    if rec.get('finish_reason') != 'stop':
        reasons.append('finish_reason_' + str(rec.get('finish_reason')))
    if not (rec.get('reasoning_content') or '').strip():
        reasons.append('empty_reasoning')
    try:
        g.update(grade_fn(rec.get('text') or '', row['gold']))
        if not g.get('has_boxed'):
            reasons.append('no_boxed')
        elif not g.get('correct'):
            reasons.append('wrong')
    except Exception as e:
        g['grade_error'] = type(e).__name__ + ': ' + str(e)[:200]
        reasons.append('grade_error')
    g['boxed_count'] = (rec.get('text') or '').count('\\boxed')
    try:
        n = count_fn(PROMPT + row['problem'], rec.get('reasoning_content') or '', rec.get('text') or '')
    except Exception as e:
        n, g['count_error'] = None, type(e).__name__ + ': ' + str(e)[:200]
        reasons.append('count_error')
    g['qwen_total_tokens'] = n
    if n is not None and n > QWEN_LIMIT:
        reasons.append('over_32k')
    g['reasons'] = reasons
    g['admitted'] = not reasons
    return g


def run(api_key, root, batch, budget, grade_fn=None, count_fn=None, transport=None, workers=WORKERS):
    root = Path(root)
    assert threading.current_thread() is threading.main_thread(), 'Call run() from the main thread (math-verify uses signal.alarm).'
    assert isinstance(api_key, str) and api_key.strip(), 'Secret is empty.'
    if grade_fn is None:
        from base_probe_worker import math_grade as grade_fn
    if count_fn is None:
        count_fn = qwen_counter()
    post = transport or urllib_transport(api_key)
    problems = read_jsonl(root / 'train2000.jsonl')
    assert len(problems) == len({p['id'] for p in problems}) == 2000, 'Expected the 2,000 selected problems.'
    by_id = {p['id']: p for p in problems}
    rows = [p for p in problems if p['batch'] == batch]
    assert rows, 'Unknown batch: ' + str(batch)

    raw_path, grade_path = root / 'raw_responses.jsonl', root / 'grades.jsonl'
    raw, grades = read_jsonl(raw_path), read_jsonl(grade_path)
    lock = threading.Lock()
    priced = lambda recs: sum(r['priced_at_peak_usd'] for r in recs)
    state = {'total_spent': priced(raw), 'batch_spent': priced([r for r in raw if r['batch'] == batch]),
             'errors': 0, 'sent': 0, 'stopped': None, 'seq': max((r['seq'] for r in raw), default=-1) + 1}
    call = {'batch': batch, 'started_at_utc': utc(), 'budget_usd': budget, 'total_cap_usd': TOTAL_CAP,
            'ledger_spent_before_usd': round(state['total_spent'], 6), 'batch_spent_before_usd': round(state['batch_spent'], 6),
            'script_version': SCRIPT_VERSION, 'workers': workers, 'api_key_saved': False}
    print('START', json.dumps(call), flush=True)

    def answered(pid, attempt):
        return any(r['id'] == pid and r['attempt'] == attempt and not r['error'] for r in raw)

    def append(path, rec):  # caller holds the lock
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def one(row, attempt):
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': PROMPT + row['problem']}],
                'temperature': .6, 'max_tokens': MAX_TOKENS, 'stream': False}
        reserve = ((len(json.dumps(body).encode()) + 512) * IN_RATE + MAX_TOKENS * OUT_RATE) / 1e6
        with lock:
            if state['errors'] >= MAX_ERRORS:
                state['stopped'] = state['stopped'] or 'error limit'
                return
            if state['batch_spent'] + reserve > budget or state['total_spent'] + reserve > TOTAL_CAP:
                state['stopped'] = state['stopped'] or 'budget'
                return
            state['batch_spent'] += reserve
            state['total_spent'] += reserve
            state['sent'] += 1
        out = {'id': row['id'], 'batch': batch, 'attempt': attempt, 'utc': utc(), 'http_status': None, 'error': None}
        t = time.perf_counter()
        try:
            out['http_status'], b = post(body)
            ch = b['choices'][0]; m = ch.get('message', {})
            out.update(text=m.get('content') or '', reasoning_content=m.get('reasoning_content') or '',
                       finish_reason=ch.get('finish_reason'), usage=b.get('usage'), model_returned=b.get('model'),
                       system_fingerprint=b.get('system_fingerprint'), response_id=b.get('id'))
        except urllib.error.HTTPError as e:
            out.update(http_status=e.code, error='HTTPError: ' + e.read(1024).decode('utf-8', 'replace').replace(api_key, '[REDACTED]'))
        except Exception as e:
            out['error'] = type(e).__name__ + ': ' + str(e)[:200].replace(api_key, '[REDACTED]')
        out['seconds'] = round(time.perf_counter() - t, 3)
        u = out.get('usage') or {}
        with lock:
            if not out['error'] and 'prompt_tokens' in u and 'completion_tokens' in u:
                cost = (u['prompt_tokens'] * IN_RATE + u['completion_tokens'] * OUT_RATE) / 1e6
            else:
                cost = reserve  # unknown charge: keep the full reservation
                if not out['error']:
                    out['error'] = 'no usage in response'
            out['priced_at_peak_usd'] = cost
            state['batch_spent'] += cost - reserve
            state['total_spent'] += cost - reserve
            if out['error']:
                state['errors'] += 1
            out['seq'] = state['seq']; state['seq'] += 1
            append(raw_path, out)  # the raw response is on disk before any grading
            raw.append(out)
            print('DONE', batch, 'attempt', attempt, out['id'], 'error' if out['error'] else out.get('finish_reason'),
                  'batch_spent', round(state['batch_spent'], 4), flush=True)

    def grade_pending():
        done = {(g['id'], g['attempt']) for g in grades}
        for r in sorted(raw, key=lambda r: r['seq']):
            if r['batch'] != batch or r['error'] or (r['id'], r['attempt']) in done:
                continue
            g = admission(r, by_id[r['id']], grade_fn, count_fn)
            append(grade_path, g)
            grades.append(g)
            done.add((r['id'], r['attempt']))

    def admitted(pid, attempt):
        return next((g['admitted'] for g in grades if g['id'] == pid and g['attempt'] == attempt), None)

    try:
        grade_pending()  # responses saved by a call that stopped before grading
        for attempt in (1, 2):
            for round_ in (1, 2):  # round 2 resends the requests of round 1 that ended in an error
                if attempt == 1:
                    todo = [r for r in rows if not answered(r['id'], 1)]
                else:
                    todo = [r for r in rows if admitted(r['id'], 1) is False and not answered(r['id'], 2)]
                if round_ == 1:
                    call['to_send_attempt_%d' % attempt] = len(todo)
                if not todo or state['stopped']:
                    break
                with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                    list(pool.map(lambda r: one(r, attempt), todo))
                grade_pending()
    finally:
        call.update(ended_at_utc=utc(), sent=state['sent'], request_errors=state['errors'],
                    stopped=state['stopped'], ledger_spent_after_usd=round(state['total_spent'], 6),
                    batch_spent_after_usd=round(state['batch_spent'], 6))
        summary = write_derived(root, problems, raw, grades, call)
        for p in root.iterdir():
            if p.is_file() and p.suffix in {'.json', '.jsonl', '.py', '.txt'}:
                assert api_key not in p.read_text(encoding='utf-8', errors='replace'), 'key found in ' + p.name
    print('END', json.dumps(call), flush=True)
    return summary


def status_of(pid, raw_errors, grade_at):
    """Final state of one problem: kept (attempt), dropped (reasons per attempt) or pending."""
    g = {a: grade_at.get((pid, a)) for a in (1, 2)}
    if g[1] and g[1]['admitted']:
        return {'status': 'kept', 'kept_attempt': 1}
    if g[1] and g[2]:
        return {'status': 'kept', 'kept_attempt': 2} if g[2]['admitted'] else \
            {'status': 'dropped', 'reasons_attempt_1': g[1]['reasons'], 'reasons_attempt_2': g[2]['reasons']}
    return {'status': 'pending', 'reasons_attempt_1': g[1]['reasons'] if g[1] else None,
            'request_errors': raw_errors.get(pid, 0)}


def write_derived(root, problems, raw, grades, call):
    calls = read_jsonl(root / 'calls.jsonl') + [call]
    (root / 'calls.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in calls), encoding='utf-8')
    grade_at = {(g['id'], g['attempt']): g for g in grades}
    raw_at = {(r['id'], r['attempt']): r for r in raw if not r['error']}
    raw_errors = collections.Counter(r['id'] for r in raw if r['error'])
    by_batch = {p['id']: p['batch'] for p in problems}
    statuses, final = [], []
    for p in problems:
        s = {'id': p['id'], 'batch': p['batch'], 'level': p['level'], 'subject': p['subject'],
             **status_of(p['id'], raw_errors, grade_at)}
        statuses.append(s)
        if s['status'] == 'kept':
            r = raw_at[(p['id'], s['kept_attempt'])]
            g = grade_at[(p['id'], s['kept_attempt'])]
            final.append({'id': p['id'], 'level': p['level'], 'subject': p['subject'], 'gold': p['gold'],
                          'problem': p['problem'], 'prompt': PROMPT + p['problem'], 'attempt': s['kept_attempt'],
                          'response_seq': r['seq'], 'response_id': r.get('response_id'),
                          'reasoning_content': r['reasoning_content'], 'content': r['text'],
                          'extracted_answer': g.get('extracted_answer'), 'qwen_total_tokens': g['qwen_total_tokens'],
                          'usage': r['usage']})
    (root / 'problem_status.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in statuses), encoding='utf-8')
    (root / 'final_traces.jsonl').write_text(''.join(json.dumps(f, ensure_ascii=False) + '\n' for f in final), encoding='utf-8')

    def part(batch=None):
        st = [s for s in statuses if batch is None or s['batch'] == batch]
        rr = [r for r in raw if batch is None or r['batch'] == batch]
        gg = [g for g in grades if batch is None or by_batch[g['id']] == batch]
        ok = [r for r in rr if not r['error']]
        a1 = [g for g in gg if g['attempt'] == 1]
        reasons = lambda sel: dict(sorted(collections.Counter(x for g in sel for x in g['reasons']).items()))
        kept = [f for f in final if batch is None or by_batch.get(f['id']) == batch]
        comp = [r['usage']['completion_tokens'] for r in ok]
        qwen = [f['qwen_total_tokens'] for f in kept]
        return {
            'problems': len(st), 'kept': sum(s['status'] == 'kept' for s in st),
            'kept_on_attempt_2': sum(s.get('kept_attempt') == 2 for s in st),
            'dropped': sum(s['status'] == 'dropped' for s in st), 'pending': sum(s['status'] == 'pending' for s in st),
            'dropped_ids': [s['id'] for s in st if s['status'] == 'dropped'],
            'responses': len(ok), 'request_errors': len(rr) - len(ok),
            'attempt_1_graded': len(a1), 'attempt_1_admitted': sum(g['admitted'] for g in a1),
            'attempt_1_failure_reasons': reasons(a1),
            'attempt_2_failure_reasons': reasons([g for g in gg if g['attempt'] == 2]),
            'multi_boxed_responses': sum(g['boxed_count'] > 1 for g in gg),
            'input_tokens': sum(r['usage']['prompt_tokens'] for r in ok),
            'output_tokens': sum(comp), 'reasoning_tokens': sum((r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) for r in ok),
            'priced_at_peak_usd': round(sum(r['priced_at_peak_usd'] for r in rr), 6),
            'mean_completion_tokens': round(statistics.mean(comp), 1) if comp else None,
            'max_completion_tokens': max(comp, default=None),
            'kept_qwen_tokens': {'mean': round(statistics.mean(qwen), 1) if qwen else None,
                                 'median': statistics.median(qwen) if qwen else None, 'max': max(qwen, default=None),
                                 'over_8k': sum(q > 8192 for q in qwen), 'over_16k': sum(q > 16384 for q in qwen)},
            'by_level_kept': {str(l): f"{sum(s['status'] == 'kept' for s in st if s['level'] == l)}/{sum(s['level'] == l for s in st)}"
                              for l in range(1, 6)},
            'seconds': {'median': statistics.median([r['seconds'] for r in rr]) if rr else None,
                        'max': max((r['seconds'] for r in rr), default=None)}}
    summary = {'updated_at_utc': utc(), 'script_version': SCRIPT_VERSION, 'model': MODEL,
               'thinking': 'not sent (API default: thinking)', 'effort': 'not sent (default: high)', 'temperature': .6,
               'max_tokens': MAX_TOKENS, 'qwen_limit': QWEN_LIMIT, 'rates_peak_usd_per_million': [IN_RATE, OUT_RATE],
               'total_cap_usd': TOTAL_CAP, 'prompt': PROMPT,
               'model_returned': sorted({r.get('model_returned') for r in raw if not r['error']} - {None}),
               'system_fingerprint': sorted({r.get('system_fingerprint') for r in raw if not r['error']} - {None}),
               'all': part(), 'by_batch': {b: part(b) for b in sorted({p['batch'] for p in problems})}}
    (root / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name != 'manifest.json' and p.suffix in {'.py', '.json', '.jsonl'})
    (root / 'manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=1))
    return summary
