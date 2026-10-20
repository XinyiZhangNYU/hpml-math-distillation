"""Heavy cleaning (group 4): unit splitting, delete-only application of labels, agreement scoring and the diff page.

claude_label.py builds the final group-4 labels on split_units(), parse_response(), apply_plan(), agreement() and
render_html() from this module. run() is the automatic annotator (deepseek-flash, T=0, thinking disabled) that was
compared with human labels during development; it is not part of the final pipeline.

Label rules:
- "abandoned" is only work that the reasoning itself gives up, refutes or finds wrong. Correct work that is redone
  later by another method is the main line; the later redo is the repeat.
- Every "abandoned" must name the unit where the reasoning gives it up or finds it wrong (dropped_at). The script
  deletes it only if dropped_at lies inside the range or on the unit right after it; otherwise that item is dropped.
- main_derivation must end at the unit where the final answer value is stated for the first time, and must be
  short: no re-derivations or re-checks inside it.
- A deletion that overlaps main_derivation loses only the overlapping units.
- The short planning of the final answer at the end of the trace is not marked.
- Rows may include a second annotated set (heldout5) that no prompt revision has looked at; agreement is reported
  per annotated set.

- reasoning_content is cut into numbered units (split_units). Nothing else is ever touched: the final solution
  (`text`, the API's `content`) is copied byte for byte.
- The script deletes whole units, so the result is always a subsequence of the original. Safety rules:
  units inside main_derivation are never deleted; a repeat whose same_as is missing, not before it, or deleted is
  dropped; an abandoned item without a valid dropped_at is dropped; an invalid response deletes nothing; if more
  than MAX_DELETE_SHARE of the reasoning characters would go, nothing is deleted. Every case is recorded.
- run() is PAID: one request per input row, budget USD 0.20 at peak rates, no retries, at most 3 failed requests.
  The key is passed in memory and never written. process() and everything below it are free and offline.
"""
import concurrent.futures
import datetime
import hashlib
import html
import json
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-flash'
IN_RATE, OUT_RATE = .30, 1.20  # USD per million tokens at peak
BUDGET = 0.20
MAX_TOKENS = 4096
WORKERS = 4
MAX_ERRORS = 3
MAX_DELETE_SHARE = 0.80  # 0.60 would refuse the human annotation of precalculus/train/642 (75%)
LABELS = ('abandoned', 'repeat')
HUMAN_DELETE = {'abandoned', 'repeat'}
N_ROWS = 30  # human5 + long20 + heldout5

PROMPT = """You are annotating the reasoning trace that a model wrote while solving a math problem. The trace is split into numbered units. Your annotation will be used to delete units, so mark only units whose removal loses nothing the solution needs.

Mark only these two kinds of units:
1. "abandoned": an approach, guess or reading of the problem that the reasoning itself gives up, refutes, or finds to be wrong or going nowhere. For every abandoned range give "dropped_at": the unit where the reasoning says so (for example a guess refuted by a counterexample, a computation found wrong, "this is not working", "let's try another way"); it must be inside the range or the unit right after it. Correct work is never abandoned, even if a later part reaches the same result by a simpler or cleaner method: the first correct derivation is the main line, and the later one is a repeat. Work whose results are used later is not abandoned.
2. "repeat": a re-derivation, re-check, re-reading or re-statement of something that was already derived, checked or read earlier in the trace, by the same or another method. The earlier occurrence is always kept and only the later occurrences are marked. A write-up or summary of the solution inside the reasoning that redoes steps already done is a repeat. Never mark a unit because the same content appears again later in the trace. For every repeat give "same_as": the earlier units it repeats. They must come before the repeat and must not be marked themselves.

Do not mark anything else: not the main line of reasoning, not the first check of a result, not planning remarks, and not the short planning of the final answer at the end of the trace. When unsure, do not mark.

Also give "main_derivation": the contiguous units where the final answer is obtained for the first time. Its last unit is the first unit that states the final answer value. Keep it short: it contains no re-derivations or re-checks. Work before it that the answer uses is kept as the main line, not marked. These units are never deleted.

Return only a JSON object, with no other text:
{"main_derivation": {"first": i, "last": j}, "delete": [{"first": i, "last": j, "label": "abandoned", "dropped_at": m, "why": "at most 15 words"}, {"first": i, "last": j, "label": "repeat", "same_as": {"first": k, "last": l}, "why": "at most 15 words"}]}
Ranges are inclusive unit numbers. Delete ranges must not overlap each other. Use "delete": [] if nothing should be deleted."""
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def split_units(text):
    """[start, end) character spans that tile text. A unit ends after a newline run, or after . ! ? plus spaces."""
    bounds = [0]
    for m in re.finditer(r'\n+|(?<=[.!?])[ \t]+', text):
        if m.end() < len(text) and text[bounds[-1]:m.end()].strip():
            bounds.append(m.end())
    units = list(zip(bounds, bounds[1:] + [len(text)]))
    assert ''.join(text[s:e] for s, e in units) == text
    return units


def build_prompt(row, units):
    t = row['reasoning_content']
    lines = '\n'.join(f'[{i}] ' + ' '.join(t[s:e].split()) for i, (s, e) in enumerate(units))
    return (PROMPT + '\n\nProblem:\n' + row['problem'] + '\n\nFinal solution shown to the user after the reasoning:\n'
            + row['text'] + '\n\nReasoning units (' + str(len(units)) + ', numbered from 0):\n' + lines)


def parse_response(raw, n):
    """Validated plan, or ValueError. Integers given as digit strings are accepted."""
    s = raw.strip()
    a, b = s.find('{'), s.rfind('}')
    if a < 0 or b < a:
        raise ValueError('no JSON object')
    obj = json.loads(s[a:b + 1])

    def rng(d, what):
        if not isinstance(d, dict):
            raise ValueError(what + ' is not an object')
        try:
            i, j = (int(d[k]) if isinstance(d[k], str) and d[k].strip().isdigit() else d[k] for k in ('first', 'last'))
        except KeyError:
            raise ValueError(what + ' lacks first/last')
        if not (isinstance(i, int) and isinstance(j, int) and 0 <= i <= j < n):
            raise ValueError(f'{what} range out of bounds: {d}')
        return i, j

    fd = rng(obj.get('main_derivation'), 'main_derivation')
    dels = obj.get('delete')
    if not isinstance(dels, list):
        raise ValueError('delete is not a list')
    out, used = [], set()
    for d in dels:
        i, j = rng(d, 'delete item')
        if d.get('label') not in LABELS:
            raise ValueError(f'bad label: {d.get("label")}')
        if used & set(range(i, j + 1)):
            raise ValueError(f'overlapping delete ranges at {i}-{j}')
        used |= set(range(i, j + 1))
        item = {'first': i, 'last': j, 'label': d['label'], 'why': str(d.get('why', ''))[:200]}
        if d['label'] == 'repeat':  # a missing or malformed same_as drops this item in apply_plan, not the whole response
            try:
                k, l = rng(d.get('same_as'), 'same_as')
                item['same_as'] = {'first': k, 'last': l}
            except ValueError as e:
                item['same_as_error'] = str(e)[:200]
        else:  # likewise a missing or malformed dropped_at
            m = d.get('dropped_at')
            m = int(m) if isinstance(m, str) and m.strip().isdigit() else m
            if isinstance(m, int) and not isinstance(m, bool) and 0 <= m < n:
                item['dropped_at'] = m
            else:
                item['dropped_at_error'] = f'bad dropped_at: {str(d.get("dropped_at"))[:50]}'
        out.append(item)
    return {'main_derivation': {'first': fd[0], 'last': fd[1]}, 'delete': sorted(out, key=lambda d: d['first'])}


def apply_plan(text, units, plan):
    md_first, md_last = plan['main_derivation']['first'], plan['main_derivation']['last']
    keep_items, dropped, trimmed = [], [], []
    for d in plan['delete']:
        if d['first'] >= md_first and d['last'] <= md_last:
            dropped.append(dict(d, reason='inside main_derivation'))
            continue
        if d['first'] <= md_last and d['last'] >= md_first:  # keep the main_derivation units, delete the rest
            if d['first'] < md_first and d['last'] > md_last:
                dropped.append(dict(d, reason='contains main_derivation'))
                continue
            trimmed.append({'first': d['first'], 'last': d['last']})
            d = dict(d, last=md_first - 1) if d['first'] < md_first else dict(d, first=md_last + 1)
        if d['label'] == 'abandoned' and not ('dropped_at' in d and d['first'] <= d['dropped_at'] <= d['last'] + 1):
            dropped.append(dict(d, reason='abandoned without valid dropped_at'))
        elif d['label'] == 'repeat' and 'same_as' not in d:
            dropped.append(dict(d, reason='repeat without valid same_as'))
        elif d['label'] == 'repeat' and d['same_as']['last'] >= d['first']:
            dropped.append(dict(d, reason='same_as not before the repeat'))
        else:
            keep_items.append(d)
    while True:  # a repeat is deleted only while what it repeats stays; dropping items only shrinks the deletion
        deleted = {i: d['label'] for d in keep_items for i in range(d['first'], d['last'] + 1)}
        bad = [d for d in keep_items if d['label'] == 'repeat'
               and set(range(d['same_as']['first'], d['same_as']['last'] + 1)) & set(deleted)]
        if not bad:
            break
        for d in bad:
            keep_items.remove(d)
            dropped.append(dict(d, reason='same_as is itself deleted'))
    dropped.sort(key=lambda d: d['first'])
    chars = sum(units[i][1] - units[i][0] for i in deleted)
    status = 'cleaned' if deleted else 'nothing_deleted'
    if chars > MAX_DELETE_SHARE * len(text):
        status, deleted = 'cap_exceeded_kept_whole', {}
    spans = []
    for i in sorted(deleted):
        s, e = units[i]
        if spans and spans[-1][1] == s and spans[-1][2] == deleted[i]:
            spans[-1][1] = e
        else:
            spans.append([s, e, deleted[i]])
    cleaned = ''.join(text[s:e] for i, (s, e) in enumerate(units) if i not in deleted)
    return {'status': status, 'deleted_units': sorted(deleted), 'deleted_spans': spans, 'dropped': dropped, 'trimmed': trimmed,
            'proposed_delete_share_chars': round(chars / len(text), 4) if text else 0.0,
            'delete_share_chars': round(sum(e - s for s, e, _ in spans) / len(text), 4) if text else 0.0,
            'cleaned_reasoning': cleaned}


def is_subsequence(small, big):
    it = iter(big)
    return all(c in it for c in small)


def masks(n, spans):
    m = bytearray(n)
    for s, e, *_ in spans:
        m[s:e] = b'\x01' * (e - s)
    return m


def prf_counts(tp, p, g, n):
    r = lambda x: round(x, 3)
    return {'precision': r(tp / p) if p else None, 'recall': r(tp / g) if g else None,
            'f1': r(2 * tp / (p + g)) if p + g else None, 'pred_share': r(p / n), 'human_share': r(g / n),
            'counts': {'tp': tp, 'pred': p, 'human': g, 'chars': n}}


def prf(pred, gold):
    return prf_counts(sum(1 for p, g in zip(pred, gold) if p and g), sum(pred), sum(gold), len(gold))


def agreement(text, units, spans, human_segs):
    """Character-level agreement with the human deletions (abandoned + repeat), plus the unit-split ceiling."""
    n = len(text)
    gold = masks(n, [(s['start'], s['end']) for s in human_segs if s['label'] in HUMAN_DELETE])
    snapped = [(s, e) for s, e in units if 2 * sum(gold[s:e]) > e - s]
    by_label = {}
    for lab in sorted(HUMAN_DELETE):
        g = masks(n, [(s['start'], s['end']) for s in human_segs if s['label'] == lab])
        hit = sum(1 for p, q in zip(masks(n, spans), g) if p and q)
        by_label[lab] = {'human_chars': sum(g), 'recall': round(hit / sum(g), 3) if sum(g) else None}
    return {'model': prf(masks(n, spans), gold), 'unit_ceiling': prf(masks(n, snapped), gold), 'recall_by_label': by_label}


def process(root, rows, responses, ann):
    """Free and offline: parse each raw response, apply it, check it, score the annotated sets, write the result files."""
    root = Path(root)
    results, report = [], {'prompt_sha256': PROMPT_SHA256, 'max_delete_share': MAX_DELETE_SHARE, 'traces': {}}
    for row in rows:
        text = row['reasoning_content']
        units = split_units(text)
        raw = responses.get(row['id'])
        res = {'id': row['id'], 'set': row['set'], 'n_units': len(units)}
        if raw is None or raw.get('error'):
            res.update(status='no_response', error=raw and raw.get('error'))
            plan = None
        else:
            try:
                plan = parse_response(raw.get('text', ''), len(units))
            except (ValueError, json.JSONDecodeError) as e:
                plan = None
                res.update(status='invalid_response', error=str(e)[:300])
        if plan:
            res['plan'] = plan
            res.update(apply_plan(text, units, plan))
        else:
            res.update(deleted_units=[], deleted_spans=[], dropped=[], trimmed=[], delete_share_chars=0.0, cleaned_reasoning=text)
        res['text'] = row['text']
        assert is_subsequence(res['cleaned_reasoning'], text), row['id']
        assert res['text'] == row['text']
        res['deleted_tokens_est'] = round(res['delete_share_chars'] * row['reasoning_tokens'])
        if row['id'] in ann:
            res['agreement'] = agreement(text, units, res['deleted_spans'], ann[row['id']])
        results.append(res)
        report['traces'][row['id']] = {k: res.get(k) for k in ('set', 'status', 'n_units', 'delete_share_chars',
                                                                 'deleted_tokens_est', 'agreement', 'error')}
        report['traces'][row['id']]['n_dropped'] = len(res['dropped'])
        report['traces'][row['id']]['n_trimmed'] = len(res.get('trimmed', []))
        report['traces'][row['id']]['dropped_reasons'] = sorted({d['reason'] for d in res['dropped']})
    report['pooled_chars_by_set'] = {}
    for name in sorted({r['set'] for r in results if 'agreement' in r}):  # human5 (seen while revising) and heldout5
        h = [r for r in results if 'agreement' in r and r['set'] == name]
        report['pooled_chars_by_set'][name] = {
            key: prf_counts(*(sum(r['agreement'][key]['counts'][c] for r in h) for c in ('tp', 'pred', 'human', 'chars')))
            for key in ('model', 'unit_ceiling')}
    report['by_status'] = {s: sum(r.get('status') == s for r in results) for s in sorted({r.get('status') for r in results})}
    tot = sum(r['reasoning_tokens'] for r in rows)
    report['deleted_tokens_est_total'] = sum(r['deleted_tokens_est'] for r in results)
    report['reasoning_tokens_total'] = tot
    (root / 'heavy_pilot_cleaned.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results), encoding='utf-8')
    (root / 'heavy_pilot_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    (root / 'heavy_pilot_diff.html').write_text(render_html(rows, results, ann, report), encoding='utf-8')
    return report


def rows_by_id(rows):
    return {r['id']: r for r in rows}


CSS = """:root{--bg:#fbfaf7;--fg:#1f2328;--muted:#646b73;--card:#ffffff;--line:#e3e0d8;--ab:#b3261e;--ab-bg:#fbe9e7;
--rp:#8a5a00;--rp-bg:#fff4d6;--fd:#1b6e3a;--fd-bg:#e6f4ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;--fg:#e6e6e6;--muted:#9aa1a9;--card:#1e2126;
--line:#30343a;--ab:#ff8a80;--ab-bg:#3a1f1d;--rp:#ffd36e;--rp-bg:#3a3018;--fd:#7fd8a0;--fd-bg:#183025}}
:root[data-theme="dark"]{--bg:#16181c;--fg:#e6e6e6;--muted:#9aa1a9;--card:#1e2126;--line:#30343a;--ab:#ff8a80;--ab-bg:#3a1f1d;
--rp:#ffd36e;--rp-bg:#3a3018;--fd:#7fd8a0;--fd-bg:#183025}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}h1{font-size:22px}h2{font-size:17px;margin:0 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.meta{color:var(--muted);font-size:13px}.cols{display:grid;grid-template-columns:1fr;gap:12px}
@media (min-width:900px){.cols.two{grid-template-columns:1fr 1fr}}
.trace{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,Consolas,monospace}
.u.abandoned{background:var(--ab-bg);color:var(--ab);text-decoration:line-through}
.u.repeat{background:var(--rp-bg);color:var(--rp);text-decoration:line-through}
.u.fd{background:var(--fd-bg)}.tag{font-size:11px;font-weight:600;padding:0 4px;border-radius:4px;margin-right:4px}
.tag.abandoned{color:var(--ab)}.tag.repeat{color:var(--rp)}.tag.fd{color:var(--fd)}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}table{border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left}"""


def render_html(rows, results, ann, report):
    e = html.escape
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Heavy cleaning</title><style>', CSS, '</style></head><body><main><h1>Heavy cleaning: what the annotator deletes</h1>',
           '<p class="meta">Struck through: deleted units (<span class="tag abandoned">abandoned</span><span class="tag repeat">repeat</span>). ',
           'Green: the main derivation (where the answer is first obtained), which is never deleted. Hover a deleted unit for the annotator\'s reason. ',
           'For the hand-annotated traces the right column shows the human labels.</p>']
    for name, pooled in report.get('pooled_chars_by_set', {}).items():
        out.append(f'<div class="card"><h2>Agreement with the human deletions ({e(name)}, characters)</h2><table><tr><th></th><th>precision</th><th>recall</th><th>F1</th><th>deleted share</th><th>human share</th></tr>')
        for k, v in pooled.items():
            out.append(f'<tr><td>{e(k)}</td><td>{v["precision"]}</td><td>{v["recall"]}</td><td>{v["f1"]}</td><td>{v["pred_share"]}</td><td>{v["human_share"]}</td></tr>')
        out.append('</table><p class="meta">unit_ceiling: the human deletions snapped to whole units, the best a unit-level annotator can reach.</p></div>')
    by = rows_by_id(rows)
    for res in results:
        row = by[res['id']]
        t = row['reasoning_content']
        units = split_units(t)
        lab = {}
        for s, eend, l in res['deleted_spans']:
            for i, (us, ue) in enumerate(units):
                if us >= s and ue <= eend:
                    lab[i] = l
        why = {}
        for d in (res.get('plan') or {}).get('delete', []):
            for i in range(d['first'], d['last'] + 1):
                why[i] = (d['label'] + (f' of {d["same_as"]["first"]}-{d["same_as"]["last"]}' if 'same_as' in d else '')
                          + (f' dropped at {d["dropped_at"]}' if 'dropped_at' in d else '') + ': ' + d['why'])
        fd = set()
        if res.get('plan'):
            fd = set(range(res['plan']['main_derivation']['first'], res['plan']['main_derivation']['last'] + 1))
        spans = []
        for i, (s, eend) in enumerate(units):
            cls = 'u ' + (lab.get(i) or ('fd' if i in fd else ''))
            title = f' title="[{i}] {e(why.get(i, ""))}"' if i in why else f' title="[{i}]"'
            spans.append(f'<span class="{cls.strip()}"{title}>{e(t[s:eend])}</span>')
        hum = ann.get(res['id'])
        out.append(f'<div class="card"><h2>{e(res["id"])}</h2><div class="meta">{e(row["set"])} · L{row["level"]} · '
                   f'{row["reasoning_tokens"]:,} reasoning tokens · {res["n_units"]} units · status {e(str(res.get("status")))} · '
                   f'deleted {res["delete_share_chars"]:.0%} of characters (~{res["deleted_tokens_est"]:,} tokens)'
                   + (f' · {len(res["dropped"])} deletion(s) refused by the script' if res['dropped'] else '')
                   + (f' · error: {e(str(res.get("error")))}' if res.get('error') else '') + '</div>')
        if 'agreement' in res:
            m = res['agreement']['model']
            out.append(f'<div class="meta">vs human: precision {m["precision"]}, recall {m["recall"]}, F1 {m["f1"]}; human deleted {m["human_share"]:.0%}</div>')
        out.append(f'<details><summary>Problem and final solution</summary><div class="trace">{e(row["problem"])}\n\n— final solution —\n{e(row["text"])}</div></details>')
        out.append(f'<div class="cols{" two" if hum else ""}"><div class="trace">{"".join(spans)}</div>')
        if hum:
            hs = ''.join(f'<span class="u {s["label"] if s["label"] in HUMAN_DELETE else ""}"><span class="tag {s["label"] if s["label"] in HUMAN_DELETE else ""}">{e(s["label"])}</span>{e(t[s["start"]:s["end"]])}</span>' for s in hum)
            out.append(f'<div class="trace">{hs}</div>')
        out.append('</div></div>')
    out.append('</main></body></html>')
    return ''.join(out)


def run(api_key, root, budget=BUDGET):
    """PAID. One annotation request per row of heavy_pilot_inputs.jsonl, then process()."""
    root = Path(root)
    assert isinstance(api_key, str) and api_key.strip(), 'Secret is empty.'
    status_path = root / 'heavy_pilot_status.json'
    assert not status_path.exists(), 'Ledger exists: inspect it instead of repeating paid requests.'
    rows = [json.loads(x) for x in (root / 'heavy_pilot_inputs.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    ann = json.loads((root / 'human_annotations.json').read_text(encoding='utf-8'))['traces']
    assert len({r['id'] for r in rows}) == len(rows) == N_ROWS and set(ann) <= {r['id'] for r in rows}
    lock = threading.Lock()
    status = {'started_at_utc': utc(), 'state': 'running', 'model': MODEL, 'thinking': 'disabled', 'temperature': 0,
              'max_tokens': MAX_TOKENS, 'budget_usd': budget, 'script_version': 3, 'prompt_sha256': PROMPT_SHA256,
              'rates_peak_usd_per_million': [IN_RATE, OUT_RATE], 'spent_conservative_usd': 0.0, 'done': 0,
              'errors': 0, 'api_key_saved': False}
    responses = {}

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    def request(body):
        reserve = ((len(json.dumps(body).encode()) + 512) * IN_RATE + body['max_tokens'] * OUT_RATE) / 1e6
        with lock:
            if status['errors'] >= MAX_ERRORS or status['spent_conservative_usd'] + reserve > budget:
                return None
            status['spent_conservative_usd'] += reserve
        out = {'utc': utc(), 'http_status': None, 'error': None}
        t = time.perf_counter()
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method='POST',
            headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out['http_status'] = r.status
                b = json.loads(r.read())
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
            if 'prompt_tokens' in u and 'completion_tokens' in u:
                priced = (u['prompt_tokens'] * IN_RATE + u['completion_tokens'] * OUT_RATE) / 1e6
                status['spent_conservative_usd'] += priced - reserve
                out['priced_at_peak_usd'] = priced
            else:
                out['priced_at_peak_usd'] = reserve
            if out['error']:
                status['errors'] += 1
        return out

    def write_outputs():
        (root / 'heavy_pilot_outputs.jsonl').write_text(
            ''.join(json.dumps(dict(id=k, **v), ensure_ascii=False) + '\n' for k, v in sorted(responses.items())), encoding='utf-8')

    def one(row):
        prompt = build_prompt(row, split_units(row['reasoning_content']))
        out = request({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'thinking': {'type': 'disabled'},
                       'temperature': 0, 'max_tokens': MAX_TOKENS, 'stream': False})
        if out is None:
            return
        out['prompt_sha256'] = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        rt = ((out.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        out['thinking_disabled_ok'] = not out.get('reasoning_content') and not rt
        with lock:  # save the raw response before any processing
            responses[row['id']] = out
            status['done'] = len(responses)
            write_outputs(); save()
            print('DONE', status['done'], '/', len(rows), 'spent', round(status['spent_conservative_usd'], 4), flush=True)

    try:
        save()
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(one, rows))
        status['state'] = 'processing'; save()
        status['thinking_not_disabled'] = sorted(k for k, v in responses.items() if not v['error'] and not v['thinking_disabled_ok'])
        status['truncated'] = sorted(k for k, v in responses.items() if v.get('finish_reason') == 'length')
        status['report'] = process(root, rows, responses, ann)
        status['stop_reason'] = 'finished' if len(responses) == len(rows) and not status['errors'] else 'stopped early or with errors'
    finally:
        status.update(state='finished', ended_at_utc=utc())
        save()
        files = [p for p in root.iterdir() if p.is_file() and p.suffix in {'.py', '.json', '.jsonl', '.html'}
                 and p.name != 'heavy_pilot_manifest.json']
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (root / 'heavy_pilot_manifest.json').write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(root / 'hpml_heavy_pilot_r3_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files + [root / 'heavy_pilot_manifest.json']:
                z.write(p, p.name)
    return status
