"""Light cleaning (groups 3 and 4). Free, offline.

Usage (python -I; OUT_DIR is new or empty):
  python -I light_clean.py FINAL_TRACES.jsonl OUT_DIR
Writes OUT_DIR/light_report.json (per-rule counts, every trigger) and OUT_DIR/heavy_inputs.jsonl: the light-cleaned
rows in the fields claude_label.py reads (id, problem, text, reasoning_content, reasoning_tokens, set, level).

Rules, applied to reasoning_content only (content, the final solution, is never touched):
- L1 loop: the same run of >= 20 words appears >= 3 times back to back; the first copy stays, the later ones go.
- L2 garbage: U+FFFD, private use, unassigned code points, control characters other than newline and tab; deleted.
- L3 script switch: a writing system that the problem does not use (CJK, kana, Hangul, Cyrillic, Arabic, Hebrew,
  Thai, Devanagari, ...); Greek and math symbols do not count; the sentence holding it is deleted.
Safety valve: if a trace would lose more than 10% of its reasoning characters, it is not cleaned but reported as
broken (admission failure: resample once, drop from all groups if still broken). The script stops with exit code 2
in that case and writes no heavy_inputs.jsonl.
"""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

MIN_LOOP_WORDS, LOOP_COPIES, MAX_SHARE = 20, 3, 0.10
FOREIGN = ('CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL', 'CYRILLIC', 'ARABIC', 'HEBREW', 'THAI', 'DEVANAGARI',
           'ARMENIAN', 'GEORGIAN', 'BENGALI', 'TAMIL', 'ETHIOPIC', 'KHMER', 'LAO', 'MYANMAR', 'TIBETAN')
SENTENCE = re.compile(r'[^.!?。！？\n]*(?:[.!?。！？]+|\n|$)')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def is_garbage(ch):
    cat = unicodedata.category(ch)
    return ch == chr(0xFFFD) or cat in ('Co', 'Cn') or (cat == 'Cc' and ch not in '\n\t')


def script_of(ch):
    if not unicodedata.category(ch).startswith('L'):
        return None
    name = unicodedata.name(ch, '')
    return next((s for s in FOREIGN if name.startswith(s)), None)


def loops(text):
    """Character spans of the 2nd+ copies of runs of >= 20 words repeated >= 3 times back to back."""
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]
    toks = [w[2] for w in words]
    n, spans, seen = len(toks), [], {}
    for i in range(n - MIN_LOOP_WORDS + 1):  # positions of each 20-word shingle
        seen.setdefault(tuple(toks[i:i + MIN_LOOP_WORDS]), []).append(i)
    covered = -1
    for pos in seen.values():
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                i, p = pos[a], pos[b] - pos[a]
                if p < MIN_LOOP_WORDS or i < covered or i + LOOP_COPIES * p > n:
                    continue
                k = 1
                while i + (k + 1) * p <= n and toks[i + k * p:i + (k + 1) * p] == toks[i:i + p]:
                    k += 1
                if k >= LOOP_COPIES:
                    spans.append((words[i + p][0], words[i + k * p - 1][1], p, k))
                    covered = i + k * p
    return spans


def clean(row):
    text, problem = row['reasoning_content'], row['problem']
    hits = {'L1': [], 'L2': [], 'L3': []}
    drop = set()
    for s, e, p, k in loops(text):
        hits['L1'].append({'start': s, 'end': e, 'period_words': p, 'copies': k})
        drop.update(range(s, e))
    for i, ch in enumerate(text):
        if is_garbage(ch):
            hits['L2'].append({'at': i, 'code': f'U+{ord(ch):04X}'})
            drop.add(i)
    allowed = {script_of(c) for c in problem} - {None}
    for m in SENTENCE.finditer(text):
        bad = {script_of(c) for c in m.group()} - {None} - allowed
        if bad and m.group():
            hits['L3'].append({'start': m.start(), 'end': m.end(), 'scripts': sorted(bad)})
            drop.update(range(m.start(), m.end()))
    share = len(drop) / max(len(text), 1)
    cleaned = ''.join(c for i, c in enumerate(text) if i not in drop) if drop else text
    return cleaned, hits, share


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'OUT_DIR must be empty'
    raw = Path(src).read_bytes()
    rows = [json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
    report = {'source': str(src), 'source_sha256': sha(raw), 'tool_sha256': sha(Path(__file__).read_bytes()),
              'n_rows': len(rows), 'rules_triggered': {'L1': 0, 'L2': 0, 'L3': 0}, 'traces_changed': 0,
              'broken_over_10pct': [], 'changes': {}}
    inputs = []
    for r in rows:
        cleaned, hits, share = clean(r)
        for k, v in hits.items():
            report['rules_triggered'][k] += bool(v)
        if share > MAX_SHARE:
            report['broken_over_10pct'].append({'id': r['id'], 'share': round(share, 4)})
        elif cleaned != r['reasoning_content']:
            report['traces_changed'] += 1
        if any(hits.values()):
            report['changes'][r['id']] = {'share': round(share, 4), **{k: v for k, v in hits.items() if v}}
        inputs.append({'id': r['id'], 'set': 'teacher_gen', 'level': r['level'], 'subject': r['subject'],
                       'problem': r['problem'], 'text': r['content'], 'reasoning_content': cleaned,
                       'reasoning_tokens': r['usage']['completion_tokens_details']['reasoning_tokens'],
                       'light_changed': cleaned != r['reasoning_content']})
    report['light_equals_raw'] = report['traces_changed'] == 0 and not report['broken_over_10pct']
    if not report['broken_over_10pct']:
        body = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in inputs).encode('utf-8')
        (out / 'heavy_inputs.jsonl').write_bytes(body)
        report['heavy_inputs_sha256'] = sha(body)
    (out / 'light_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('n_rows', 'rules_triggered', 'traces_changed', 'broken_over_10pct',
                                             'light_equals_raw')}))
    if report['broken_over_10pct']:
        sys.exit(2)


if __name__ == '__main__':
    main(*sys.argv[1:3])
