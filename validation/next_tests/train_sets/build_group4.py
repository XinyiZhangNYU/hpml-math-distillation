"""Build the group 4 (heavy-clean) training set from the 1,971 kept teacher traces; offline, no API.

Group 4 = each kept trace once, in final_traces.jsonl order. The 797 labeled traces (>=300 reasoning tokens) use
the Claude heavy-clean result (heavy_claude_full/work/results/batch_NNN_cleaned.jsonl); the 1,174 short traces use
the light-clean version, which equals the raw trace (light_clean_full/light_report.json). Only reasoning_content
changes; content (final solution) is copied unchanged. Group 3 is final_traces.jsonl itself and is not copied.

Checks (any failure stops): input hashes; every cleaned file matches results_manifest.json; the labeled and short
id sets are disjoint and cover all 1,971 ids; each cleaned reasoning is a subsequence of the raw reasoning; content is
byte-identical to the raw content. Qwen token counts here are estimates; the exact recount needs the Qwen tokenizer
(count_group4_tokens.py, Colab).

usage: python -I build_group4.py <next_tests dir>
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

RAW_SHA = '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e'
HEAVY_INPUTS_SHA = 'b3a23d381c175b5365db7a6f41a8dbd144a939d8bd0b54a45597a5f9eaf87069'
PREPARE_SHA = 'd9f0bec559481d54c85413c6abc793c9ac1cb1a911a64503e747889ba73a902d'
RESULTS_MANIFEST_SHA = '99d34593d6d2664c085382245c762f22be6f27aa9cb273daa5e761afb63133d9'
N_BATCHES = 80
ACCEPTED_GATE_FAILS = {  # batches whose calibration item scored below the gate; their labels were checked and kept
    'batch_005', 'batch_019', 'batch_025', 'batch_052', 'batch_055', 'batch_059',
    'batch_064', 'batch_069', 'batch_074', 'batch_079'}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    return sha_bytes(Path(p).read_bytes())


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]


def is_subsequence(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def main(nt):
    nt = Path(nt)
    tr = nt / 'teacher_results'
    raw_p = tr / 'teacher_gen_b4/final_traces.jsonl'
    light_p = tr / 'light_clean_full/light_report.json'
    hin_p = tr / 'light_clean_full/heavy_inputs.jsonl'
    work = tr / 'heavy_claude_full/work'
    assert sha(raw_p) == RAW_SHA, 'final_traces.jsonl changed'
    assert sha(hin_p) == HEAVY_INPUTS_SHA, 'heavy_inputs.jsonl changed'
    assert sha(work / 'prepare.json') == PREPARE_SHA, 'prepare.json changed'
    assert sha(work / 'results_manifest.json') == RESULTS_MANIFEST_SHA, 'results_manifest.json changed'
    light = json.loads(light_p.read_text(encoding='utf-8'))
    assert light['source_sha256'] == RAW_SHA and light['light_equals_raw'] and light['traces_changed'] == 0
    assert light['heavy_inputs_sha256'] == HEAVY_INPUTS_SHA
    prep = json.loads((work / 'prepare.json').read_text(encoding='utf-8'))
    manifest = json.loads((work / 'results_manifest.json').read_text(encoding='utf-8'))
    mfiles = {e['path']: e for e in manifest['files']}

    raw = read_jsonl(raw_p)
    hin = {r['id']: r for r in read_jsonl(hin_p)}
    assert len(raw) == len(hin) == 1971 and len({r['id'] for r in raw}) == 1971
    for r in raw:  # light = raw, so the heavy-clean input is the raw trace
        h = hin[r['id']]
        assert h['reasoning_content'] == r['reasoning_content'] and h['text'] == r['content'], r['id']

    cleaned, batch_of, gate = {}, {}, {}
    for n in range(1, N_BATCHES + 1):
        tag = f'batch_{n:03d}'
        for suffix in ('_cleaned.jsonl', '_report.json'):
            rel = f'results/{tag}{suffix}'
            ent = mfiles['work/' + rel]
            assert sha(work / rel) == ent['sha256'], rel + ' differs from results_manifest.json'
        rep = json.loads((work / f'results/{tag}_report.json').read_text(encoding='utf-8'))
        gate[tag] = rep['calibration']['passed']
        for c in read_jsonl(work / f'results/{tag}_cleaned.jsonl'):
            assert not c['calibration'] and c['id'] not in cleaned, c['id']
            cleaned[c['id']] = c
            batch_of[c['id']] = tag
    failed = {t for t, ok in gate.items() if not ok}
    assert failed == ACCEPTED_GATE_FAILS, f'gate failures differ from the accepted list: {sorted(failed ^ ACCEPTED_GATE_FAILS)}'

    short = set(prep['short_ids'])
    ids = [r['id'] for r in raw]
    assert len(cleaned) == prep['n_annotated'] == 797 and len(short) == prep['n_short_kept_light'] == 1174
    assert not (short & set(cleaned)) and short | set(cleaned) == set(ids)

    out, stats = [], {'cleaned': 0, 'nothing_deleted': 0, 'kept_whole_over_cap': 0, 'short_not_labeled': 0}
    raw_chars = new_chars = 0
    for r in raw:
        rid, text = r['id'], r['reasoning_content']
        row = {k: v for k, v in r.items() if k not in ('reasoning_content', 'qwen_total_tokens')}
        if rid in cleaned:
            c = cleaned[rid]
            new = c['cleaned_reasoning']
            assert c['text'] == r['content'], rid
            assert is_subsequence(new, text), rid
            status = c['status']
            deleted_est = c['deleted_tokens_est']
            heavy = {'status': status, 'batch': batch_of[rid], 'delete_share_chars': c['delete_share_chars'],
                     'deleted_spans': c['deleted_spans'], 'deleted_reasoning_tokens_est': deleted_est}
        else:
            new, status, deleted_est = text, 'short_not_labeled', 0
            heavy = {'status': status}
        stats[status if status in stats else 'kept_whole_over_cap'] += 1
        heavy['raw_reasoning_sha256'] = sha_bytes(text.encode('utf-8'))
        heavy['reasoning_sha256'] = sha_bytes(new.encode('utf-8'))
        row['reasoning_content'] = new
        row['qwen_total_tokens_raw'] = r['qwen_total_tokens']
        row['qwen_total_tokens_est'] = r['qwen_total_tokens'] - deleted_est
        row['heavy_clean'] = heavy
        raw_chars += len(text)
        new_chars += len(new)
        out.append(row)

    dest = tr / 'group4_heavy_v1'
    dest.mkdir(exist_ok=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out).encode('utf-8')
    (dest / 'group4_heavy.jsonl').write_bytes(body)
    q_raw = [r['qwen_total_tokens_raw'] for r in out]
    q_est = [r['qwen_total_tokens_est'] for r in out]
    lab = [r for r in out if r['heavy_clean']['status'] != 'short_not_labeled']
    report = {
        'tool_sha256': sha(__file__),
        'inputs': {'final_traces.jsonl': RAW_SHA, 'heavy_inputs.jsonl': HEAVY_INPUTS_SHA,
                   'prepare.json': PREPARE_SHA, 'results_manifest.json': RESULTS_MANIFEST_SHA},
        'output': {'file': 'group4_heavy.jsonl', 'rows': len(out), 'bytes': len(body), 'sha256': sha_bytes(body)},
        'status_counts': stats,
        'calibration_gate_failed_batches_accepted': sorted(ACCEPTED_GATE_FAILS),
        'reasoning_chars': {'raw': raw_chars, 'group4': new_chars,
                            'deleted_share': round(1 - new_chars / raw_chars, 4)},
        'labeled_delete_share_chars': {'mean': round(statistics.mean(r['heavy_clean']['delete_share_chars'] for r in lab), 4),
                                       'median': statistics.median(r['heavy_clean']['delete_share_chars'] for r in lab)},
        'qwen_total_tokens': {
            'note': 'group 3 = raw (exact, from teacher_gen); group 4 = raw minus DeepSeek-reasoning-token estimate of the '
                    'deleted share (estimate; exact recount pending, count_group4_tokens.py)',
            'group3_exact': {'sum': sum(q_raw), 'mean': round(statistics.mean(q_raw), 1),
                             'median': statistics.median(q_raw), 'max': max(q_raw)},
            'group4_est': {'sum': sum(q_est), 'mean': round(statistics.mean(q_est), 1),
                           'median': statistics.median(q_est), 'max': max(q_est)}},
        'checks': ['input hashes', '160 result files match results_manifest.json', 'gate failures = accepted list',
                   'labeled 797 + short 1,174 = 1,971, disjoint', 'light = raw for all 1,971',
                   'cleaned reasoning is a subsequence of raw', 'content byte-identical to raw'],
    }
    (dest / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main(sys.argv[1])
