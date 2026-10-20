"""Offline dry run of claude_label.py (free, no API). Usage: python -I dry_run_claude_label.py EMPTY_WORK_DIR

Uses the 25 development rows (human5 + long20) as stand-in inputs. Checks:
1. examples renders all 5 human5 traces.
2. prepare: short traces are skipped, every batch has exactly one calibration item.
3. apply with the reference labels on the calibration item and no deletions elsewhere: F1 1.0, gate passes.
4. apply rejects labels without dropped_at / same_as, with a deletion inside main_derivation, or with a missing key (exit 1).
5. apply with labels that delete everything kept in the calibration trace fails the gate (exit 2).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / 'claude_label.py'
sys.path.insert(0, str(HERE))
import claude_label as cl

work = Path(sys.argv[1])
assert work.is_dir() and not any(work.iterdir()), 'give an empty directory'
hc, pilot_rows, _ = cl.load_pilot()
ref = json.loads((cl.PILOT / 'heldout_labels_r3.json').read_text(encoding='utf-8'))['labelers']['claude']


def tool(*args):
    p = subprocess.run([sys.executable, '-I', str(TOOL), *map(str, args)], capture_output=True, text=True, encoding='utf-8')
    return p.returncode, p.stdout + p.stderr


def plan_from_codes(codes):
    """Unit codes ('.', 'a', 'r') -> label object; main_derivation is the last kept unit (outside every deletion)."""
    md = max(i for i, c in enumerate(codes) if c == '.')
    first_kept = codes.index('.')
    dels, i = [], 0
    while i < len(codes):
        if codes[i] == '.':
            i += 1
            continue
        j = i
        while j + 1 < len(codes) and codes[j + 1] == codes[i]:
            j += 1
        if codes[i] == 'a':
            dels.append({'first': i, 'last': j, 'label': 'abandoned', 'kind': 'branch', 'dropped_at': j, 'why': 'test'})
        else:
            assert first_kept < i
            dels.append({'first': i, 'last': j, 'label': 'repeat', 'kind': 'reverify',
                         'same_as': {'first': first_kept, 'last': first_kept}, 'why': 'test'})
        i = j + 1
    return {'main_derivation': {'first': md, 'last': md}, 'delete': dels}


# 1
code, out = tool('examples', work / 'examples.md')
assert code == 0, out
assert (work / 'examples.md').read_text(encoding='utf-8').count('\n## ') == 5

# 2
inputs = work / 'inputs.jsonl'
inputs.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pilot_rows if r['set'] != 'heldout5'),
                  encoding='utf-8')
job = work / 'job'
code, out = tool('prepare', inputs, job, 10)
assert code == 0, out
prep = json.loads((job / 'prepare.json').read_text(encoding='utf-8'))
assert prep['n_rows'] == 25 and prep['n_annotated'] + prep['n_short_kept_light'] == 25, prep
assert prep['short_ids'] == ['algebra/train/273'], prep['short_ids']  # the only pilot row under 300 tokens
keys = {}
for n in range(1, prep['n_batches'] + 1):
    keys[n] = json.loads((job / 'batches' / f'batch_{n:03d}_key.json').read_text(encoding='utf-8'))
    assert sum(v['calibration'] for v in keys[n].values()) == 1
    md = (job / 'batches' / f'batch_{n:03d}.md').read_text(encoding='utf-8')
    assert not any(v['id'] in md for v in keys[n].values()), 'ids must not appear in the batch file'
assert code == 0 and tool('prepare', inputs, job)[0] != 0, 'prepare must refuse to reshuffle'

# 3
by_id = {r['id']: r for r in pilot_rows}
guide = cl.sha(cl.GUIDE)


def write_labels(n, labels):
    (job / 'labels' / f'batch_{n:03d}.json').write_text(json.dumps({'guide_sha256': guide, 'labels': labels}), encoding='utf-8')


def base_labels(n):
    labels = {}
    for k, v in keys[n].items():
        units = hc.split_units(by_id[v['id']]['reasoning_content'])
        labels[k] = (plan_from_codes(ref[v['id']]) if v['calibration']
                     else {'main_derivation': {'first': len(units) - 1, 'last': len(units) - 1}, 'delete': []})
    return labels


for n in keys:
    write_labels(n, base_labels(n))
    code, out = tool('apply', job, n)
    assert code == 0, out
    rep = json.loads((job / 'results' / f'batch_{n:03d}_report.json').read_text(encoding='utf-8'))
    assert rep['calibration']['f1'] == 1.0 and rep['calibration']['passed'], rep['calibration']
    assert rep['by_status'] == {'nothing_deleted': len(keys[n]) - 1}, rep['by_status']
    print('batch', n, 'ok', json.dumps(rep['calibration']))

# 4
cal_key = next(k for k, v in keys[1].items() if v['calibration'])
other = next(k for k, v in keys[1].items() if not v['calibration'])
bad = base_labels(1)
item = bad[cal_key]['delete'][0]
field = 'dropped_at' if item['label'] == 'abandoned' else 'same_as'
del item[field]
write_labels(1, bad)
code, out = tool('apply', job, 1)
assert code == 1 and 'lacks ' + field in out, out
bad = base_labels(1)
last = bad[other]['main_derivation']['last']
bad[other]['delete'] = [{'first': last, 'last': last, 'label': 'repeat', 'kind': 'tail', 'same_as': {'first': 0, 'last': 0}, 'why': 'x'}]
write_labels(1, bad)
code, out = tool('apply', job, 1)
assert code == 1 and 'inside main_derivation' in out, out
bad = base_labels(1)
del bad[other]
write_labels(1, bad)
code, out = tool('apply', job, 1)
assert code == 1 and 'missing' in out, out

# 5
bad = base_labels(1)
codes = ref[keys[1][cal_key]['id']]
kept = [i for i, c in enumerate(codes) if c == '.']
md = kept[-1]
bad[cal_key] = {'main_derivation': {'first': md, 'last': md},
                'delete': [{'first': kept[0] + 1, 'last': md - 1, 'label': 'repeat', 'kind': 'reverify',
                            'same_as': {'first': kept[0], 'last': kept[0]}, 'why': 'x'}]}
write_labels(1, bad)
code, out = tool('apply', job, 1)
assert code == 2 and 'GATE FAILED' in out, out
write_labels(1, base_labels(1))
assert tool('apply', job, 1)[0] == 0
print('DRY RUN OK: examples, prepare, apply, strict checks and calibration gate')
