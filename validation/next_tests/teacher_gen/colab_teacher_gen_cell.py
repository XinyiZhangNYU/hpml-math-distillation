# HPML teacher generation, one batch per run (PAID). Set BATCH and BUDGET below. b0 = pilot, 100 problems, about
# USD 0.14; b1-b4 = 475 each, about USD 0.56-0.60 each (peak rates). Upload teacher_gen.py, train2000.jsonl and base_probe_worker.py to /content first.
# Rerunning the cell with the same BATCH resumes from the ledger in /content/hpml_teacher_gen; it never repeats a paid response.
from pathlib import Path
import hashlib, shutil, subprocess, sys, json
from google.colab import userdata
BATCH, BUDGET = 'b0', 0.80  # the budget must leave room for 8 worst-case reservations (8 x USD 0.079)
G = Path('/content/hpml_teacher_gen'); G.mkdir(exist_ok=True)
API_KEY = userdata.get('DEEPSEEK_API_KEY')  # read first: a missing Secret stops the cell before anything is sent
assert API_KEY and API_KEY.strip(), 'Secret DEEPSEEK_API_KEY is empty.'
expected = {'teacher_gen.py': None,
            'train2000.jsonl': '99c149864bf011488ddc61e2f98144d4e1473a97422a19a79e05e0cc66287826',
            'base_probe_worker.py': None}  # SHA-256 checked for the data file only
for name, digest in expected.items():
    if not (G/name).exists():
        shutil.copyfile(Path('/content')/name, G/name)
    assert digest is None or hashlib.sha256((G/name).read_bytes()).hexdigest() == digest, name
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'math-verify==0.9.0', 'transformers==5.17.0',
                'tokenizers==0.23.2', 'huggingface_hub==1.33.0'], check=True)
sys.path.insert(0, str(G))
import teacher_gen
from base_probe_worker import math_grade
count = teacher_gen.qwen_counter()
# free self-test of the real grader and counter before any request
row = next(json.loads(x) for x in (G/'train2000.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(x)['batch'] == BATCH)
fake = lambda text: {'id': row['id'], 'attempt': 0, 'seq': -1, 'finish_reason': 'stop', 'reasoning_content': 'r', 'text': text}
good = teacher_gen.admission(fake('\\boxed{' + row['gold'] + '}'), row, math_grade, count)
bad = teacher_gen.admission(fake('\\boxed{' + row['gold'] + '+1}'), row, math_grade, count)
assert good['admitted'] and bad['reasons'] == ['wrong'] and 0 < good['qwen_total_tokens'] < 2000, (good, bad)
print('SELF_TEST_OK', good['qwen_total_tokens'])
print('TEACHER_GEN_STARTED', BATCH, 'budget USD', BUDGET, '; files verified; key not saved.', flush=True)
st = teacher_gen.run(API_KEY, G, BATCH, BUDGET, grade_fn=math_grade, count_fn=count)
b = st['by_batch'][BATCH]
print('TEACHER_GEN_FINISHED', BATCH)
print(json.dumps({k: v for k, v in b.items() if k != 'dropped_ids'}))
print('DROPPED', b['dropped_ids'])
print('LEDGER_TOTAL_USD', st['all']['priced_at_peak_usd'])
