# HPML stage 2 smoke test (one Colab A100 runtime, no API key, about 1-1.5 compute hours). Nothing here is a result:
# the adapters are disposable and the dev numbers only measure time, memory and correctness of the pipeline.
#   1. group 3, lr 1e-4, 30 steps, with the padding check   2. group 2, lr 1e-4, 15 steps (short sequences)
#   3. merge the group 3 adapter into the base weights    4. vLLM on MATH dev 100, 1 sample, 16k tokens, with the
#      check that vLLM on the merged weights reproduces the adapter's loss.
# Files come from the public GitHub repository and the public Drive folder and are checked by SHA-256.
# Run this cell once; it blocks until the last stage ends, then writes hpml_stage2_smoke_evidence.zip.
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, sys, time, zipfile
ROOT = Path('/content/hpml_stage2_smoke'); ROOT.mkdir(exist_ok=True)
assert not (ROOT/'smoke_stages.json').exists(), 'Results exist: inspect them instead of rerunning.'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
(ROOT/'nvidia_smi.txt').write_text(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/train_lora.py': 'd83e35f15327a70e5ad58fda9301b28c57a8d0120b18445094c26ff62691da8e',
    'training/merge_lora.py': 'a673c2f739da01136e7cece66fe6d2e48e38f8f73c175dbb52dba664f5b5feb1',
    'training/dev_eval.py': '31520942316a4e80b5add1595fe4665d5a43eb65524e80d477d1433619c40cc9',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f',
    'base_probe/math_dev_100.jsonl': '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0'}
DRIVE_FILES = {  # public Drive file id, SHA-256
    'final_traces.jsonl': ('1y4Aa985j91jEcvzimgnZderzazaRddjF', '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e'),
    'group4_heavy.jsonl': ('1IQAKEYo48J39Wq33-y-rdUPS7GcXud8W', '8ac76e67953c530714e15ef21411112fc45133db6875fd0e6a6b5f45ed7c286b'),
    'group2_human.jsonl': ('1yytOP7ChuoAVIgaGbuy9QIVzfWLnF7EK', '176dc0b04b9d599870e37172ff06872c0faeacad2ed4f7f1a3b33d21dabe8f93')}
subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT/'repo')], check=True)
WORK, DATA = ROOT/'work', ROOT/'data'; WORK.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
for path, digest in REPO_FILES.items():
    assert sha(ROOT/'repo'/path) == digest, path
    shutil.copyfile(ROOT/'repo'/path, WORK/Path(path).name)
for name, (fid, digest) in DRIVE_FILES.items():
    subprocess.run(['curl', '-sL', '-o', str(DATA/name), f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'], check=True)
    assert sha(DATA/name) == digest, name
print('FILES_OK', sorted(REPO_FILES), sorted(DRIVE_FILES), flush=True)

def venv(name, packages):
    env = ROOT/name
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(env)], check=True)
    py = str(env/'bin/python')
    with (ROOT/f'{name}_install.log').open('w') as log:
        subprocess.run([py, '-m', 'pip', 'install', *packages], stdout=log, stderr=subprocess.STDOUT, check=True)
    (ROOT/f'{name}_pip_freeze.txt').write_text(subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    return py

stages = []
def stage(name, py, args, timeout):
    if stages and stages[-1]['exit'] != 0:
        stages.append({'stage': name, 'exit': 'skipped'}); return
    print('STAGE_START', name, datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
    t = time.perf_counter()
    with (ROOT/f'{name}.log').open('w') as log:
        try: code = subprocess.run([py, '-u', *args], cwd=WORK, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired: code = f'timeout_{timeout}s'
    stages.append({'stage': name, 'exit': code, 'seconds': time.perf_counter() - t})
    (ROOT/'smoke_stages.json').write_text(json.dumps(stages, indent=2))
    print('STAGE_END', json.dumps(stages[-1]), flush=True)

t0 = time.perf_counter()
train_py = venv('trainvenv', ['unsloth==2026.10.5', 'unsloth_zoo==2026.10.5', 'transformers==5.17.0', 'peft==0.21.1',
                              'torch==2.11.0', 'xformers==0.0.35', 'ninja'])
stages.append({'stage': 'install_train', 'exit': 0, 'seconds': time.perf_counter() - t0})
RUNS = ROOT/'runs'
stage('train_g3', train_py, ['train_lora.py', '--data', str(DATA), '--group', '3', '--lr', '1e-4', '--epochs', '1',
                             '--max-steps', '30', '--padding-check', '--out', str(RUNS/'smoke_g3')], 2700)
stage('train_g2', train_py, ['train_lora.py', '--data', str(DATA), '--group', '2', '--lr', '1e-4', '--epochs', '1',
                             '--max-steps', '15', '--out', str(RUNS/'smoke_g2')], 1800)
stage('merge_g3', train_py, ['merge_lora.py', str(RUNS/'smoke_g3'/'step30'), str(ROOT/'merged_g3')], 1800)
if stages[-1]['exit'] == 0:
    t0 = time.perf_counter()
    vllm_py = venv('vllmvenv', ['vllm==0.21.0', 'transformers==5.17.0', 'tokenizers==0.23.2', 'math-verify==0.9.0', 'ninja'])
    stages.append({'stage': 'install_vllm', 'exit': 0, 'seconds': time.perf_counter() - t0})
    stage('dev_g3', vllm_py, ['dev_eval.py', '--model', str(ROOT/'merged_g3'), '--tag', 'smoke_g3_step30', '--prompt', 'chat',
                              '--samples', '1', '--max-tokens', '16384', '--check-loss', str(RUNS/'smoke_g3'/'step30'/'check_loss.json'),
                              '--out', str(RUNS/'dev_smoke_g3')], 3000)
(ROOT/'smoke_stages.json').write_text(json.dumps(stages, indent=2))
keep = [p for p in ROOT.iterdir() if p.is_file()] + [p for p in RUNS.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl')
        and p.name != 'adapter_config.json'] + [ROOT/'merged_g3'/'merge_report.json']
with zipfile.ZipFile(ROOT/'hpml_stage2_smoke_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in keep:
        if p.exists() and p.name != 'hpml_stage2_smoke_evidence.zip':
            z.write(p, str(p.relative_to(ROOT)))
for s in stages: print(json.dumps(s))
for name in ['smoke_g3', 'smoke_g2']:
    r = RUNS/name/'train_report.json'
    if r.exists():
        r = json.loads(r.read_text()); st = r['steps'][5:] or r['steps']
        print(name, 'padding_check', r.get('padding_check', {}).get('pass'), 'median step %.1f s' % sorted(x['seconds'] for x in st)[len(st)//2] if st else '',
              'tokens/s %.0f' % (sum(x['real_tokens'] for x in st) / sum(x['seconds'] for x in st)) if st else '',
              'peak alloc %.1f GiB' % max((x['peak_allocated_gib'] for x in r['steps']), default=0))
d = RUNS/'dev_smoke_g3'/'dev_summary.json'
if d.exists():
    d = json.loads(d.read_text()); print('dev', {k: d.get(k) for k in ('accuracy', 'truncated', 'mean_output_tokens', 'generation_wall_seconds', 'load_seconds')},
                                        'check_loss', (d.get('check_loss') or {}).get('max_relative_difference'))
print('SMOKE_DONE - download hpml_stage2_smoke_evidence.zip from', ROOT, 'then delete the runtime.')
