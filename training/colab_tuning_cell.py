# HPML stage 2 tuning (one Colab A100 runtime, no API key, about 7.5-8 compute hours on a 40 GB A100).
# Learning rate {1e-4, 2e-4} x epochs {1, 2, 3} for training groups 3, 4 and 2; every adapter is merged into the base
# weights and evaluated on MATH dev 100 (4 samples, 16k tokens); then condition 1 (base, Minerva 4-shot) on dev.
# run_tuning.py does the work in the background, so stopping this cell does not stop the run: running the cell again
# follows the log, or, if the runtime restarted with its files still there, resumes after the last finished job.
# Files come from the public GitHub repository and the public Drive folder and are checked by SHA-256.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time
ROOT = Path('/content/hpml_stage2_tuning'); ROOT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/train_lora.py': 'd83e35f15327a70e5ad58fda9301b28c57a8d0120b18445094c26ff62691da8e',
    'training/merge_lora.py': 'a673c2f739da01136e7cece66fe6d2e48e38f8f73c175dbb52dba664f5b5feb1',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/run_tuning.py': '32735e7cc75e64f708f9bc6854cd8f23f5783dc1e003b77aca338fcd4df4bbe9',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f',
    'base_probe/math_dev_100.jsonl': '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0'}
DRIVE_FILES = {  # public Drive file id, SHA-256
    'final_traces.jsonl': ('1y4Aa985j91jEcvzimgnZderzazaRddjF', '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e'),
    'group4_heavy.jsonl': ('1IQAKEYo48J39Wq33-y-rdUPS7GcXud8W', '8ac76e67953c530714e15ef21411112fc45133db6875fd0e6a6b5f45ed7c286b'),
    'group2_human.jsonl': ('1yytOP7ChuoAVIgaGbuy9QIVzfWLnF7EK', '176dc0b04b9d599870e37172ff06872c0faeacad2ed4f7f1a3b33d21dabe8f93')}
PID, LOG = ROOT/'run_tuning.pid', ROOT/'run_tuning.log'

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc = None
if not (PID.exists() and alive(int(PID.read_text()))):
    shutil.rmtree(ROOT/'repo', ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT/'repo')], check=True)
    WORK, DATA = ROOT/'work', ROOT/'data'; WORK.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    for path, digest in REPO_FILES.items():
        assert sha(ROOT/'repo'/path) == digest, path
        shutil.copyfile(ROOT/'repo'/path, WORK/Path(path).name)
    for name, (fid, digest) in DRIVE_FILES.items():
        if not (DATA/name).exists() or sha(DATA/name) != digest:
            subprocess.run(['curl', '-sL', '-o', str(DATA/name), f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'], check=True)
        assert sha(DATA/name) == digest, name
    with (ROOT/'nvidia_smi.txt').open('a') as f: f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    print('FILES_OK', sorted(REPO_FILES), sorted(DRIVE_FILES), flush=True)
    proc = subprocess.Popen([sys.executable, '-u', str(WORK/'run_tuning.py'), str(ROOT)], stdout=LOG.open('a'),
                            stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid))
pid, shown, beat = int(PID.read_text()), 0, 0.0
while True:
    text = LOG.read_text() if LOG.exists() else ''
    print(text[shown:], end='', flush=True); shown = len(text)
    running = proc.poll() is None if proc else alive(pid)
    if not running: break
    if time.time() - beat > 600 and (ROOT/'current_job.json').exists():  # heartbeat: last line of the running job
        cur = json.loads((ROOT/'current_job.json').read_text()); lines = Path(cur['log']).read_text(errors='replace').replace('\r', '\n').strip().splitlines()
        print('HEARTBEAT', time.strftime('%H:%M:%S'), cur['job'], (lines[-1] if lines else '')[:300], flush=True); beat = time.time()
    time.sleep(30)
print('RUN_TUNING_EXITED', proc.returncode if proc else 'not started here')
print('Next: read tuning_summary.json; keep the runtime until the selected adapters are saved.')
