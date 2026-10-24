# HPML systems stage (one Colab A100, no API key, no training): LoRA training cost from the records of the final run,
# the offline early-stop estimate, and the vLLM scheduling benchmark on the merged condition 3 / seed 1 model
# (DESIGN.md §11-§12). Needs Google Drive mounted at /content/drive with the final run in MyDrive/hpml_final; the
# results are copied to MyDrive/hpml_sched after every job. run_sched.py works in the background, so stopping this
# cell does not stop the run; running the cell again follows the log. On a new runtime the finished results are first
# restored from Drive, and the run resumes after the last finished job.
# Scripts come from the public GitHub repository and are checked by SHA-256.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time
ROOT = Path('/content/hpml_sched'); ROOT.mkdir(exist_ok=True)
FINAL = Path('/content/drive/MyDrive/hpml_final'); DRIVE_OUT = Path('/content/drive/MyDrive/hpml_sched')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
assert (FINAL / 'final_state.json').exists() and (FINAL / 'adapters/g3_s1/check_loss.json').exists(), 'Mount Google Drive first.'
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/merge_lora.py': 'a673c2f739da01136e7cece66fe6d2e48e38f8f73c175dbb52dba664f5b5feb1',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/final_eval.py': '24ab0e81e8974578b95680dcda724390316b08ae3d6ba712a39452e56fd5c814',
    'training/sched_bench.py': 'b46ce2f1e3d3dd6517226b4ed42c589f0fee52daa79cfcb28f0540c0945fba70',
    'training/lora_cost.py': 'a25d45620de79f49a0e44b1de989bb767442e5239d653cb54a8bb8ac6c473d2d',
    'training/early_stop_estimate.py': 'e49823d2a4ad94bc1eceb3df43e40e4362a1ea09eb0f45ca4a748c64cfd4fd3f',
    'training/run_sched.py': '055381464844a42803150f758b5d7b2805c39dbaddabef81cbdbe075eb5eafdd',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f',
    'base_probe/math_dev_100.jsonl': '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0'}
PID, LOG = Path('/content/run_sched.pid'), ROOT / 'run_sched.log'  # the pid file stays outside ROOT (ROOT is synced)

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc, shown = None, 0
if not (PID.exists() and alive(int(PID.read_text()))):
    if DRIVE_OUT.exists() and not (ROOT / 'sched_state.json').exists():
        subprocess.run(['rsync', '-rt', f'{DRIVE_OUT}/', f'{ROOT}/'], check=True)
        print('RESTORED_FROM_DRIVE', sorted(p.name for p in ROOT.iterdir()), flush=True)
    shutil.rmtree(ROOT / 'repo', ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT / 'repo')], check=True)
    WORK = ROOT / 'work'; WORK.mkdir(exist_ok=True)
    for path, digest in REPO_FILES.items():
        assert sha(ROOT / 'repo' / path) == digest, path
        shutil.copyfile(ROOT / 'repo' / path, WORK / Path(path).name)
    with (ROOT / 'nvidia_smi.txt').open('a') as f: f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    print('FILES_OK', sorted(REPO_FILES), flush=True)
    shown = len(LOG.read_text()) if LOG.exists() else 0
    proc = subprocess.Popen([sys.executable, '-u', str(WORK / 'run_sched.py'), str(ROOT), str(FINAL), str(DRIVE_OUT)],
                            stdout=LOG.open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid))
pid, beat = int(PID.read_text()), 0.0
while True:
    text = LOG.read_text() if LOG.exists() else ''
    print(text[shown:], end='', flush=True); shown = len(text)
    running = proc.poll() is None if proc else alive(pid)
    if not running: break
    if time.time() - beat > 600 and (ROOT / 'current_job.json').exists():  # heartbeat: last line of the running job
        cur = json.loads((ROOT / 'current_job.json').read_text()); lines = Path(cur['log']).read_text(errors='replace').replace('\r', '\n').strip().splitlines()
        print('HEARTBEAT', time.strftime('%H:%M:%S'), cur['job'], (lines[-1] if lines else '')[:300], flush=True); beat = time.time()
    time.sleep(30)
print('RUN_SCHED_EXITED', proc.returncode if proc else 'not started here')
print('Next: check that MyDrive/hpml_sched holds bench/bench_summary.json and analysis/, then delete the runtime.')
