# HPML scheduling extension (DESIGN.md §11.3; one Colab A100, no API key, no training): five cells of one fixed and
# one queue run each, on the merged condition 3 / seed 1 and condition 4 / seed 1 models. Needs Google Drive mounted
# at /content/drive with the final run in MyDrive/hpml_final; the results are copied to MyDrive/hpml_sched_ext every
# 10 minutes and after every job. run_ext.py works in the background, so stopping this cell does not stop the run;
# running the cell again follows the log. On a new runtime the finished results are first restored from Drive.
# Before each cell the run waits: read the compute unit balance (Runtime > View resources) and answer 'go', or 'stop'
# if it is below 12 units.
# Scripts come from the public GitHub repository and are checked by SHA-256.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time
ROOT = Path('/content/hpml_ext'); ROOT.mkdir(exist_ok=True)
FINAL = Path('/content/drive/MyDrive/hpml_final'); DRIVE_OUT = Path('/content/drive/MyDrive/hpml_sched_ext')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
assert all((FINAL / f'adapters/{a}/check_loss.json').exists() for a in ('g3_s1', 'g4_s1')), 'Mount Google Drive first.'
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/merge_lora.py': 'a673c2f739da01136e7cece66fe6d2e48e38f8f73c175dbb52dba664f5b5feb1',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/final_eval.py': '24ab0e81e8974578b95680dcda724390316b08ae3d6ba712a39452e56fd5c814',
    'training/sched_bench.py': 'b46ce2f1e3d3dd6517226b4ed42c589f0fee52daa79cfcb28f0540c0945fba70',
    'training/sched_ext.py': 'a91d922b7360405d0b98384e3d6f1adb76e7a5c5e8dfb122e0059ebd6569a2e9',
    'training/run_ext.py': 'ba473d5567b01ece579978284313602c5e55b1355a9ca0103c36cd293f8a7632',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f',
    'base_probe/math_dev_100.jsonl': '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0'}
PID, LOG = Path('/content/run_ext.pid'), ROOT / 'run_ext.log'  # the pid file stays outside ROOT (ROOT is synced)

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc, shown = None, 0
if not (PID.exists() and alive(int(PID.read_text()))):
    if DRIVE_OUT.exists() and not (ROOT / 'ext_state.json').exists():
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
    proc = subprocess.Popen([sys.executable, '-u', str(WORK / 'run_ext.py'), str(ROOT), str(FINAL), str(DRIVE_OUT)],
                            stdout=LOG.open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid))
pid, beat, WAIT = int(PID.read_text()), 0.0, ROOT / 'ext/wait_go.json'
while True:
    text = LOG.read_text() if LOG.exists() else ''
    print(text[shown:], end='', flush=True); shown = len(text)
    running = proc.poll() is None if proc else alive(pid)
    if not running: break
    try: cell = json.loads(WAIT.read_text())['cell'] if WAIT.exists() else None
    except ValueError: cell = None
    if cell is not None and not (ROOT / f'ext/go/cell{cell}').exists():
        answer = ''
        while answer not in ('go', 'stop'):
            answer = input(f'WAIT_GO cell {cell}: read the balance (Runtime > View resources); go, or stop below 12 units: ').strip().lower()
        (ROOT / 'ext/go').mkdir(parents=True, exist_ok=True); (ROOT / f'ext/go/cell{cell}').write_text(answer)
        print('ANSWERED', cell, answer, time.strftime('%H:%M:%S'), flush=True)
    if time.time() - beat > 600 and (ROOT / 'current_job.json').exists():  # heartbeat: last line of the running job
        cur = json.loads((ROOT / 'current_job.json').read_text()); lines = Path(cur['log']).read_text(errors='replace').replace('\r', '\n').strip().splitlines()
        print('HEARTBEAT', time.strftime('%H:%M:%S'), cur['job'], (lines[-1] if lines else '')[:300], flush=True); beat = time.time()
    time.sleep(15)
print('RUN_EXT_EXITED', proc.returncode if proc else 'not started here')
print('Next: check that MyDrive/hpml_sched_ext holds ext/cell*/cell_summary.json (or ext/not_run.json), then delete the runtime.')
