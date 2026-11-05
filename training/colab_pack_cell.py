# HPML training packing and fused kernels (DESIGN.md §11.4; one Colab A100, no API key, no evaluation): the gates, then
# arms R, F and P of groups 2 and 3 (epoch 1, seed 1, lr 2e-4), then the summary. Needs Google Drive mounted at
# /content/drive with the seed-1 tuning runs in MyDrive/hpml_stage2_tuning/runs; the results are copied to
# MyDrive/hpml_pack every 10 minutes and after every job. run_pack.py works in the background, so stopping this cell
# does not stop the run; running the cell again follows the log. On a new runtime the finished results are first
# restored from Drive. Before each arm the run waits: read the compute unit balance (Runtime > View resources) and
# answer 'go', or 'stop' if it is below 8 units.
# Files come from the public GitHub repository and the public Drive folder and are checked by SHA-256.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time
ROOT = Path('/content/hpml_pack'); ROOT.mkdir(exist_ok=True)
SEED1 = Path('/content/drive/MyDrive/hpml_stage2_tuning/runs'); DRIVE_OUT = Path('/content/drive/MyDrive/hpml_pack')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
assert all((SEED1 / f'g{g}_lr2e-4/train_report.json').exists() for g in '23'), 'Mount Google Drive first.'
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/train_lora.py': 'b8fcde9d6350746c281593dcb9f8435b9a9ad0ea332c4451335ec7f8ea1db660',
    'training/pack_bench.py': 'a0bc19a4adaf343b2e0a99a1efea0123f4e3f0c6531268d84fcffca9103651f2',
    'training/run_pack.py': 'f43bc2f8b89c2fbb3b940dd5f2bb5fd41bba59499783e072f0a28142cba5cd7e'}
DRIVE_FILES = {  # public Drive file id, SHA-256
    'final_traces.jsonl': ('1y4Aa985j91jEcvzimgnZderzazaRddjF', '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e'),
    'group2_human.jsonl': ('1yytOP7ChuoAVIgaGbuy9QIVzfWLnF7EK', '176dc0b04b9d599870e37172ff06872c0faeacad2ed4f7f1a3b33d21dabe8f93')}
PID, LOG = Path('/content/run_pack.pid'), ROOT / 'run_pack.log'  # the pid file stays outside ROOT (ROOT is synced)

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc, shown = None, 0
if not (PID.exists() and alive(int(PID.read_text()))):
    if DRIVE_OUT.exists() and not (ROOT / 'pack_state.json').exists():
        subprocess.run(['rsync', '-rt', f'{DRIVE_OUT}/', f'{ROOT}/'], check=True)
        print('RESTORED_FROM_DRIVE', sorted(p.name for p in ROOT.iterdir()), flush=True)
    shutil.rmtree(ROOT / 'repo', ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT / 'repo')], check=True)
    WORK, DATA = ROOT / 'work', ROOT / 'data'
    for d in (WORK, DATA): d.mkdir(exist_ok=True)
    for path, digest in REPO_FILES.items():
        assert sha(ROOT / 'repo' / path) == digest, path
        shutil.copyfile(ROOT / 'repo' / path, WORK / Path(path).name)
    for name, (fid, digest) in DRIVE_FILES.items():
        if not (DATA / name).exists() or sha(DATA / name) != digest:
            subprocess.run(['curl', '-sL', '-o', str(DATA / name), f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'], check=True)
        assert sha(DATA / name) == digest, name
    with (ROOT / 'nvidia_smi.txt').open('a') as f: f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    print('FILES_OK', sorted(REPO_FILES), sorted(DRIVE_FILES), flush=True)
    shown = len(LOG.read_text()) if LOG.exists() else 0
    proc = subprocess.Popen([sys.executable, '-u', str(WORK / 'run_pack.py'), str(ROOT), str(SEED1), str(DRIVE_OUT)],
                            stdout=LOG.open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid))
pid, beat, WAIT = int(PID.read_text()), 0.0, ROOT / 'wait_go.json'
while True:
    text = LOG.read_text() if LOG.exists() else ''
    print(text[shown:], end='', flush=True); shown = len(text)
    running = proc.poll() is None if proc else alive(pid)
    if not running: break
    try: arm = json.loads(WAIT.read_text())['arm'] if WAIT.exists() else None
    except ValueError: arm = None
    if arm is not None and not (ROOT / f'go/{arm}').exists():
        answer = ''
        while answer not in ('go', 'stop'):
            answer = input(f'WAIT_GO {arm}: read the balance (Runtime > View resources); go, or stop below 8 units: ').strip().lower()
        (ROOT / 'go').mkdir(exist_ok=True); (ROOT / f'go/{arm}').write_text(answer)
        print('ANSWERED', arm, answer, time.strftime('%H:%M:%S'), flush=True)
    if time.time() - beat > 600 and (ROOT / 'current_job.json').exists():  # heartbeat: last line of the running job
        cur = json.loads((ROOT / 'current_job.json').read_text()); lines = Path(cur['log']).read_text(errors='replace').replace('\r', '\n').strip().splitlines()
        print('HEARTBEAT', time.strftime('%H:%M:%S'), cur['job'], (lines[-1] if lines else '')[:300], flush=True); beat = time.time()
    time.sleep(15)
print('RUN_PACK_EXITED', proc.returncode if proc else 'not started here')
print('Next: read pack_summary.txt; check that MyDrive/hpml_pack holds runs/*/train_report.json and gates/, then delete the runtime.')
