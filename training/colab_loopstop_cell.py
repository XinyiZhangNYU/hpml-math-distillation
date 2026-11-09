# HPML online loop early stop (DESIGN.md §14; one Colab A100, no API key, no training): off and on runs of the §12
# loop rule on tier 2 x 2 samples, for the merged condition 4 / seed 1 and condition 3 / seed 1 models. Needs Google
# Drive mounted at /content/drive with the final run in MyDrive/hpml_final; the results are copied to
# MyDrive/hpml_loopstop every 10 minutes and after every job. run_loopstop.py works in the background, so stopping this
# cell does not stop the run; running the cell again follows the log. On a new runtime the finished results are first
# restored from Drive. Before each model's engine loads the run waits: read the compute unit balance
# (Runtime > View resources) and answer 'go', or 'stop' if it is below 12 units.
# Scripts and test files come from the public GitHub repository and are checked by SHA-256.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time, zipfile
from google.colab import drive; drive.mount('/content/drive')
ROOT = Path('/content/hpml_loopstop'); ROOT.mkdir(exist_ok=True)
FINAL = Path('/content/drive/MyDrive/hpml_final'); DRIVE_OUT = Path('/content/drive/MyDrive/hpml_loopstop')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
assert all((FINAL / f'adapters/{a}/check_loss.json').exists() for a in ('g3_s1', 'g4_s1')), 'Mount Google Drive first.'
assert all((FINAL / f'evals/{r}/tier2/summary.json').exists() for r in ('c3_s1', 'c4_s1')), 'Saved tier 2 outputs missing.'
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/merge_lora.py': 'a673c2f739da01136e7cece66fe6d2e48e38f8f73c175dbb52dba664f5b5feb1',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/final_eval.py': '24ab0e81e8974578b95680dcda724390316b08ae3d6ba712a39452e56fd5c814',
    'training/sched_bench.py': 'b46ce2f1e3d3dd6517226b4ed42c589f0fee52daa79cfcb28f0540c0945fba70',
    'training/sched_ext.py': 'a91d922b7360405d0b98384e3d6f1adb76e7a5c5e8dfb122e0059ebd6569a2e9',
    'training/early_stop_estimate.py': 'e49823d2a4ad94bc1eceb3df43e40e4362a1ea09eb0f45ca4a748c64cfd4fd3f',
    'training/loopstop_bench.py': 'c055add965f5877320ec91955d640de45a718ef026839187a48f184c8a408150',
    'training/run_loopstop.py': 'aa7ce3676a56bdc8d7d8158accfcb6be1c7c40af3791e4fb1fa31df365efb2c3',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f'}
TEST_FILES = {  # repository path -> SHA-256; the zip holds the tier 2 files (final_eval.py checks them)
    'validation/next_tests/test_freeze/scoring_exclusions.json': 'c268021888738f9c22f5791024f0954a9dc79f54ff5f386be17e961bc392b62c',
    'validation/next_tests/test_freeze/stage1_run1/test_freeze_stage1.zip': '87055cbc387afdfd92165017c5a341ba9c8fb6f29f1cb99131be4a7bdd773b71'}
PID, LOG = Path('/content/run_loopstop.pid'), ROOT / 'run_loopstop.log'  # the pid file stays outside ROOT (ROOT is synced)

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc, shown = None, 0
if not (PID.exists() and alive(int(PID.read_text()))):
    if DRIVE_OUT.exists() and not (ROOT / 'ls_state.json').exists():
        subprocess.run(['rsync', '-rt', f'{DRIVE_OUT}/', f'{ROOT}/'], check=True)
        print('RESTORED_FROM_DRIVE', sorted(p.name for p in ROOT.iterdir()), flush=True)
    shutil.rmtree(ROOT / 'repo', ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT / 'repo')], check=True)
    WORK, TESTS = ROOT / 'work', ROOT / 'tests'
    for d in (WORK, TESTS): d.mkdir(exist_ok=True)
    for path, digest in {**REPO_FILES, **TEST_FILES}.items():
        assert sha(ROOT / 'repo' / path) == digest, path
    for path in REPO_FILES:
        shutil.copyfile(ROOT / 'repo' / path, WORK / Path(path).name)
    shutil.copyfile(ROOT / 'repo' / list(TEST_FILES)[0], TESTS / 'scoring_exclusions.json')
    with zipfile.ZipFile(ROOT / 'repo' / list(TEST_FILES)[1]) as z:
        for name in ('tier2_aime2026.jsonl', 'tier2_hmmt_feb2026.jsonl'):
            (TESTS / name).write_bytes(z.read(name))
    with (ROOT / 'nvidia_smi.txt').open('a') as f: f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    print('FILES_OK', sorted(REPO_FILES), sorted(p.name for p in TESTS.iterdir()), flush=True)
    shown = len(LOG.read_text()) if LOG.exists() else 0
    proc = subprocess.Popen([sys.executable, '-u', str(WORK / 'run_loopstop.py'), str(ROOT), str(FINAL), str(DRIVE_OUT)],
                            stdout=LOG.open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid))
pid, beat, WAIT = int(PID.read_text()), 0.0, ROOT / 'ls/wait_go.json'
while True:
    text = LOG.read_text() if LOG.exists() else ''
    print(text[shown:], end='', flush=True); shown = len(text)
    running = proc.poll() is None if proc else alive(pid)
    if not running: break
    try: cell = json.loads(WAIT.read_text())['cell'] if WAIT.exists() else None
    except ValueError: cell = None
    if cell is not None and not (ROOT / f'ls/go/cell{cell}').exists():
        answer = ''
        while answer not in ('go', 'stop'):
            answer = input(f'WAIT_GO model {cell} (1 = condition 4, 2 = condition 3): read the balance (Runtime > View resources); go, or stop below 12 units: ').strip().lower()
        (ROOT / 'ls/go').mkdir(parents=True, exist_ok=True); (ROOT / f'ls/go/cell{cell}').write_text(answer)
        print('ANSWERED', cell, answer, time.strftime('%H:%M:%S'), flush=True)
    if time.time() - beat > 600 and (ROOT / 'current_job.json').exists():  # heartbeat: last line of the running job
        cur = json.loads((ROOT / 'current_job.json').read_text()); lines = Path(cur['log']).read_text(errors='replace').replace('\r', '\n').strip().splitlines()
        print('HEARTBEAT', time.strftime('%H:%M:%S'), cur['job'], (lines[-1] if lines else '')[:300], flush=True); beat = time.time()
    time.sleep(15)
print('RUN_LOOPSTOP_EXITED', proc.returncode if proc else 'not started here')
print('Next: check that MyDrive/hpml_loopstop holds ls/c4 and ls/c3 with model_summary.json (or not_run.json), then delete the runtime.')
