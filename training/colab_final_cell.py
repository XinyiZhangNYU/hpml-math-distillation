# HPML final stage (one Colab A100, no API key): training seeds 2 and 3 of groups 3, 4 and 2, and the final evaluation
# of the 9 merged models and of condition 1 on the frozen test tiers (DESIGN.md §8-§9), then final_stats.py.
# Needs Google Drive mounted at /content/drive: the seed-1 adapters come from MyDrive/hpml_stage2_tuning/runs, and the
# results are copied to MyDrive/hpml_final after every job. run_final.py works in the background, so stopping this
# cell does not stop the run; running the cell again follows the log. On a new runtime the finished results are first
# restored from Drive, and the run resumes after the last finished job.
# Files come from the public GitHub repository and the public Drive folder and are checked by SHA-256.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time, zipfile
ROOT = Path('/content/hpml_final'); ROOT.mkdir(exist_ok=True)
SEED1 = Path('/content/drive/MyDrive/hpml_stage2_tuning/runs'); DRIVE_OUT = Path('/content/drive/MyDrive/hpml_final')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
print('ACTUAL_GPU', gpu); assert 'A100' in gpu, 'Expected an A100: inspect before spending compute.'
assert all((SEED1 / f'g{g}_lr2e-4/epoch1/check_loss.json').exists() for g in '342'), 'Mount Google Drive first.'
REPO_FILES = {  # repository path -> SHA-256
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'training/train_lora.py': 'b8fcde9d6350746c281593dcb9f8435b9a9ad0ea332c4451335ec7f8ea1db660',
    'training/merge_lora.py': 'a673c2f739da01136e7cece66fe6d2e48e38f8f73c175dbb52dba664f5b5feb1',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/final_eval.py': '24ab0e81e8974578b95680dcda724390316b08ae3d6ba712a39452e56fd5c814',
    'training/final_stats.py': 'd4a9e2eec31a573df98f83a4815301b86342c757fbadf87cfb8c6219556a7cb3',
    'training/run_final.py': '83a7904cda8b41e955b5cb82c8492bccc84f08bbe96d86b402ec6075ca3efd84',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f'}
TEST_FILES = {  # repository path -> SHA-256; the zip holds the tier 2 and tier 3 files (final_eval.py checks them)
    'validation/next_tests/test_freeze/stage2_run1/tier1_variants.jsonl': 'c5e1eb3cea9ddcabdadd93952951cc1673968083c113662926c8c2245e16331b',
    'validation/next_tests/test_freeze/scoring_exclusions.json': 'c268021888738f9c22f5791024f0954a9dc79f54ff5f386be17e961bc392b62c',
    'validation/next_tests/test_freeze/stage1_run1/test_freeze_stage1.zip': '87055cbc387afdfd92165017c5a341ba9c8fb6f29f1cb99131be4a7bdd773b71'}
DRIVE_FILES = {  # public Drive file id, SHA-256
    'final_traces.jsonl': ('1y4Aa985j91jEcvzimgnZderzazaRddjF', '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e'),
    'group4_heavy.jsonl': ('1IQAKEYo48J39Wq33-y-rdUPS7GcXud8W', '8ac76e67953c530714e15ef21411112fc45133db6875fd0e6a6b5f45ed7c286b'),
    'group2_human.jsonl': ('1yytOP7ChuoAVIgaGbuy9QIVzfWLnF7EK', '176dc0b04b9d599870e37172ff06872c0faeacad2ed4f7f1a3b33d21dabe8f93')}
PID, LOG = Path('/content/run_final.pid'), ROOT / 'run_final.log'  # the pid file stays outside ROOT (ROOT is synced)

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc, shown = None, 0
if not (PID.exists() and alive(int(PID.read_text()))):
    if DRIVE_OUT.exists() and not (ROOT / 'final_state.json').exists():
        subprocess.run(['rsync', '-rt', f'{DRIVE_OUT}/', f'{ROOT}/'], check=True)
        print('RESTORED_FROM_DRIVE', sorted(p.name for p in ROOT.iterdir()), flush=True)
    shutil.rmtree(ROOT / 'repo', ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT / 'repo')], check=True)
    WORK, DATA, TESTS = ROOT / 'work', ROOT / 'data', ROOT / 'tests'
    for d in (WORK, DATA, TESTS): d.mkdir(exist_ok=True)
    for path, digest in {**REPO_FILES, **TEST_FILES}.items():
        assert sha(ROOT / 'repo' / path) == digest, path
    for path in REPO_FILES:
        shutil.copyfile(ROOT / 'repo' / path, WORK / Path(path).name)
    for path in list(TEST_FILES)[:2]:
        shutil.copyfile(ROOT / 'repo' / path, TESTS / Path(path).name)
    with zipfile.ZipFile(ROOT / 'repo' / list(TEST_FILES)[2]) as z:
        for name in ('tier2_aime2026.jsonl', 'tier2_hmmt_feb2026.jsonl', 'tier3_math500.jsonl'):
            (TESTS / name).write_bytes(z.read(name))
    for name, (fid, digest) in DRIVE_FILES.items():
        if not (DATA / name).exists() or sha(DATA / name) != digest:
            subprocess.run(['curl', '-sL', '-o', str(DATA / name), f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'], check=True)
        assert sha(DATA / name) == digest, name
    with (ROOT / 'nvidia_smi.txt').open('a') as f: f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    print('FILES_OK', sorted(REPO_FILES), sorted(p.name for p in TESTS.iterdir()), sorted(DRIVE_FILES), flush=True)
    shown = len(LOG.read_text()) if LOG.exists() else 0
    proc = subprocess.Popen([sys.executable, '-u', str(WORK / 'run_final.py'), str(ROOT), str(SEED1), str(DRIVE_OUT)],
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
print('RUN_FINAL_EXITED', proc.returncode if proc else 'not started here')
print('Next: read final_report.txt; check that MyDrive/hpml_final holds every result, then delete the runtime.')
