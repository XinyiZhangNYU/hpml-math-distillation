# HPML loop-rule sensitivity (DESIGN.md §12.1; Colab CPU runtime, no GPU, no API key, no generation). Needs Google
# Drive mounted at /content/drive with the final run in MyDrive/hpml_final and the §12 estimate in
# MyDrive/hpml_sched/analysis; the result, early_stop_sensitivity.json, is written there (with its log and pip freeze).
# The script works in the background, so stopping this cell does not stop it; running the cell again follows the log.
# Scripts come from the public GitHub repository and are checked by SHA-256.
from pathlib import Path
import hashlib, os, shutil, subprocess, sys, time
ROOT = Path('/content/hpml_sens'); ROOT.mkdir(exist_ok=True)
EVALS = Path('/content/drive/MyDrive/hpml_final/evals'); ANALYSIS = Path('/content/drive/MyDrive/hpml_sched/analysis')
ESTIMATE, RESULT = ANALYSIS / 'early_stop_estimate.json', ANALYSIS / 'early_stop_sensitivity.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert EVALS.exists() and ESTIMATE.exists(), 'Mount Google Drive first.'
print('CPUS', os.cpu_count())
REPO_FILES = {  # repository path -> SHA-256
    'training/early_stop_sensitivity.py': 'f4162c3ea68011ff299baeab790884c2d80dffa0153a068bc4397dc537a1ab6d',
    'training/early_stop_estimate.py': 'e49823d2a4ad94bc1eceb3df43e40e4362a1ea09eb0f45ca4a748c64cfd4fd3f',
    'training/final_eval.py': '24ab0e81e8974578b95680dcda724390316b08ae3d6ba712a39452e56fd5c814',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f',
    'base_probe/math_dev_100.jsonl': '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0'}
PID, LOG, WORK = Path('/content/sens.pid'), ROOT / 'early_stop_sensitivity.log', ROOT / 'work'

def alive(pid):
    status = Path(f'/proc/{pid}/status')
    return status.exists() and 'State:\tZ' not in status.read_text()

proc, shown = None, 0
if RESULT.exists():
    print('ALREADY_DONE', RESULT)
elif not (PID.exists() and alive(int(PID.read_text()))):
    shutil.rmtree(ROOT / 'repo', ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/XinyiZhangNYU/hpml-math-distillation', str(ROOT / 'repo')], check=True)
    WORK.mkdir(exist_ok=True)
    for path, digest in REPO_FILES.items():
        assert sha(ROOT / 'repo' / path) == digest, path
        shutil.copyfile(ROOT / 'repo' / path, WORK / Path(path).name)
    env = ROOT / 'venv'; py = str(env / 'bin/python')
    if not Path(py).exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(env)], check=True)
        subprocess.run([py, '-m', 'pip', 'install', '-q', 'math-verify==0.9.0'], check=True)
    (ROOT / 'pip_freeze.txt').write_text(subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    print('FILES_OK', sorted(REPO_FILES), flush=True)
    proc = subprocess.Popen([py, '-I', '-u', str(WORK / 'early_stop_sensitivity.py'), str(EVALS), str(ESTIMATE), str(ANALYSIS)],
                            stdout=LOG.open('w'), stderr=subprocess.STDOUT, start_new_session=True, cwd='/content')
    PID.write_text(str(proc.pid))
if PID.exists() and not RESULT.exists() or proc:
    pid = int(PID.read_text())
    while True:
        text = LOG.read_text() if LOG.exists() else ''
        print(text[shown:], end='', flush=True); shown = len(text)
        if not (proc.poll() is None if proc else alive(pid)): break
        time.sleep(20)
    for f in (LOG, ROOT / 'pip_freeze.txt'):
        shutil.copyfile(f, ANALYSIS / ('early_stop_sensitivity_' + f.name if f.name == 'pip_freeze.txt' else f.name))
    print('EXITED', proc.returncode if proc else 'not started here', 'RESULT', RESULT.exists())
    print('Next: delete the runtime.')
