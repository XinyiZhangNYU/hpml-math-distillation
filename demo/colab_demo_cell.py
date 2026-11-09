# HPML demo extraction (DESIGN.md §13; Colab CPU runtime, no GPU, no API key, no generation). Needs Google Drive
# mounted at /content/drive with the final run in MyDrive/hpml_final; writes demo_data.json, its log and pip freeze to
# MyDrive/hpml_demo. Scripts come from the public GitHub repository and are checked by SHA-256.
from pathlib import Path
import hashlib, shutil, subprocess, sys
ROOT = Path('/content/hpml_demo'); ROOT.mkdir(exist_ok=True)
EVALS, OUT = Path('/content/drive/MyDrive/hpml_final/evals'), Path('/content/drive/MyDrive/hpml_demo')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert EVALS.exists(), 'Mount Google Drive first.'
REPO_FILES = {  # repository path -> SHA-256
    'demo/demo_extract.py': '2e81ede132ca18d6ce7eeecdc575c3d67d73b87183646ae68afba8d275cda4f0',
    'training/early_stop_estimate.py': 'e49823d2a4ad94bc1eceb3df43e40e4362a1ea09eb0f45ca4a748c64cfd4fd3f',
    'training/final_eval.py': '24ab0e81e8974578b95680dcda724390316b08ae3d6ba712a39452e56fd5c814',
    'training/dev_eval.py': '4d1aeb89b558a0209c162b23760b859ff60e5fa4c7121fe619d28f49d3313490',
    'training/sft_data.py': '9f4c31efaa7083aa3be60ef584609195024cebaf5c11b2f31b2370fbc214f3f3',
    'validation/eval_common.py': '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76',
    'base_probe/base_probe_worker.py': 'e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f'}
WORK = ROOT / 'work'
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
LOG = ROOT / 'demo_extract.log'
with LOG.open('w') as f:
    rc = subprocess.run([py, '-I', '-u', str(WORK / 'demo_extract.py'), str(EVALS), str(OUT)], stdout=f,
                        stderr=subprocess.STDOUT, cwd='/content').returncode
print(LOG.read_text())
OUT.mkdir(parents=True, exist_ok=True)
shutil.copyfile(LOG, OUT / LOG.name); shutil.copyfile(ROOT / 'pip_freeze.txt', OUT / 'demo_extract_pip_freeze.txt')
print('EXITED', rc, 'SIZE_MB', round((OUT / 'demo_data.json').stat().st_size / 2**20, 2) if (OUT / 'demo_data.json').exists() else None)
print('Next: delete the runtime.')
