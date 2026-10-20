# HPML BASE PROBE 9B-Base: few-shot MATH dev 100 + MBPP val 90; no training, no API.
from pathlib import Path
import json,sys,subprocess,hashlib,shutil
PROBE_ROOT=Path('/content/hpml_base_probe'); PROBE_ROOT.mkdir(exist_ok=True)
assert not (PROBE_ROOT/'9b-base_status.json').exists(), 'Inspect existing job; never duplicate it.'
expected={"base_probe_worker.py": "e8e636e90a32621a8d7d4e0bbff498e74258cb3bddcc9757c89257a507a8703f", "base_probe_controller.py": "3d8eedb423cb2556d2a99c0077d120a6bb6dfb872028643820de162c509b1075", "math_dev_100.jsonl": "0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0", "mbpp_dev_90.jsonl": "aeded3158024aafbdfcb71210c4b679a4586b3c4b441e60dc3c51ef3369ea812", "mbpp_shots.json": "edc4d9baae9fb0645acd5fb901dd224ff4d3794e21a2fa7d2f22c402cb11a250"}
for name,digest in expected.items():
    src=Path('/content')/name
    assert hashlib.sha256(src.read_bytes()).hexdigest()==digest, name
    if name.endswith('.py'): compile(src.read_text(),name,'exec')
    shutil.move(str(src),str(PROBE_ROOT/name))
print('PROBE_FILES_OK',sorted(expected))
print('ACTUAL_GPU',subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip())
subprocess.run([sys.executable,'-m','venv','--system-site-packages','--without-pip',str(PROBE_ROOT/'.venv')],check=True)
PROBE_PY=str(PROBE_ROOT/'.venv/bin/python')
probe_cmd=(f"{PROBE_PY} -m pip install vllm==0.21.0 transformers==5.17.0 tokenizers==0.23.2 math-verify==0.9.0 ninja > {PROBE_ROOT}/install.log 2>&1 "
           f"&& PROBE_STUDENT=9b-base {PROBE_PY} -u {PROBE_ROOT}/base_probe_controller.py > {PROBE_ROOT}/controller.log 2>&1")
probe_proc=subprocess.Popen(['bash','-c',probe_cmd],start_new_session=True)
print('PROBE_PIPELINE_STARTED',probe_proc.pid,'install then 9B-Base probe; hard limit 2400 s after install')
