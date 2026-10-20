"""Bounded owned child process, GPU sampling, manifest and sealed export for one student."""
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT/'.venv/bin/python')
STUDENT = os.environ.get('PROBE_STUDENT','9b-base')
LIMIT = 2400


def run():
    status_path = ROOT/f'{STUDENT}_status.json'
    assert not status_path.exists(), 'Inspect existing stage before any restart.'
    status = {'student':STUDENT,'state':'running','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hard_wall_limit_seconds':LIMIT,'shutdown_required_after_export':True}
    samples = []; started = time.perf_counter()
    def save():
        status_path.write_text(json.dumps(status,indent=2))
    save()
    with (ROOT/f'{STUDENT}.log').open('w') as log:
        process = subprocess.Popen([PY,'-u',str(ROOT/'base_probe_worker.py')],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            if time.perf_counter()-started > LIMIT:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                status['state'] = 'wall_limit_reached'; break
            try:
                raw = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).strip()
                memory, util = map(float,raw.split(','))
                samples.append({'seconds':time.perf_counter()-started,'gpu_used_gib':memory/1024,'gpu_utilization_pct':util})
            except Exception: pass
            time.sleep(2)
    if status['state'] == 'running': status['state'] = 'finished' if process.returncode==0 else 'failed'
    status.update(exit_code=process.returncode, wall_seconds=time.perf_counter()-started,
        ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sampled_device_peak_gib=max((r['gpu_used_gib'] for r in samples),default=None))
    summary = ROOT/f'{STUDENT}_summary.json'
    if summary.exists(): status['complete'] = json.loads(summary.read_text()).get('complete',False)
    save()
    (ROOT/f'{STUDENT}_resources.json').write_text(json.dumps(samples))
    subprocess.run([PY,'-m','pip','freeze'],stdout=(ROOT/'requirements-resolved.txt').open('w'),check=False)
    artifacts = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.jsonl','.log','.txt'}]
    manifest = {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts}
    (ROOT/f'{STUDENT}_evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
    bundle = ROOT/f'hpml_base_probe_{STUDENT}_evidence.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in artifacts+[ROOT/f'{STUDENT}_evidence_manifest.json']: z.write(p,p.name)
    print('PROBE_CONTROLLER_FINISHED',json.dumps(status),flush=True)
    print('EVIDENCE_ZIP_SHA256',hashlib.sha256(bundle.read_bytes()).hexdigest(),flush=True)


if __name__ == '__main__': run()
