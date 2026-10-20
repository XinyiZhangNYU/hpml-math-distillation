"""Run bounded stages in separate processes, releasing GPU memory between them."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile

ROOT=Path('/content/hpml_validation')
PY=str(ROOT/'.venv/bin/python')
status={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stages':{},'complete':False,'teacher_api':'not_run_missing_user_supplied_access_and_budget'}
if '--vllm-only' in sys.argv:
    prior_status = json.loads((ROOT/'attempts/attempt-02/run_status.json').read_text())
    if prior_status.get('stages', {}).get('hf', {}).get('state') == 'finished':
        status['stages']['hf'] = prior_status['stages']['hf']
        status['retained_hf_evidence'] = 'attempts/attempt-02 (same session, source, model, and prompts)'
samples=[]
stop=threading.Event()

def save():
    (ROOT/'run_status.json').write_text(json.dumps(status,indent=2))

def monitor():
    while not stop.is_set():
        try:
            val=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            mem,util=map(float,val.split(','))
            import psutil
            samples.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'stage':status.get('active_stage'),'gpu_used_gib':mem/1024,'gpu_util_pct':util,
                            'host_used_gib':psutil.virtual_memory().used/1024**3})
        except Exception:
            pass
        stop.wait(1)

def stage(name,limit):
    status['active_stage']=name
    status['stages'][name]={'state':'running','time_limit_seconds':limit}
    save()
    start=time.perf_counter()
    with (ROOT/(name+'.log')).open('w') as log:
        p=subprocess.Popen([PY,'-u',str(ROOT/'validation_worker.py'),name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=p.wait(timeout=limit)
            result={'state':'finished' if rc==0 else 'failed','exit_code':rc}
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL)
                p.wait()
            result={'state':'time_limit_reached'}
    result['wall_seconds']=time.perf_counter()-start
    stage_samples=[x for x in samples if x['stage']==name]
    if stage_samples:
        result['sampled_device_peak_gib']=max(x['gpu_used_gib'] for x in stage_samples)
        result['sampled_host_peak_used_gib']=max(x['host_used_gib'] for x in stage_samples)
    status['stages'][name]=result
    save()

try:
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    stage('vllm',2400)
    # Short paired generation probes, not another accuracy study.
    if '--vllm-only' not in sys.argv and (ROOT/'validation_100.jsonl').exists():stage('hf',600)
    status['complete']=all(x['state']=='finished' for x in status['stages'].values())
finally:
    stop.set()
    if 'thread' in globals():thread.join(timeout=3)
    status['active_stage']=None
    status['ended_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if (ROOT/'vllm_summary.json').exists():
        status['baseline_100_both_modes_complete']=json.loads((ROOT/'vllm_summary.json').read_text()).get('complete',False)
    save()
    (ROOT/'resource_samples.json').write_text(json.dumps(samples))
    subprocess.run([PY,'-m','pip','freeze'],stdout=(ROOT/'requirements-resolved.txt').open('w'),check=False)
    artifacts=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.jsonl','.log','.txt'}]
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts}
    (ROOT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
    prior=list((ROOT/'attempts').rglob('*')) if (ROOT/'attempts').exists() else []
    with zipfile.ZipFile(ROOT/'hpml_validation_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in artifacts+[ROOT/'evidence_manifest.json']+[p for p in prior if p.is_file()]:z.write(p,p.relative_to(ROOT))
    print('VALIDATION_CONTROLLER_FINISHED',json.dumps(status),flush=True)
