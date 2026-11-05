"""Scheduling extension on one Colab A100 (DESIGN.md §11.3); no training and no API calls.

Jobs, in this order: merge of the condition 3 / seed 1 adapter; sched_ext.py at 64 slots (cell 1), 32 slots (cell 2)
and 128 slots (cell 3); the merged condition 3 weights are deleted; merge of the condition 4 / seed 1 adapter;
sched_ext.py at 64 slots (cells 4 and 5). Each cell waits for the hand-read balance decision (see sched_ext.py); after
a 'stop' every later cell is recorded as not run and nothing else starts. A finished job is skipped when this is run
again (sched_ext.py also skips finished runs). Everything except the merged weights, the adapters, the virtual
environment and the repository checkout is copied to DRIVE_OUT every 10 minutes and after every job.

usage: python -u run_ext.py ROOT FINAL_DIR DRIVE_OUT   (ROOT holds work/ with the checked scripts; FINAL_DIR is the
       Drive folder of the final run with adapters/g3_s1 and adapters/g4_s1)
"""
import datetime
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT, FINAL_DIR, DRIVE_OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WORK, LOGS, EXT = ROOT / 'work', ROOT / 'logs', ROOT / 'ext'
VLLM_PACKAGES = ['vllm==0.21.0', 'transformers==5.17.0', 'tokenizers==0.23.2', 'math-verify==0.9.0', 'ninja']
NOT_SYNCED = ['merged_*', 'adapter_*', 'vllmvenv', 'repo']
STATE = ROOT / 'ext_state.json'
# (job, adapter, slots, [(cell, samples per problem, run order)]), DESIGN.md §11.3
PLAN = [('ext_g3_slots64', 'g3_s1', 64, [(1, 6, 'fixed,queue')]),
        ('ext_g3_slots32', 'g3_s1', 32, [(2, 2, 'queue,fixed')]),
        ('ext_g3_slots128', 'g3_s1', 128, [(3, 2, 'fixed,queue')]),
        ('ext_g4_slots64', 'g4_s1', 64, [(4, 2, 'queue,fixed'), (5, 6, 'fixed,queue')])]
state = json.loads(STATE.read_text()) if STATE.exists() else {'jobs': {}}
lock = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def log(*parts):
    print(now()[:19], *parts, flush=True)


def sync():
    """Copy results to Drive; a failed copy is logged and the run goes on."""
    with lock:
        try:
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(['rsync', '-rt', *[f'--exclude=/{x}' for x in NOT_SYNCED], f'{ROOT}/', f'{DRIVE_OUT}/'],
                                 capture_output=True, text=True, timeout=1800)
            if res.returncode:
                log('SYNC_FAILED', res.returncode, res.stderr[-500:])
        except Exception as e:  # noqa: BLE001
            log('SYNC_FAILED', repr(e))


def sync_loop():
    while True:
        time.sleep(600)
        sync()


def venv(name, packages):
    env, freeze = ROOT / name, ROOT / f'{name}_pip_freeze.txt'
    py = str(env / 'bin/python')
    if freeze.exists() and Path(py).exists():
        return py
    shutil.rmtree(env, ignore_errors=True)
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(env)], check=True)
    with (LOGS / f'{name}_install.log').open('w') as out:
        subprocess.run([py, '-m', 'pip', 'install', *packages], stdout=out, stderr=subprocess.STDOUT, check=True)
    freeze.write_text(subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    return py


def job(name, py, args, timeout, done):
    """Runs one stage unless done() already holds."""
    if done():
        log('JOB_SKIP_DONE', name)
        return True
    tries = state['jobs'].get(name, {}).get('tries', 0) + 1
    path = LOGS / f'{name}.try{tries}.log'
    save(ROOT / 'current_job.json', {'job': name, 'log': str(path), 'started_at_utc': now()})
    log('JOB_START', name)
    t = time.perf_counter()
    with path.open('w') as out_log:
        try:
            code = subprocess.run([py, '-u', *args], cwd=WORK, stdout=out_log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = f'timeout_{timeout}s'
    ok = code == 0 and done()
    state['jobs'][name] = {'tries': tries, 'exit': code, 'ok': ok, 'seconds': time.perf_counter() - t,
                           'ended_at_utc': now(), 'log': path.name}
    save(STATE, state)
    log('JOB_END', name, json.dumps(state['jobs'][name]))
    sync()
    return ok


def not_run():
    p = EXT / 'not_run.json'
    return set(json.loads(p.read_text())['cells']) if p.exists() else set()


def cells_done(cells):
    return all((EXT / f'cell{c}' / 'cell_summary.json').exists() or c in not_run() for c, _, _ in cells)


def merged(adapter, py):
    """Copies the adapter from Drive and merges it; any other merged weights are deleted first (disk)."""
    a, m = ROOT / f'adapter_{adapter}', ROOT / f'merged_{adapter}'
    for other in ROOT.glob('merged_*'):
        if other != m:
            shutil.rmtree(other, ignore_errors=True)
    if not (a / 'check_loss.json').exists():
        shutil.rmtree(a, ignore_errors=True)
        shutil.copytree(FINAL_DIR / 'adapters' / adapter, f'{a}.tmp')
        Path(f'{a}.tmp').rename(a)
    if not (m / 'merge_report.json').exists():
        shutil.rmtree(m, ignore_errors=True)
    ok = job(f'merge_{adapter}', py, ['merge_lora.py', str(a), str(m)], 1800, lambda: (m / 'merge_report.json').exists())
    return (a, m) if ok else None


def main():
    for d in (LOGS, EXT):
        d.mkdir(parents=True, exist_ok=True)
    log('EXT_START', f'free disk {shutil.disk_usage(ROOT).free / 1024 ** 3:.0f} GiB')
    threading.Thread(target=sync_loop, daemon=True).start()
    py = venv('vllmvenv', VLLM_PACKAGES)
    log('VENV_READY')
    for name, adapter, slots, cells in PLAN:
        if not_run():
            later = sorted({c for _, _, _, cs in PLAN for c, _, _ in cs} - {c for c in range(1, 6)
                                                                            if (EXT / f'cell{c}' / 'cell_summary.json').exists()})
            save(EXT / 'not_run.json', {'cells': later, 'utc': now()})
            log('STOPPED_BY_GATE', later)
            break
        if cells_done(cells):
            log('JOB_SKIP_DONE', name)
            continue
        am = merged(adapter, py)
        if am is None:
            log('MERGE_FAILED', adapter)
            break
        args = ['sched_ext.py', '--model', str(am[1]), '--check-loss', str(am[0] / 'check_loss.json'),
                '--out', str(EXT), '--slots', str(slots)]
        for c, k, modes in cells:
            args += ['--cell', str(c), str(k), modes]
        if not job(name, py, args, 12 * 3600, lambda: cells_done(cells)):
            log('JOB_FAILED', name)
            break
    if all((EXT / f'cell{c}' / 'cell_summary.json').exists() or c in not_run() for c in range(1, 6)):
        for m in ROOT.glob('merged_*'):
            shutil.rmtree(m, ignore_errors=True)
    sync()
    log('EXT_DONE', json.dumps({k: v['ok'] for k, v in state['jobs'].items()}), 'not run', sorted(not_run()))


if __name__ == '__main__':
    main()
