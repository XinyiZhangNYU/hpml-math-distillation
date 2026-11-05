"""Packing and fused-kernel timing on one Colab A100 (DESIGN.md §11.4): the gates, then arms R, F and P of group 2
and then of group 3 (epoch 1, seed 1, lr 2e-4), then the summary.

Two virtual environments: rvenv is the §8 training stack (TRAIN_PACKAGES, as run_final.py); fvenv adds the fused
linear-attention kernels (flash-linear-attention, causal-conv1d) under the same pins. Arm R is train_lora.py,
unchanged, in rvenv; arms F and P are pack_bench.py in fvenv. Gates: pack_bench.py --mode gate-ref (rvenv), then
--mode gate (fvenv), whose gate.json names the arms allowed; if the gate job fails, only R runs.

Before each arm the run writes ROOT/wait_go.json and waits for ROOT/go/ARM ('go' or 'stop', written by the Colab cell
after the compute unit balance was read; below 8 units the answer is 'stop'). On 'stop' this and every later arm is
recorded as not run. A finished job is skipped when this is run again. Results are copied to DRIVE_OUT every 10
minutes and after every job (not the virtual environments, the data or the adapters that train_lora.py saves).

usage: python -u run_pack.py ROOT SEED1_DIR DRIVE_OUT   (ROOT holds work/ with the checked scripts and data/ with
       the training groups; SEED1_DIR is the runs directory of the tuning, for the seed-1 timing record)
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

GROUPS = ['2', '3']
ARMS = ['R', 'F', 'P']
LR = '2e-4'
TRAIN_PACKAGES = ['unsloth==2026.10.5', 'unsloth_zoo==2026.10.5', 'transformers==5.17.0', 'peft==0.21.1',
                  'torch==2.11.0', 'xformers==0.0.35', 'ninja']
FUSED_PACKAGES = ['flash-linear-attention', 'causal-conv1d']  # versions recorded in fvenv_pip_freeze.txt
NOT_SYNCED = ['rvenv', 'fvenv', 'data', 'epoch1']
if __name__ == '__main__':
    ROOT, SEED1_DIR, DRIVE_OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    WORK, DATA, RUNS, GATES, LOGS, GO = (ROOT / n for n in ('work', 'data', 'runs', 'gates', 'logs', 'go'))
    STATE = ROOT / 'pack_state.json'
    state = json.loads(STATE.read_text()) if STATE.exists() else {'jobs': {}, 'not_run': {}}
LOCK = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def log(*parts):
    print(now()[:19], *parts, flush=True)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def sync():
    """Copy results to Drive; a failed copy is logged and the run goes on."""
    with LOCK:
        try:
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(['rsync', '-rt', *[f'--exclude={x}' for x in NOT_SYNCED], f'{ROOT}/', f'{DRIVE_OUT}/'],
                                 capture_output=True, text=True, timeout=1800)
            if res.returncode:
                log('SYNC_FAILED', res.returncode, res.stderr[-500:])
        except Exception as e:  # noqa: BLE001
            log('SYNC_FAILED', repr(e))


def sync_loop():
    while True:
        time.sleep(600)
        sync()


def venv(name, packages, extra=()):
    env, freeze = ROOT / name, ROOT / f'{name}_pip_freeze.txt'
    py = str(env / 'bin/python')
    if freeze.exists() and Path(py).exists():
        return py
    shutil.rmtree(env, ignore_errors=True)
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(env)], check=True)
    pins = ROOT / 'pins.txt'
    pins.write_text('\n'.join(p for p in TRAIN_PACKAGES if '==' in p) + '\n')
    t = time.perf_counter()
    with (LOGS / f'{name}_install.log').open('w') as out:
        subprocess.run([py, '-m', 'pip', 'install', *packages], stdout=out, stderr=subprocess.STDOUT, check=True)
        for p in extra:  # one at a time, the §8 pins held; causal-conv1d builds against the installed torch
            subprocess.run([py, '-m', 'pip', 'install', '--no-build-isolation', '-c', str(pins), p], stdout=out,
                           stderr=subprocess.STDOUT, check=True, env={**os.environ, 'MAX_JOBS': '8'})
    freeze.write_text(subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    log('VENV_READY', name, f'{time.perf_counter() - t:.0f}s')
    return py


def job(name, py, args, timeout, done, out=None):
    """Runs one stage unless done() already holds; a leftover output of a failed try is kept aside."""
    if done():
        log('JOB_SKIP_DONE', name)
        return True
    tries = state['jobs'].get(name, {}).get('tries', 0) + 1
    if out is not None and Path(out).exists():
        Path(out).rename(f'{out}.failed{tries - 1}')
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


def wait_go(name):
    go = GO / name
    if not go.exists():
        save(ROOT / 'wait_go.json', {'arm': name, 'since_utc': now()})
        log('WAIT_GO', name)
        while not go.exists():
            time.sleep(10)
    (ROOT / 'wait_go.json').unlink(missing_ok=True)
    answer = go.read_text().strip()
    log('GATE_ANSWER', name, answer)
    assert answer in ('go', 'stop'), answer
    return answer == 'go'


def finished(run):
    return (read(run / 'train_report.json') or {}).get('state') == 'finished'


def arm_args(g, arm, run):
    if arm == 'R':
        return ['train_lora.py', '--data', str(DATA), '--group', g, '--lr', LR, '--epochs', '1', '--out', str(run)]
    return ['pack_bench.py', '--mode', arm, '--data', str(DATA), '--group', g, '--out', str(run)]


def not_run(name, reason):
    state['not_run'][name] = reason
    save(STATE, state)
    log('NOT_RUN', name, reason)


def main():
    for d in (LOGS, RUNS, GATES, GO):
        d.mkdir(parents=True, exist_ok=True)
    log('PACK_START', f'free disk {shutil.disk_usage(ROOT).free / 1024 ** 3:.0f} GiB')
    threading.Thread(target=sync_loop, daemon=True).start()
    r_py = venv('rvenv', TRAIN_PACKAGES)
    f_py = venv('fvenv', TRAIN_PACKAGES, FUSED_PACKAGES)
    sync()
    ref, gate = GATES / 'ref.json', GATES / 'gate.json'
    job('gate_ref', r_py, ['pack_bench.py', '--mode', 'gate-ref', '--data', str(DATA), '--out', str(ref)], 2 * 3600,
        ref.exists)
    if ref.exists():
        job('gate', f_py, ['pack_bench.py', '--mode', 'gate', '--data', str(DATA), '--ref', str(ref), '--out', str(gate)],
            2 * 3600, gate.exists)
    allowed = (read(gate) or {}).get('arms_allowed', ['R'])
    log('ARMS_ALLOWED', allowed, json.dumps((read(gate) or {}).get('gates')))
    stopped = False
    for g in GROUPS:
        for arm in ARMS:
            name, run = f'g{g}_{arm}', RUNS / f'g{g}_{arm}'
            if finished(run):
                log('JOB_SKIP_DONE', name)
                continue
            if arm not in allowed:
                not_run(name, 'gate')
                continue
            if stopped or not wait_go(name):
                stopped = True
                not_run(name, 'balance')
                continue
            job(name, r_py if arm == 'R' else f_py, arm_args(g, arm, run), 3 * 3600, lambda: finished(run), out=run)
    res = subprocess.run([sys.executable, str(WORK / 'pack_bench.py'), '--mode', 'summary', '--runs', str(RUNS),
                          '--gates', str(GATES), '--record', str(SEED1_DIR), '--out', str(ROOT / 'pack_summary.json')],
                         capture_output=True, text=True, cwd=WORK)
    (ROOT / 'pack_summary.txt').write_text(res.stdout + res.stderr)
    log('SUMMARY', res.returncode, '\n' + res.stdout + res.stderr[-2000:])
    sync()
    log('PACK_DONE')


if __name__ == '__main__':
    main()
