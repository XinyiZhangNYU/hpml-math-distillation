"""Systems stage on one Colab A100 (DESIGN.md §11-§12), after the final run; no training and no API calls.

Jobs, in this order: lora_cost.py over the records of the final run (CPU); early_stop_estimate.py over its saved
outputs (CPU, exploratory, §12); the merge of the condition 3 / seed 1 adapter; sched_bench.py on the merged model
(GPU). The two CPU jobs run before the benchmark so that nothing else uses the machine while it is timed. A finished
job is skipped when this is run again (sched_bench.py also skips finished runs); a failed job is recorded and the next
job still runs, except that the benchmark needs the merge. After every job, everything except the merged weights, the
virtual environment and the repository checkout is copied to DRIVE_OUT.

usage: python -u run_sched.py ROOT FINAL_DIR DRIVE_OUT   (ROOT holds work/ with the checked scripts; FINAL_DIR is the
       Drive folder of the final run: adapters/, runs/, evals/, final_state.json)
"""
import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT, FINAL_DIR, DRIVE_OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WORK, LOGS, ANALYSIS, BENCH = ROOT / 'work', ROOT / 'logs', ROOT / 'analysis', ROOT / 'bench'
ADAPTER, MERGED = ROOT / 'adapter_g3_s1', ROOT / 'merged'
VLLM_PACKAGES = ['vllm==0.21.0', 'transformers==5.17.0', 'tokenizers==0.23.2', 'math-verify==0.9.0', 'ninja']
NOT_SYNCED = ['merged', 'vllmvenv', 'repo']
STATE = ROOT / 'sched_state.json'
state = json.loads(STATE.read_text()) if STATE.exists() else {'jobs': {}}


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
    try:
        DRIVE_OUT.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(['rsync', '-rt', *[f'--exclude=/{x}' for x in NOT_SYNCED], f'{ROOT}/', f'{DRIVE_OUT}/'],
                             capture_output=True, text=True, timeout=1800)
        if res.returncode:
            log('SYNC_FAILED', res.returncode, res.stderr[-500:])
    except Exception as e:  # noqa: BLE001
        log('SYNC_FAILED', repr(e))


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


def bench_done():
    return (BENCH / 'bench_summary.json').exists()


def main():
    for d in (LOGS, ANALYSIS, BENCH):
        d.mkdir(parents=True, exist_ok=True)
    log('SCHED_START', f'free disk {shutil.disk_usage(ROOT).free / 1024 ** 3:.0f} GiB')
    job('lora_cost', sys.executable, ['lora_cost.py', str(FINAL_DIR), str(ANALYSIS)], 900,
        lambda: (ANALYSIS / 'lora_cost.json').exists())
    vllm_py = venv('vllmvenv', VLLM_PACKAGES)
    log('VENV_READY')
    sync()
    job('early_stop', vllm_py, ['early_stop_estimate.py', str(FINAL_DIR / 'evals'), str(ANALYSIS)], 2 * 3600,
        lambda: (ANALYSIS / 'early_stop_estimate.json').exists())
    if not bench_done():
        if not (ADAPTER / 'check_loss.json').exists():
            shutil.rmtree(ADAPTER, ignore_errors=True)
            shutil.copytree(FINAL_DIR / 'adapters' / 'g3_s1', f'{ADAPTER}.tmp')
            Path(f'{ADAPTER}.tmp').rename(ADAPTER)
        if not (MERGED / 'merge_report.json').exists():
            shutil.rmtree(MERGED, ignore_errors=True)
        if job('merge_g3_s1', vllm_py, ['merge_lora.py', str(ADAPTER), str(MERGED)], 1800,
               lambda: (MERGED / 'merge_report.json').exists()):
            job('bench', vllm_py, ['sched_bench.py', '--model', str(MERGED), '--check-loss', str(ADAPTER / 'check_loss.json'),
                                   '--out', str(BENCH)], 8 * 3600, bench_done)
    if bench_done():  # kept after a failed benchmark, so that a rerun on this runtime skips the merge
        shutil.rmtree(MERGED, ignore_errors=True)
    sync()
    log('SCHED_DONE', json.dumps({k: v['ok'] for k, v in state['jobs'].items()}))


if __name__ == '__main__':
    main()
