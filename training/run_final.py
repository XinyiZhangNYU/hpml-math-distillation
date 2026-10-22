"""Final stage on one Colab A100 (DESIGN.md §8-§9): training seeds 2 and 3 of groups 3, 4 and 2, the final
evaluation of the 9 merged models and of condition 1, then final_stats.py over all runs.

Seed 1 is the tuning run: its 2e-4 epoch-1 adapters are copied from SEED1_DIR (the runs directory of the tuning) into
adapters/. Seeds 2 and 3 are 1-epoch runs at 2e-4 with their own LoRA seed and question order. Order: the seed-1
models are evaluated first, then condition 1, then each further seed is trained for all three groups and its models
are evaluated. Each model is merged into the base weights, evaluated by final_eval.py with the loss check, and its
merged weights are deleted. A finished job is skipped when this is run again; final_eval.py also skips finished
tiers. A failed job is recorded and only the jobs that need its output are skipped. After every job, everything
except the merged weights, the virtual environments and the training data is copied to DRIVE_OUT.

usage: python -u run_final.py ROOT SEED1_DIR DRIVE_OUT   (ROOT holds work/ with the checked scripts, data/ with the
       training groups and tests/ with the frozen test files)
"""
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT, SEED1_DIR, DRIVE_OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WORK, DATA, TESTS = ROOT / 'work', ROOT / 'data', ROOT / 'tests'
RUNS, ADAPTERS, EVALS, LOGS = ROOT / 'runs', ROOT / 'adapters', ROOT / 'evals', ROOT / 'logs'
GROUPS = ['3', '4', '2']
LR = '2e-4'
SEEDS = {1: (91432026, 20261021), 2: (91432027, 20261022), 3: (91432028, 20261023)}  # LoRA init, question order
TIERS = ['tier1', 'tier2', 'tier3']
TRAIN_PACKAGES = ['unsloth==2026.10.5', 'unsloth_zoo==2026.10.5', 'transformers==5.17.0', 'peft==0.21.1',
                  'torch==2.11.0', 'xformers==0.0.35', 'ninja']
VLLM_PACKAGES = ['vllm==0.21.0', 'transformers==5.17.0', 'tokenizers==0.23.2', 'math-verify==0.9.0', 'ninja']
NOT_SYNCED = ['merged', 'trainvenv', 'vllmvenv', 'data']
STATE = ROOT / 'final_state.json'
state = json.loads(STATE.read_text()) if STATE.exists() else {'jobs': {}}


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


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


def job(name, py, args, timeout, done, out=None):
    """Runs one stage unless done() already holds; a leftover output directory of a failed try is kept aside."""
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


def adapter(g, s):
    return ADAPTERS / f'g{g}_s1' if s == 1 else RUNS / f'train_g{g}_s{s}' / 'epoch1'


def eval_done(out):
    return all((read(out / t / 'summary.json') or {}).get('complete') for t in TIERS)


def copy_seed1():
    for g in GROUPS:
        dst = ADAPTERS / f'g{g}_s1'
        if not (dst / 'check_loss.json').exists():
            src = SEED1_DIR / f'g{g}_lr{LR}'
            report = read(src / 'train_report.json')
            h = report['hyper']
            assert report['state'] == 'finished' and report['group'] == g and float(h['lr']) == float(LR), src
            assert (h['lora_seed'], h['order_seed']) == SEEDS[1], h
            shutil.rmtree(f'{dst}.tmp', ignore_errors=True)
            shutil.copytree(src / 'epoch1', f'{dst}.tmp')
            shutil.copyfile(src / 'train_report.json', Path(f'{dst}.tmp') / 'train_report.json')
            Path(f'{dst}.tmp').rename(dst)
        log('SEED1_ADAPTER', g, {p.name: sha(p)[:16] for p in sorted(dst.iterdir()) if p.is_file()})


def evaluate(g, s, train_py, vllm_py):
    out, merged = EVALS / f'c{g}_s{s}', ROOT / 'merged'
    if eval_done(out):
        log('JOB_SKIP_DONE', out.name)
        return
    if not (adapter(g, s) / 'check_loss.json').exists():
        log('JOB_SKIP_NO_ADAPTER', out.name)
        return
    shutil.rmtree(merged, ignore_errors=True)
    if job(f'merge_g{g}_s{s}', train_py, ['merge_lora.py', str(adapter(g, s)), str(merged)], 1800,
           lambda: (merged / 'merge_report.json').exists()):
        job(f'eval_c{g}_s{s}', vllm_py, ['final_eval.py', '--condition', g, '--train-seed', str(s), '--model', str(merged),
                                         '--tests', str(TESTS), '--check-loss', str(adapter(g, s) / 'check_loss.json'),
                                         '--out', str(out)], 10 * 3600, lambda: eval_done(out))
    shutil.rmtree(merged, ignore_errors=True)


def main():
    for d in (LOGS, RUNS, ADAPTERS, EVALS):
        d.mkdir(parents=True, exist_ok=True)
    log('FINAL_START', f'free disk {shutil.disk_usage(ROOT).free / 1024 ** 3:.0f} GiB')
    copy_seed1()
    train_py = venv('trainvenv', TRAIN_PACKAGES)
    vllm_py = venv('vllmvenv', VLLM_PACKAGES)
    log('VENVS_READY')
    sync()
    for g in GROUPS:
        evaluate(g, 1, train_py, vllm_py)
    c1 = EVALS / 'c1'
    job('eval_c1', vllm_py, ['final_eval.py', '--condition', '1', '--tests', str(TESTS), '--out', str(c1)], 4 * 3600,
        lambda: eval_done(c1))
    for s in (2, 3):
        for g in GROUPS:
            run = RUNS / f'train_g{g}_s{s}'
            job(f'train_g{g}_s{s}', train_py,
                ['train_lora.py', '--data', str(DATA), '--group', g, '--lr', LR, '--epochs', '1', '--seed', str(SEEDS[s][0]),
                 '--order-seed', str(SEEDS[s][1]), '--out', str(run)], 3 * 3600,
                lambda: (read(run / 'train_report.json') or {}).get('state') == 'finished', out=run)
        for g in GROUPS:
            evaluate(g, s, train_py, vllm_py)
    runs = [EVALS / f'c{g}_s{s}' for g in GROUPS for s in SEEDS] + [c1]
    missing = [r.name for r in runs if not eval_done(r)]
    if missing:
        log('FINAL_INCOMPLETE', missing)
    else:
        res = subprocess.run([sys.executable, str(WORK / 'final_stats.py'), str(ROOT / 'final_report.json'), *map(str, runs)],
                             capture_output=True, text=True)
        (ROOT / 'final_report.txt').write_text(res.stdout + res.stderr)
        log('FINAL_STATS', res.returncode, '\n' + res.stdout + res.stderr[-2000:])
    sync()
    log('FINAL_DONE')


if __name__ == '__main__':
    main()
