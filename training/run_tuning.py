"""Stage 2 tuning on one Colab A100: the learning-rate x epoch grid for training groups 3, 4 and 2, and the dev
evaluations that pick each group's setting. colab_tuning_cell.py checks every file and starts this in the background.

Order: for each group (3, 4, 2) and learning rate (1e-4, 2e-4): train 3 epochs (constant learning rate after warmup,
so the epoch-k adapter is what a k-epoch run gives), then for each epoch merge the adapter into the base weights,
evaluate the merged weights on MATH dev 100 (4 samples, 16k tokens, with the loss check) and delete the merged
weights. Last, condition 1 (base model, Minerva 4-shot) on dev as a reference. A finished job is skipped when this is
run again; a failed job is recorded and only the jobs that need its output are skipped. A training run that failed
is trained again from the start on the next run, and its epochs are evaluated again from the new adapters.

Selection (fixed before any dev result): in each group, the highest mean dev accuracy over the 6 settings; settings
within 1 percentage point of it go to fewer epochs, then the lower learning rate. tuning_summary.json is rewritten
after every evaluation.

usage: python -u run_tuning.py ROOT   (ROOT holds work/ with the checked scripts and data/ with the training groups)
"""
import datetime
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(sys.argv[1])
WORK, DATA, RUNS, LOGS = ROOT / 'work', ROOT / 'data', ROOT / 'runs', ROOT / 'logs'
GROUPS = ['3', '4', '2']
LRS = ['1e-4', '2e-4']
EPOCHS = 3
DEV = ['--samples', '4', '--max-tokens', '16384']
TRAIN_PACKAGES = ['unsloth==2026.10.5', 'unsloth_zoo==2026.10.5', 'transformers==5.17.0', 'peft==0.21.1',
                  'torch==2.11.0', 'xformers==0.0.35', 'ninja']
VLLM_PACKAGES = ['vllm==0.21.0', 'transformers==5.17.0', 'tokenizers==0.23.2', 'math-verify==0.9.0', 'ninja']
STATE = ROOT / 'tuning_state.json'
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


def venv(name, packages):
    env, freeze = ROOT / name, ROOT / f'{name}_pip_freeze.txt'
    py = str(env / 'bin/python')
    if freeze.exists():
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
    return ok


def dev_row(tag, path):
    s = read(path / 'dev_summary.json')
    if not (s and s.get('complete')):
        return None
    keep = ('accuracy', 'per_sample_accuracy', 'truncated', 'think_closed', 'correct_with_unclosed_think',
            'mean_output_tokens', 'median_output_tokens', 'prefix_accuracy', 'by_level', 'generation_wall_seconds',
            'load_seconds')
    row = {'tag': tag, **{k: s[k] for k in keep}}
    if s.get('check_loss'):
        row['check_loss'] = {k: s['check_loss'][k] for k in ('max_relative_difference', 'pass')}
    return row


def summarize():
    table, selection = [], {}
    for g in GROUPS:
        rows = []
        for lr in LRS:
            for e in range(1, EPOCHS + 1):
                row = dev_row(f'g{g}_lr{lr}_epoch{e}', RUNS / f'dev_g{g}_lr{lr}_epoch{e}')
                if row:
                    rows.append({'group': g, 'lr': lr, 'epochs': e, **row})
        table += rows
        if len(rows) < len(LRS) * EPOCHS:
            selection[g] = {'status': f'{len(rows)} of {len(LRS) * EPOCHS} settings evaluated'}
            continue
        best = max(r['accuracy'] for r in rows)
        near = [r for r in rows if r['accuracy'] >= best - 0.01 - 1e-9]
        pick = min(near, key=lambda r: (r['epochs'], float(r['lr'])))
        selection[g] = {'status': 'selected', 'lr': pick['lr'], 'epochs': pick['epochs'], 'accuracy': pick['accuracy'],
                        'best_accuracy': best, 'within_1pp': [r['tag'] for r in near],
                        'all_check_loss_pass': all(r.get('check_loss', {}).get('pass') for r in rows)}
    save(ROOT / 'tuning_summary.json', {
        'updated_at_utc': now(), 'rule': 'per group: highest mean dev accuracy; within 1 pp, fewer epochs, then lower lr',
        'selection': selection, 'dev': table,
        'condition1_reference': dev_row('cond1_base_minerva', RUNS / 'dev_cond1_base_minerva')})
    log('SUMMARY', json.dumps({g: selection[g] for g in GROUPS}))


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    log('TUNING_START', f'free disk {free:.0f} GiB')
    train_py = venv('trainvenv', TRAIN_PACKAGES)
    vllm_py = venv('vllmvenv', VLLM_PACKAGES)
    log('VENVS_READY')
    for g in GROUPS:
        for lr in LRS:
            run = RUNS / f'g{g}_lr{lr}'
            trained = lambda: (read(run / 'train_report.json') or {}).get('state') == 'finished'
            if run.exists() and not trained():  # a retrain replaces the adapters, so their earlier evaluations no longer apply
                for e in range(1, EPOCHS + 1):
                    dev = RUNS / f'dev_g{g}_lr{lr}_epoch{e}'
                    if dev.exists():
                        dev.rename(f'{dev}.stale_{now()[:19].replace(":", "")}')
            job(f'train_g{g}_lr{lr}', train_py,
                ['train_lora.py', '--data', str(DATA), '--group', g, '--lr', lr, '--epochs', str(EPOCHS), '--out', str(run)],
                4 * 3600, trained, out=run)
            for e in range(1, EPOCHS + 1):
                adapter, merged = run / f'epoch{e}', ROOT / 'merged'
                dev = RUNS / f'dev_g{g}_lr{lr}_epoch{e}'
                if dev_row('', dev):
                    log('JOB_SKIP_DONE', dev.name)
                    continue
                if not (adapter / 'check_loss.json').exists():
                    log('JOB_SKIP_NO_ADAPTER', dev.name)
                    continue
                shutil.rmtree(merged, ignore_errors=True)
                if job(f'merge_g{g}_lr{lr}_epoch{e}', train_py, ['merge_lora.py', str(adapter), str(merged)], 1800,
                       lambda: (merged / 'merge_report.json').exists()):
                    job(dev.name, vllm_py, ['dev_eval.py', '--model', str(merged), '--tag', f'g{g}_lr{lr}_epoch{e}',
                                            '--prompt', 'chat', *DEV, '--check-loss', str(adapter / 'check_loss.json'),
                                            '--out', str(dev)], 2 * 3600, lambda: bool(dev_row('', dev)), out=dev)
                shutil.rmtree(merged, ignore_errors=True)
                summarize()
    cond1 = RUNS / 'dev_cond1_base_minerva'
    job(cond1.name, vllm_py, ['dev_eval.py', '--model', 'base', '--tag', 'cond1_base_minerva', '--prompt', 'minerva', *DEV,
                              '--out', str(cond1)], 2 * 3600, lambda: bool(dev_row('', cond1)), out=cond1)
    summarize()
    keep = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.log')]
    keep += [p for p in (LOGS.iterdir())] + [p for p in RUNS.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl')]
    with zipfile.ZipFile(ROOT / 'hpml_stage2_tuning_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in keep:
            z.write(p, str(p.relative_to(ROOT)))
    log('TUNING_DONE', 'evidence zip and adapters are in', ROOT)


if __name__ == '__main__':
    main()
