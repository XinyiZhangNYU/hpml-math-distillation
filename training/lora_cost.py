"""LoRA training and evaluation cost of the final run (DESIGN.md §11), from the records it saved; nothing is rerun.

Per group and seed: epoch-1 training time (sum of step times, and the epoch wall time with saving and the loss check),
real tokens and real tokens per second, peak memory (PyTorch allocated, and whole-device used sampled every 0.2 s),
host peak RSS and model load time, from train_report.json (seed 1: the tuning run, whose epoch-1 adapter is the 1-epoch
model, so only its epoch-1 rows count). Merge and evaluation job times come from final_state.json, and generation time
and throughput per tier from the evaluation summaries.

usage: python -I lora_cost.py FINAL_DIR OUT   (FINAL_DIR = the synced final run: adapters/, runs/, evals/, final_state.json)
"""
import json
import sys
from pathlib import Path


def main(final_dir, out):
    d, out = Path(final_dir), Path(out)
    state = json.loads((d / 'final_state.json').read_text())['jobs']
    res = {'training': {}, 'evaluation': {}}
    for g in ('2', '3', '4'):
        for s in (1, 2, 3):
            p = d / 'adapters' / f'g{g}_s1' / 'train_report.json' if s == 1 else d / 'runs' / f'train_g{g}_s{s}' / 'train_report.json'
            r = json.loads(p.read_text())
            ep = r['epochs_done'][0]
            steps = [x for x in r['steps'] if x['epoch'] == 1]
            assert r['state'] == 'finished' and ep['tag'] == 'epoch1' and len(steps) == ep['steps'], (p, ep['tag'], len(steps))
            res['training'][f'g{g}_s{s}'] = {
                'gpu': r.get('gpu'), 'load_seconds': r.get('load_seconds'), 'steps': ep['steps'],
                'train_seconds': ep['train_seconds'], 'epoch_wall_seconds': ep['seconds'], 'real_tokens': ep['real_tokens'],
                'real_tokens_per_second': ep['real_tokens'] / ep['train_seconds'],
                'padded_tokens': sum(x['padded_tokens'] for x in steps),
                'peak_allocated_gib': max(x['peak_allocated_gib'] for x in steps),
                'device_used_peak_gib_sampled': max(x['device_used_peak_gib_sampled'] for x in steps),
                'host_peak_rss_gib': max(x['host_peak_rss_gib'] for x in steps),
                'trainable_parameters': r.get('trainable_parameters'),
                'job_seconds': state.get(f'train_g{g}_s{s}', {}).get('seconds'),
                'merge_job_seconds': state.get(f'merge_g{g}_s{s}', {}).get('seconds')}
    for name in ['c1'] + [f'c{g}_s{s}' for g in '342' for s in (1, 2, 3)]:
        tiers = {}
        for t in ('tier1', 'tier2', 'tier3'):
            sm = json.loads((d / 'evals' / name / t / 'summary.json').read_text())
            tiers[t] = {k: sm.get(k) for k in ('generation_wall_seconds', 'output_tokens', 'aggregate_output_tokens_per_second',
                                               'mean_output_tokens', 'truncated', 'load_seconds')}
        res['evaluation'][name] = {'job_seconds': state.get(f'eval_{name}', {}).get('seconds'), 'tiers': tiers}
    by_group = {}
    for g in '234':
        rows = [res['training'][f'g{g}_s{s}'] for s in (1, 2, 3)]
        by_group[g] = {k: sum(r[k] for r in rows) / 3 for k in ('train_seconds', 'real_tokens', 'real_tokens_per_second',
                                                               'peak_allocated_gib', 'device_used_peak_gib_sampled')}
    res['training_mean_by_group'] = by_group
    out.mkdir(parents=True, exist_ok=True)
    (out / 'lora_cost.json').write_text(json.dumps(res, indent=1))
    print('LORA_COST', json.dumps(by_group), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
