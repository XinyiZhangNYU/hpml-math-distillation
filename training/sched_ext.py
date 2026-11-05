"""Scheduling extension (DESIGN.md §11.3): one engine, one or more cells of one fixed and one queue run each.

Reuses the step loop, statistics and comparison of sched_bench.py (§11) unchanged; only the workload size (MATH dev
100 x k samples in the §11 order and seed rule), the slot count (max_num_seqs = fixed group size) and the runs vary.
Budget 32,768 for every run.

Before each cell starts, the Colab compute unit balance is read by hand (it cannot be read from the runtime): the
script writes OUT/wait_go.json and waits for OUT/go/cellN, which holds 'go' or 'stop' (written by the Colab cell after
the balance was read; below 12 units the answer is 'stop'). On 'stop' this and every later cell is recorded as not
run in OUT/not_run.json. The gate of an engine's first cell comes before the engine loads.

Per cell, OUT/cellN/runs/NAME.json and NAME.outputs.jsonl.gz as in sched_bench.py, and OUT/cellN/cell_summary.json.

usage: python -u sched_ext.py --model MERGED_DIR --check-loss ADAPTER_DIR/check_loss.json --out DIR --slots S
       --cell N K MODES [--cell ...]   (K: samples per problem; MODES: e.g. fixed,queue in run order)
"""
import argparse
import gzip
import json
import time
import traceback
from pathlib import Path

import sched_bench as sb  # noqa: E402  (step loop, statistics, comparison and frozen-file check of §11)
from sched_bench import dev_eval, final_eval, sft_data  # noqa: E402,F401

BUDGET = 32768


def gate(out, cell):
    """Waits for the hand-read balance decision of `cell`; True to run it."""
    go = out / 'go' / f'cell{cell}'
    if not go.exists():
        dev_eval.save(out / 'wait_go.json', {'cell': cell, 'since_utc': sb.now_utc()})
        print('WAIT_GO', cell, flush=True)
        while not go.exists():
            time.sleep(10)
    answer = go.read_text().strip()
    (out / 'wait_go.json').unlink(missing_ok=True)
    print('GATE', cell, answer, flush=True)
    assert answer in ('go', 'stop'), answer
    return answer == 'go'


def stop_here(out, cells):
    p = out / 'not_run.json'
    prev = json.loads(p.read_text()) if p.exists() else {'cells': []}
    prev['cells'] = sorted(set(prev['cells']) | {c for c, _, _ in cells})
    prev['utc'] = sb.now_utc()
    dev_eval.save(p, prev)
    print('NOT_RUN', prev['cells'], flush=True)


def run(a):
    import importlib.metadata as metadata
    import torch
    from vllm import LLM, SamplingParams
    out = Path(a.out)
    cells = [(int(c), int(k), modes.split(',')) for c, k, modes in a.cell]
    assert all(set(m) == {'fixed', 'queue'} and len(m) == 2 for _, _, m in cells), cells
    todo = [c for c in cells if not (out / f'cell{c[0]}' / 'cell_summary.json').exists()]
    if not todo:
        print('CELLS_DONE_ALREADY', [c for c, _, _ in cells], flush=True)
        return
    if not gate(out, todo[0][0]):
        stop_here(out, todo)
        return
    shas = {n: sb.sha(sb.HERE / n) for n in sb.FROZEN}
    assert all(shas[n].startswith(sb.FROZEN[n]) for n in sb.FROZEN), shas
    sb.GROUP = a.slots  # run_stats() reads the group size and slot count from here
    rows = [json.loads(s) for s in (sb.HERE / 'math_dev_100.jsonl').read_text().splitlines()]
    assert len(rows) == 100
    tok = sft_data.load_tokenizer()
    stop = dict(stop_token_ids=[tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')])
    meta_p = out / f'engine_slots{a.slots}_{Path(a.model).name}.json'
    meta = {'utc_start': sb.now_utc(), 'model': a.model, 'slots': a.slots, 'max_model_len': sb.MAX_MODEL_LEN,
            'engine': dev_eval.ENGINE, 'sampling': dev_eval.SAMPLING, 'seed_rule': '91432026 + item index + 100000*sample',
            'stop': stop, 'cells': cells, 'frozen_sha256': shas, 'script_sha256': sb.sha(Path(__file__)),
            'sched_bench_sha256': sb.sha(sb.HERE / 'sched_bench.py'), 'gpu': torch.cuda.get_device_name(0),
            'versions': {n: metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']}}
    merge = Path(a.model) / 'merge_report.json'
    meta['merge_report'] = json.loads(merge.read_text()) if merge.exists() else None
    t = time.perf_counter()
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=sb.MAX_MODEL_LEN, max_num_seqs=a.slots, **dev_eval.ENGINE,
              enforce_eager=False, enable_prefix_caching=False, language_model_only=True, seed=sb.SEED,
              disable_log_stats=False)
    meta['load_seconds'] = time.perf_counter() - t
    engine = llm.llm_engine
    try:
        cc = engine.vllm_config.cache_config
        meta['kv_cache'] = {'num_gpu_blocks': cc.num_gpu_blocks, 'block_size': cc.block_size}
    except Exception as e:  # noqa: BLE001  (informational only)
        meta['kv_cache'] = repr(e)
    meta['check_loss'] = dev_eval.check_nll(llm, json.loads(Path(a.check_loss).read_text()))
    dev_eval.save(meta_p, meta)
    print('CHECK_LOSS', json.dumps({k: meta['check_loss'][k] for k in ('max_relative_difference', 'pass')}), flush=True)
    assert meta['check_loss']['pass'], 'The merged weights do not reproduce the adapter loss.'
    prompts0 = [sft_data.chat_prompt(rows[i]['problem']) for i in range(8)]
    ptok0 = [len(tok.encode(p, add_special_tokens=False)) for p in prompts0]
    wp = [SamplingParams(**dev_eval.SAMPLING, max_tokens=256, seed=sb.SEED + i, **stop) for i in range(8)]
    w = sb.drive(engine, prompts0, wp, ptok0, 0, 'warmup')
    g = llm.generate(prompts0, [p.clone() for p in wp], use_tqdm=False)
    meta['warmup'] = {'step_loop_tokens': w['tokens'], 'generate_tokens': [len(x.outputs[0].token_ids) for x in g],
                      'identical_text': sum(x == y.outputs[0].text for x, y in zip(w['text'], g)),
                      'step_loop_seconds': w['wall']}
    dev_eval.save(meta_p, meta)
    print('WARMUP', json.dumps(meta['warmup']), flush=True)
    assert all(e is not None for e in w['end']) and all(0 < x <= 256 for x in w['tokens']), 'step loop check failed'
    gpu = sb.GpuMonitor()
    gpu.start()
    for n_cell, (cell, k, modes) in enumerate(todo):
        if n_cell and not gate(out, cell):
            stop_here(out, todo[n_cell:])
            break
        cdir = out / f'cell{cell}'
        (cdir / 'runs').mkdir(parents=True, exist_ok=True)
        items = [(i, s) for s in range(k) for i in range(len(rows))]
        prompts = [sft_data.chat_prompt(rows[i]['problem']) for i, _ in items]
        ptok = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
        names = [f'cell{cell}_{m}' for m in modes]
        for name, mode in zip(names, modes):
            p = cdir / 'runs' / f'{name}.json'
            if p.exists() and json.loads(p.read_text()).get('complete'):
                print('RUN_SKIP_DONE', name, flush=True)
                continue
            params = [SamplingParams(**dev_eval.SAMPLING, max_tokens=BUDGET, seed=sb.SEED + i + 100000 * s, **stop)
                      for i, s in items]
            print('RUN_START', name, sb.now_utc(), flush=True)
            utc, c0 = sb.now_utc(), sb.engine_counters(llm)
            res = sb.drive(engine, prompts, params, ptok, a.slots if mode == 'fixed' else 0, name)
            c1 = sb.engine_counters(llm)
            stats, scores = sb.run_stats(res, gpu.between(res['start'], res['start'] + res['wall']), items, rows,
                                         BUDGET, mode)
            stats.update({'name': name, 'cell': cell, 'slots': a.slots, 'samples': k, 'model': a.model,
                          'utc_start': utc, 'utc_end': sb.now_utc(),
                          'engine_counters': {x: c1[x] - c0[x] if isinstance(c1[x], (int, float)) and x in c0 else c1[x]
                                              for x in c1},
                          'trace_columns': ['seconds', 'submitted_unfinished', 'running', 'resident_tokens'],
                          'trace': res['trace'],
                          'gpu_trace_mib': [[round(t - res['start'], 1), m] for t, m in gpu.samples
                                            if res['start'] <= t <= res['start'] + res['wall']]})
            recs = [{'k': j, 'id': rows[i]['id'], 'sample': s, 'seed': sb.SEED + i + 100000 * s,
                     'prompt_tokens': ptok[j], 'output_tokens': res['tokens'][j], 'finish_reason': res['reason'][j],
                     'first_token_s': res['first'][j], 'finish_s': res['end'][j], 'correct': scores[j]['correct'],
                     'think_closed': scores[j]['think_closed'], 'text': res['text'][j]} for j, (i, s) in enumerate(items)]
            with gzip.open(cdir / 'runs' / f'{name}.outputs.jsonl.gz', 'wt', encoding='utf-8') as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in recs)
            stats['outputs_sha256'] = sb.sha(cdir / 'runs' / f'{name}.outputs.jsonl.gz')
            stats['complete'] = True
            dev_eval.save(p, stats)
            print('RUN_END', name, json.dumps({x: stats[x] for x in ('generation_wall_seconds', 'output_tokens',
                                                                      'output_tokens_per_second', 'busy_slot_share',
                                                                      'truncated', 'accuracy')}), flush=True)
        results, texts = {}, {}
        for name in names:
            results[name] = json.loads((cdir / 'runs' / f'{name}.json').read_text())
            with gzip.open(cdir / 'runs' / f'{name}.outputs.jsonl.gz', 'rt', encoding='utf-8') as f:
                texts[name] = [json.loads(x)['text'] for x in f]
        summary = {'utc': sb.now_utc(), 'cell': cell, 'model': a.model, 'slots': a.slots, 'requests': len(items),
                   'compare': sb.compare(results, texts)[str(BUDGET)],
                   'runs': {n: {x: r[x] for x in ('mode', 'generation_wall_seconds', 'output_tokens',
                                                  'output_tokens_per_second', 'busy_slot_share', 'finish_seconds',
                                                  'truncated', 'accuracy', 'peak_running', 'peak_resident_tokens',
                                                  'engine_counters')}
                            for n, r in results.items()}}
        dev_eval.save(cdir / 'cell_summary.json', summary)
        print('CELL_DONE', cell, json.dumps(summary['compare']), flush=True)
    gpu.stop.set()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--check-loss', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slots', type=int, required=True)
    ap.add_argument('--cell', nargs=3, action='append', required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f'ext_error_slots{args.slots}.json').write_text(
            json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
