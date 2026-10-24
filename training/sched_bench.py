"""vLLM request scheduling benchmark (DESIGN.md §11): fixed-size request groups versus a continuously refilled queue.

Workload: MATH dev 100 x 2 samples = 200 requests, in the order of dev_eval.py (every problem for sample 0, then for
sample 1), on the merged condition 3 / seed 1 model, with the chat prompt of sft_data.py and the sampling of DESIGN.md
§9 (T 0.6, top_p 0.95, top_k 20; request seed 91432026 + item index + 100000 * sample). Engine as in final_eval.py:
max_model_len 32,768 + 2,048, 2,048-token prefill chunks, 0.80 GPU memory, no prefix caching, CUDA graphs, and
max_num_seqs 64. One engine serves every run.

  fixed: requests are submitted 64 at a time in workload order; the next group goes in when every request of the
         previous group has finished.
  queue: all requests are submitted at once; the engine starts a waiting request as soon as one of its 64 slots is free.

Both modes drive the same engine through add_request() and step(), so they differ only in when requests are submitted.
Runs, in this order: budget 32,768: fixed, queue, queue, fixed; budget 8,192: queue, fixed.

Before the runs: the loss check of the merged weights (dev_eval.check_nll; the benchmark stops if it fails) and a
warm-up of 8 short requests, driven once by the step loop and once by LLM.generate() to check the step loop.

Per run, OUT/runs/NAME.json: generation wall time, output tokens, throughput, every request's first-token and finish
time from the start of the run, the busy slot share, a once-a-second trace of submitted and running requests, resident
tokens (prompt + generated tokens of the running requests) and GPU memory used (nvidia-smi), vLLM's preemption and
token counters over the run, and the accuracy under the §9 rule. OUT/runs/NAME.outputs.jsonl.gz holds the outputs with text. OUT/bench_summary.json compares the modes.

usage: python -u sched_bench.py --model MERGED_DIR --check-loss ADAPTER_DIR/check_loss.json --out DIR
"""
import argparse
import datetime
import gzip
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev_eval  # noqa: E402  (sampling, engine settings and loss check of the dev evaluation)
import final_eval  # noqa: E402  (§9 scoring rule)
import sft_data  # noqa: E402

SEED, SAMPLES, GROUP, MAX_MODEL_LEN = dev_eval.SEED, 2, 64, 32768 + 2048
RUNS = [('t32k_fixed_1', 32768, 'fixed'), ('t32k_queue_1', 32768, 'queue'), ('t32k_queue_2', 32768, 'queue'),
        ('t32k_fixed_2', 32768, 'fixed'), ('t8k_queue_1', 8192, 'queue'), ('t8k_fixed_1', 8192, 'fixed')]
FROZEN = {'eval_common.py': '37aa9b22', 'base_probe_worker.py': 'e8e636e9', 'dev_eval.py': '4d1aeb89',
          'final_eval.py': '24ab0e81', 'sft_data.py': '9f4c31ef', 'math_dev_100.jsonl': '0453023e'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class GpuMonitor(threading.Thread):
    """Whole-device memory used (nvidia-smi, MiB) about once a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=10).stdout
                self.samples.append((time.perf_counter(), int(out.split()[0])))
            except Exception:  # noqa: BLE001  (a missed sample is not an error)
                pass
            self.stop.wait(1.0)

    def between(self, t0, t1):
        return [m for t, m in self.samples if t0 <= t <= t1]


def drive(engine, prompts, params, ptok, group, tag):
    """Submits requests in groups of `group` (0: all at once), steps the engine until every request has finished.

    Times are seconds from the first submission. A request counts as running from its first generated token to its
    finish; resident tokens are the prompt and generated tokens of the running requests."""
    from vllm.sampling_params import RequestOutputKind
    n = len(prompts)
    batches = [list(range(i, min(i + group, n))) for i in range(0, n, group)] if group else [list(range(n))]
    first, end, ntok, reason, pieces = [None] * n, [None] * n, [0] * n, [None] * n, [[] for _ in range(n)]
    live, trace, groups = set(), [], []
    nxt, last_trace, last_beat = 0, -1.0, 0.0
    start = time.perf_counter()
    while nxt < len(batches) or live:
        if not live:
            t = time.perf_counter() - start
            if groups:
                groups[-1]['end'] = t
            groups.append({'start': t, 'requests': len(batches[nxt])})
            for k in batches[nxt]:
                p = params[k].clone()
                p.output_kind = RequestOutputKind.DELTA
                engine.add_request(f'r{k:04d}.{tag}', prompts[k], p)
                live.add(k)
            nxt += 1
        outs = engine.step()
        now = time.perf_counter() - start
        for o in outs:
            k = int(o.request_id[1:5])
            c = o.outputs[0]
            if c.token_ids and first[k] is None:
                first[k] = now
            ntok[k] += len(c.token_ids)
            pieces[k].append(c.text)
            if o.finished:
                end[k], reason[k] = now, c.finish_reason
                live.discard(k)
        if now - last_trace >= 1.0:
            running = [k for k in live if first[k] is not None]
            trace.append([round(now, 2), len(live), len(running), sum(ptok[k] + ntok[k] for k in running)])
            last_trace = now
        if now - last_beat >= 60:
            print('PROGRESS', tag, f'{now:.0f}s', 'finished', sum(e is not None for e in end), 'of', n,
                  'live', len(live), flush=True)
            last_beat = now
    wall = time.perf_counter() - start
    groups[-1]['end'] = wall
    return {'start': start, 'wall': wall, 'first': first, 'end': end, 'tokens': ntok, 'reason': reason,
            'text': [''.join(x) for x in pieces], 'trace': trace, 'groups': groups}


def engine_counters(llm):
    """vLLM's own counters (preemptions, prompt and generated tokens, requests); informational, so a failure is kept."""
    try:
        return {m.name: m.value for m in llm.get_metrics()
                if isinstance(getattr(m, 'value', None), (int, float))
                and any(x in m.name for x in ('preempt', 'prompt_tokens', 'generation_tokens', 'request_success'))}
    except Exception as e:  # noqa: BLE001
        return {'error': repr(e)}


def quant(xs):
    xs = sorted(xs)
    return {'mean': statistics.mean(xs), 'p50': xs[len(xs) // 2], 'p90': xs[min(len(xs) - 1, int(0.9 * len(xs)))],
            'max': xs[-1]}


def run_stats(res, gpu_mib, items, rows, budget, mode):
    n = len(items)
    scores = [final_eval.score(res['text'][k], rows[i]['gold'], True) for k, (i, _) in enumerate(items)]
    busy = sum(res['end'][k] - res['first'][k] for k in range(n) if res['first'][k] is not None)
    tokens = sum(res['tokens'])
    out = {'budget': budget, 'mode': mode, 'requests': n, 'group': GROUP if mode == 'fixed' else None,
           'generation_wall_seconds': res['wall'], 'output_tokens': tokens,
           'output_tokens_per_second': tokens / res['wall'],
           'truncated': sum(r == 'length' for r in res['reason']), 'mean_output_tokens': tokens / n,
           'max_output_tokens': max(res['tokens']),
           'finish_seconds': quant(res['end']), 'first_token_seconds': quant([x for x in res['first'] if x is not None]),
           'busy_slot_share': busy / (GROUP * res['wall']),
           'peak_running': max((t[2] for t in res['trace']), default=0),
           'peak_resident_tokens': max((t[3] for t in res['trace']), default=0),
           'gpu_memory_used_mib': {'max': max(gpu_mib), 'min': min(gpu_mib), 'samples': len(gpu_mib)} if gpu_mib else None,
           'accuracy': sum(s['correct'] for s in scores) / n, 'think_closed': sum(s['think_closed'] for s in scores) / n}
    if mode == 'fixed':
        out['groups'] = [{**g, 'seconds': g['end'] - g['start'],
                          'max_output_tokens': max(res['tokens'][k] for k in range(j * GROUP, j * GROUP + g['requests']))}
                         for j, g in enumerate(res['groups'])]
    return out, scores


def compare(results, texts):
    """Mean of each budget and mode, the queue/fixed ratios, and how many outputs are identical between runs."""
    out = {}
    for budget in sorted({r['budget'] for r in results.values()}, reverse=True):
        rs = {m: [r for r in results.values() if r['budget'] == budget and r['mode'] == m] for m in ('fixed', 'queue')}
        mean = {m: {k: statistics.mean(r[k] for r in v) for k in ('generation_wall_seconds', 'output_tokens_per_second',
                                                                  'busy_slot_share', 'accuracy')}
                for m, v in rs.items() if v}
        entry = {'runs': {m: len(v) for m, v in rs.items()}, 'mean': mean}
        if len(mean) == 2:
            entry['queue_over_fixed_throughput'] = (mean['queue']['output_tokens_per_second']
                                                    / mean['fixed']['output_tokens_per_second'])
            entry['fixed_over_queue_wall'] = mean['fixed']['generation_wall_seconds'] / mean['queue']['generation_wall_seconds']
        names = [n for n in texts if results[n]['budget'] == budget]
        entry['identical_outputs'] = {f'{a}~{b}': sum(x == y for x, y in zip(texts[a], texts[b])) / len(texts[a])
                                      for i, a in enumerate(names) for b in names[i + 1:]}
        out[str(budget)] = entry
    return out


def run(a):
    import importlib.metadata as metadata
    import torch
    from vllm import LLM, SamplingParams
    out = Path(a.out)
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    shas = {n: sha(HERE / n) for n in FROZEN}
    assert all(shas[n].startswith(FROZEN[n]) for n in FROZEN), shas
    rows = [json.loads(s) for s in (HERE / 'math_dev_100.jsonl').read_text().splitlines()]
    assert len(rows) == 100
    tok = sft_data.load_tokenizer()
    stop = dict(stop_token_ids=[tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')])
    items = [(i, s) for s in range(SAMPLES) for i in range(len(rows))]
    prompts = [sft_data.chat_prompt(rows[i]['problem']) for i, _ in items]
    ptok = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
    meta_p = out / 'bench_meta.json'
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.setdefault('sessions', []).append({
        'utc_start': now_utc(), 'model': a.model, 'workload': 'MATH dev 100 x 2 samples, dev_eval order',
        'requests': len(items), 'group': GROUP, 'max_num_seqs': GROUP, 'max_model_len': MAX_MODEL_LEN,
        'engine': dev_eval.ENGINE, 'sampling': dev_eval.SAMPLING, 'seed_rule': '91432026 + item index + 100000*sample',
        'stop': stop, 'prompt_tokens': {'total': sum(ptok), 'max': max(ptok)}, 'runs': RUNS, 'frozen_sha256': shas,
        'script_sha256': sha(Path(__file__)), 'gpu': torch.cuda.get_device_name(0),
        'versions': {n: metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']}})
    session = meta['sessions'][-1]
    merge = Path(a.model) / 'merge_report.json'
    session['merge_report'] = json.loads(merge.read_text()) if merge.exists() else None
    dev_eval.save(meta_p, meta)
    t = time.perf_counter()
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=MAX_MODEL_LEN, max_num_seqs=GROUP, **dev_eval.ENGINE,
              enforce_eager=False, enable_prefix_caching=False, language_model_only=True, seed=SEED, disable_log_stats=False)
    session['load_seconds'] = time.perf_counter() - t
    engine = llm.llm_engine
    try:
        cc = engine.vllm_config.cache_config
        session['kv_cache'] = {'num_gpu_blocks': cc.num_gpu_blocks, 'block_size': cc.block_size}
    except Exception as e:  # noqa: BLE001  (informational only)
        session['kv_cache'] = repr(e)
    session['check_loss'] = dev_eval.check_nll(llm, json.loads(Path(a.check_loss).read_text()))
    dev_eval.save(meta_p, meta)
    print('CHECK_LOSS', json.dumps({k: session['check_loss'][k] for k in ('max_relative_difference', 'pass')}), flush=True)
    assert session['check_loss']['pass'], 'The merged weights do not reproduce the adapter loss.'
    # Warm-up, and a check of the step loop against LLM.generate on the same 8 short requests.
    wp = [SamplingParams(**dev_eval.SAMPLING, max_tokens=256, seed=SEED + i, **stop) for i in range(8)]
    w = drive(engine, prompts[:8], wp, ptok, 0, 'warmup')
    g = llm.generate(prompts[:8], [p.clone() for p in wp], use_tqdm=False)
    session['warmup'] = {'step_loop_tokens': w['tokens'], 'generate_tokens': [len(x.outputs[0].token_ids) for x in g],
                         'identical_text': sum(x == y.outputs[0].text for x, y in zip(w['text'], g)),
                         'step_loop_seconds': w['wall']}
    dev_eval.save(meta_p, meta)
    print('WARMUP', json.dumps(session['warmup']), flush=True)
    assert all(e is not None for e in w['end']) and all(0 < x <= 256 for x in w['tokens']), 'step loop check failed'
    gpu = GpuMonitor()
    gpu.start()
    for name, budget, mode in RUNS:
        p = out / 'runs' / f'{name}.json'
        if p.exists() and json.loads(p.read_text()).get('complete'):
            print('RUN_SKIP_DONE', name, flush=True)
            continue
        params = [SamplingParams(**dev_eval.SAMPLING, max_tokens=budget, seed=SEED + i + 100000 * s, **stop) for i, s in items]
        print('RUN_START', name, now_utc(), flush=True)
        utc, c0 = now_utc(), engine_counters(llm)
        res = drive(engine, prompts, params, ptok, GROUP if mode == 'fixed' else 0, name)
        c1 = engine_counters(llm)
        stats, scores = run_stats(res, gpu.between(res['start'], res['start'] + res['wall']), items, rows, budget, mode)
        stats.update({'name': name, 'utc_start': utc, 'utc_end': now_utc(), 'session': len(meta['sessions']),
                      'engine_counters': {k: c1[k] - c0[k] if isinstance(c1[k], (int, float)) and k in c0 else c1[k]
                                          for k in c1},
                      'trace_columns': ['seconds', 'submitted_unfinished', 'running', 'resident_tokens'],
                      'trace': res['trace'],
                      'gpu_trace_mib': [[round(t - res['start'], 1), m] for t, m in gpu.samples
                                        if res['start'] <= t <= res['start'] + res['wall']]})
        recs = [{'k': k, 'id': rows[i]['id'], 'sample': s, 'seed': SEED + i + 100000 * s, 'prompt_tokens': ptok[k],
                 'output_tokens': res['tokens'][k], 'finish_reason': res['reason'][k], 'first_token_s': res['first'][k],
                 'finish_s': res['end'][k], 'correct': scores[k]['correct'], 'think_closed': scores[k]['think_closed'],
                 'text': res['text'][k]} for k, (i, s) in enumerate(items)]
        with gzip.open(out / 'runs' / f'{name}.outputs.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in recs)
        stats['outputs_sha256'] = sha(out / 'runs' / f'{name}.outputs.jsonl.gz')
        stats['complete'] = True
        dev_eval.save(p, stats)
        print('RUN_END', name, json.dumps({k: stats[k] for k in ('generation_wall_seconds', 'output_tokens',
                                                                  'output_tokens_per_second', 'busy_slot_share',
                                                                  'truncated', 'accuracy')}), flush=True)
    gpu.stop.set()
    results, texts = {}, {}
    for name, _, _ in RUNS:
        results[name] = json.loads((out / 'runs' / f'{name}.json').read_text())
        with gzip.open(out / 'runs' / f'{name}.outputs.jsonl.gz', 'rt', encoding='utf-8') as f:
            texts[name] = [json.loads(x)['text'] for x in f]
    summary = {'utc': now_utc(), 'by_budget': compare(results, texts),
               'runs': {n: {k: r[k] for k in ('budget', 'mode', 'generation_wall_seconds', 'output_tokens',
                                              'output_tokens_per_second', 'busy_slot_share', 'finish_seconds',
                                              'truncated', 'accuracy', 'gpu_memory_used_mib', 'peak_resident_tokens',
                                              'engine_counters')}
                        for n, r in results.items()}}
    dev_eval.save(out / 'bench_summary.json', summary)
    print('BENCH_DONE', json.dumps(summary['by_budget']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--check-loss', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'bench_error.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
