"""Development evaluation on MATH dev 100 with vLLM 0.21.0 (tuning only; no test set is read).

--prompt chat: the training chat template of sft_data.py, zero-shot, stop at <|im_end|> (conditions 2-4).
--prompt minerva: the 4-shot prompt of base_probe/base_probe_worker.py, stop at "\\nProblem:" (condition 1).
Sampling: T 0.6, top_p 0.95, top_k 20, seed 91432026 + item index + 100000 * sample. Grading: math-verify through
validation/eval_common.py (last \\boxed{} after </think>). Every output is also re-graded after truncation at each
--budgets value below --max-tokens. think_closed records whether a chat output closed its think block.

--check-loss ADAPTER_DIR/check_loss.json: before generating, recompute the target NLL of the four check sequences from
the loaded weights (vLLM prompt logprobs) and compare it with the value the trained adapter gave in train_lora.py.

usage: python -u dev_eval.py --model MERGED_DIR|base --prompt chat|minerva --samples 4 --max-tokens 16384 --out DIR
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_data  # noqa: E402
import eval_common  # noqa: E402  (frozen grader, copied next to this file)
import base_probe_worker  # noqa: E402  (Minerva shots and stop string, copied next to this file)

SEED = 91432026
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)
# Prompt log-probs (the --check-loss pass) take chunk tokens x 248,320 vocab x fp32 outside vLLM's memory budget;
# 2,048-token chunks and 0.80 utilization leave room for that on a 40 GB A100.
ENGINE = dict(max_num_batched_tokens=2048, gpu_memory_utilization=.80)
FROZEN = {'eval_common.py': '37aa9b22', 'base_probe_worker.py': 'e8e636e9', 'math_dev_100.jsonl': '0453023e'}


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def check_nll(llm, rows):
    from vllm import SamplingParams
    outs = llm.generate([{'prompt_token_ids': r['input_ids']} for r in rows],
                        SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
    res = []
    for r, o in zip(rows, outs):
        lp = o.prompt_logprobs
        nll = -sum(lp[j][r['input_ids'][j]].logprob for j in range(r['n_prompt'], len(r['input_ids'])))
        res.append({'id': r['id'], 'trained_nll_sum': r['nll_sum'], 'vllm_nll_sum': nll, 'tokens': r['tokens'],
                    'relative_difference': abs(nll - r['nll_sum']) / r['nll_sum']})
    worst = max(x['relative_difference'] for x in res)
    return {'rows': res, 'max_relative_difference': worst, 'pass': worst < 0.03}


def run(a):
    import importlib.metadata as metadata
    import torch
    from vllm import LLM, SamplingParams
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_p = out / 'dev_summary.json'
    assert not summary_p.exists(), 'Results exist here: inspect them instead of rerunning.'
    shas = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in FROZEN}
    assert all(shas[n].startswith(FROZEN[n]) for n in FROZEN), shas
    rows = [json.loads(s) for s in (HERE / 'math_dev_100.jsonl').read_text().splitlines()]
    assert len(rows) == 100
    tok = sft_data.load_tokenizer()
    if a.prompt == 'chat':
        prompts = [sft_data.chat_prompt(r['problem']) for r in rows]
        stop = dict(stop_token_ids=[tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')])
    else:
        prompts = [base_probe_worker.math_prompt(r['problem']) for r in rows]
        stop = dict(stop=base_probe_worker.MATH_STOP)
    max_prompt = max(len(tok.encode(p, add_special_tokens=False)) for p in prompts)
    assert max_prompt <= 2048, max_prompt
    budgets = sorted(b for b in a.budgets if b < a.max_tokens)
    model_kw = dict(model=sft_data.TOKENIZER, revision=sft_data.REVISION) if a.model == 'base' else dict(model=a.model)
    summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model': a.model, 'tag': a.tag,
               'prompt': a.prompt, 'samples': a.samples, 'max_tokens': a.max_tokens, 'budgets': budgets,
               'sampling': SAMPLING, 'seed_rule': '91432026 + item index + 100000*sample', 'stop': {k: v for k, v in stop.items()},
               'max_prompt_tokens': max_prompt, 'max_num_seqs': a.max_num_seqs, 'engine': ENGINE, 'frozen_sha256': shas,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'gpu': torch.cuda.get_device_name(0),
               'versions': {n: metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']},
               'test_sets_read': False, 'complete': False}
    if a.model != 'base':
        merge = Path(a.model) / 'merge_report.json'
        summary['merge_report'] = json.loads(merge.read_text()) if merge.exists() else None
    save(summary_p, summary)
    t = time.perf_counter()
    llm = LLM(**model_kw, dtype='bfloat16', max_model_len=a.max_tokens + 2048, max_num_seqs=a.max_num_seqs,
              **ENGINE, enforce_eager=False,
              enable_prefix_caching=False, language_model_only=True, seed=SEED, disable_log_stats=False)
    summary['load_seconds'] = time.perf_counter() - t
    if a.check_loss:
        summary['check_loss'] = check_nll(llm, json.loads(Path(a.check_loss).read_text()))
        save(summary_p, summary)
        print('CHECK_LOSS', json.dumps({k: summary['check_loss'][k] for k in ('max_relative_difference', 'pass')}), flush=True)
    items = [(i, s) for s in range(a.samples) for i in range(len(rows))]
    params = [SamplingParams(**SAMPLING, max_tokens=a.max_tokens, seed=SEED + i + 100000 * s, **stop) for i, s in items]
    t = time.perf_counter()
    outputs = llm.generate([prompts[i] for i, _ in items], params, use_tqdm=True)
    wall = time.perf_counter() - t
    records = []
    for (i, s), res in zip(items, outputs):
        o, r = res.outputs[0], rows[i]
        rec = {'id': r['id'], 'level': r['level'], 'subject': r['subject'], 'sample': s, 'seed': SEED + i + 100000 * s,
               'gold': r['gold'], 'text': o.text, 'output_tokens': len(o.token_ids), 'finish_reason': o.finish_reason,
               'think_closed': '</think>' in o.text, **eval_common.grade(o.text, r['gold'])}
        rec['prefix'] = {}
        for b in budgets:
            text = o.text if len(o.token_ids) <= b else tok.decode(o.token_ids[:b], skip_special_tokens=True)
            rec['prefix'][str(b)] = eval_common.grade(text, r['gold'])['correct']
        records.append(rec)
    (out / 'dev_outputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    n = len(records)
    summary.update({
        'n': n, 'generation_wall_seconds': wall, 'output_tokens': sum(r['output_tokens'] for r in records),
        'accuracy': sum(r['correct'] for r in records) / n,
        'per_sample_accuracy': [sum(r['correct'] for r in records if r['sample'] == s) / len(rows) for s in range(a.samples)],
        'truncated': sum(r['finish_reason'] == 'length' for r in records) / n,
        'think_closed': sum(r['think_closed'] for r in records) / n,
        'correct_with_unclosed_think': sum(r['correct'] and not r['think_closed'] for r in records),
        'mean_output_tokens': sum(r['output_tokens'] for r in records) / n,
        'median_output_tokens': sorted(r['output_tokens'] for r in records)[n // 2],
        'prefix_accuracy': {str(b): sum(r['prefix'][str(b)] for r in records) / n for b in budgets},
        'by_level': {str(l): sum(r['correct'] for r in records if r['level'] == l) / sum(r['level'] == l for r in records)
                     for l in range(1, 6)},
        'complete': True})
    summary['aggregate_output_tokens_per_second'] = summary['output_tokens'] / wall
    save(summary_p, summary)
    print('DEV_EVAL_FINISHED', json.dumps({k: summary[k] for k in ('tag', 'accuracy', 'truncated', 'mean_output_tokens',
                                                                     'prefix_accuracy', 'generation_wall_seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--prompt', required=True, choices=['chat', 'minerva'])
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--budgets', type=int, nargs='*', default=[4096, 8192, 16384])
    p.add_argument('--max-num-seqs', type=int, default=64)
    p.add_argument('--check-loss')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'dev_error.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
