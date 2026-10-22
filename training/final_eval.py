"""Final evaluation (DESIGN.md §9) of one model on the frozen test tiers with vLLM 0.21.0.

--condition 1: the base model with the Minerva 4-shot prompt of base_probe/base_probe_worker.py, stop at
"\\nProblem:". --condition 2|3|4 --train-seed 1|2|3 --model MERGED_DIR: the merged adapter with the training chat
template of sft_data.py, zero-shot, stop at <|im_end|>. Samples per problem: 8 (tiers 1 and 2), 4 (tier 3).
Budget: 32,768 generated tokens; every output is scored again after truncation at 4k, 8k and 16k tokens. Sampling
and engine settings are those of dev_eval.py (T 0.6, top_p 0.95, top_k 20; 2,048-token prefill chunks, 0.80 GPU
memory). Request seed: 91432026 + 1,000,000 * tier + item index + 100,000 * sample, the same for every model.

Grading: math-verify through the frozen validation/eval_common.py, on the last \\boxed{} after </think>. For
conditions 2-4 an output, or a truncated prefix, whose think block is not closed is wrong (§9); correct_eval_common
keeps the grader's verdict without that rule. Ids in scoring_exclusions.json are generated and recorded with
scored: false, and are left out of every accuracy.

--tests DIR holds the frozen test files (checked by SHA-256): tier1_variants.jsonl, tier2_aime2026.jsonl,
tier2_hmmt_feb2026.jsonl, tier3_math500.jsonl and scoring_exclusions.json. Each tier goes to OUT/TIER/:
scores.jsonl (one row per output, without text), outputs.jsonl.gz (with text) and summary.json. A tier whose summary
says complete is skipped when this is run again. final_stats.py combines the runs.

usage: python -u final_eval.py --condition 3 --train-seed 1 --model MERGED_DIR --tests DIR --out DIR
       [--tiers tier1 tier2 tier3] [--check-loss ADAPTER_DIR/check_loss.json]
       python -u final_eval.py --condition 1 --tests DIR --out DIR
"""
import argparse
import datetime
import gzip
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev_eval  # noqa: E402  (sampling, engine settings and the loss check of the dev evaluation)
import eval_common  # noqa: E402  (frozen grader, copied next to this file)

SEED = dev_eval.SEED
MAX_TOKENS = 32768
BUDGETS = [4096, 8192, 16384]
TIERS = {  # tier -> (number, files, samples per problem)
    'tier1': (1, ['tier1_variants.jsonl'], 8),
    'tier2': (2, ['tier2_aime2026.jsonl', 'tier2_hmmt_feb2026.jsonl'], 8),
    'tier3': (3, ['tier3_math500.jsonl'], 4)}
TEST_FILES = {
    'tier1_variants.jsonl': 'c5e1eb3cea9ddcabdadd93952951cc1673968083c113662926c8c2245e16331b',
    'tier2_aime2026.jsonl': '67dbd42cc6327589a164a1f831b47690903af619124ea87e49e55863d27759da',
    'tier2_hmmt_feb2026.jsonl': '40f1a4430432c608f832d1e9cc2c66780e1c3f34d071c05da59cb138dea6fddc',
    'tier3_math500.jsonl': 'f9608716204039b465c396c7a3a753b2f3bb1976933237c5064d5a4e55cc385b',
    'scoring_exclusions.json': 'c268021888738f9c22f5791024f0954a9dc79f54ff5f386be17e961bc392b62c'}
FROZEN = {'eval_common.py': '37aa9b22', 'base_probe_worker.py': 'e8e636e9', 'dev_eval.py': '4d1aeb89'}
COUNTS = {'tier1': 90, 'tier2': 63, 'tier3': 500}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_tier(tests, tier):
    """Rows of a tier with id, problem, gold, cluster (tier 1: source problem; else the problem) and subset."""
    rows = []
    for name in TIERS[tier][1]:
        for line in (Path(tests) / name).read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if tier == 'tier1':
                cluster, subset = r['family'], r['type']
            elif tier == 'tier2':
                cluster, subset = r['id'], r['id'].split('/')[0]
            else:
                cluster, subset = r['id'], 'level' + str(r['level'])
            rows.append({'id': r['id'], 'problem': r['problem'], 'gold': r['answer'], 'cluster': cluster,
                         'subset': subset})
    assert len(rows) == COUNTS[tier] and len({r['id'] for r in rows}) == len(rows), (tier, len(rows))
    return rows


def excluded_ids(tests):
    return {x['id'] for x in json.loads((Path(tests) / 'scoring_exclusions.json').read_text())['exclusions']}


def score(text, gold, chat):
    """§9 rule: for chat outputs, a think block that is not closed makes the output wrong."""
    g = eval_common.grade(text, gold)
    closed = '</think>' in text
    return {'think_closed': closed, 'correct': bool(g['correct'] and (closed or not chat)),
            'correct_eval_common': g['correct'], 'extracted_answer': g['extracted_answer'], 'parseable': g['parseable']}


def score_prefixes(text, token_ids, gold, chat, decode, budgets=BUDGETS):
    """Score the output again after truncation at each budget; a shorter output is scored whole."""
    res = {}
    for b in budgets:
        t = text if len(token_ids) <= b else decode(token_ids[:b])
        s = score(t, gold, chat)
        res[str(b)] = {'correct': s['correct'], 'think_closed': s['think_closed']}
    return res


def problem_mean(records, key=lambda r: r['correct']):
    """Mean over problems of the mean over samples, scored rows only."""
    by = {}
    for r in records:
        if r['scored']:
            by.setdefault(r['id'], []).append(key(r))
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def summarize(records, chat):
    scored = [r for r in records if r['scored']]
    n = len(scored)
    tokens = sorted(r['output_tokens'] for r in scored)
    out = {'outputs': len(records), 'scored_outputs': n, 'scored_problems': len({r['id'] for r in scored}),
           'accuracy': problem_mean(records),
           'accuracy_eval_common': problem_mean(records, lambda r: r['correct_eval_common']),
           'truncated': sum(r['finish_reason'] == 'length' for r in scored) / n,
           'mean_output_tokens': sum(tokens) / n, 'median_output_tokens': statistics.median(tokens),
           'prefix_accuracy': {b: problem_mean(records, lambda r, b=b: r['prefix'][b]['correct']) for b in map(str, BUDGETS)},
           'prefix_truncated': {str(b): sum(r['output_tokens'] > b for r in scored) / n for b in BUDGETS},
           'by_subset': {s: problem_mean([r for r in records if r['subset'] == s])
                         for s in sorted({r['subset'] for r in scored})},
           'excluded': {i: [r['correct'] for r in records if r['id'] == i] for i in sorted({r['id'] for r in records if not r['scored']})}}
    if chat:
        out['think_closed'] = sum(r['think_closed'] for r in scored) / n
        out['correct_with_unclosed_think'] = sum(r['correct_eval_common'] and not r['think_closed'] for r in scored)
    return out


def run(a):
    import importlib.metadata as metadata
    import sft_data
    import base_probe_worker
    import torch
    from vllm import LLM, SamplingParams
    chat = a.condition != 1
    assert (a.model == 'base') == (not chat) and (a.train_seed is None) == (not chat), 'condition 1 = base model, no seed'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shas = {n: sha(HERE / n) for n in FROZEN}
    assert all(shas[n].startswith(FROZEN[n]) for n in FROZEN), shas
    test_shas = {n: sha(Path(a.tests) / n) for n in TEST_FILES}
    assert test_shas == TEST_FILES, test_shas
    skip = excluded_ids(a.tests)
    tiers = [t for t in a.tiers if not (out / t / 'summary.json').exists()
             or not json.loads((out / t / 'summary.json').read_text()).get('complete')]
    print('TIERS_TO_RUN', tiers, flush=True)
    if not tiers:
        return
    tok = sft_data.load_tokenizer()
    if chat:
        build = sft_data.chat_prompt
        stop = dict(stop_token_ids=[tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')])
    else:
        build = base_probe_worker.math_prompt
        stop = dict(stop=base_probe_worker.MATH_STOP)
    rows = {t: load_tier(a.tests, t) for t in tiers}
    prompts = {t: [build(r['problem']) for r in rows[t]] for t in tiers}
    max_prompt = max(len(tok.encode(p, add_special_tokens=False)) for t in tiers for p in prompts[t])
    assert max_prompt <= 2048, max_prompt
    model_kw = dict(model=sft_data.TOKENIZER, revision=sft_data.REVISION) if not chat else dict(model=a.model)
    meta = {'condition': a.condition, 'train_seed': a.train_seed, 'model': a.model,
            'prompt': 'chat' if chat else 'minerva', 'max_tokens': MAX_TOKENS, 'budgets': BUDGETS,
            'sampling': dev_eval.SAMPLING, 'engine': dev_eval.ENGINE, 'max_num_seqs': a.max_num_seqs,
            'seed_rule': '91432026 + 1000000*tier + item index + 100000*sample',
            'stop': stop, 'max_prompt_tokens': max_prompt, 'think_rule': 'unclosed think block = wrong' if chat else None,
            'frozen_sha256': shas, 'test_sha256': test_shas, 'excluded_ids': sorted(skip),
            'script_sha256': sha(Path(__file__)), 'gpu': torch.cuda.get_device_name(0),
            'versions': {n: metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']}}
    if chat:
        merge = Path(a.model) / 'merge_report.json'
        meta['merge_report'] = json.loads(merge.read_text()) if merge.exists() else None
    t0 = time.perf_counter()
    llm = LLM(**model_kw, dtype='bfloat16', max_model_len=MAX_TOKENS + 2048, max_num_seqs=a.max_num_seqs,
              **dev_eval.ENGINE, enforce_eager=False, enable_prefix_caching=False, language_model_only=True, seed=SEED,
              disable_log_stats=False)
    meta['load_seconds'] = time.perf_counter() - t0
    if a.check_loss:
        meta['check_loss'] = dev_eval.check_nll(llm, json.loads(Path(a.check_loss).read_text()))
        print('CHECK_LOSS', json.dumps({k: meta['check_loss'][k] for k in ('max_relative_difference', 'pass')}), flush=True)
        dev_eval.save(out / 'check_loss.json', meta['check_loss'])
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    for t in tiers:
        number, _, samples = TIERS[t]
        d = out / t
        d.mkdir(exist_ok=True)
        summary = {**meta, 'tier': t, 'samples': samples, 'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'complete': False}
        dev_eval.save(d / 'summary.json', summary)
        items = [(i, s) for s in range(samples) for i in range(len(rows[t]))]
        seed = lambda i, s: SEED + 1000000 * number + i + 100000 * s  # noqa: E731
        params = [SamplingParams(**dev_eval.SAMPLING, max_tokens=MAX_TOKENS, seed=seed(i, s), **stop) for i, s in items]
        print('TIER_START', t, len(items), flush=True)
        t1 = time.perf_counter()
        outputs = llm.generate([prompts[t][i] for i, _ in items], params, use_tqdm=True)
        wall = time.perf_counter() - t1
        records, texts = [], []
        for (i, s), res in zip(items, outputs):
            o, r = res.outputs[0], rows[t][i]
            ids = list(o.token_ids)
            rec = {'condition': a.condition, 'train_seed': a.train_seed, 'tier': t, 'id': r['id'], 'cluster': r['cluster'],
                   'subset': r['subset'], 'scored': r['id'] not in skip, 'sample': s, 'seed': seed(i, s), 'gold': r['gold'],
                   'output_tokens': len(ids), 'finish_reason': o.finish_reason, **score(o.text, r['gold'], chat),
                   'prefix': score_prefixes(o.text, ids, r['gold'], chat, decode)}
            records.append(rec)
            texts.append({'id': r['id'], 'sample': s, 'text': o.text})
        (d / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        with gzip.open(d / 'outputs.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + '\n' for x in texts)
        summary.update(summarize(records, chat))
        summary.update({'generation_wall_seconds': wall, 'output_tokens': sum(r['output_tokens'] for r in records),
                        'scores_sha256': sha(d / 'scores.jsonl'), 'outputs_sha256': sha(d / 'outputs.jsonl.gz'),
                        'utc_end': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete': True})
        summary['aggregate_output_tokens_per_second'] = summary['output_tokens'] / wall
        dev_eval.save(d / 'summary.json', summary)
        print('TIER_FINISHED', json.dumps({k: summary[k] for k in ('tier', 'accuracy', 'truncated', 'prefix_accuracy',
                                                                     'mean_output_tokens', 'generation_wall_seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--condition', type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument('--train-seed', type=int, choices=[1, 2, 3])
    p.add_argument('--model', default='base')
    p.add_argument('--tests', required=True)
    p.add_argument('--tiers', nargs='+', default=list(TIERS), choices=list(TIERS))
    p.add_argument('--max-num-seqs', type=int, default=64)
    p.add_argument('--check-loss')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'final_error.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
