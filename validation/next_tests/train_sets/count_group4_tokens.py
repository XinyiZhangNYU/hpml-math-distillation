"""Exact Qwen token counts of the training sets; tokenizer files only, no model, no API.

Uses the same SFT layout, tokenizer and revision as teacher_gen.py. Group 3 counts must equal qwen_total_tokens in
final_traces.jsonl (sanity check of the setup); group 4 counts replace the estimate in group4_heavy.jsonl. If
group2_human.jsonl is given, group 2 is counted in two layouts: the solution after an empty think block
(group2_empty_think) and the solution with no think block (group2_no_think).

usage (needs transformers): python count_group4_tokens.py final_traces.jsonl group4_heavy.jsonl out.json [group2_human.jsonl]
"""
import hashlib
import json
import os
import statistics
import sys

PROMPT = 'Solve the following mathematical problem. Explain the necessary steps and put the final answer in \\boxed{}.\n\n'
TOKENIZER = 'Qwen/Qwen3.5-9B-Base'
TOKENIZER_REVISION = '68c46c4b3498877f3ef123c856ecfde50c39f404'
QWEN_LIMIT = 32768


def layout(prompt, reasoning, answer):  # identical to teacher_gen.layout
    head = '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    return head, head + '<think>\n' + reasoning.strip() + '\n</think>\n\n' + answer.strip() + '<|im_end|>\n'


def main(raw_p, g4_p, out_p, g2_p=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)

    def count(r):
        head, full = layout(PROMPT + r['problem'], r['reasoning_content'], r['content'])
        head_ids = tok(head, add_special_tokens=False)['input_ids']
        ids = tok(full, add_special_tokens=False)['input_ids']
        assert ids[:len(head_ids)] == head_ids, 'prompt boundary moved'
        return len(ids)

    raw = [json.loads(line) for line in open(raw_p, encoding='utf-8')]
    g4 = [json.loads(line) for line in open(g4_p, encoding='utf-8')]
    assert [r['id'] for r in raw] == [r['id'] for r in g4]
    n3 = [count(r) for r in raw]
    mism = [r['id'] for r, n in zip(raw, n3) if n != r['qwen_total_tokens']]
    assert not mism, f'group 3 recount differs from teacher_gen for {len(mism)} traces, e.g. {mism[:5]}'
    n4 = [count(r) for r in g4]
    g2_counts = {}
    if g2_p:
        g2 = [json.loads(line) for line in open(g2_p, encoding='utf-8')]
        assert [r['id'] for r in g2] == [r['id'] for r in raw]
        g2_counts['group2_empty_think'] = [count({'problem': r['problem'], 'reasoning_content': '',
                                                  'content': r['solution']}) for r in g2]

        def count_no_think(r):
            head = layout(PROMPT + r['problem'], '', '')[0]
            return len(tok(head + r['solution'].strip() + '<|im_end|>\n', add_special_tokens=False)['input_ids'])
        g2_counts['group2_no_think'] = [count_no_think(r) for r in g2]

    def summ(xs):
        return {'sum': sum(xs), 'mean': round(statistics.mean(xs), 1), 'median': statistics.median(xs),
                'max': max(xs), 'over_8k': sum(x > 8192 for x in xs), 'over_16k': sum(x > 16384 for x in xs),
                'over_limit': sum(x > QWEN_LIMIT for x in xs)}
    out = {'tokenizer': TOKENIZER, 'revision': TOKENIZER_REVISION,
           'inputs_sha256': {os.path.basename(p): hashlib.sha256(open(p, 'rb').read()).hexdigest()
                             for p in (raw_p, g4_p, g2_p) if p},
           'group3': summ(n3), 'group4': summ(n4), **{k: summ(v) for k, v in g2_counts.items()},
           'group4_per_trace': {r['id']: n for r, n in zip(g4, n4)}}
    json.dump(out, open(out_p, 'w', encoding='utf-8', newline='\n'), indent=1)
    print(json.dumps({k: out[k] for k in ('group3', 'group4', *g2_counts)}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
