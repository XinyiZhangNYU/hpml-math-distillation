"""SFT sequences, loss masks and step plans for groups 2-4. Tokenizer only: no model, no GPU, no API.

The layout is the one of teacher_gen.py and count_group4_tokens.py (DESIGN section 3). The loss covers every token
after the prompt, from <think> through the final <|im_end|>\n. Group 2 puts the human solution after an empty think
block. encode() checks that the prompt boundary does not move and that the group totals equal the frozen counts.

Step plan: one epoch visits the 1,971 questions in a seeded order that depends only on (seed, epoch), so every group
and learning rate sees the questions in the same order. Each optimizer step takes the next STEP_EXAMPLES questions
(the last step of an epoch takes the remainder). Inside a step the examples are sorted by length and cut into
microbatches of similar length whose padded size (examples x longest) stays within TOKEN_BUDGET; the cut minimizes
padded tokens plus MICROBATCH_OVERHEAD per microbatch. Padding is on the right and is masked out of both attention
and loss. Examples are never packed together. The microbatch split changes speed and memory, and the step's loss only up to rounding.

usage (local check): python sft_data.py DATA_DIR   (prints per-group token and step statistics)
"""
import hashlib
import json
import random
import sys
from pathlib import Path

PROMPT = 'Solve the following mathematical problem. Explain the necessary steps and put the final answer in \\boxed{}.\n\n'
TOKENIZER = 'Qwen/Qwen3.5-9B-Base'
REVISION = '68c46c4b3498877f3ef123c856ecfde50c39f404'
LIMIT = 32768
N_QUESTIONS = 1971
STEP_EXAMPLES = 16
TOKEN_BUDGET = 32768
ORDER_SEED = 20261021
MICROBATCH_OVERHEAD = 1024  # padded-token cost charged per extra forward/backward pass when splitting a step
GROUPS = {  # file, SHA-256, (reasoning, answer) fields, frozen total tokens (token_counts_v1)
    '2': ('group2_human.jsonl', '176dc0b04b9d599870e37172ff06872c0faeacad2ed4f7f1a3b33d21dabe8f93',
          lambda r: ('', r['solution']), 611428),
    '3': ('final_traces.jsonl', '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e',
          lambda r: (r['reasoning_content'], r['content']), 2420158),
    '4': ('group4_heavy.jsonl', '8ac76e67953c530714e15ef21411112fc45133db6875fd0e6a6b5f45ed7c286b',
          lambda r: (r['reasoning_content'], r['content']), 1750516),
}


def layout(prompt, reasoning, answer):  # identical to teacher_gen.layout
    head = '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    return head, head + '<think>\n' + reasoning.strip() + '\n</think>\n\n' + answer.strip() + '<|im_end|>\n'


def chat_prompt(problem):
    return layout(PROMPT + problem, '', '')[0]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION)


def load_group(data_dir, group):
    name, digest, fields, _ = GROUPS[group]
    raw = (Path(data_dir) / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    rows = [json.loads(s) for s in raw.decode('utf-8').splitlines()]
    assert len(rows) == N_QUESTIONS
    return rows


def encode(tok, rows, group):
    """Returns [{'id', 'input_ids', 'n_prompt'}]; the loss covers input_ids[n_prompt:]."""
    fields, total = GROUPS[group][2], GROUPS[group][3]
    end = tok('<|im_end|>\n', add_special_tokens=False)['input_ids']
    out = []
    for r in rows:
        head, full = layout(PROMPT + r['problem'], *fields(r))
        head_ids = tok(head, add_special_tokens=False)['input_ids']
        ids = tok(full, add_special_tokens=False)['input_ids']
        assert ids[:len(head_ids)] == head_ids, ('prompt boundary moved', r['id'])
        assert ids[-len(end):] == end and len(ids) <= LIMIT, r['id']
        out.append({'id': r['id'], 'input_ids': ids, 'n_prompt': len(head_ids)})
    assert sum(len(x['input_ids']) for x in out) == total, ('group total differs from token_counts_v1', group)
    return out


def epoch_order(epoch, seed=ORDER_SEED, n=N_QUESTIONS):
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


def step_plan(lengths, epoch, seed=ORDER_SEED, step_examples=STEP_EXAMPLES, budget=TOKEN_BUDGET):
    """Returns the steps of one epoch; each step is a list of microbatches, each a list of example indices."""
    order = epoch_order(epoch, seed, len(lengths))
    return [split(sorted(order[k:k + step_examples], key=lambda i: (-lengths[i], i)), lengths, budget)
            for k in range(0, len(order), step_examples)]


def split(chunk, lengths, budget=TOKEN_BUDGET, overhead=MICROBATCH_OVERHEAD):
    """Cuts a longest-first list into consecutive microbatches, minimizing padded tokens + overhead per microbatch
    (dynamic programming), with padded size (examples x longest) <= budget unless the microbatch is one example."""
    n = len(chunk)
    best, cut = [0] + [None] * n, [0] * (n + 1)
    for k in range(1, n + 1):
        for j in range(k):
            size = (k - j) * lengths[chunk[j]]
            if k - j > 1 and size > budget:
                continue
            c = best[j] + size + overhead
            if best[k] is None or c < best[k]:
                best[k], cut[k] = c, j
    micro, k = [], n
    while k:
        micro.append(chunk[cut[k]:k])
        k = cut[k]
    return micro[::-1]


def plan_stats(examples, epochs=3, seed=ORDER_SEED):
    lengths = [len(x['input_ids']) for x in examples]
    out = []
    for e in range(1, epochs + 1):
        steps = step_plan(lengths, e, seed)
        real = sum(lengths)
        padded = sum(len(m) * lengths[m[0]] for s in steps for m in s)
        out.append({'epoch': e, 'steps': len(steps), 'microbatches': sum(len(s) for s in steps),
                    'max_microbatches_per_step': max(len(s) for s in steps),
                    'real_tokens': real, 'padded_tokens': padded, 'padding_fraction': round(1 - real / padded, 4),
                    'supervised_tokens': sum(len(x['input_ids']) - x['n_prompt'] for x in examples)})
    return out


if __name__ == '__main__':
    tok = load_tokenizer()
    data_dir = sys.argv[1]
    ids = None
    for g in GROUPS:
        ex = encode(tok, load_group(data_dir, g), g)
        if ids is None:
            ids = [x['id'] for x in ex]
        assert [x['id'] for x in ex] == ids, 'groups must list the questions in the same order'
        print('group', g, json.dumps(plan_stats(ex)))
