"""Packing and fused-kernel timing of LoRA training (DESIGN.md §11.4); one Colab A100, no API, no evaluation.

Arm R is train_lora.py itself (§8 stack: PyTorch fallbacks for the linear-attention layers). This script runs the rest:

  gate-ref  (§8 stack) summed target NLL of the four §8 check sequences of group 3, each alone; and the leakage
            control: the examples of the packed rows of the first GATE_STEPS steps of group 3, concatenated without
            boundaries, against the same examples alone (expected to differ).
  gate      (fused stack) gate 1: on one forward pass every full-attention layer ran the per-example flash
            attention below and every linear-attention layer ran the fla delta rule and the causal_conv1d
            convolution, with no PyTorch fallback warning; gate 2: the check NLLs within TOL of gate-ref; gate 3: on
            the same packed rows, with boundaries, every example within TOL of the same example alone, and the
            training call on each packed row within TOL of the sum of its examples alone.
  F, P      (fused stack) timed epoch 1 of one group, seed 1, lr 2e-4, the optimizer, schedule, LoRA setting and
            loss of train_lora.py. F: the §8 microbatches (right padding). P: the examples of each step, longest
            first, first-fit into rows of at most 32,768 tokens, one row per forward pass.
  summary   (CPU) primary and secondary results from the run records (runs/g{2,3}_{R,F,P}/train_report.json).

Fused stack. Before the model code is imported, fla.ops.gated_delta_rule.chunk_gated_delta_rule and
causal_conv1d.causal_conv1d_fn are wrapped (same signature): the wrappers count calls and pass the sequence
boundaries of a packed row (cu_seqlens; seq_idx). transformers 5.17.0 resolves these two names when the Qwen3.5
modeling code is imported, so the counts show that the fused kernels ran. The full-attention layers use the
attention function 'seg_flash' registered here: PyTorch SDPA restricted to its flash-attention backend, causal, one
call per example (per real row length for F), padded positions set to zero. Its mask function returns None, and
the model is called without attention_mask (right padding cannot reach a real token in a causal model; the loss
masks it).

usage: python -u pack_bench.py --mode gate-ref --data DIR --out FILE
       python -u pack_bench.py --mode gate --data DIR --ref FILE --out FILE
       python -u pack_bench.py --mode F|P --data DIR --group 2|3 --out DIR
       python -u pack_bench.py --mode summary --runs DIR --gates DIR [--record DIR] --out FILE
"""
import argparse
import datetime
import functools
import gc
import hashlib
import json
import logging
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft_data  # noqa: E402

GATE_GROUP = '3'
GATE_STEPS = 4
TOL = 1e-2
BUDGET = sft_data.TOKEN_BUDGET
LR, EPOCH, WARMUP = 2e-4, 1, 10
LORA_SEED, ORDER_SEED = 91432026, sft_data.ORDER_SEED  # seed 1 (§8)
CHUNK = 4096  # rows of logits per cross-entropy chunk in the gates


class Ctx:
    segments = None  # per row: [(start, end)] of the real tokens of each example
    cu = None        # cu_seqlens of a packed row (int64)
    seq_idx = None   # (1, L) int32 example index of each token of a packed row


CTX = Ctx()
COUNTS = {'attention': 0, 'chunk': 0, 'conv': 0}
FALLBACK = []
WRAPPED = {}  # id of an original kernel entry point -> its wrapper
ATTN_CAST = [0]


class FallbackCatch(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if 'falling back' in msg:
            FALLBACK.append(msg[:300])


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def versions():
    import importlib.metadata as metadata
    out = {}
    for n in ['torch', 'transformers', 'peft', 'unsloth', 'unsloth_zoo', 'xformers', 'triton', 'fla-core',
              'flash-linear-attention', 'causal-conv1d', 'kernels', 'einops']:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def pack(step, lengths, budget=BUDGET):
    """First-fit decreasing: the examples of one step, longest first, into rows of at most budget tokens."""
    rows, room = [], []
    for i in sorted(step, key=lambda i: (-lengths[i], i)):
        for r in range(len(rows)):
            if room[r] >= lengths[i]:
                rows[r].append(i)
                room[r] -= lengths[i]
                break
        else:
            rows.append([i])
            room.append(budget - lengths[i])
    return rows


def plan(lengths, arm, seed=ORDER_SEED):
    """Epoch 1: per step, the microbatches of §8 (R, F) or the packed rows (P); the same examples per step."""
    steps = sft_data.step_plan(lengths, EPOCH, seed)
    if arm == 'P':
        return [pack([i for m in s for i in m], lengths) for s in steps]
    return steps


# ---------------------------------------------------------------- fused stack

def fused_setup():
    """Wraps the two kernel entry points before transformers imports the Qwen3.5 modeling code."""
    import fla.ops
    import fla.ops.gated_delta_rule as gdr
    import causal_conv1d
    chunk0, conv0 = gdr.chunk_gated_delta_rule, causal_conv1d.causal_conv1d_fn

    @functools.wraps(chunk0)
    def chunk(*a, **kw):
        COUNTS['chunk'] += 1
        if CTX.cu is not None:
            kw['cu_seqlens'] = CTX.cu
        return chunk0(*a, **kw)

    @functools.wraps(conv0)
    def conv(*a, **kw):
        COUNTS['conv'] += 1
        if CTX.seq_idx is not None:
            kw['seq_idx'] = CTX.seq_idx
        return conv0(*a, **kw)

    WRAPPED.update({id(chunk0): chunk, id(conv0): conv})
    return {'chunk_gated_delta_rule': f'{chunk0.__module__}.{chunk0.__qualname__}',
            'causal_conv1d_fn': f'{conv0.__module__}.{conv0.__qualname__}', 'replaced_in': rewire()}


def rewire():
    """Points every module-level name that holds an original kernel entry point at its wrapper."""
    done = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(('fla', 'causal_conv1d', 'transformers.models.qwen3_5', 'unsloth')):
            continue
        for attr, value in list(vars(mod).items()):
            if id(value) in WRAPPED and value is not WRAPPED[id(value)]:
                setattr(mod, attr, WRAPPED[id(value)])
                done.append(f'{name}.{attr}')
    return done


def seg_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query (B, H, L, D), key/value (B, Hkv, L, D) -> (B, L, H, D); causal flash attention inside each segment."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    COUNTS['attention'] += 1
    assert CTX.segments is not None and len(CTX.segments) == query.shape[0], 'segments not set'
    if query.dtype not in (torch.bfloat16, torch.float16):  # the flash backend takes 16-bit inputs only
        ATTN_CAST[0] += 1
        query, key, value = query.to(torch.bfloat16), key.to(torch.bfloat16), value.to(torch.bfloat16)
    rep = query.shape[1] // key.shape[1]
    if rep > 1:  # same head mapping as repeat_kv
        key, value = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)
    out, length = [], query.shape[2]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for r, segs in enumerate(CTX.segments):
            parts, end = [], 0
            for a, b in segs:
                assert a == end
                parts.append(F.scaled_dot_product_attention(query[r:r + 1, :, a:b], key[r:r + 1, :, a:b],
                                                            value[r:r + 1, :, a:b], is_causal=True, scale=scaling))
                end = b
            if end < length:
                parts.append(query.new_zeros(1, query.shape[1], length - end, query.shape[3]))
            out.append(torch.cat(parts, dim=2))
    return torch.cat(out, dim=0).transpose(1, 2).contiguous(), None


def use_seg_attention(model):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register('seg_flash', seg_attention)
    AttentionMaskInterface.register('seg_flash', lambda *a, **k: None)
    before = set()
    for m in model.modules():
        cfg = getattr(m, 'config', None)
        if cfg is not None and hasattr(cfg, '_attn_implementation'):
            before.add(str(cfg._attn_implementation))
            cfg._attn_implementation = 'seg_flash'
    return sorted(before)


# ---------------------------------------------------------------- model and rows

def load_model(torch, fused):
    from unsloth import FastLanguageModel
    import train_lora
    model, _ = FastLanguageModel.from_pretrained(
        model_name=sft_data.TOKENIZER, revision=sft_data.REVISION, max_seq_length=sft_data.LIMIT, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False, device_map={'': 'cuda:0'})
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, bias='none',
                                             target_modules=train_lora.TARGETS, use_gradient_checkpointing='unsloth',
                                             random_state=LORA_SEED)
    FastLanguageModel.for_training(model)
    attn_before = {'attention_before': use_seg_attention(model), 'rewired_after_load': rewire()} if fused else None
    return model, attn_before


def text_parts(model):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    inner = base.model
    return getattr(inner, 'language_model', inner), base.get_output_embeddings(), base.config


def layer_types(config):
    for c in (config, getattr(config, 'text_config', None)):
        if c is not None and getattr(c, 'layer_types', None):
            return list(c.layer_types)
    return None


def row_inputs(torch, examples, idx, kind, fused):
    """One row. kind 'alone' (one example), 'packed' (boundaries) or 'naive' (concatenated, no boundaries).
    Returns the model inputs and sets CTX for the fused stack."""
    ids, labels, pos, seq, segs, cu = [], [], [], [], [], [0]
    for j, i in enumerate(idx):
        x, p = examples[i]['input_ids'], examples[i]['n_prompt']
        a = len(ids)
        ids += x
        labels += [-100] * p + x[p:]
        pos += range(len(x))
        seq += [j] * len(x)
        segs.append((a, a + len(x), p))
        cu.append(a + len(x))
    t_ids = torch.tensor([ids], dtype=torch.long, device='cuda')
    t_lab = torch.tensor([labels], dtype=torch.long, device='cuda')
    kw = {'input_ids': t_ids, 'use_cache': False}
    CTX.segments, CTX.cu, CTX.seq_idx = None, None, None
    if kind == 'packed':
        assert fused
        kw['position_ids'] = torch.tensor([pos], dtype=torch.long, device='cuda')
        CTX.segments = [[(a, b) for a, b, _ in segs]]
        CTX.cu = torch.tensor(cu, dtype=torch.long, device='cuda')
        CTX.seq_idx = torch.tensor([seq], dtype=torch.int32, device='cuda')
    elif fused:  # alone on the fused stack
        assert kind == 'alone' and len(idx) == 1
        CTX.segments = [[(0, len(ids))]]
    else:  # §8 stack: alone or naive, attention mask of ones as in train_lora.py
        kw['attention_mask'] = torch.ones_like(t_ids)
    n = int((t_lab[:, 1:] != -100).sum())
    return kw, t_lab, segs, n


def segment_nll(torch, model, kw, segs):
    """Summed target NLL of each example of a row, from the final hidden states (chunked lm_head, fp32)."""
    import torch.nn.functional as F
    tm, head, _ = text_parts(model)
    ids = kw['input_ids'][0]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        h = tm(**kw).last_hidden_state[0]
        out = []
        for a, b, p in segs:
            s = 0.0
            for c in range(a + p - 1, b - 1, CHUNK):
                e = min(c + CHUNK, b - 1)
                s += float(F.cross_entropy(head(h[c:e]).float(), ids[c + 1:e + 1], reduction='sum'))
            out.append(s)
    return out


def alone_nll(torch, model, examples, i, fused):
    kw, _, segs, _ = row_inputs(torch, examples, [i], 'alone', fused)
    return segment_nll(torch, model, kw, segs)[0]


def check_ids(examples):  # the four §8 check sequences (train_lora.check_loss)
    import train_lora
    return [i for i in sft_data.epoch_order(1) if len(examples[i]['input_ids']) <= 4096][:train_lora.CHECK_EXAMPLES]


def gate_rows(lengths):
    rows = [(k + 1, r + 1, row) for k, s in enumerate(plan(lengths, 'P')[:GATE_STEPS]) for r, row in enumerate(s)]
    return [x for x in rows if len(x[2]) > 1]


def rel(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------- gates

def run_gate(a, fused):
    import torch
    logging.getLogger('transformers').addHandler(FallbackCatch())
    kernels = fused_setup() if fused else None
    tok = sft_data.load_tokenizer()
    examples = sft_data.encode(tok, sft_data.load_group(a.data, GATE_GROUP), GATE_GROUP)
    lengths = [len(x['input_ids']) for x in examples]
    model, attn_before = load_model(torch, fused)
    model.eval()
    out = {'mode': a.mode, 'started_at_utc': now(), 'versions': versions(), 'gpu': torch.cuda.get_device_name(0),
           'kernels': kernels, 'attention_before': attn_before, 'tolerance': TOL, 'group': GATE_GROUP,
           'env_USE_HUB_KERNELS': os.environ.get('USE_HUB_KERNELS')}
    types = layer_types(text_parts(model)[2])
    out['layer_types'] = None if types is None else {t: types.count(t) for t in sorted(set(types))}
    ids = check_ids(examples)
    COUNTS.update(attention=0, chunk=0, conv=0)
    out['check'] = [{'id': examples[i]['id'], 'nll': alone_nll(torch, model, examples, i, fused)} for i in ids]
    alone = {}
    rows = gate_rows(lengths)
    out['rows'] = []
    for step, r, row in rows:
        for i in row:
            if i not in alone:
                alone[i] = alone_nll(torch, model, examples, i, fused)
        kw, labels, segs, n = row_inputs(torch, examples, row, 'packed' if fused else 'naive', fused)
        packed = segment_nll(torch, model, kw, segs)
        rec = {'step': step, 'row': r, 'tokens': segs[-1][1],
               'examples': [{'id': examples[i]['id'], 'alone': alone[i], 'in_row': v, 'rel': rel(v, alone[i])}
                            for i, v in zip(row, packed)]}
        if fused:  # the training call on the packed row
            kw, labels, segs, n = row_inputs(torch, examples, row, 'packed', fused)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**kw, labels=labels).loss
            total = sum(alone[i] for i in row)
            rec['training_call'] = {'loss_sum': float(loss) * n, 'alone_sum': total, 'rel': rel(float(loss) * n, total)}
        out['rows'].append(rec)
        print('GATE_ROW', json.dumps({k: rec[k] for k in ('step', 'row', 'tokens')}),
              max(e['rel'] for e in rec['examples']), flush=True)
    worst = max(e['rel'] for x in out['rows'] for e in x['examples'])
    out['max_example_rel'] = worst
    out['fallback_warnings'] = FALLBACK
    out['attention_inputs_cast_to_bf16'] = ATTN_CAST[0]
    if fused:
        # gate 1: one forward pass of the first check sequence, counted
        COUNTS.update(attention=0, chunk=0, conv=0)
        alone_nll(torch, model, examples, ids[0], fused)
        out['counts_one_forward'] = dict(COUNTS)
        n_full = None if types is None else types.count('full_attention')
        n_lin = None if types is None else types.count('linear_attention')
        g1 = (not FALLBACK and n_full and n_lin and COUNTS['attention'] == n_full
              and COUNTS['chunk'] == n_lin and COUNTS['conv'] == n_lin)
        ref = json.loads(Path(a.ref).read_text())
        ref_nll = {c['id']: c['nll'] for c in ref['check']}
        out['check_vs_ref'] = [{'id': c['id'], 'fused': c['nll'], 'ref': ref_nll[c['id']],
                                'rel': rel(c['nll'], ref_nll[c['id']])} for c in out['check']]
        g2 = all(c['rel'] <= TOL for c in out['check_vs_ref'])
        g3 = bool(out['rows']) and worst <= TOL and all(x['training_call']['rel'] <= TOL for x in out['rows'])
        out['gates'] = {'1_kernels_in_use': bool(g1), '2_same_function': g2, '3_isolation': g3}
        out['arms_allowed'] = ['R'] + (['F'] + (['P'] if g3 else []) if g1 and g2 else [])
    else:
        out['control_detects_leakage'] = worst > TOL
    out['ended_at_utc'] = now()
    save(a.out, out)
    print('GATE_DONE', json.dumps({k: out.get(k) for k in ('mode', 'gates', 'arms_allowed', 'max_example_rel',
                                                            'counts_one_forward', 'control_detects_leakage')}), flush=True)


# ---------------------------------------------------------------- timed runs

def run_timed(a):
    import resource
    import torch
    import train_lora
    logging.getLogger('transformers').addHandler(FallbackCatch())
    kernels = fused_setup()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report_p = out / 'train_report.json'
    assert not report_p.exists(), 'A report exists here: inspect it instead of rerunning.'
    tok = sft_data.load_tokenizer()
    examples = sft_data.encode(tok, sft_data.load_group(a.data, a.group), a.group)
    lengths = [len(x['input_ids']) for x in examples]
    pad_id = tok.convert_tokens_to_ids('<|endoftext|>')
    steps = plan(lengths, a.mode)
    assert [sorted(i for m in s for i in m) for s in steps] == \
        [sorted(i for m in s for i in m) for s in plan(lengths, 'R')], 'arms must see the same examples per step'
    report = {'state': 'loading', 'arm': a.mode, 'started_at_utc': now(), 'group': a.group,
              'model': sft_data.TOKENIZER, 'revision': sft_data.REVISION, 'data_sha256': sft_data.GROUPS[a.group][1],
              'script_sha256': {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
                                for n in ('pack_bench.py', 'train_lora.py', 'sft_data.py')},
              'gpu': torch.cuda.get_device_name(0), 'kernels': kernels,
              'hyper': {'lr': LR, 'epochs': 1, 'warmup_steps': WARMUP, 'lora_seed': LORA_SEED, 'order_seed': ORDER_SEED,
                        'packing': 'first-fit rows of at most %d tokens, one row per pass' % BUDGET if a.mode == 'P'
                        else 'none; right padding inside a microbatch (as §8)'},
              'versions': versions(), 'env_USE_HUB_KERNELS': os.environ.get('USE_HUB_KERNELS'),
              'steps': [], 'epochs_done': []}
    save(report_p, report)
    t = time.perf_counter()
    model, report['attention_before'] = load_model(torch, True)
    report['load_seconds'] = time.perf_counter() - t
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    report['trainable_parameters'] = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / WARMUP))
    report['state'] = 'training'
    save(report_p, report)
    device = train_lora.DevicePeak(torch)
    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.perf_counter()
    for k, micro in enumerate(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        parts = []
        for m in micro:
            if a.mode == 'P':
                kw, labels, segs, n = row_inputs(torch, examples, m, 'packed', True)
                parts.append((kw, labels, n, (CTX.segments, CTX.cu, CTX.seq_idx), segs[-1][1], kw['input_ids'].numel()))
            else:
                ids, mask, labels, n = train_lora.batch(torch, examples, m, pad_id)
                segs = [[(0, len(examples[i]['input_ids']))] for i in m]
                parts.append(({'input_ids': ids, 'use_cache': False}, labels, n, (segs, None, None),
                              int(mask.sum()), ids.numel()))
        n_step = sum(p[2] for p in parts)
        total = 0.0
        for kw, labels, n, ctx, _, _ in parts:
            CTX.segments, CTX.cu, CTX.seq_idx = ctx
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**kw, labels=labels).loss
            assert torch.isfinite(loss).item()
            (loss * n / n_step).backward()
            total += float(loss.detach()) * n
            del loss
        CTX.segments, CTX.cu, CTX.seq_idx = None, None, None
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lr = sched.get_last_lr()[0]
        optimizer.step()
        sched.step()
        torch.cuda.synchronize()
        row = {'epoch': 1, 'step': k + 1, 'global_step': k + 1, 'lr': lr, 'loss': total / n_step,
               'grad_norm': float(norm), 'examples': sum(len(m) for m in micro), 'microbatches': len(micro),
               'real_tokens': sum(p[4] for p in parts), 'padded_tokens': sum(p[5] for p in parts),
               'loss_tokens': n_step, 'seconds': time.perf_counter() - start,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
               'device_used_peak_gib_sampled': device.peak / 1024 ** 3,
               'host_peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
        if k == 0:
            report['counts_after_step1'] = dict(COUNTS)
            assert COUNTS['attention'] and COUNTS['chunk'] and COUNTS['conv'] and not FALLBACK, \
                ('fused path not used', COUNTS, FALLBACK)
        report['steps'].append(row)
        print('TRAIN_STEP', json.dumps(row), flush=True)
        del parts
        if (k + 1) % 10 == 0:
            save(report_p, report)
    rows = report['steps']
    report['epochs_done'].append({'tag': 'epoch1', 'steps': len(rows), 'seconds': time.perf_counter() - epoch_start,
                                  'train_seconds': sum(r['seconds'] for r in rows),
                                  'real_tokens': sum(r['real_tokens'] for r in rows),
                                  'mean_loss': sum(r['loss'] * r['loss_tokens'] for r in rows)
                                  / sum(r['loss_tokens'] for r in rows)})
    device.stop.set()
    report['counts_total'] = dict(COUNTS)
    report['fallback_warnings'] = FALLBACK
    report['attention_inputs_cast_to_bf16'] = ATTN_CAST[0]
    report['state'] = 'finished'
    report['ended_at_utc'] = now()
    save(report_p, report)
    print('TRAIN_FINISHED', json.dumps(report['epochs_done'][-1]), flush=True)
    gc.collect()


# ---------------------------------------------------------------- summary (CPU)

def summarize(a):
    runs, gates = Path(a.runs), Path(a.gates)
    read = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    out = {'gates': read(gates / 'gate.json'), 'gate_ref': read(gates / 'ref.json'), 'groups': {}}
    for k in ('gates', 'gate_ref'):
        if out[k]:
            out[k] = {x: out[k].get(x) for x in ('gates', 'arms_allowed', 'max_example_rel', 'control_detects_leakage',
                                                 'counts_one_forward', 'check_vs_ref', 'kernels', 'versions',
                                                 'attention_before', 'fallback_warnings')}
    for g in ('2', '3'):
        arms = {}
        for arm in 'RFP':
            r = read(runs / f'g{g}_{arm}' / 'train_report.json')
            if r is None or r.get('state') != 'finished':
                arms[arm] = None
                continue
            steps = [s for s in r['steps'] if s['epoch'] == 1]
            e = r['epochs_done'][0]
            arms[arm] = {'train_seconds': e['train_seconds'], 'real_tokens': e['real_tokens'],
                         'real_tokens_per_s': e['real_tokens'] / e['train_seconds'], 'steps': len(steps),
                         'forward_passes': sum(s['microbatches'] for s in steps),
                         'processed_tokens': sum(s['padded_tokens'] for s in steps),
                         'peak_allocated_gib': max(s['peak_allocated_gib'] for s in steps),
                         'device_used_peak_gib_sampled': max(s['device_used_peak_gib_sampled'] for s in steps),
                         'load_seconds': r.get('load_seconds'), 'mean_loss': e['mean_loss'],
                         '_t': [s['seconds'] for s in steps], '_loss': [s['loss'] for s in steps]}
        res = {'arms': arms, 'ratios': {}}
        for name, x, y in (('R_over_F', 'R', 'F'), ('F_over_P', 'F', 'P'), ('R_over_P', 'R', 'P')):
            if arms[x] and arms[y]:
                tx, ty = arms[x]['_t'], arms[y]['_t']
                assert len(tx) == len(ty)
                dl = [abs(p - q) for p, q in zip(arms[x]['_loss'], arms[y]['_loss'])]
                res['ratios'][name] = {'train_seconds': arms[x]['train_seconds'] / arms[y]['train_seconds'],
                                       'median_step_ratio_after_5': statistics.median(p / q for p, q in
                                                                                      zip(tx[5:], ty[5:])),
                                       'loss_abs_diff_step1': dl[0], 'loss_abs_diff_max': max(dl),
                                       'loss_abs_diff_mean': sum(dl) / len(dl)}
        if a.record and arms['R']:
            rec = read(Path(a.record) / f'g{g}_lr2e-4' / 'train_report.json')
            if rec:
                t0 = rec['epochs_done'][0]['train_seconds']
                res['R_vs_record'] = {'record_train_seconds': t0, 'this_session': arms['R']['train_seconds'],
                                      'difference': arms['R']['train_seconds'] / t0 - 1,
                                      'session_effect': abs(arms['R']['train_seconds'] / t0 - 1) > 0.10}
        for v in arms.values():
            if v:
                v.pop('_t'), v.pop('_loss')
        out['groups'][g] = res
    save(a.out, out)
    for g, res in out['groups'].items():
        print('GROUP', g, json.dumps({k: (v and round(v['train_seconds'], 1)) for k, v in res['arms'].items()}),
              json.dumps({k: round(v['train_seconds'], 3) for k, v in res['ratios'].items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['gate-ref', 'gate', 'F', 'P', 'summary'])
    p.add_argument('--data')
    p.add_argument('--group', choices=['2', '3'])
    p.add_argument('--ref')
    p.add_argument('--runs')
    p.add_argument('--gates')
    p.add_argument('--record')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        if args.mode == 'summary':
            summarize(args)
        elif args.mode in ('gate-ref', 'gate'):
            run_gate(args, fused=args.mode == 'gate')
        else:
            run_timed(args)
    except Exception:
        err = Path(args.out)
        err = (err if args.mode in ('F', 'P') else err.parent)
        err.mkdir(parents=True, exist_ok=True)
        (err / f'error_{args.mode}.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
