"""BF16 LoRA training of Qwen3.5-9B-Base on one training group (Colab A100, the stack of the capacity test).

Adapter: r 16, alpha 16, dropout 0 on the attention, MLP and linear-attention projections, Unsloth gradient
checkpointing, LoRA init seed = --seed (same as the capacity test); --order-seed sets the question order (default
sft_data.ORDER_SEED, the order of the tuning runs), while the four check questions always come from the default order.
Optimizer: AdamW, weight decay 0, gradient clip 1.0, linear warmup over --warmup steps, then a constant learning
rate, so the adapter saved after epoch k is what a k-epoch run would give and one run covers the epoch grid.
Loss: token-level mean over all supervised tokens of a step (sft_data.py builds the sequences, masks and
microbatches).

--max-steps N stops early (smoke test). --single puts one example in each microbatch (no padding at all).
--padding-check first compares, on the first padded microbatch, the summed loss of the padded batch with the summed
loss of its examples run one at a time. Each saved adapter directory also gets check_loss.json (the target NLL of
four fixed questions), which dev_eval.py reproduces from the merged weights.

usage: python -u train_lora.py --data DIR --group 3 --lr 1e-4 --epochs 3 --out DIR [--seed N] [--order-seed N]
       [--max-steps N] [--padding-check] [--single]
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft_data  # noqa: E402

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
           'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']
CHECK_EXAMPLES = 4


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


class DevicePeak:
    """Samples whole-device used memory every 0.2 s (includes the CUDA context and allocator slack)."""
    def __init__(self, torch):
        self.torch, self.peak, self.stop = torch, 0, threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while not self.stop.wait(.2):
            free, total = self.torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)


def batch(torch, examples, idx, pad_id):
    width = max(len(examples[i]['input_ids']) for i in idx)
    ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), width), -100, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for row, i in enumerate(idx):
        x = examples[i]['input_ids']
        ids[row, :len(x)] = torch.tensor(x)
        labels[row, examples[i]['n_prompt']:len(x)] = torch.tensor(x[examples[i]['n_prompt']:])
        mask[row, :len(x)] = 1
    n_loss = int((labels[:, 1:] != -100).sum())  # the model shifts labels by one
    return ids.cuda(), mask.cuda(), labels.cuda(), n_loss


def summed_loss(torch, model, examples, idx, pad_id):
    ids, mask, labels, n = batch(torch, examples, idx, pad_id)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    return loss, n


def padding_check(torch, model, examples, steps, pad_id):
    micro = next((m for s in steps for m in s if len(m) > 1), None)
    if micro is None:
        return {'skipped': 'no microbatch with more than one example'}
    with torch.no_grad():
        loss, n = summed_loss(torch, model, examples, micro, pad_id)
        padded = float(loss) * n
        single = 0.0
        for i in micro:
            l1, n1 = summed_loss(torch, model, examples, [i], pad_id)
            single += float(l1) * n1
    rel = abs(padded - single) / abs(single)
    return {'examples': len(micro), 'lengths': [len(examples[i]['input_ids']) for i in micro],
            'padded_sum': padded, 'one_at_a_time_sum': single, 'relative_difference': rel, 'pass': rel < 1e-2}


def check_loss(torch, model, examples, pad_id):
    """Summed target NLL of the first CHECK_EXAMPLES epoch-1 questions of at most 4,096 tokens, one at a time.
    dev_eval.py recomputes it from the merged weights in vLLM to show that the merge and the vLLM load reproduce
    the trained adapter."""
    rows = []
    picked = [i for i in sft_data.epoch_order(1) if len(examples[i]['input_ids']) <= 4096][:CHECK_EXAMPLES]
    with torch.no_grad():
        for i in picked:
            loss, n = summed_loss(torch, model, examples, [i], pad_id)
            rows.append({'id': examples[i]['id'], 'nll_sum': float(loss) * n, 'tokens': n,
                         'n_prompt': examples[i]['n_prompt'], 'input_ids': examples[i]['input_ids']})
    return rows


def run(a):
    import importlib.metadata as metadata
    from unsloth import FastLanguageModel
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report_p = out / 'train_report.json'
    assert not report_p.exists(), 'A report exists here: inspect it instead of rerunning.'
    tok = sft_data.load_tokenizer()
    examples = sft_data.encode(tok, sft_data.load_group(a.data, a.group), a.group)
    lengths = [len(x['input_ids']) for x in examples]
    pad_id = tok.convert_tokens_to_ids('<|endoftext|>')
    plans = {e: sft_data.step_plan(lengths, e, a.order_seed) for e in range(1, a.epochs + 1)}
    if a.single:  # one example per microbatch (no padding); same steps and loss
        plans = {e: [[[i] for m in s for i in m] for s in p] for e, p in plans.items()}
    total_steps = sum(len(p) for p in plans.values())

    def version(n):
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            return None
    report = {'state': 'loading', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'group': a.group, 'model': sft_data.TOKENIZER, 'revision': sft_data.REVISION,
              'data_sha256': sft_data.GROUPS[a.group][1],
              'script_sha256': {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
                                for n in ('train_lora.py', 'sft_data.py')},
              'gpu': torch.cuda.get_device_name(0), 'physical_gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
              'hyper': {'lr': a.lr, 'epochs': a.epochs, 'warmup_steps': a.warmup, 'schedule': 'linear warmup, then constant',
                        'optimizer': 'AdamW betas (0.9, 0.999) eps 1e-8', 'weight_decay': 0.0, 'grad_clip': 1.0,
                        'step_examples': sft_data.STEP_EXAMPLES, 'token_budget': sft_data.TOKEN_BUDGET,
                        'microbatch_overhead': sft_data.MICROBATCH_OVERHEAD,
                        'order_seed': a.order_seed, 'lora_seed': a.seed, 'rank': 16, 'alpha': 16, 'dropout': 0.0,
                        'targets': TARGETS, 'precision': 'bfloat16', 'gradient_checkpointing': 'unsloth',
                        'packing': 'none; ' + ('one example per microbatch' if a.single else 'right padding inside a microbatch, masked')},
              'max_steps': a.max_steps, 'steps_per_epoch': {e: len(p) for e, p in plans.items()},
              'versions': {n: version(n) for n in ['torch', 'transformers', 'peft', 'unsloth', 'unsloth_zoo', 'xformers']},
              'steps': [], 'epochs_done': []}
    save(report_p, report)
    t = time.perf_counter()
    model, _ = FastLanguageModel.from_pretrained(
        model_name=sft_data.TOKENIZER, revision=sft_data.REVISION, max_seq_length=sft_data.LIMIT, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False, device_map={'': 'cuda:0'})
    report['load_seconds'] = time.perf_counter() - t
    assert max(max(x['input_ids']) for x in examples) < model.get_input_embeddings().num_embeddings
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, bias='none',
                                             target_modules=TARGETS, use_gradient_checkpointing='unsloth',
                                             random_state=a.seed)
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    assert not any(('visual' in n.lower() or 'vision' in n.lower()) and p.requires_grad for n, p in model.named_parameters())
    report['trainable_parameters'] = sum(p.numel() for p in trainable)
    FastLanguageModel.for_training(model)
    report['check_loss_before_training'] = [{k: r[k] for k in ('id', 'nll_sum', 'tokens')}
                                            for r in check_loss(torch, model, examples, pad_id)]
    if a.padding_check:
        report['padding_check'] = padding_check(torch, model, examples, plans[1], pad_id)
        save(report_p, report)
        print('PADDING_CHECK', json.dumps(report['padding_check']), flush=True)
        assert report['padding_check'].get('pass', True), 'Padded and one-at-a-time losses differ.'
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / a.warmup))
    report['state'] = 'training'
    save(report_p, report)
    device = DevicePeak(torch)
    torch.cuda.reset_peak_memory_stats()
    done = 0
    for epoch in range(1, a.epochs + 1):
        epoch_start = time.perf_counter()
        for k, micro in enumerate(plans[epoch]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            parts = [batch(torch, examples, m, pad_id) for m in micro]
            n_step = sum(p[3] for p in parts)
            total = 0.0
            for ids, mask, labels, n in parts:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
                assert torch.isfinite(loss).item()
                (loss * n / n_step).backward()
                total += float(loss.detach()) * n
                del loss
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            lr = sched.get_last_lr()[0]
            optimizer.step()
            sched.step()
            torch.cuda.synchronize()
            done += 1
            row = {'epoch': epoch, 'step': k + 1, 'global_step': done, 'lr': lr, 'loss': total / n_step,
                   'grad_norm': float(norm), 'examples': sum(len(m) for m in micro), 'microbatches': len(micro),
                   'real_tokens': sum(int(p[1].sum()) for p in parts), 'padded_tokens': sum(p[0].numel() for p in parts),
                   'loss_tokens': n_step, 'seconds': time.perf_counter() - start,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
                   'device_used_peak_gib_sampled': device.peak / 1024 ** 3,
                   'host_peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
            report['steps'].append(row)
            print('TRAIN_STEP', json.dumps(row), flush=True)
            del parts
            if done % 10 == 0:
                save(report_p, report)
            if a.max_steps and done >= a.max_steps:
                break
        tag = f'epoch{epoch}' if not (a.max_steps and done >= a.max_steps) else f'step{done}'
        model.save_pretrained(str(out / tag))
        save(out / tag / 'check_loss.json', check_loss(torch, model, examples, pad_id))
        rows = [r for r in report['steps'] if r['epoch'] == epoch]
        report['epochs_done'].append({'tag': tag, 'steps': len(rows), 'seconds': time.perf_counter() - epoch_start,
                                      'train_seconds': sum(r['seconds'] for r in rows),
                                      'real_tokens': sum(r['real_tokens'] for r in rows),
                                      'mean_loss': sum(r['loss'] * r['loss_tokens'] for r in rows) / sum(r['loss_tokens'] for r in rows)})
        save(report_p, report)
        print('EPOCH_SAVED', json.dumps(report['epochs_done'][-1]), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if a.max_steps and done >= a.max_steps:
            break
    device.stop.set()
    report['state'] = 'finished'
    report['total_planned_steps'] = total_steps
    report['ended_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(report_p, report)
    print('TRAIN_FINISHED', json.dumps({k: report[k] for k in ('group', 'epochs_done', 'load_seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--group', required=True, choices=sorted(sft_data.GROUPS))
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=91432026)
    p.add_argument('--order-seed', type=int, default=sft_data.ORDER_SEED)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--padding-check', action='store_true')
    p.add_argument('--single', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'train_error.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
