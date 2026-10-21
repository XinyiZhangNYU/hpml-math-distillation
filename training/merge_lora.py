"""Merges a LoRA adapter into the pinned BF16 base checkpoint, shard by shard: W' = W + (alpha / r) * B @ A.

Every adapter module must map to exactly one base tensor with matching shape; anything unmatched stops the merge.
The merge is done in float32 and cast back to the base dtype. Non-weight files (config, tokenizer, processor) are
copied from the base snapshot unchanged, so vLLM loads the merged model exactly as it loads the base model.

usage: python merge_lora.py ADAPTER_DIR OUT_DIR
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-9B-Base'
REVISION = '68c46c4b3498877f3ef123c856ecfde50c39f404'
EXPECTED_MODULES = 248
EXPECTED_LORA_PARAMS = 43278336


def main(adapter_dir, out_dir):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    adapter_dir, out = Path(adapter_dir), Path(out_dir)
    assert not out.exists(), 'Output exists: inspect it instead of merging again.'
    cfg = json.loads((adapter_dir / 'adapter_config.json').read_text())
    assert cfg['r'] == 16 and cfg['lora_alpha'] == 16 and not cfg.get('use_dora') and not cfg.get('use_rslora'), cfg
    scale = cfg['lora_alpha'] / cfg['r']
    lora = load_file(str(adapter_dir / 'adapter_model.safetensors'))
    pairs = {}
    for k in lora:
        m = re.fullmatch(r'base_model\.model\.(.+)\.lora_([AB])\.weight', k)
        assert m, ('unexpected adapter tensor', k)
        pairs.setdefault(m.group(1) + '.weight', {})[m.group(2)] = lora[k]
    assert all(set(v) == {'A', 'B'} for v in pairs.values())
    # 32 MLPs x 3 + 24 linear-attention layers x 5 + 8 full-attention layers x 4; with r 16 these hold the
    # 43,278,336 trainable parameters counted in the capacity test.
    assert len(pairs) == EXPECTED_MODULES, len(pairs)
    assert sum(v['A'].numel() + v['B'].numel() for v in pairs.values()) == EXPECTED_LORA_PARAMS
    base = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model']))
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    if not any(k in index for k in pairs):  # adapter saved from the text-only class: model.layers.* -> model.language_model.layers.*
        pairs = {re.sub(r'^model\.layers\.', 'model.language_model.layers.', k): v for k, v in pairs.items()}
        report_names = 'text-only names mapped to model.language_model.*'
    else:
        report_names = 'names match the checkpoint'
    missing = sorted(k for k in pairs if k not in index)
    assert not missing, ('adapter modules with no base tensor', len(missing), missing[:5])
    out.mkdir(parents=True)
    report = {'adapter_dir': str(adapter_dir), 'scale': scale, 'modules': len(pairs), 'names': report_names,
              'modules_by_type': {t: sum(k.endswith('.' + t + '.weight') for k in pairs) for t in
                                  sorted({k.split('.')[-2] for k in pairs})}, 'shards': {}}
    assert not any('.visual.' in k or k.startswith('mtp.') for k in pairs), 'adapter touches vision or MTP weights'
    for shard in sorted(set(index.values())):
        tensors = load_file(str(base / shard))
        merged = 0
        for k in [k for k in tensors if k in pairs]:
            w, a, b = tensors[k], pairs[k]['A'], pairs[k]['B']
            assert w.shape == (b.shape[0], a.shape[1]), (k, w.shape, a.shape, b.shape)
            tensors[k] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype).contiguous()
            merged += 1
        save_file(tensors, str(out / shard), metadata={'format': 'pt'})
        report['shards'][shard] = {'tensors': len(tensors), 'merged': merged}
        del tensors
    assert sum(s['merged'] for s in report['shards'].values()) == len(pairs)
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith('.safetensors'):
            shutil.copyfile(p, out / p.name)
    report['adapter_sha256'] = hashlib.sha256((adapter_dir / 'adapter_model.safetensors').read_bytes()).hexdigest()
    (out / 'merge_report.json').write_text(json.dumps(report, indent=2))
    print('MERGED', json.dumps({'modules': len(pairs), 'shards': len(report['shards'])}), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
