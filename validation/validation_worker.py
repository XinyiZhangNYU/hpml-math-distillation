import os
import sys
os.environ['PATH'] = str(__import__('pathlib').Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
import json
import time
import traceback
from pathlib import Path
from eval_common import ROOT, MODEL, REVISION, SEED, PROMPT, PARAMS, write_json, parser_checks, prepare_data, load_rows, grade, summarize


def run_vllm():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    parser_checks()
    rows = prepare_data() if not (ROOT/'validation_100.jsonl').exists() else load_rows()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    protocol = {'model': MODEL, 'revision': REVISION, 'seed': SEED, 'sampling': PARAMS,
                'validation_size': 100, 'final_test_accessed': False,
                'engine': 'vllm', 'dtype': 'bfloat16', 'max_model_len': 8192,
                'max_num_seqs': 16, 'gpu_memory_utilization': 0.85, 'enforce_eager': False,
                'prefix_caching': False, 'language_model_only': True,
                'time_limit_seconds': 1800, 'training': 'No training or model adaptation in this test.'}
    write_json('protocol.json', protocol)
    started = time.perf_counter()
    llm = LLM(model=MODEL, revision=REVISION, dtype='bfloat16',
              max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=8192,
              gpu_memory_utilization=0.85, enforce_eager=False,
              enable_prefix_caching=False, language_model_only=True,
              seed=SEED, disable_log_stats=True)
    load_seconds = time.perf_counter()-started
    warm_prompt = tok.apply_chat_template([{'role':'user','content':'Calculate 17 times 23. Put the final answer in \\boxed{}.'}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for i in range(2):
        t = time.perf_counter()
        out = llm.generate([warm_prompt], SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)[0]
        print('VLLM_WARMUP', i, 'seconds', time.perf_counter()-t, 'tokens', len(out.outputs[0].token_ids), flush=True)
    completed = {}
    all_start = time.perf_counter()
    for mode in [False, True]:
        mode_name = 'thinking' if mode else 'nonthinking'
        collected = []
        mode_wall = 0.0
        for begin in range(0, len(rows), 16):
            if time.perf_counter()-all_start > 1800:
                print('VALIDATION_TIME_LIMIT', flush=True)
                break
            batch = rows[begin:begin+16]
            prompts = [tok.apply_chat_template([{'role':'user','content':PROMPT+x['problem']}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=mode) for x in batch]
            params = SamplingParams(**PARAMS, seed=SEED+begin)
            t = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter()-t
            mode_wall += elapsed
            for case, result in zip(batch, outputs):
                output = result.outputs[0]
                scored = grade(output.text, case['gold'])
                record = {k:case[k] for k in ['id','subject','level','gold','problem_sha256']}
                record.update(mode=mode_name, text=output.text, input_tokens=len(result.prompt_token_ids),
                              output_tokens=len(output.token_ids), finish_reason=output.finish_reason,
                              batch_seconds=elapsed, **scored)
                collected.append(record)
            (ROOT/(mode_name+'_outputs.jsonl')).write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in collected))
            summary = summarize(collected, mode_wall)
            completed[mode_name] = summary
            write_json('vllm_summary.json', {'load_seconds':load_seconds, 'modes':completed,
                       'complete':all(x.get('n')==100 for x in completed.values()) and len(completed)==2})
            print('VALIDATION_PROGRESS', mode_name, len(collected), 'of',len(rows),
                  'accuracy',round(summary['accuracy'],4),'tokens',summary['total_output_tokens'],
                  'wall_seconds',round(mode_wall,2),flush=True)
    # Same five batch-one prompts separate latency from the baseline batch throughput.
    probes = rows[:5]
    latency = []
    for case in probes:
        prompt = tok.apply_chat_template([{'role':'user','content':PROMPT+case['problem']}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
        t = time.perf_counter()
        output = llm.generate([prompt], SamplingParams(temperature=0,max_tokens=512),use_tqdm=False)[0].outputs[0]
        elapsed = time.perf_counter()-t
        latency.append({'id':case['id'],'seconds':elapsed,'tokens':len(output.token_ids),
                        'text':output.text,'finish_reason':output.finish_reason})
    write_json('vllm_batch1_probes.json', latency)
    print('VLLM_VALIDATION_FINISHED', json.dumps(completed), flush=True)


def run_hf():
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
            dtype=torch.bfloat16, device_map={'':'cuda:0'}, low_cpu_mem_usage=True,
            attn_implementation='sdpa')
    model.eval()
    write_json('hf_load.json',{'seconds':time.perf_counter()-start,'torch':torch.__version__,
            'scope':'Unmodified BF16 checkpoint, SDPA, optional GDN packages as installed; includes frozen vision weights.'})
    rows = load_rows()[:5]
    outputs=[]
    deadline=time.perf_counter()+300
    for index,case in enumerate([{'id':'warmup','problem':'Calculate 17 times 23.'}]*2+rows):
        if index>=2 and time.perf_counter()>deadline:
            break
        prompt=tok.apply_chat_template([{'role':'user','content':PROMPT+case['problem']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=tok(prompt,return_tensors='pt').to('cuda')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t=time.perf_counter()
        with torch.inference_mode():
            result=model.generate(**encoded,max_new_tokens=64 if index<2 else 512,
                     do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        ids=result[0,encoded['input_ids'].shape[1]:].tolist()
        record={'id':case['id'],'warmup':index<2,'seconds':elapsed,'tokens':len(ids),
                'text':tok.decode(ids,skip_special_tokens=True),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'host_ram_total_gib':__import__('psutil').virtual_memory().total/1024**3}
        outputs.append(record)
        write_json('hf_batch1_probes.json',outputs)
        print('HF_PROBE',index,json.dumps({k:v for k,v in record.items() if k!='text'}),flush=True)
    print('HF_VALIDATION_FINISHED',flush=True)


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    try:
        stage=sys.argv[1]
        if stage=='vllm':run_vllm()
        elif stage=='hf':run_hf()
        else:raise ValueError(stage)
    except Exception:
        write_json(sys.argv[1]+'_error.json',{'traceback':traceback.format_exc()})
        raise
