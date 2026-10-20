"""Base-model capability probe: few-shot MATH dev and MBPP validation, no training."""
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
ROOT = Path(__file__).resolve().parent
MODELS = {'9b-base':('Qwen/Qwen3.5-9B-Base','68c46c4b3498877f3ef123c856ecfde50c39f404'),
          '4b-base':('Qwen/Qwen3.5-4B-Base','1001bb4d826a52d1f399e183466143f4da7b741b')}
STUDENT = os.environ.get('PROBE_STUDENT','9b-base')
MODEL, REVISION = MODELS[STUDENT]
SEED = 91432026
CONTEXT, CONCURRENCY = 8192, 32
MATH_CAP, CODE_CAP = 4096, 1024
PREFIX_BUDGETS = [1024, 2048, 4096]
# Primary metric (pre-registered): mean accuracy of the two T=0.6 samples. Greedy is a reference.
SETTINGS = [('greedy', dict(temperature=0.0), 1),
            ('t0.6', dict(temperature=0.6, top_p=0.95, top_k=20), 2)]
DECISION_RULE = {'primary_metric':'t0.6 mean accuracy over 2 seeds',
    'student_band':[0.20, 0.70], 'teacher_minus_student_min_pp':20,
    'tie_break':['prefer math (validated pipeline)', 'prefer the student with the larger teacher gap']}

MATH_SHOTS = [
    (r'Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.',
     r'The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.'),
    (r'If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$',
     r'We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$'),
    (r'Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?',
     r'If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \begin{align*} 30n&=480\\ \Rightarrow\qquad n&=480/30=\boxed{16} \end{align*}'),
    (r'If the system of equations \begin{align*} 6x-4y&=a,\\ 6y-9x &=b. \end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ assuming $b$ is nonzero.',
     r'If we multiply the first equation by $-\frac{3}{2}$, we obtain $$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$'),
]
MATH_STOP = ['\nProblem:']
CODE_STOP = ['[DONE]', '\nYou are an expert Python programmer']


def save(name, value):
    p = ROOT/name
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(p)


def boxed(text):
    found = None
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    found = text[start:i]; break
    return found


def math_grade(text, gold):
    from math_verify import parse, verify, LatexExtractionConfig
    def parsed(s):
        a = boxed(s)
        return a, (parse('$'+a+'$', extraction_config=[LatexExtractionConfig()]) if a is not None else [])
    answer, pred = parsed(text)
    _, expected = parsed('\\boxed{'+str(gold)+'}')
    if not expected: raise ValueError('Unparseable reference: '+str(gold))
    try: correct = bool(pred and verify(expected, pred))
    except Exception: correct = False
    return {'extracted_answer':answer, 'has_boxed':answer is not None, 'correct':correct}


def math_prompt(problem):
    shots = ''.join(f'Problem:\n{p}\n\nSolution:\n{s}\n\n' for p, s in MATH_SHOTS)
    return shots+f'Problem:\n{problem}\n\nSolution:\n'


def code_block(row, with_code):
    tests = '\n'.join(row['test_list'])
    head = f"You are an expert Python programmer, and here is your task: {row['text']} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    return head+(row['code'].replace('\r\n','\n').strip()+'\n[DONE]\n\n' if with_code else '')


def code_prompt(row, shots):
    return ''.join(code_block(s, True) for s in shots)+code_block(row, False)


def extract_code(text):
    text = text.split('[DONE]')[0]
    fence = re.search(r'```(?:python)?\n(.*?)```', text, re.S)
    return (fence.group(1) if fence else text).strip()


def run_tests(code, row, timeout=10):
    program = (row.get('test_setup_code') or '')+'\n'+code+'\n\n'+'\n'.join(row['test_list'])+'\n'
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'candidate.py'
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable,'-I',str(path)], cwd=d, capture_output=True, text=True, timeout=timeout)
            return {'passed':r.returncode==0, 'error':None if r.returncode==0 else r.stderr[-400:]}
        except subprocess.TimeoutExpired:
            return {'passed':False, 'error':'timeout'}


def summarize(records, wall, task):
    n = len(records)
    out = {'n':n, 'correct':sum(r['correct'] for r in records), 'accuracy':sum(r['correct'] for r in records)/n,
        'truncated':sum(r['finish_reason']=='length' for r in records),
        'output_tokens':sum(r['output_tokens'] for r in records),
        'mean_output_tokens':sum(r['output_tokens'] for r in records)/n,
        'generation_wall_seconds':wall,
        'aggregate_output_tokens_per_second':sum(r['output_tokens'] for r in records)/wall}
    samples = sorted({r['sample'] for r in records})
    out['per_sample_accuracy'] = [sum(r['correct'] for r in records if r['sample']==s)/sum(r['sample']==s for r in records) for s in samples]
    ids = sorted({r['id'] for r in records}, key=str)
    out['any_sample_correct'] = sum(any(r['correct'] for r in records if r['id']==i) for i in ids)/len(ids)
    if task == 'math':
        out['has_boxed'] = sum(r['has_boxed'] for r in records)
        out['by_level'] = {str(l):{'n':sum(r['level']==l for r in records),
            'correct':sum(r['correct'] for r in records if r['level']==l)} for l in range(1,6)}
    return out


def run():
    import importlib.metadata as metadata
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tag = STUDENT
    assert not (ROOT/f'{tag}_summary.json').exists(), 'Inspect existing results instead of duplicating a run.'
    math_rows = [json.loads(s) for s in (ROOT/'math_dev_100.jsonl').read_text().splitlines()]
    code_rows = [json.loads(s) for s in (ROOT/'mbpp_dev_90.jsonl').read_text().splitlines()]
    shots = json.loads((ROOT/'mbpp_shots.json').read_text())
    assert len(math_rows)==100 and len(code_rows)==90 and len(shots)==3
    identity = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu':torch.cuda.get_device_name(0), 'physical_gpu_bytes':torch.cuda.get_device_properties(0).total_memory,
        'versions':{n:metadata.version(n) for n in ['torch','transformers','vllm','math-verify','tokenizers']}}
    save(f'{tag}_runtime_identity.json', identity)
    assert 'A100' in identity['gpu'], 'Unexpected GPU: inspect before consuming compute.'
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tasks = {'math':[math_prompt(r['problem']) for r in math_rows],
             'code':[code_prompt(r, shots) for r in code_rows]}
    lengths = {k:max(len(tok.encode(p, add_special_tokens=False)) for p in v) for k, v in tasks.items()}
    assert lengths['math']+MATH_CAP <= CONTEXT and lengths['code']+CODE_CAP <= CONTEXT, lengths
    protocol = {'student':tag,'model':MODEL,'revision':REVISION,'prompting':'raw few-shot, no chat template',
        'math':{'source':'100 MATH train-split development problems (same set as the post-trained 9B run)',
                'shots':'Minerva 4-shot','max_tokens':MATH_CAP,'stop':MATH_STOP,'prefix_budgets':PREFIX_BUDGETS},
        'code':{'source':'MBPP full validation split, task_ids 511-600','shots':'MBPP task_ids 2,3,4',
                'max_tokens':CODE_CAP,'stop':CODE_STOP,'grader':'all test_list asserts pass, 10 s timeout'},
        'settings':[{'name':n,**p,'samples':k} for n, p, k in SETTINGS],
        'seed_rule':'91432026 + item index + 100000*sample','max_model_len':CONTEXT,'max_num_seqs':CONCURRENCY,
        'max_prompt_tokens':lengths,'decision_rule':DECISION_RULE,'training_performed':False,'final_test_accessed':False,
        'data_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['math_dev_100.jsonl','mbpp_dev_90.jsonl','mbpp_shots.json']}}
    save(f'{tag}_protocol.json', protocol)
    t = time.perf_counter()
    llm = LLM(model=MODEL, revision=REVISION, dtype='bfloat16', max_model_len=CONTEXT,
        max_num_seqs=CONCURRENCY, max_num_batched_tokens=8192, gpu_memory_utilization=.85,
        enforce_eager=False, enable_prefix_caching=False, language_model_only=True,
        seed=SEED, disable_log_stats=False)
    summary = {'student':tag,'load_seconds':time.perf_counter()-t,'results':{},'complete':False}
    print('PROBE_MODEL_LOADED', json.dumps(summary), flush=True)
    for index in range(2):
        out = llm.generate(['Problem:\nWhat is $17\\cdot 23$?\n\nSolution:\n'], SamplingParams(temperature=0,max_tokens=64), use_tqdm=False)
        print('KERNEL_WARMUP', index, len(out[0].outputs[0].token_ids), flush=True)
    for task, rows, cap, stop in [('math',math_rows,MATH_CAP,MATH_STOP), ('code',code_rows,CODE_CAP,CODE_STOP)]:
        for name, params, k in SETTINGS:
            items = [(i, s) for s in range(k) for i in range(len(rows))]
            sampling = [SamplingParams(**params, max_tokens=cap, stop=stop, seed=SEED+i+100000*s) for i, s in items]
            print('QUEUE_START', task, name, len(items), flush=True)
            t = time.perf_counter()
            outputs = llm.generate([tasks[task][i] for i, _ in items], sampling, use_tqdm=True)
            wall = time.perf_counter()-t
            records, prefixes = [], []
            codes = [extract_code(o.outputs[0].text) for o in outputs] if task=='code' else None
            if task == 'code':
                with concurrent.futures.ThreadPoolExecutor(8) as pool:
                    results = list(pool.map(lambda a: run_tests(a[0], rows[a[1][0]]), zip(codes, items)))
            for j, ((i, s), result) in enumerate(zip(items, outputs)):
                o, row = result.outputs[0], rows[i]
                rec = {'id':row['id'] if task=='math' else row['task_id'], 'sample':s, 'seed':SEED+i+100000*s,
                    'text':o.text, 'output_tokens':len(o.token_ids), 'output_token_ids':list(o.token_ids),
                    'input_tokens':len(result.prompt_token_ids), 'finish_reason':o.finish_reason, 'stop_reason':str(o.stop_reason)}
                if task == 'math':
                    rec.update(level=row['level'], subject=row['subject'], gold=row['gold'], **math_grade(o.text, row['gold']))
                    for b in PREFIX_BUDGETS:
                        text = o.text if len(o.token_ids)<=b else tok.decode(o.token_ids[:b], skip_special_tokens=True)
                        prefixes.append({'id':rec['id'],'sample':s,'budget':b,**math_grade(text,row['gold'])})
                else:
                    rec.update(code=codes[j], correct=results[j]['passed'], error=results[j]['error'])
                records.append(rec)
            (ROOT/f'{tag}_{task}_{name}_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            stats = summarize(records, wall, task)
            if prefixes:
                stats['prefix_points'] = [{'budget':b,'mean_accuracy':sum(p['correct'] for p in prefixes if p['budget']==b)/len(records)} for b in PREFIX_BUDGETS]
            summary['results'][f'{task}/{name}'] = stats
            save(f'{tag}_summary.json', summary)
            print('QUEUE_FINISHED', task, name, json.dumps(stats), flush=True)
    summary['complete'] = len(summary['results'])==4
    save(f'{tag}_summary.json', summary)
    print('PROBE_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    try: run()
    except Exception:
        save(f'{STUDENT}_error.json', {'traceback':traceback.format_exc()})
        raise
