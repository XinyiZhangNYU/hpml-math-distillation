"""Build the group 2 (human solutions) training set: the MATH reference solution of each of the 1,971 kept questions.

Group 2 = the same questions as groups 3 and 4, in final_traces.jsonl order. Each id (subject/train/i) is row i of
that subject's train parquet file at the pinned hendrycks_math revision, the same rows that prepare_data() and
select_length.py enumerate. The solution text is copied unchanged.

Checks (any failure stops): input hashes; parquet hashes; for every id the problem text and its sha256 equal the
train2000.jsonl row, the level matches, and the last \\boxed{} of the solution equals the recorded gold answer.

usage:
  python -I build_group2.py fetch <math_dir>                  download the 7 train parquet files (network, free)
  python -I build_group2.py build <next_tests dir> <math_dir>  offline; writes teacher_results/group2_human_v1/
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

DATASET = 'EleutherAI/hendrycks_math'
REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'
SUBJECTS = ['algebra', 'counting_and_probability', 'geometry',
            'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
PARQUET = '{}/train-00000-of-00001.parquet'
PARQUET_SHA = {
    'algebra': '73347f5b55869f21ff094361025dbd12d291e971aa4d72142eacee10438e449f',
    'counting_and_probability': 'd37ae017930df41678edee50dcde18c17fb06afaff9eb505c1b95f1002bfcb31',
    'geometry': '9dbf5e5b55499e696e0ba72da9d85d4f0be3acbf7eaa6cb014b0fc78bb63784f',
    'intermediate_algebra': '0c87f3798ebdfb915b1f9b6fc13c1a8d064c3cb91631ccbe85b83ed5d71150f6',
    'number_theory': '81c08649149b39811f2d1d7ddf5dfccf6c4c8044a2e169cfb24db09160c941f4',
    'prealgebra': '765b526fb5c0b8385cb7be2c16bf4f6f28cf71f2bc15c826abfbb5fa7f8616cc',
    'precalculus': '28bb1ff95525c8d32521509bef91de76339659f5467a381e6e8deea204538f0e',
}
RAW_SHA = '5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e'
TRAIN2000_SHA = '99c149864bf011488ddc61e2f98144d4e1473a97422a19a79e05e0cc66287826'
N_KEPT = 1971


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]


def fetch(math_dir):
    from huggingface_hub import hf_hub_download
    math_dir = Path(math_dir)
    for s in SUBJECTS:
        p = hf_hub_download(DATASET, PARQUET.format(s), repo_type='dataset', revision=REVISION, local_dir=math_dir)
        print(PARQUET.format(s), sha(p))


def load_subject(math_dir, subject):
    import pyarrow.parquet as pq
    p = Path(math_dir) / PARQUET.format(subject)
    assert sha(p) == PARQUET_SHA[subject], PARQUET.format(subject) + ' changed'
    return pq.read_table(p).to_pylist(), sha(p)


def build(nt, math_dir):
    nt = Path(nt)
    sys.path.insert(0, str(nt.parent))
    from eval_common import boxed
    tr = nt / 'teacher_results'
    raw_p = tr / 'teacher_gen_b4/final_traces.jsonl'
    t2k_p = nt / 'teacher_gen/train2000.jsonl'
    assert sha(raw_p) == RAW_SHA, 'final_traces.jsonl changed'
    assert sha(t2k_p) == TRAIN2000_SHA, 'train2000.jsonl changed'
    kept = [r['id'] for r in read_jsonl(raw_p)]
    assert len(kept) == len(set(kept)) == N_KEPT
    t2k = {r['id']: r for r in read_jsonl(t2k_p)}

    rows, parquet_sha = {}, {}
    for s in SUBJECTS:
        rows[s], parquet_sha[s] = load_subject(math_dir, s)

    out = []
    for i in kept:
        subject, split, idx = i.split('/')
        assert split == 'train'
        src, ref = rows[subject][int(idx)], t2k[i]
        problem = src['problem']
        assert problem == ref['problem'], ('problem text differs', i)
        assert hashlib.sha256(' '.join(problem.split()).encode()).hexdigest() == ref['problem_sha256'], i
        assert str(ref['level']) in str(src['level']), ('level differs', i)
        assert boxed(src['solution'].split('</think>')[-1]) == ref['gold'], ('gold differs', i)
        out.append({'id': i, 'subject': subject, 'level': ref['level'], 'problem': problem, 'gold': ref['gold'],
                    'problem_sha256': ref['problem_sha256'], 'solution': src['solution']})

    dest = tr / 'group2_human_v1'
    dest.mkdir(parents=True, exist_ok=True)
    out_p = dest / 'group2_human.jsonl'
    out_p.write_bytes(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out).encode('utf-8'))
    chars = [len(r['solution']) for r in out]
    report = {
        'dataset': DATASET, 'revision': REVISION, 'parquet_sha256': parquet_sha,
        'inputs_sha256': {'final_traces.jsonl': RAW_SHA, 'train2000.jsonl': TRAIN2000_SHA},
        'rows': len(out), 'order': 'final_traces.jsonl',
        'checks': ['problem text and sha256 equal train2000.jsonl', 'level matches',
                   'last \\boxed{} of the solution equals gold'],
        'solution_chars': {'sum': sum(chars), 'mean': round(statistics.mean(chars), 1),
                           'median': statistics.median(chars), 'max': max(chars)},
        'output_sha256': sha(out_p),
    }
    (dest / 'build_report.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'fetch':
        fetch(sys.argv[2])
    elif sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
