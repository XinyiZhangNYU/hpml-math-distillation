# HPML test-set freeze, stage 1 (free, CPU runtime, no API key, no model). Downloads the pinned public datasets
# (MATH train/test, MATH-500, MathArena AIME 2026 / HMMT Feb 2026; about 6 MB) and writes the frozen tiers,
# the variant sources and the training pool deduplicated against them. Upload the files listed in FILES to
# /content first (from this repository, same names). Terminate the runtime after saving the zip.
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
P = Path('/content/hpml_project')  # mirrors the repository layout
FILES = {  # uploaded name: (path in the repository, SHA-256 for data files)
    'freeze_tests.py': ('validation/next_tests/test_freeze/freeze_tests.py', None),
    'select_length.py': ('validation/next_tests/length_tests/select_length.py', None),
    'eval_common.py': ('validation/eval_common.py', '37aa9b224b5e3e95eb520496881aad7daccb78a5463044396c8db99dab155e76'),
    'validation_100.jsonl': ('validation/evidence/validation_100.jsonl', '0453023eb394f9000314de4dc28ed3a73e8c1c876f2a70c02216f61ebf9cb1b0'),
    'teacher_candidates_50.jsonl': ('validation/evidence/teacher_candidates_50.jsonl', '89a3fa804de3510c5d65b12a5a36687b493eaae2dc1db1b85e3510e360b3269b'),
    'teacher_math_50.jsonl': ('validation/next_tests/teacher_math50/teacher_math_50.jsonl', 'a1a423640fe001862b74153d32acbda219993d1c418796d3208059a7f9bdc930'),
    'math_perturb_hard.jsonl': ('validation/next_tests/perturb_materials/math_perturb_hard.jsonl', '418ca51e02f3b639a919dda72b5201ce35ac6c6d7d31c3231735beb4ea012dd7'),
    'math_perturb_simple.jsonl': ('validation/next_tests/perturb_materials/math_perturb_simple.jsonl', '2130821d4235dc1f5d962bee7f84a096651bf98fcb17b6dcfd3d0a48875ba5bb')}
for name, (rel, digest) in FILES.items():
    data = (Path('/content')/name).read_bytes()
    assert digest is None or hashlib.sha256(data).hexdigest() == digest, name
    (P/rel).parent.mkdir(parents=True, exist_ok=True); (P/rel).write_bytes(data)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'math-verify==0.9.0', 'datasets==4.8.5', 'huggingface_hub==1.33.0'], check=True)
out = Path('/content/test_freeze_stage1'); assert not out.exists(), 'Output folder exists: inspect it.'
subprocess.run([sys.executable, '-I', str(P/'validation/next_tests/test_freeze/freeze_tests.py'), str(P), str(out)], check=True)
m = json.loads((out/'freeze_manifest.json').read_text())
print(json.dumps({k: m[k] for k in ('pool', 'tiers', 'variant_sources')}, indent=1))
shutil.make_archive('/content/test_freeze_stage1', 'zip', out)
print('ZIP_SHA256', hashlib.sha256(Path('/content/test_freeze_stage1.zip').read_bytes()).hexdigest())
