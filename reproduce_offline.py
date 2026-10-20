"""Re-run every free, offline step of the frozen pipeline and compare the outputs with the committed files.

No API call, no model, no GPU, no network. Needs Python 3.10+ with sympy (test-freeze stage 2).

Usage: python -I reproduce_offline.py [WORK_DIR]
WORK_DIR must not exist; a temporary directory is used if it is omitted. Nothing inside the repository is written
(apart from Python's own __pycache__).

Steps:
  1. test freeze stage 2: tier-1 variants, final training pool, dedup report, manifest (byte-identical);
  2. train2000 selection from the regenerated pool (byte-identical);
  3. teacher_gen.py dry run with a fake transport (ledger, resume, retry and budget logic);
  4. claude_label.py dry run (prepare / apply / strict checks / calibration gate).
"""
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NT = ROOT / 'validation/next_tests'
FREEZE = NT / 'test_freeze'
POOL_SHA = 'd451355f171602e0ca55b48f4090f7829d99585df8d73bf65ec85b0011ea4a54'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, expect=None):
    res = subprocess.run([sys.executable, '-I', *map(str, args)], capture_output=True, text=True, encoding='utf-8')
    if res.returncode != 0 or (expect and expect not in res.stdout):
        sys.stdout.write(res.stdout[-3000:])
        sys.stderr.write(res.stderr[-3000:])
        raise SystemExit(f'FAILED: {Path(args[0]).name}')
    return res.stdout


def same(a, b, label):
    ok = sha(a) == sha(b)
    print(f'  {"ok  " if ok else "DIFF"} {label}')
    return ok


def main(work):
    work.mkdir(parents=True)
    ok = True

    print('1. test freeze stage 2')
    stage1 = work / 'stage1'
    with zipfile.ZipFile(FREEZE / 'stage1_run1/test_freeze_stage1.zip') as z:
        z.extractall(stage1)
    run([FREEZE / 'freeze_stage2.py', ROOT, stage1, work / 'stage2'])
    for name in ('tier1_variants.jsonl', 'stage2_dedup.json', 'freeze_manifest.json'):
        ok &= same(work / 'stage2' / name, FREEZE / 'stage2_run1' / name, name)
    pool_ok = sha(work / 'stage2/train_eligible_pool.jsonl') == POOL_SHA
    print(f'  {"ok  " if pool_ok else "DIFF"} train_eligible_pool.jsonl (6,356 rows, {POOL_SHA[:16]}...)')
    ok &= pool_ok

    print('2. train2000 selection')
    root2 = work / 'root'
    dest = root2 / 'validation/next_tests/test_freeze/stage2_run1'
    dest.mkdir(parents=True)
    (dest / 'train_eligible_pool.jsonl').write_bytes((work / 'stage2/train_eligible_pool.jsonl').read_bytes())
    run([NT / 'teacher_gen/select_train2000.py', root2, work / 'train2000'])
    for name in ('train2000.jsonl', 'train2000_selection.json'):
        ok &= same(work / 'train2000' / name, NT / 'teacher_gen' / name, name)

    print('3. teacher_gen.py dry run')
    run([NT / 'teacher_gen/dry_run_teacher_gen.py', NT / 'teacher_gen', work / 'dry_teacher_gen'], 'DRY_RUN_PASSED')
    print('  ok   DRY_RUN_PASSED')

    print('4. claude_label.py dry run')
    (work / 'dry_claude_label').mkdir()
    run([NT / 'heavy_clean_claude/dry_run_claude_label.py', work / 'dry_claude_label'], 'DRY RUN OK')
    print('  ok   DRY RUN OK')

    print('ALL OK' if ok else 'SOME OUTPUTS DIFFER')
    return 0 if ok else 1


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(tempfile.mkdtemp()) / 'work'
    assert not target.exists(), 'WORK_DIR exists'
    print('work dir:', target)
    sys.exit(main(target))
