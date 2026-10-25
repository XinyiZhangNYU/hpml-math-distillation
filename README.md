# Distilling thinking-mode math reasoning into Qwen3.5-9B-Base

NYU ECE-GY 9143 High Performance Machine Learning, Fall 2026, by Xinyi Zhang and Siyuan Wang.

We distill reasoning traces from **DeepSeek-V4.1-Flash** (thinking mode) into **Qwen3.5-9B-Base** with LoRA. The
comparison is against fine-tuning on MATH's human solutions, and we test whether deleting abandoned attempts and
repeated work from the traces helps. Evaluation is on 90 post-release variants of MATH test problems, AIME 2026,
HMMT February 2026, and MATH-500 as a reference. A systems part compares vLLM request scheduling strategies.

The frozen design (models, prompt, admission rules, cleaning rules, conditions, test sets, evaluation protocol)
is in **[DESIGN.md](DESIGN.md)**.

## Status

| Stage | Status |
|---|---|
| Student and task selection (base probes, development checks) | done |
| Test sets frozen and deduplicated against training data | done |
| 2,000 training questions selected | done |
| Teacher traces generated (1,971 kept, USD 2.68) | done |
| Rule-based check of the raw traces (groups 3 and 4) | done (0 changes, so group 3 trains on the raw traces) |
| Cleaning: deleting abandoned attempts and repeated work (group 4; 797 traces) | done |
| Group 4 training file built | done |
| Group 2 (human solutions) training file | done |
| Exact token counts of groups 2–4 | done |
| Hyperparameter tuning on MATH dev 100 (18 checkpoints) | done: 2e-4, 1 epoch for groups 2–4 |
| Training setup and evaluation protocol frozen (DESIGN.md §8–§9) | done |
| LoRA training, 3 seeds per group | done |
| Final evaluation scripts (`final_eval.py`, `final_stats.py`, `run_final.py`) | done |
| Final evaluation of the 4 conditions (3 seeds; results in DESIGN.md §10) | done |
| Why tier 2 outputs hit the 32k budget (exploratory, post hoc; DESIGN.md §10.1) | done: most truncations are loops |
| vLLM scheduling experiment (DESIGN.md §11) | done: the queue gives 2.05× the throughput of fixed groups at 32k |
| LoRA training and evaluation cost (DESIGN.md §11.2) | done |
| Loop early-stop estimate (exploratory, offline; DESIGN.md §12) | done |

## Quick check (offline, about 30 s, no GPU and no API)

```bash
python -I reproduce_offline.py
```

This re-runs every free, deterministic step from the committed inputs and compares the output with the committed
files byte for byte:

- test-freeze stage 2 (the 90 variants and the 6,356-problem training pool);
- the 2,000-question sample;
- dry runs of the teacher-generation ledger;
- dry runs of the cleaning label checker.

It needs Python ≥ 3.10 with `sympy`, and the last line should read `ALL OK`.

## Repository layout

Scripts find their inputs by relative path, and every data input is checked against a recorded SHA-256.

```
DESIGN.md                      frozen design
reproduce_offline.py           one-shot offline reproduction check
base_probe/                    few-shot probes of 9B-Base / 4B-Base on MATH dev 100 and MBPP val 90 (Colab GPU)
training/                      LoRA training, adapter merge and dev evaluation (Colab GPU); not frozen yet
validation/
  eval_common.py               MATH loading, answer parsing and grading (math-verify); defines the train pool
  validation_{worker,controller}.py   post-trained Qwen3.5-9B check + vLLM vs HF throughput (Colab GPU)
  evidence/                    development sets: validation_100, teacher_candidates_50
  next_tests/
    perturb_materials/         MATH-Perturb hard/simple (only used for near-duplicate removal)
    teacher_math50/            the 50 development problems used to check the teacher
    length_tests/              select_length.py (rebuilds the MATH train pool; shared by the freeze scripts)
    test_freeze/               test-set freeze, stage 1 (Colab CPU) and stage 2 (local); frozen outputs; checks
    teacher_gen/               train2000 selection, teacher generation (paid API) + offline dry run
    light_clean/               rule-based check L1–L3
    heavy_clean_pilot/         sentence splitter / deletion engine (heavy_clean_v3.py) + reference labels
    heavy_clean_claude/        cleaning label tool, labeling guide (frozen), dry run
    train_sets/                group 2 and group 4 builders, exact Qwen token counts
```

## Data kept outside the repository

The generated traces and labels are too large for the repository. They are shared read-only on Google Drive:
[hpml-math-distillation-data](https://drive.google.com/drive/folders/1t2iVSALy0f7qxFcHDKLTAJD8VnmTwPON). The
folder has the six subfolders below; download them into `validation/next_tests/teacher_results/`. Each file is
listed with its SHA-256 so that any copy can be verified; scripts expect them under these relative paths:

| Path (under `validation/next_tests/teacher_results/`) | Bytes | SHA-256 | What |
|---|---|---|---|
| `teacher_gen_b4/final_traces.jsonl` | 7,858,345 | `5b14ebf90959eeeefd9147dbee3fb88c062f7229abbaead57bda4ca7f634c51e` | 1,971 admitted traces = **group 3** |
| `teacher_gen_b4/raw_responses.jsonl` | 7,794,897 | `c9a582846f346fdd4d08d798f5aa17b9a47780161fafbff56ada0bc8cae7e363` | full paid-response ledger (2,053) |
| `teacher_gen_b4/grades.jsonl` | 551,097 | `76c7a5384319d8067a0a59314f10e89051f6605ef4457b97e1b2655bc0ce4138` | admission grading of every response |
| `teacher_gen_b4/manifest.json`, `summary.json` | | see the files | hashes, costs, counts |
| `light_clean_full/heavy_inputs.jsonl` | 6,612,630 | `b3a23d381c175b5365db7a6f41a8dbd144a939d8bd0b54a45597a5f9eaf87069` | raw traces after the rule-based check (unchanged) |
| `light_clean_full/light_report.json` | 509 | `19ed609a7b56a8e96ed681becb133acea2304538b774a2b9a36550a8a91a2c3a` | L1–L3 counts (all 0) |
| `heavy_claude_full/heavy_claude_full_work.zip` | 7,098,575 | `7b3b4f4eaed38c4a41f0b4228494f3437a51e5b17acc80a2b02972025feeb601` | cleaning `work/`: 80 batches, labels, results, diff pages |
| `group4_heavy_v1/group4_heavy.jsonl` | 6,625,578 | `8ac76e67953c530714e15ef21411112fc45133db6875fd0e6a6b5f45ed7c286b` | 1,971 rows = **group 4** |
| `group4_heavy_v1/build_report.json` | 1,869 | see the file | build checks and token estimates |
| `group2_human_v1/group2_human.jsonl` | 1,818,505 | `176dc0b04b9d599870e37172ff06872c0faeacad2ed4f7f1a3b33d21dabe8f93` | 1,971 rows = **group 2** |
| `group2_human_v1/build_report.json` | 1,298 | `bacc8ea9e07eeb46b55dc5324255c2e1faf3f8639d48f0ccd1206e154d414def` | group 2 build checks |
| `token_counts_v1/token_counts.json` | 68,558 | `089f2d60fba026ea494ec1a4e16ab07cf48dc7475f767a51e6c1bcbbc7e52ad0` | exact token counts of groups 2–4 |

Unzip `heavy_claude_full_work.zip` inside `heavy_claude_full/` so that `heavy_claude_full/work/` exists. The
`group2_human_v1/` and `token_counts_v1/` files can also be rebuilt exactly (steps 12 and 13 below).

## Environment

| Where | Used for | Pins |
|---|---|---|
| Local, any OS, Python ≥ 3.10 | stage 2, selection, cleaning, group 4, dry runs | standard library + `sympy` |
| Colab CPU (Linux) | test-freeze stage 1, math-verify checks, teacher API runs | `math-verify==0.9.0`, `datasets==4.8.5`, `huggingface_hub==1.33.0`, `transformers==5.17.0`, `tokenizers==0.23.2` |
| Colab A100 | base probes / validation | `vllm==0.21.0`, `transformers==5.17.0`, `tokenizers==0.23.2`, `math-verify==0.9.0` |

- math-verify uses `signal.alarm`, so anything that grades must run on Linux in the main thread.
- Each Colab cell installs its own pins and checks the recorded SHA-256 of its inputs before running.
- Run scripts with `python -I`, and write outputs to a new directory outside the repository; most scripts
  refuse to overwrite.
- API keys: the paid cells read `DEEPSEEK_API_KEY` from **Colab Secrets** only. No script writes the key, and
  the teacher script checks every file it writes for it. Do not commit keys.

## Full reproduction, step by step

Steps marked **paid** call the DeepSeek API and cost money. Their outputs are archived, so re-running them is
only needed to regenerate the data from scratch. Teacher outputs are sampled (T=0.6), so a re-run gives
different traces. Use the archived ones to reproduce the training sets.

### A. Development evidence (why this student and teacher)

1. **Base probes** (Colab A100, about 1 h each). Upload the five files listed in `expected` of
   `base_probe/colab_probe_setup_cell.py` to `/content`, run the setup cell, then poll with
   `colab_probe_monitor_cell.py`. The `_4b` variants do the same for 4B-Base. Settings: Minerva-style 4-shot,
   4,096 tokens, T=0.6, two seeds. Result: MATH dev 100 at 54.0% (9B) and 50.0% (4B); MBPP at 67.2% and 59.4%.
2. **Post-trained 9B check and throughput** (Colab A100): `validation/validation_controller.py` runs
   `validation_worker.py` in bounded stages under `/content/hpml_validation`. Result: about 90% on dev, and
   72.84 tok/s with vLLM versus 16.74 tok/s with HF at batch 1.
3. **Teacher on dev** (paid API): 48/50 in non-thinking mode on `teacher_math50/teacher_math_50.jsonl`, and
   20/20 in thinking mode on 20 of those problems.

### B. Test-set freeze and training pool

4. **Stage 1** (Colab CPU, free, about 6 MB of downloads): upload the files listed in `FILES` of
   `test_freeze/colab_freeze_cell.py` and run the cell. It rebuilds the MATH train pool, loads the pinned
   test sets, picks the variant sources and deduplicates. The archived output is
   `test_freeze/stage1_run1/test_freeze_stage1.zip` (`87055cbc…`).
5. **Stage 2** (local, free). Extract the stage 1 zip into a new directory `STAGE1`, then:
   ```bash
   python -I validation/next_tests/test_freeze/freeze_stage2.py . STAGE1 OUT_STAGE2
   ```
   This runs every variant's own answer check (`variants.py`), deduplicates the variants against the pool, and
   writes:
   - `tier1_variants.jsonl` (90);
   - the final `train_eligible_pool.jsonl` (6,356 problems, `d451355f…`);
   - `stage2_dedup.json` and `freeze_manifest.json`.

   The pool itself is not committed, because this step regenerates it exactly. Copy it to
   `validation/next_tests/test_freeze/stage2_run1/` for the next step.
6. **Answer checks of the variants**:
   - `test_freeze/math_verify_run1/math_verify_check.py EVAL_COMMON TIER1_JSONL OUT_JSON` (Linux): 90/90
     parse, 90/90 self-correct, 90/90 wrong values rejected;
   - `test_freeze/blind_review_run1/symbolic_proofs.py`: SymPy proofs for two min/max families.

   The blind solves by two model families agreed on 90/90. One ambiguous item is excluded from scoring, see
   `test_freeze/scoring_exclusions.json`.

### C. Teacher data

7. **Select the 2,000 questions** (local, free):
   ```bash
   python -I validation/next_tests/teacher_gen/select_train2000.py . OUT_TRAIN2000
   ```
   It removes multiple-choice problems and draws a seeded simple random sample. The output must equal the
   committed `teacher_gen/train2000.jsonl` (`99c14986…`).
8. **Generate the traces** (paid, Colab CPU, about USD 0.6 per batch of 475). Upload `teacher_gen.py`,
   `train2000.jsonl` and `base_probe/base_probe_worker.py`, then set `BATCH` (b0, then b1–b4) and `BUDGET` in
   `teacher_gen/colab_teacher_gen_cell.py` and run it.
   - The ledger is append-only and resumable, and a paid response is never repeated.
   - Admission, the single resample and the drop rule follow DESIGN.md §3.
   - Check the logic offline first:
     `python -I validation/next_tests/teacher_gen/dry_run_teacher_gen.py validation/next_tests/teacher_gen NEW_DIR`.

### D. Cleaning and training sets

9. **Rule-based check** (local, free):
    ```bash
    python -I validation/next_tests/light_clean/light_clean.py FINAL_TRACES.jsonl OUT_LIGHT
    ```
    It writes `light_report.json` and `heavy_inputs.jsonl`, which must match the hashes in the data table.
10. **Cleaning** (local, free). Follow `heavy_clean_claude/LABELING_GUIDE.md`.
    - Labeling: `claude_label.py prepare heavy_inputs.jsonl WORK_DIR 10 20261019` creates the batches, and the
      labels are written to `WORK_DIR/labels/batch_NNN.json`.
    - The labels are Claude's judgment under the frozen guide and cannot be regenerated exactly. The archived
      labels are the source of truth.
    - Deletion and the calibration gate are deterministic and can be re-run from the archived labels:
      `python -I validation/next_tests/heavy_clean_claude/claude_label.py apply WORK_DIR NNN`.
    - Dry run: `dry_run_claude_label.py EMPTY_DIR`.
11. **Build group 4** (local, free). With the data above in place:
    ```bash
    python -I validation/next_tests/train_sets/build_group4.py validation/next_tests
    ```
    It checks every input hash, writes `teacher_results/group4_heavy_v1/`, and verifies that each cleaned
    reasoning is a subsequence of the raw reasoning and that `content` is unchanged.
12. **Build group 2** (local, free; about 3 MB of downloads, needs `huggingface_hub` and `pyarrow`):
    ```bash
    python -I validation/next_tests/train_sets/build_group2.py fetch MATH_DIR
    python -I validation/next_tests/train_sets/build_group2.py build validation/next_tests MATH_DIR
    ```
    `fetch` downloads the 7 MATH train parquet files at the pinned revision; `build` checks their hashes, takes
    the reference solution of each kept question in `final_traces.jsonl` order, checks that the problem text,
    level and gold answer match `train2000.jsonl`, and writes `teacher_results/group2_human_v1/`. The output is
    deterministic and must match the hashes in the data table.
13. **Exact token counts** (any machine with `transformers==5.17.0`; tokenizer files only, about 10 s):
    ```bash
    python count_group4_tokens.py final_traces.jsonl group4_heavy.jsonl token_counts.json group2_human.jsonl
    ```
    It asserts that the group 3 recount equals the recorded `qwen_total_tokens`. Results are in DESIGN.md §5.

### E. Training and evaluation

The scripts in `training/` run on a Colab A100 with the pinned stacks of the capacity test (training: unsloth
2026.10.5, transformers 5.17.0, peft 0.21.1, torch 2.11.0) and of the base probes (evaluation: vLLM 0.21.0,
math-verify 0.9.0). The training setup, the tuning results and the final evaluation protocol are frozen in
DESIGN.md §8–§9.

| Script | What it does |
|---|---|
| `sft_data.py` | Builds the §3 sequences and loss masks, checks the group totals against `token_counts.json`, and plans the steps: a seeded question order shared by all groups, 16 questions per step, length-sorted microbatches with right padding, no packing. Tokenizer only. |
| `train_lora.py` | BF16 LoRA r 16 on one group: AdamW, warmup then constant learning rate, token-mean loss per step, an adapter saved after every epoch. `--seed` sets the LoRA initialization and `--order-seed` the question order. |
| `merge_lora.py` | Merges an adapter into the pinned base weights shard by shard, so vLLM loads the result like the base model. |
| `dev_eval.py` | vLLM generation on MATH dev 100 and grading with `validation/eval_common.py`, with re-grading after truncation at 4k / 8k / 16k tokens. It first checks that vLLM on the merged weights reproduces the adapter's loss on four fixed training sequences. |
| `colab_smoke_cell.py` | One Colab cell: a short run of groups 3 and 2, a merge, and a dev evaluation, to measure time and memory. It fetches every input from this repository and the Drive folder and checks the SHA-256 of each. |
| `run_tuning.py` | The tuning grid: learning rate {1e-4, 2e-4} × epochs {1, 2, 3} for groups 3, 4 and 2, each adapter merged and evaluated on MATH dev 100 (4 samples, 16k tokens), plus condition 1 on dev as a reference. Picks each group's setting by the rule fixed in advance (highest mean dev accuracy; within 1 point, fewer epochs, then the lower learning rate). Skips finished jobs when run again. |
| `colab_tuning_cell.py` | One Colab cell that checks every input like the smoke cell, starts `run_tuning.py` in the background and follows its log. It pins `train_lora.py` as of commit `1d71b47` (the tuning runs); the current script only adds `--order-seed`, and its defaults give the same training. |
| `final_eval.py` | The final evaluation of one model (DESIGN.md §9) on the frozen test tiers: 8 samples per problem in tiers 1 and 2, 4 in tier 3, 32k tokens, rescored at 4k / 8k / 16k. Conditions 2–4 use the training chat template and count an output whose think block is not closed as wrong; condition 1 uses the Minerva 4-shot prompt. Sampling and engine settings come from `dev_eval.py`; the request seed is 91432026 + 1,000,000 × tier + item index + 100,000 × sample, the same for every model. It checks every test file by SHA-256, records the excluded tier-1 item without scoring it, and writes per-output scores, the output texts and a summary for each tier. |
| `final_stats.py` | Combines the runs: accuracy per condition, tier and budget (mean over problems of the mean over seeds and samples), truncation rate, output length, and paired, clustered 95% intervals (Miller 2024; tier 1 clustered by source problem). CPU only. |
| `run_final.py` | The final stage: copies the seed-1 adapters of the tuning run, trains seeds 2 and 3 (groups 3, 4 and 2; 2e-4, 1 epoch), merges and evaluates the 9 models with `final_eval.py` (with the loss check), evaluates condition 1, and runs `final_stats.py`. Skips finished jobs when run again and copies the results to Drive after every job. |
| `colab_final_cell.py` | One Colab cell that checks every input by SHA-256 (scripts and test files from this repository, training data from the Drive folder), restores earlier results from Drive on a new runtime, starts `run_final.py` in the background and follows its log. Needs Google Drive mounted. |
| `colab_trunc_check.py` | Post-hoc look at the tier 2 outputs of conditions 3 and 4 that hit the 32k budget (DESIGN.md §10.1): exact loops, zlib ratio of the tail, and every `oxed{}` graded with math-verify. Reads the final run in `MyDrive/hpml_final` and writes `analysis/trunc_tier2/` there. CPU only. |
| `sched_bench.py` | The scheduling benchmark of DESIGN.md §11: fixed groups of 64 requests versus a continuously refilled queue on one vLLM engine, MATH dev 100 × 2 samples, at 32k and 8k tokens. Records throughput, per-request first-token and finish times, slot use, resident tokens, preemptions and GPU memory, and checks accuracy. |
| `lora_cost.py` | LoRA training and evaluation cost from the records of the final run (time, tokens per second, peak memory). CPU only. |
| `early_stop_estimate.py` | Offline, exploratory estimate of a loop early-stop rule on the saved final outputs (DESIGN.md §12). CPU only. |
| `run_sched.py` | Runs `lora_cost.py`, `early_stop_estimate.py`, the merge of the condition 3 / seed 1 adapter and `sched_bench.py`, skipping finished jobs, and copies the results to Drive after every job. |
| `colab_sched_cell.py` | One Colab cell that checks the scripts by SHA-256, starts `run_sched.py` in the background and follows its log. Needs Google Drive mounted with the final run in `MyDrive/hpml_final`. |

## References

1. X. Ren, B. Wu, L. Liu. *I Learn Better If You Speak My Language.* EMNLP 2024. https://arxiv.org/abs/2402.11192
2. E. Miller. *Adding Error Bars to Evals.* arXiv 2024. https://arxiv.org/abs/2411.00640
3. D. Hendrycks et al. *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS 2021. https://arxiv.org/abs/2103.03874
4. Y. Kim, A. M. Rush. *Sequence-Level Knowledge Distillation.* EMNLP 2016. https://arxiv.org/abs/1606.07947
5. N. Ho, L. Schmid, S.-Y. Yun. *Large Language Models Are Reasoning Teachers.* ACL 2023. https://arxiv.org/abs/2212.10071
6. K. Huang et al. *MATH-Perturb.* ICML 2025. https://arxiv.org/abs/2502.06453
7. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021. https://arxiv.org/abs/2106.09685
8. I. Mirzadeh et al. *GSM-Symbolic.* ICLR 2025. https://arxiv.org/abs/2410.05229
9. M. Balunović et al. *MathArena: Evaluating LLMs on Uncontaminated Math Competitions.* NeurIPS 2025 D&B. https://arxiv.org/abs/2505.23281
10. M. Wu et al. *Reasoning or Memorization? Unreliable Results of RL Due to Data Contamination.* AAAI 2026. https://arxiv.org/abs/2507.10532
11. W. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP 2023. https://arxiv.org/abs/2309.06180
12. G.-I. Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022.

Data sources: MATH (`EleutherAI/hendrycks_math`), MATH-500 (`HuggingFaceH4/MATH-500`), MathArena AIME 2026 and
HMMT February 2026, MATH-Perturb, and MBPP, each under its own license. Teacher outputs come from the DeepSeek API.
The cleaning deletions were chosen by Claude (Anthropic): Claude only selected which teacher sentences to
delete and wrote no training text. They are used for course research only.
