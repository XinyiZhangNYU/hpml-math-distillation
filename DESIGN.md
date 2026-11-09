# Frozen design

Everything in sections 1–9 is frozen: the data, prompts, rules, test sets, training setup and evaluation
protocol are fixed and pinned by SHA-256, and they will not change from here on. Section 10 lists what is still open.

Research question: can thinking-mode reasoning traces from DeepSeek-V4.1-Flash teach Qwen3.5-9B-Base to solve
math problems it fails, beyond fine-tuning on human solutions? Does deleting abandoned attempts and repeated work
help, and do the gains transfer to new problems?

## 1. Models

| Role | Model | Pin |
|---|---|---|
| Student | `Qwen/Qwen3.5-9B-Base` | revision `68c46c4b3498877f3ef123c856ecfde50c39f404` (model and tokenizer) |
| Teacher | DeepSeek-V4.1-Flash via the DeepSeek API, model id `deepseek-flash` | system fingerprint `e72d3c7f663d3e1383b00b3884cb4501` on every response |

Why these models (development evidence, `base_probe/`, `validation/`):

- MATH dev 100: 9B-Base 54.0%, 4B-Base 50.0%. MBPP val 90: 67.2% and 59.4%. Settings: Minerva-style 4-shot,
  4,096 tokens, T=0.6, mean of two seeds, vLLM 0.21.0. Math leaves more room than code, so the task is math.
- The post-trained Qwen3.5-9B solves about 90% of the dev set, so the Base model is the student.
- On 50 dev problems the teacher scored 48/50 (non-thinking) against the student's 57%. In thinking mode it scored
  20/20 on a 20-problem pilot.
- Decoding throughput for the post-trained 9B, batch 1: vLLM 72.84 tok/s vs HF `generate` 16.74 tok/s.

## 2. Training questions

- Source: `EleutherAI/hendrycks_math` at revision `21a5633873b6a120296cce3e2df9d5550074f4a3`, train split. The pool is
  rebuilt exactly as `validation/eval_common.py::prepare_data()` builds it. Exclusions: 707 diagram problems,
  5 unparseable references, 1 duplicate, 2 with no level.
- Removed from the pool:
  - the 150 development problems (`validation_100`, `teacher_candidates_50`, `teacher_math_50`);
  - near-duplicates (word-trigram Jaccard ≥ 0.5) of development problems and of MATH-Perturb hard/simple (204);
  - any problem with the same normalized text as a test item, or Jaccard ≥ 0.5 with one, in any tier
    (section 6), including reserve variant sources.

  This leaves a training-eligible pool of **6,356** problems (`d451355f…`).
- Multiple-choice problems are removed by a fixed text rule before generation (92 removed; 6,264 remain).
- Simple random sample: `SEED = 20261019`, N = **2,000** → `train2000.jsonl` (`99c14986…`). It was generated in
  batches b0 = 100 and b1–b4 = 475 each.

## 3. Teacher generation

- Prompt (user turn, no system prompt):
  `Solve the following mathematical problem. Explain the necessary steps and put the final answer in \boxed{}.\n\n` + problem.
  The reference answer is never shown to the teacher.
- Request: no `thinking` field (the API default is thinking mode), effort not sent (default high), `temperature`
  0.6, `max_tokens` 65,536, timeout 1,800 s, non-streaming. One trace per problem.
- **Admission** (all must hold):
  1. `finish_reason == "stop"`;
  2. `reasoning_content` is non-empty;
  3. `content` contains `\boxed`;
  4. the last box is graded correct against the gold answer by `math_grade` (`base_probe/base_probe_worker.py`,
     math-verify 0.9.0), with no normalization added;
  5. the Qwen SFT sequence is ≤ **32,768** tokens. The layout is
     `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n`.
- A problem that fails admission is resampled once with the same request. If it fails again, it is
  dropped from **all** groups. Nothing is truncated.
- Result: 2,000 problems → **1,971** kept (24 of them on the second attempt) and 29 dropped (all wrong answers).
  Cost: USD 2.6841 at peak rates. Kept Qwen sequence length: mean 1,228, median 534, max 23,538 tokens.

## 4. Cleaning (delete-only; `content` is never changed)

**Rule-based check** (groups 3 and 4; script `light_clean.py`):

| Rule | What it deletes |
|---|---|
| L1 loops | Any run of ≥ 20 words repeated ≥ 3 times back to back. The first copy is kept. |
| L2 garbage | U+FFFD, private-use and unassigned code points, and control characters other than `\n` and `\t`. |
| L3 script switch | A sentence in a writing system the problem does not use. Greek letters and math symbols do not count. |

There is a safety valve: a trace that would lose more than 10% of its reasoning counts as an admission failure.

- "Content unrelated to the problem" is not a rule, because it has no mechanical definition.
- Result: 0 triggers, so groups 3 and 4 start from the raw traces, byte-identical.

**Cleaning** (group 4 only; script `claude_label.py`):

- Scope: only traces with ≥ 300 reasoning tokens (797 traces). Shorter traces stay raw.
- The reasoning is split into sentence units (`heavy_clean_v3.split_units`), and only whole units are deleted.
- What gets deleted, per the frozen `LABELING_GUIDE.md`:
  - abandoned branches (`dropped_at` required);
  - repeated work (`same_as` required, and it must point to earlier text that is kept). This covers second and
    later re-verification, re-reading, and a tail re-derivation.
- What is never deleted:
  - the main derivation, i.e. up to the first statement of the final answer;
  - the first check of each result;
  - set-up that is used later;
  - the final solution.

  At most 80% of a trace may be deleted.
- Claude labels the deletions under the frozen guide. Labels are archived per batch with their hashes.
  The deletions are reproducible from the labels, but the labeling itself cannot be re-run.
- Calibration: each batch of 10 has one hidden reference trace, rotating among 5 traces. The gate is
  precision ≥ 0.70 and F1 ≥ 0.60.
  - 70 of 80 batches passed.
  - The labels of the other 10 batches were checked against the guide and kept; their scores are recorded.
  - Calibration is not blind after batch 5, because references were visible on earlier diff pages.
- Result:
  - 768 traces had deletions and 29 had nothing to delete.
  - 39.1% of all reasoning characters were removed; per labeled trace the mean is 35.3%.
  - Group 4 is `group4_heavy.jsonl` (`8ac76e67…`), 1,971 rows.

Automatic checks (all passed):

- every cleaned reasoning is a subsequence of the raw reasoning;
- `content` is byte-identical to the raw `content`;
- every input and result file matches its recorded hash;
- the labeled and short id sets are disjoint and cover all 1,971 ids.

## 5. Conditions

| # | Condition | Training targets |
|---|---|---|
| 1 | 9B-Base, no training | — |
| 2 | LoRA on human solutions | MATH reference solutions of the 1,971 kept questions (`group2_human.jsonl`) |
| 3 | LoRA on raw traces (main) | `final_traces.jsonl` |
| 4 | LoRA on cleaned traces | `group4_heavy.jsonl` |

Groups 2–4 share:

- the same 1,971 questions, adapter configuration, training budget and hyperparameter search;
- 32,768-token sequences;
- the sequence layout of §3. Group 2 puts the human solution after an empty think block
  (`<think>\n\n</think>\n\n{solution}<|im_end|>\n`). The loss covers everything after the prompt, from `<think>`
  through the final `<|im_end|>`.

Because group 4 is shorter, the 3-vs-4 difference mixes "cleaner" with "shorter". The total number of training
tokens is reported for every group. Exact counts with the pinned tokenizer, prompt tokens included
(`teacher_results/token_counts_v1/token_counts.json`):

| Group | Total tokens | Mean | Median | Max | Over 8k / 16k / 32k |
|---|---|---|---|---|---|
| 2 (human solutions, empty think block) | 611,428 | 310.2 | 242 | 2,640 | 0 / 0 / 0 |
| 3 (raw traces) | 2,420,158 | 1,227.9 | 534 | 23,538 | 44 / 8 / 0 |
| 4 (cleaned traces) | 1,750,516 | 888.1 | 512 | 20,162 | 12 / 2 / 0 |

The group 3 recount equals the `qwen_total_tokens` recorded at admission for all 1,971 traces. Group 4 has
72.3% of group 3's tokens. No sequence needs truncation.

## 6. Test sets (deduplicated against training data)

| Tier | Content | Pin |
|---|---|---|
| 1 (main) | 90 variants of 30 MATH test problems: for each, one wording, one numbers and one condition/target change. They were written after the student's release. | `tier1_variants.jsonl` `c5e1eb3c…` |
| 2 | MathArena AIME 2026 (30) and HMMT February 2026 (33) | dataset revisions `d61f4868…` / `589a53fe…`; files `67dbd42c…` / `40f1a443…` |
| 3 (reference) | MATH-500, possibly contaminated | revision `6e4ed1a2…`; file `f9608716…` |

- Variant sources: 6 primary and 4 reserve per level, chosen in seeded order. Diagram problems, unparseable
  references and MATH-500 members or near-duplicates are skipped. No reserve was needed.
- How each variant answer was checked:
  - Its own Python check (brute force or SymPy) must agree with the recorded value and with the parsed LaTeX
    answer.
  - A wording variant must also keep the source answer.
  - math-verify parses all 90 answers, grades each as correct against itself, and grades a constructed wrong
    value as wrong.
  - Blind solves by Claude Opus and by GPT agree with the reference on 90/90.
- **Scoring exclusion:** `variant/algebra/test/158/numbers` is excluded as ambiguous (integer versus real sides),
  so tier 1 is scored on **89** items. It is reported separately and not replaced
  (`test_freeze/scoring_exclusions.json`).
- Rounding convention for the rounding families: round half up, as in the MATH source answers.

## 7. Evaluation protocol (frozen parts)

- Prompts: conditions 2–4 get the training chat template of §3, zero-shot. Condition 1 (no training) gets the
  Minerva-style 4-shot prompt of the base probes (`base_probe/base_probe_worker.py`: raw text, no chat
  template, stop at `\nProblem:`), the standard way to evaluate a base model. The 1-vs-trained comparison
  therefore includes the prompt change; the comparisons among groups 2–4 do not.
- Grader: math-verify 0.9.0 through `validation/eval_common.py` (`37aa9b22…`), applied to the last `\boxed{}`.
- Generation budget: up to 32k tokens. The same outputs are also scored after truncation at 4k, 8k and 16k
  tokens.
- Reported per tier: accuracy, truncation rate and output length.
- Statistics: several samples per problem are averaged. The conditions are compared with paired, clustered
  confidence intervals following Miller (2024).
- Systems experiment: on vLLM, fixed-size request groups versus a continuously refilled queue. We measure
  throughput, latency, memory, and LoRA training cost.

## 8. Training (frozen after tuning on MATH dev 100)

Scripts: `training/sft_data.py` (`9f4c31ef…`), `training/train_lora.py` (`b8fcde9d…`), `training/merge_lora.py`
(`a673c2f7…`). Stack: unsloth 2026.10.5, transformers 5.17.0, peft 0.21.1, torch 2.11.0, one A100 40 GB.

| Item | Setting |
|---|---|
| Adapter | LoRA r 16, alpha 16, dropout 0, BF16, on 12 projection types (attention, MLP and linear attention): 248 modules, 43,278,336 parameters |
| Step | 16 questions (1,971 questions → 124 steps per epoch; the last step has 3) |
| Loss | mean over all supervised tokens of the step (§5) |
| Microbatches | length-sorted inside a step, right padding masked out, at most 32,768 padded tokens each, no packing |
| Optimizer | AdamW (0.9, 0.999, eps 1e-8), weight decay 0, gradient clip 1.0 |
| Schedule | linear warmup over 10 steps, then constant |
| **Learning rate, epochs** | **2e-4, 1 epoch, for all three groups** |
| Evaluated weights | the adapter merged into the pinned base weights (`merge_lora.py`), loaded in vLLM like the base model |

**Training seeds: 3**, the same for every group, so groups 2–4 are paired by seed. A seed sets the LoRA
initialization (`--seed`) and the question order (`--order-seed`):

| Seed | LoRA init | Question order |
|---|---|---|
| 1 | 91432026 | 20261021 |
| 2 | 91432027 | 20261022 |
| 3 | 91432028 | 20261023 |

Seed 1 is the tuning run: with a constant learning rate after warmup, its epoch-1 adapter at 2e-4 is exactly a
1-epoch run, so it is used as is.

**How the setting was chosen.** Grid: learning rate {1e-4, 2e-4} × epochs {1, 2, 3}, seed 1, for each group.
Each adapter was merged and evaluated on MATH dev 100 (4 samples per problem, 16k tokens, T 0.6, top-p 0.95,
top-k 20, vLLM 0.21.0 with 2,048-token prefill chunks and 0.80 GPU memory). Rule, fixed before any result: in each
group, the highest mean dev accuracy; settings within 1 point of it go to fewer epochs, then the lower learning
rate. Dev accuracy:

| Group | 1e-4, 1 ep | 1e-4, 2 ep | 1e-4, 3 ep | 2e-4, 1 ep | 2e-4, 2 ep | 2e-4, 3 ep |
|---|---|---|---|---|---|---|
| 2 (human solutions) | 0.578 | 0.578 | 0.585 | **0.653** | 0.613 | 0.615 |
| 3 (raw traces) | 0.895 | 0.910 | 0.878 | **0.915** | 0.890 | 0.905 |
| 4 (cleaned traces) | 0.870 | 0.875 | 0.878 | **0.898** | 0.880 | 0.885 |

Condition 1 (Base, Minerva 4-shot) on the same dev setup: 0.558.

- Every evaluated checkpoint first checks that vLLM on the merged weights reproduces the trained adapter's target
  loss on four fixed training sequences (tolerance 3% relative). 16 of 18 pass, with at most 1.8%. Group 2 at 2e-4
  with 2 and 3 epochs gives 3.04% and 3.4%. Both come from one 209-token sequence whose loss had become very small
  (summed NLL 12.1 and 4.1; absolute differences 0.37 and 0.14). These two results are reported as they are and
  were not selected.
- These dev accuracies use the grading of `eval_common.py` alone. Under the §9 rule, 5 group 3 checkpoints each lose
  one sample (0.25 points). No selection changes.
- On one A100 40 GB, a 3-epoch tuning run took 46 minutes for group 3 and 32 minutes for group 4, model loading
  included.

## 9. Final evaluation (frozen)

- Weights: the merged adapter of each group and seed (9 models), plus the base model for condition 1.
- Prompts and grader as in §7. Sampling: T 0.6, top-p 0.95, top-k 20, vLLM 0.21.0, the engine settings of the dev
  evaluation.
- **Closed think block required** (conditions 2–4): an output whose `<think>` block is not closed is graded wrong.
  Only the last `\boxed{}` after `</think>` counts. Condition 1 has no think block.
- **Samples per problem:** tier 1 and tier 2: 8; tier 3 (MATH-500): 4. Condition 1 gets the same numbers.
- **Headline budget:** 32,768 generated tokens. The same outputs are scored again after truncation at 4k, 8k and
  16k tokens; an output cut inside its think block is wrong under the rule above.
- **Accuracy** of a condition: the mean over problems of the mean over its seeds and samples.
- **Confidence intervals** (paired, clustered, Miller 2024): tier 1 is clustered by source problem (30 clusters,
  89 scored variants); tiers 2 and 3 by problem. All seeds and samples of a problem stay in its cluster, so the
  intervals include seed variation.

## 10. Final results (3 seeds, §9)

Run with `training/run_final.py` on a Colab A100 40GB, 2026-10-22 to 2026-10-24. Each job ran once; no job was
repeated. Accuracy at the 32k budget, with paired, clustered 95% intervals:

| Condition | Tier 1 (89 variants) | Tier 2 (AIME/HMMT 2026) | Tier 3 (MATH-500) |
|---|---|---|---|
| 1 base model, 4-shot | 0.567 [0.455, 0.680] | 0.012 [0.001, 0.023] | 0.505 [0.471, 0.539] |
| 2 human solutions | 0.564 [0.447, 0.680] | 0.019 [0.003, 0.034] | 0.516 [0.484, 0.547] |
| 3 raw traces | 0.948 [0.918, 0.977] | 0.393 [0.310, 0.476] | 0.907 [0.889, 0.925] |
| 4 cleaned traces | 0.928 [0.887, 0.969] | 0.324 [0.249, 0.399] | 0.886 [0.867, 0.904] |

| Paired difference | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| 3 − 4 | +0.019 [+0.002, +0.037] | +0.069 [+0.045, +0.092] | +0.021 [+0.012, +0.031] |
| 2 − 1 | −0.004 [−0.049, +0.041] | +0.007 [−0.008, +0.021] | +0.011 [−0.010, +0.031] |

Accuracy after truncation, conditions 3 / 4:

| Budget | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| 4k | 0.900 / 0.912 | 0.211 / 0.227 | 0.847 / 0.853 |
| 8k | 0.929 / 0.923 | 0.277 / 0.277 | 0.877 / 0.873 |
| 16k | 0.942 / 0.927 | 0.339 / 0.308 | 0.896 / 0.882 |

Truncation rate and mean output length at 32k, conditions 3 / 4:

| Tier | Truncation rate | Mean output length (tokens) |
|---|---|---|
| 1 | 0.018 / 0.020 | 1,571 / 1,361 |
| 2 | 0.411 / 0.337 | 18,088 / 14,890 |
| 3 | 0.036 / 0.030 | 2,567 / 2,014 |

Condition 4 trains on 72% of the tokens of condition 3 (§5). This design cannot separate the effect of cleaner
traces from the effect of shorter ones.

### 10.1 Exploratory: why tier 2 outputs hit the 32k budget (post hoc, not pre-registered)

Question: do the tier 2 outputs of conditions 3 and 4 that reach 32,768 tokens run out of budget while still
working, or are they stuck in a loop? Script: `training/colab_trunc_check.py`, Colab CPU, over the saved outputs of
the final run (all 3 seeds, checked against the SHA-256 in each `summary.json`); no model and no API. Each truncated
output is put in one class:

- **Exact loop:** the end of the text repeats with a fixed period over at least max(2p, 1,500) characters.
- **Near-repetitive:** no exact loop, but the zlib ratio of the last 16,000 characters is below 0.2767, the 5th
  percentile of finished outputs longer than 8k tokens.
- **No repetition found:** the rest.

Every `\boxed{}` in the output is also graded with math-verify 0.9.0, as in the frozen grader (one comparison timed
out and counts as wrong).

| | Condition 3 | Condition 4 |
|---|---|---|
| Truncated / all outputs | 622 / 1,512 | 510 / 1,512 |
| Exact loop | 201 (32%) | 275 (54%) |
| Near-repetitive | 130 (21%) | 84 (16%) |
| No repetition found | 291 (47%) | 151 (30%) |
| Truncated outputs with a correct `\boxed{}` anywhere | 8 | 2 |
| Truncated on a problem this condition never solved | 282 (45%) | 195 (38%) |
| No repetition found, on a problem solved at least once (share of all outputs) | 160 (10.6%) | 92 (6.1%) |
| Loop start, median [p10, p90] (tokens) | 18,255 [6,212, 28,952] | 14,231 [5,088, 26,922] |
| Loop period, median (tokens) | 24 | 59 |
| Finished outputs over 8k tokens: count / exact loops / accuracy | 306 / 0 / 0.575 | 222 / 0 / 0.329 |

Exact loops per seed: 80 / 62 / 59 (condition 3) and 118 / 53 / 104 (condition 4). In 18 sampled outputs
(`examples.txt`, 3 per class and condition), the classes match the text. Exact loops repeat one number or sentence
thousands of times. Near-repetitive outputs mostly enumerate cases without end or restate the same doubt. Outputs
with no repetition found are mostly still computing, often restarting after finding an error; a few take the 2026
problems for older contest problems and try to recall an answer.

- Most 32k truncations are not a budget limit: 53% (condition 3) and 70% (condition 4) are loops or
  near-repetitive. Almost no truncated output had already found the answer (8 and 2).
- A larger budget has little room to help. Even if every truncated output with no repetition found, on a problem
  solved at least once, turned correct, tier 2 accuracy would rise by at most 10.6 (condition 3) and 6.1
  (condition 4) points. Going from 16k to 32k added 5.4 and 1.6 points (§10).
- Condition 4 writes shorter outputs but loops more often (54% against 32% of its truncations). The cause was not
  tested.

## 11. Systems experiment (frozen 2026-10-24, before any run)

Question: on vLLM, how much do fixed-size request groups lose against a continuously refilled queue when output
lengths vary as much as they do for these models? Script: `training/sched_bench.py`, run by `training/run_sched.py`
on one Colab A100 40GB.

- **Model and workload:** the merged condition 3 / seed 1 model (the loss check of §9 must pass first). MATH dev 100 ×
  2 samples = 200 requests, in the order of the dev evaluation (all problems for sample 0, then for sample 1), with
  the chat prompt and sampling of §9 and the request seed 91432026 + item index + 100,000 × sample. Dev problems are
  used so that no test set is generated again.
- **Engine:** vLLM 0.21.0 with the settings of the final evaluation (BF16, 2,048-token prefill chunks, 0.80 GPU
  memory, CUDA graphs, no prefix caching), 64 sequence slots, maximum length 32,768 + 2,048. One engine serves all
  runs.
- **Modes:** *fixed*: requests go in 64 at a time in workload order (groups of 64, 64, 64 and 8), and the next group
  goes in when every request of the previous one has finished. *queue*: all 200 requests go in at once, and the
  engine starts a waiting request whenever a slot is free. Both modes drive the engine through the same
  `add_request()` / `step()` loop, so they differ only in when requests are submitted. A warm-up of 8 short
  requests also checks this loop against `LLM.generate()`.
- **Runs, in this order:** budget 32,768: fixed, queue, queue, fixed; budget 8,192: queue, fixed. The 32k runs are
  ordered ABBA so that drift over the session does not favour one mode.
- **Primary result:** at 32k, generated tokens per second of generation wall time, queue over fixed (mean of the
  two runs of each mode), and the wall time of the 200 requests. With two runs per mode there is no interval: both
  runs are reported, and a difference smaller than the spread between the two runs of one mode is reported as no
  difference.
- **Secondary:** per-request first-token and finish times (mean, median, p90, max); the busy slot share (the summed
  time from first token to finish over 64 × wall time); in fixed mode, each group's time and longest output; peak
  running requests and peak resident tokens (prompt and generated tokens of the running requests) against the KV
  cache capacity; vLLM's preemption counter; whole-device GPU memory (nvidia-smi, once a second); the same at 8k.
- **Sanity checks, not results:** accuracy (§9 rule) and truncation per run, and the share of identical outputs
  between runs. The request seed is fixed, but batch composition changes the arithmetic, so outputs may differ
  between runs; accuracy should not differ beyond sampling noise.
- **LoRA training cost** (`training/lora_cost.py`, from the records of the final run, nothing is rerun): per group
  and seed, the epoch-1 training time, real (non-padding) tokens and tokens per second, peak PyTorch and
  whole-device GPU memory, host memory and load time from `train_report.json`; merge and evaluation job times from
  the final run's state file; generation time and throughput of each evaluation tier.
- **Not claimed:** results for other GPUs, models, slot counts or workloads; a fixed group size other than the slot
  count.

### 11.1 Results (2026-10-24, one session)

All six runs finished on one engine (vLLM 0.21.0, torch 2.11.0, NVIDIA A100-SXM4-40GB), after the loss check
(largest relative difference 0.00085) and the warm-up check (8/8 outputs equal to `LLM.generate()`). Model load took
313 s and is not counted.

| Run | Wall (s) | Output tokens | Tokens/s | Busy slot share | Finish p50 / p90 / max (s) | Truncated | Accuracy |
|---|---|---|---|---|---|---|---|
| 32k fixed 1 | 1,688.0 | 521,203 | 308.8 | 0.082 | 524.9 / 1,084.8 / 1,688.0 | 9 | 0.900 |
| 32k queue 1 | 642.3 | 406,805 | 633.4 | 0.198 | 25.2 / 70.8 / 642.2 | 5 | 0.915 |
| 32k queue 2 | 642.2 | 406,805 | 633.5 | 0.198 | 25.2 / 70.8 / 642.2 | 5 | 0.915 |
| 32k fixed 2 | 1,687.2 | 521,203 | 308.9 | 0.082 | 524.9 / 1,084.0 / 1,687.2 | 9 | 0.900 |
| 8k queue 1 | 177.9 | 229,726 | 1,291.4 | 0.422 | 25.2 / 70.8 / 177.9 | 12 | 0.880 |
| 8k fixed 1 | 469.6 | 263,226 | 560.6 | 0.150 | 140.5 / 293.3 / 469.6 | 16 | 0.865 |

- **Primary result:** at 32k the queue generated 2.05 times as many tokens per second as fixed groups (633.5 vs 308.8,
  means of two runs), and the 200 requests took 2.63 times as long in fixed groups (1,687.6 vs 642.2 s). The two runs
  of each mode differ by less than 0.05%, so this is a difference under the rule above. At 8k (one run each) the
  ratios are 2.30 and 2.64.
- **Why:** in fixed mode each group waits for its longest output. Each of the first three groups held an output
  that reached the 32,768-token budget, so the groups took 515.6, 541.2 and 561.9 s (8k: 131.1, 134.1 and 135.0 s).
  In the first group, 56 of the 64 requests had finished after 27 s. The last group of 8 took 69.3 s. In queue mode
  the freed slots take new requests, so the median request finishes after 25 s instead of 525 s.
- **Same work?** No. Batch composition changes the arithmetic, so the two modes produced different outputs: 39.5% of
  outputs are identical between the modes, against 100% between the two runs of the same mode. Fixed mode generated
  1.28 times as many tokens at 32k (9 truncated outputs against 5) and 1.15 times at 8k. Tokens per second already
  corrects for the amount of work; the wall-time ratio does not. Accuracy differs by 3 of 200 outputs at each budget,
  within sampling noise.
- **Busy slot share** is low in both modes (0.08 and 0.20 at 32k) because a few long outputs run nearly alone at
  the end of each group (fixed) or of the whole workload (queue). With few sequences running, each decodes about 64
  tokens per second (from the resident-token trace), so one 32,768-token output needs more than 500 s by itself; this
  sets the queue's wall time of 642 s. A larger workload would spend a smaller share of its time in this tail, so the
  ratios measured here apply to this workload size only.
- **Memory:** up to 64 requests ran at once. Peak resident tokens were 131,327 (fixed) and up to 183,411 (queue),
  against a KV cache capacity of 389,535 tokens (vLLM log: 12.45 GiB for the cache, attention blocks of 528 tokens for
  this hybrid model). There were no preemptions in any run. Whole-device GPU memory stayed at 38,130–38,132 MiB in
  every run because vLLM reserves its 0.80 share at start, so it does not separate the modes.
- vLLM's `request_success` counter read 0 in every run (its labelled form was not captured); it is not used.

### 11.2 LoRA training and evaluation cost (from the records of the final run)

Epoch 1, 124 optimizer steps, 43,278,336 trainable parameters; mean of the 3 seeds (range):

| Group | Real tokens | Training time (s) | Real tokens/s | Real / padded tokens | Peak PyTorch memory (GiB) | Peak device memory (GiB) |
|---|---|---|---|---|---|---|
| 2 human solutions | 611,428 | 263.8 (257.1–272.7) | 2,319 | 0.78 | 19.6 | 20.7 |
| 3 raw traces | 2,420,158 | 937.3 (914.5–964.6) | 2,583 | 0.91 | 25.6 (max 27.3) | 28.0 (max 29.7) |
| 4 cleaned traces | 1,750,516 | 668.0 (608.5–705.1) | 2,632 | 0.89 | 23.1 | 25.0 |

Training time follows the number of real tokens: group 4 has 0.72 times the tokens of group 3 and needs 0.71 times
the time. Group 2 has the lowest throughput and the most padding. The nine adapters took
5,607 s (1.56 h) of training in total; each merge took 62–132 s.

Evaluation (§9, all three tiers; mean per seed, condition 1 has one run):

| Condition | Generation time (h) | Output tokens (millions) | Output tokens/s |
|---|---|---|---|
| 1 base model | 2.36 | 6.61 | 777 |
| 2 human solutions | 1.82 | 5.63 | 859 |
| 3 raw traces | 4.46 | 15.37 | 956 |
| 4 cleaned traces | 3.49 | 12.50 | 995 |

The ten evaluations took 32.2 h of job time, about 20 times the training time, so generation dominates the cost of
this study. Condition 4 generates 19% fewer tokens than condition 3 and needs 22% less generation time; it is also
less accurate (§10).

### 11.3 Extension: workload size, slot count and a second model (frozen 2026-11-04, before any run)

Question: does the §11.1 gain of the queue over fixed groups hold for a larger workload, for other slot counts, and
for a model with a different output length distribution? Same scripts, extended with the workload size, the slot
count and the list of runs as arguments; one Colab A100 40GB; engine settings of §11 except `max_num_seqs`, which
equals the slot count. Nothing else changes: chat prompt, sampling, request seed rule, budget 32,768 + 2,048.

- **Second model:** the merged condition 4 / seed 1 model (the loss check of §9 must pass first). It has the same base
  model and size as condition 3, and in the final evaluation it generated 19% fewer tokens (§11.2) and looped in a
  larger share of its truncated outputs (§10.1), so only the output length distribution changes.
- **Workloads:** MATH dev 100 × k samples in the order of §11, k = 2 (200 requests, as §11) or k = 6 (600 requests;
  the first 200 are the requests of §11).
- **Cells,** each with one fixed and one queue run (fixed group size = slot count):

  | Cell | Model | Requests | Slots | First run |
  |---|---|---|---|---|
  | 1 | condition 3 | 600 | 64 | fixed |
  | 2 | condition 3 | 200 | 32 | queue |
  | 3 | condition 3 | 200 | 128 | fixed |
  | 4 | condition 4 | 200 | 64 | queue |
  | 5 | condition 4 | 600 | 64 | fixed |

  One run per mode: in §11.1 two runs of the same mode differed by less than 0.05%. The first mode alternates over
  the cells. Engines start in the order of the cells (condition 3 at 64, 32 and 128 slots, then condition 4 at 64),
  each with the warm-up check of §11. Before each cell starts, the Colab compute unit balance is read; below 12 units
  the session stops and the remaining cells are reported as not run. The order is not changed after any result is
  seen.
- **Primary result per cell:** generated tokens per second of generation wall time, queue over fixed, and the wall
  time of the workload in each mode, set next to the §11.1 values (2.05 and 2.63). With one run per mode there is no
  interval; differences between cells are described, not tested.
- **Expected direction (stated before the runs):** at 600 requests the ratio is larger than at 200, because every
  fixed group still waits for its longest output while the queue's final tail is shared by three times the work.
  At 128 slots, 128 × 34,816 tokens exceed the KV cache of 389,535 tokens, so preemptions may occur; they are
  counted and reported, not avoided.
- **Secondary and sanity checks:** as in §11 (finish times, busy slot share, group times and longest outputs, peak
  running requests and resident tokens against the KV cache, preemptions, GPU memory; accuracy and truncation per
  run).
- **Not claimed:** other GPUs, budgets or models; fixed group sizes other than the slot count; mixed workloads.
- **Estimated cost:** about 6 A100-hours (cells 1 and 5 about 1.5–2 h each, the other cells 0.5–1 h each, plus four
  engine starts and one merge), about 35 compute units at about 5.3 units per hour; the balance was 48.41 units
  before the run.

#### Results (run 2026-11-05, 03:00–09:41 UTC; all five cells ran)

Balance read before each cell: 47.61, 34.14, 26.30, 21.25 and 16.20 units, so every cell ran. Each engine passed
the loss check (largest relative difference 0.0064, 0.0023 and 0.0024 for condition 3 at 64, 32 and 128 slots;
0.0186 for condition 4) and the warm-up check (8 of 8 texts identical). Engine starts took 190–322 s; they are not
counted.

| Cell | Model | Requests | Slots | Tokens/s, queue / fixed | Wall, fixed / queue |
|---|---|---|---|---|---|
| §11.1 | condition 3 | 200 | 64 | 2.05 | 2.63 |
| 1 | condition 3 | 600 | 64 | 3.17 | 3.39 |
| 2 | condition 3 | 200 | 32 | 3.15 | 4.48 |
| 3 | condition 3 | 200 | 128 | 1.50 | 1.81 |
| 4 | condition 4 | 200 | 64 | 2.12 | 2.60 |
| 5 | condition 4 | 600 | 64 | 3.26 | 3.81 |

| Run | Wall (s) | Output tokens | Tokens/s | Busy slot share | Finish p50 / p90 / max (s) | Truncated | Accuracy | Preemptions |
|---|---|---|---|---|---|---|---|---|
| 1 fixed | 4,902.4 | 1,446,094 | 295.0 | 0.079 | 2,160.2 / 4,228.1 / 4,902.4 | 24 | 0.907 | 0 |
| 1 queue | 1,446.3 | 1,353,499 | 935.8 | 0.425 | 102.7 / 270.0 / 1,446.3 | 19 | 0.918 | 32 |
| 2 queue | 709.1 | 466,424 | 657.8 | 0.397 | 44.7 / 103.6 / 709.1 | 8 | 0.890 | 0 |
| 2 fixed | 3,177.2 | 663,252 | 208.8 | 0.104 | 1,519.8 / 2,583.5 / 3,177.2 | 14 | 0.875 | 0 |
| 3 fixed | 1,143.5 | 506,697 | 443.1 | 0.065 | 24.7 / 620.9 / 1,143.5 | 9 | 0.895 | 0 |
| 3 queue | 632.1 | 419,435 | 663.6 | 0.113 | 19.0 / 70.6 / 632.1 | 7 | 0.910 | 0 |
| 4 queue | 637.8 | 402,421 | 631.0 | 0.196 | 24.5 / 61.3 / 637.8 | 7 | 0.890 | 0 |
| 4 fixed | 1,656.8 | 493,941 | 298.1 | 0.080 | 522.3 / 1,111.6 / 1,656.8 | 11 | 0.890 | 0 |
| 5 fixed | 5,328.1 | 1,456,956 | 273.4 | 0.073 | 2,188.8 / 4,312.8 / 5,328.1 | 29 | 0.897 | 0 |
| 5 queue | 1,398.0 | 1,244,562 | 890.2 | 0.405 | 81.5 / 195.1 / 1,398.0 | 21 | 0.895 | 31 |

- **Workload size:** at 600 requests the queue's gain is larger than at 200, as expected: 3.17 (condition 3) and
  3.26 (condition 4) times the tokens per second, against 2.05 and 2.12 at 200. Fixed groups stay near 0.08 busy
  slot share at both sizes, while the queue's share rises from 0.20 to 0.40–0.43 because the final tail of a few long
  outputs is shared by three times the work.
- **Slot count:** the gain falls as the slot count rises: 3.15 at 32 slots, 2.05 at 64, 1.50 at 128. With 128 slots
  the 200 requests form only two fixed groups (128 and 72), so fixed mode waits for a long output twice instead of
  seven times at 32 slots; the queue's wall time barely changes (709, 642 and 632 s) because it is set by the
  longest single output.
- **Second model:** condition 4 at 200 requests and 64 slots gives 2.12 and 2.60, within 4% of condition 3 (2.05 and
  2.63). Its shorter outputs did not change the ratio at this size; at 600 requests it gave a slightly larger
  ratio than condition 3 (3.26 against 3.17).
- **Preemptions** occurred only in the two 600-request queue runs (32 and 31), not at 128 slots as anticipated. In a
  long queue, long outputs accumulate: the prompt and generated tokens of the started, unfinished requests peaked at
  533,615 and 582,492, above the cache capacity of 389,535 tokens. This count includes preempted requests, so it is
  not the cache occupancy. At 128 slots the peak was 197,104 (fixed) and 229,507 (queue). The tokens per second of
  the queue include the recomputation after preemption.
- **Same work?** As in §11.1, the modes produce different outputs (32.5–39.0% identical per cell), and fixed mode
  generated 1.07–1.42 times as many tokens. Accuracy differs by at most 1.5 points between the modes of a cell,
  within sampling noise. With one run per mode, the differences between cells are described, not tested.
- **Cost:** about 6.7 A100-hours and 45.4 compute units (48.17 before, 2.74 after; about 6.7 units per hour), above
  the estimate of 35 because the usage rate was higher than 5.3.
- **Records:** `MyDrive/hpml_sched_ext/ext/cellN/cell_summary.json` and `runs/` (per run: statistics with the
  once-a-second trace, and all outputs), the four `engine_*.json` files, the five answers in `go/`, and the logs and
  pip freeze one level up. `ext/wait_go.json` there is a copy from an earlier sync (the copy to Drive does not
  delete files).

### 11.4 Training: sequence packing and fused linear-attention kernels (timing only; frozen 2026-11-05, before any run)

Question: how much LoRA training time (§8, §11.2) does packing save, compared with the padded microbatches of §8,
and what do the fused linear-attention kernels contribute? This study measures time only. No adapter from this
study is evaluated, and nothing in §8–§10 changes.

**Why kernels are part of the question.** Qwen3.5-9B mixes full-attention layers and gated delta net
(linear-attention) layers. In the §8 stack the linear-attention layers ran on the PyTorch fallbacks of
transformers 5.17.0, because `causal_conv1d` and `fla` were not installed. Those fallbacks treat a row as a
single sequence. When several examples are packed into one row, the recurrent state and the short causal
convolution taps would carry over from one example into the next. Packing without leakage needs kernels that
take sequence boundaries: `cu_seqlens` for the chunked delta rule, boundary-aware causal convolution, and
variable-length flash attention in the full-attention layers. These kernels are also faster in their own
right. A third arm separates the two effects.

- **Arms** (same data, step plan, seeds, LoRA setting and loss as §8; one A100 40GB; one Colab session):

  | Arm | Linear-attention and attention kernels | Microbatches |
  |---|---|---|
  | R | §8 stack as recorded (PyTorch fallbacks) | §8: length-sorted, right padding, at most 32,768 padded tokens |
  | F | fused kernels (`fla` delta rule, `causal_conv1d`, flash attention), from the transformers kernel hub or pip, versions and sources recorded | as R |
  | P | as F | packed: the examples of a step, longest first, first-fit into rows of at most 32,768 tokens, one row per forward pass; positions restart at 0 for each example and the boundaries are passed to every layer |

  All three arms run the same 124 optimizer steps on the same 16 questions per step, with the same loss: the
  mean over all supervised tokens of the step. Only the grouping into forward passes and the kernels change.
- **Workload:** epoch 1, seed 1 (LoRA init 91432026, question order 20261021), learning rate 2e-4, for group 2
  (human solutions: the most padding, 0.78 real/padded) and group 3 (raw traces: the most tokens, 0.91).
  Order: group 2 R, F, P, then group 3 R, F, P.
- **Gates before any timed run** (F and P stack; failures are reported, not worked around by changing the design):
  1. *Kernels in use:* the run records which implementation of each linear-attention function and of the
     attention ran. If a fallback ran, F and P do not start.
  2. *Same function:* on the four check sequences of §8, the summed target NLL of the untrained model under F is
     within 1% (relative) of R.
  3. *Isolation:* on the packed rows of the first four steps of group 3, the summed NLL of every example in its
     packed row is within 1% (relative) of the same example run alone. As a control, the same check with the
     R stack and no boundaries is expected to fail, which shows that the check detects leakage.

  If gate 1 or 2 fails, only R runs. If gate 3 fails, R and F run and P is reported as not achieved in this
  stack. Code fixes that keep the arms, workload and measures unchanged are allowed and recorded.
- **Primary result per group:** training seconds of the epoch (sum of step times, model loading excluded) and
  real tokens per second, with the ratios R/F (kernels) and F/P (packing).
- **Expected direction (stated before the runs):** P is faster than F, more for group 2 than for group 3. If
  padding were the only cost, the bound is the padded/real ratio, 1.28 for group 2 and 1.10 for group 3; fewer
  forward passes can add to it. R/F is expected above 1. Its size is not predicted.
- **Secondary and sanity checks:** forward passes and processed tokens per epoch, peak PyTorch and device memory,
  median step-time ratio after the first 5 steps, and per-step loss differences between arms (the first step
  should agree closely; later steps may drift through numerics). R is compared with the seed-1 record of the
  final run (`train_report.json`). A difference above 10% is reported as a session effect; all ratios use
  this session's runs only.
- **Balance rule:** the Colab balance is read before each arm. Below 8 units the session stops, and the
  remaining arms are reported as not run. The order is not changed after any result is seen.
- **Not claimed:** the accuracy of adapters trained with packing or fused kernels (not evaluated); more than one
  epoch; other groups, GPUs or models. With one run per arm there are no intervals; the per-step times give the
  spread within a run.
- **Estimated cost:** R takes about 4.5 min (group 2) and 16 min (group 3) of training, as in §11.2. F and P
  should take less. With six model loads, the kernel setup and the gates, the total is about 1.5–2 A100-hours,
  or about 10–14 compute units at about 6.7 units per hour.

#### Results (run 2026-11-09, 08:54–10:50 UTC, one A100-SXM4-40GB; all six arms ran)

Balance read before each arm: 95.57, 95.01, 93.87, 92.77, 89.96 and 88.27 units, so every arm ran in the frozen
order. No code was changed after the freeze.

- **Stack:** flash-linear-attention and fla-core 0.6.0 and causal-conv1d 1.7.0 from pip (causal-conv1d built
  against torch 2.11.0 with the §8 pins held; the kernel hub was not used), triton 3.6.0. The fused environment
  took 1,051 s to set up and the §8 environment 83 s; neither is counted.
- **Gate 1 (kernels in use): passed.** One forward pass of the fused stack called the fla chunked delta rule 24
  times, the boundary-aware causal convolution 24 times and the per-segment flash attention 8 times, matching the
  layer counts, with no fallback warning. The §8 stack logged the `causal_conv1d` fallback warning, as recorded
  in §8.
- **Gate 2 (same function): passed.** On the four check sequences the summed target NLL under the fused stack was
  within 0.046%, 0.107%, 0.161% and 0.062% of the §8 stack.
- **Gate 3 (isolation): passed.** On the packed rows of the first four steps of group 3, the largest relative
  difference between an example in its packed row and the same example alone was 0.46%. The control (§8 stack,
  no boundaries) reached 62%, so the check detects leakage.

| Group | Arm | Training s | Real tokens/s | Forward passes | Processed tokens | Peak PyTorch / device GiB | Load s | Mean loss |
|---|---|---|---|---|---|---|---|---|
| 2 | R | 384.9 | 1,588 | 314 | 784,513 | 19.8 / 21.8 | 25.7 | 0.53768 |
| 2 | F | 400.8 | 1,526 | 314 | 784,513 | 19.9 / 28.6 | 45.1 | 0.53771 |
| 2 | P | 346.1 | 1,767 | 124 | 611,428 | 20.3 / 32.0 | 45.1 | 0.53783 |
| 3 | R | 828.8 | 2,920 | 563 | 2,652,705 | 24.2 / 25.9 | 26.3 | 0.75597 |
| 3 | F | 796.2 | 3,040 | 563 | 2,652,705 | 24.0 / 32.3 | 45.7 | 0.75603 |
| 3 | P | 837.2 | 2,891 | 131 | 2,420,158 | 26.4 / 36.7 | 45.4 | 0.75600 |

Real tokens per epoch: 611,428 (group 2) and 2,420,158 (group 3). Device memory is sampled.

| Group | R/F, epoch (median step after 5) | F/P, epoch (median step after 5) | R/P, epoch (median step after 5) |
|---|---|---|---|
| 2 | 0.960 (1.015) | 1.158 (1.381) | 1.112 (1.397) |
| 3 | 1.041 (1.047) | 0.951 (1.130) | 0.990 (1.193) |

- **Packing (F/P):** the expected direction held for group 2. Packing cut the forward passes from 314 to 124 and
  the processed tokens by 22%, and F took 1.16 times as long as P, within the padding bound of 1.28. For group 3
  it did not hold: the median step was 1.13 times faster, but the epoch was 5% slower. The gap between the
  median and the total means that a minority of packed steps were much slower than the rest. This run does not
  show why; the per-step times are in the reports.
- **Kernels (R/F):** about no effect. F was 4% slower than R for group 2 and 4% faster for group 3, against an
  expected ratio above 1. The fused stack also used 6.4–6.7 GiB more device memory while its PyTorch peak was the
  same, and loading it took about 19 s longer.
- **Same training:** the per-step losses agreed across arms: at most 0.0006 apart at the first step and 0.0041
  at any step, and the epoch mean losses differ in the fourth decimal.
- **Session effect:** R took 47% longer than the seed-1 record for group 2 (384.9 s against 261.5 s), above the
  10% threshold, and 9% less for group 3 (828.8 s against 914.5 s). As frozen, all ratios use this session only.
  R for group 2 was the first timed arm of the session.
- **Not claimed:** as listed above. With one run per arm, the differences of 4–5% are within what a session can
  vary, as the group-2 record comparison shows.
- **Cost:** about 16.7 compute units (102.17 before, 85.48 after; about 6.7 units per hour), above the estimate of
  10–14 because of the 17-minute build of the fused environment and the waits at the go prompts. This includes a
  few minutes on an L4 runtime, assigned when no A100 was free, that the GPU check stopped before any work.
- **Records:** `MyDrive/hpml_pack/pack_summary.json` and `.txt`, `gates/` (`ref.json`, `gate.json`),
  `runs/g{2,3}_{R,F,P}/train_report.json` (per-step times and losses), the six answers in `go/`, and the logs and
  pip freezes. `wait_go.json` and `current_job.json` there are copies from the last sync.

## 12. Exploratory: loop early stop (offline, chosen after the results)

`training/early_stop_estimate.py` applies a loop rule to the saved outputs of conditions 2–4 (all seeds and tiers):
every 1,000 characters, if the last 300 characters occurred within the previous 20,000 at distance p and the text
repeats with period p over the last max(2p, 1,500) characters, generation would stop there. The stopped prefix is
graded under §9 and its tokens are estimated from the output's own tokens per character. Reported per condition and
tier: outputs stopped (truncated or finished, correct before the stop or not), accuracy with and without the rule,
and the share of generated tokens saved. Nothing is generated, and the rule was chosen after the results, so this is
an estimate, not a result of the design.

Result (exploratory):

| Condition / tier | Outputs | Truncated | Stopped | Generated tokens saved | Median stop (tokens) | Accuracy, unchanged |
|---|---|---|---|---|---|---|
| 2 / tier 1 | 2,136 | 49 | 32 | 48.1% | 1,580 | 0.564 |
| 2 / tier 2 | 1,512 | 236 | 170 | 58.8% | 2,141 | 0.019 |
| 2 / tier 3 | 6,000 | 154 | 116 | 54.4% | 2,077 | 0.516 |
| 3 / tier 1 | 2,136 | 38 | 17 | 8.1% | 18,377 | 0.948 |
| 3 / tier 2 | 1,512 | 622 | 212 | 10.5% | 19,606 | 0.393 |
| 3 / tier 3 | 6,000 | 214 | 120 | 13.3% | 14,912 | 0.907 |
| 4 / tier 1 | 2,136 | 42 | 26 | 18.7% | 10,949 | 0.928 |
| 4 / tier 2 | 1,512 | 510 | 287 | 20.8% | 15,707 | 0.324 |
| 4 / tier 3 | 6,000 | 179 | 130 | 22.3% | 10,942 | 0.886 |

The rule stopped only outputs that later reached the budget, and no stop changed a grade, so accuracy is the same
with and without it in every cell. Condition 2 loops early and often, so the rule would save about half of its
tokens. Conditions 3 and 4 mostly run out of budget while still reasoning; the rule catches 34–73% of their
truncated outputs per cell, late, and would save 8–22% of their tokens. These are token estimates: generation time,
and what the freed slots would do under the queue of §11, were not measured.

### 12.1 Sensitivity and seed split of the loop rule (frozen 2026-11-05, before any run; still exploratory)

Question: is the §12 result a property of the one parameter setting picked after the results, or does it hold
around it, and does a setting chosen on one seed carry over to the other two? Script:
`training/early_stop_sensitivity.py` on a Colab CPU runtime (cell `training/colab_sens_cell.py`), over the same saved
outputs as §12 (conditions 2–4, 3 seeds, 3 tiers, read-only from `MyDrive/hpml_final/evals`); the result is
`early_stop_sensitivity.json` next to `early_stop_estimate.json` in `MyDrive/hpml_sched/analysis`. Nothing is
generated, no GPU and no API is used, and the rule, the §9 grading of the stopped prefix and the token estimate are
those of §12.

- **Grid:** each of the four parameters at half, the §12 value and double: probe 150 / 300 / 600 characters, minimum
  loop 750 / 1,500 / 3,000 characters, check every 500 / 1,000 / 2,000 characters, maximum period 10,000 / 20,000 /
  40,000 characters; all 81 combinations.
- **Reproduction check, first:** the §12 setting must reproduce `early_stop_estimate.json` in every cell (outputs,
  stops, accuracy with the rule, tokens saved). If it does not, nothing else is reported until the difference is
  explained.
- **Per setting**, per cell and per condition (tiers pooled by tokens): outputs stopped; stops of outputs that had
  finished; grade changes (correct to wrong, and wrong to correct, counted apart); share of generated tokens saved;
  share of truncated outputs stopped.
- **Sensitivity reading (fixed now):** the §12 setting is called stable if each of its 8 one-step neighbours (one
  parameter halved or doubled) has no grade change, no stop of a finished output and, in every condition, a saved
  share within 5 percentage points of the §12 setting. The change relative to the §12 share is reported next to
  each difference. Otherwise the parameters that break it are named. The whole grid is reported either way.
- **Seed split:** a setting is chosen on seed 1 alone: the largest share of tokens saved over conditions 2–4 and all
  tiers, among the settings with no grade change and no stop of a finished output on seed 1. Ties go to the more
  cautious setting (larger minimum loop, then larger probe, then larger check interval, then smaller maximum period).
  On seeds 2 and 3, each on its own, it is then reported against the §12 setting: grade changes, stops of finished
  outputs, and tokens saved per condition. The choice is said to carry over if each of seeds 2 and 3 has no grade
  change, no stop of a finished output, and in each condition a saved share within 5 percentage points of its seed-1
  value (relative change reported alongside).
- **Limits:** the three seeds share their problems and training data, so the split tests stability across training
  seed and sampling, not across problems. Both parts stay exploratory: the §12 setting itself was chosen after
  seeing all three seeds. Tokens are estimated from characters as in §12, and time is not measured.

#### Results (run 2026-11-05, Colab CPU runtime, 2 CPUs, 87 s)

- **Reproduction:** passed. The §12 setting reproduces `early_stop_estimate.json` in all nine cells, and its stop
  point equals `early_stop_estimate.stop_point()` for every output.
- **Whole grid:** none of the 81 settings changes a grade or stops a finished output, in any condition, tier or seed.
  Share of generated tokens saved, lowest to highest over the grid: condition 2 53.9–57.0%, condition 3 10.4–11.7%,
  condition 4 19.9–21.7%. The lowest is probe 150 / minimum loop 3,000 / check 2,000 / maximum period 10,000; the
  highest is minimum loop 750 / check 500 / maximum period 40,000 (probe 300 for condition 2, 600 for 3 and 4).
- **Sensitivity:** stable. The §12 setting saves 55.90% (condition 2), 11.26% (3) and 21.13% (4) of generated tokens
  and stops 318 of 439, 349 of 874 and 443 of 731 truncated outputs. None of its eight neighbours changes a grade or
  stops a finished output, and their saved shares are (difference in percentage points, relative change):

  | Neighbour | Condition 2 | Condition 3 | Condition 4 |
  |---|---|---|---|
  | probe 150 | 55.87% (−0.04, −0.1%) | 11.22% (−0.04, −0.3%) | 21.09% (−0.04, −0.2%) |
  | probe 600 | 55.93% (+0.03, +0.1%) | 11.29% (+0.03, +0.3%) | 21.17% (+0.04, +0.2%) |
  | minimum loop 750 | 56.66% (+0.75, +1.4%) | 11.54% (+0.28, +2.5%) | 21.46% (+0.33, +1.5%) |
  | minimum loop 3,000 | 54.53% (−1.38, −2.5%) | 10.72% (−0.53, −4.7%) | 20.44% (−0.69, −3.3%) |
  | check 500 | 56.21% (+0.31, +0.5%) | 11.38% (+0.12, +1.1%) | 21.31% (+0.18, +0.9%) |
  | check 2,000 | 55.16% (−0.74, −1.3%) | 11.03% (−0.23, −2.0%) | 20.75% (−0.39, −1.8%) |
  | maximum period 10,000 | 55.90% (0.00, 0.0%) | 11.26% (0.00, 0.0%) | 21.13% (0.00, 0.0%) |
  | maximum period 40,000 | 55.93% (+0.03, +0.1%) | 11.26% (0.00, 0.0%) | 21.13% (0.00, 0.0%) |

  The minimum loop length moves the saved share most; the largest difference is 1.38 points.
- **Seed split:** does not carry over under the fixed reading. On seed 1 all 81 settings have no grade change and no
  stop of a finished output, so the choice is the setting that saves most there: probe 600 / minimum loop 750 /
  check 500 / maximum period 10,000, 24.77% of the seed-1 tokens of conditions 2–4 (the §12 setting: 24.21%). On
  seeds 2 and 3 it changes no grade and stops no finished output, but two saved shares move more than 5 points from
  their seed-1 values:

  | Seed | Condition 2 | Condition 3 | Condition 4 |
  |---|---|---|---|
  | 1 (chosen there) | 58.84% | 14.07% | 24.27% |
  | 2 | 61.02% (+2.18, +3.7%) | 10.30% (−3.77, −26.8%) | 16.38% (−7.89, −32.5%) |
  | 3 | 49.71% (−9.13, −15.5%) | 10.53% (−3.54, −25.2%) | 23.83% (−0.44, −1.8%) |
  | §12 setting, seeds 1 / 2 / 3 | 57.92 / 59.98 / 48.45% | 13.63 / 9.90 / 10.07% | 23.70 / 15.92 / 23.34% |

  On seeds 2 and 3 the chosen setting saves 0.4–1.3 points more than the §12 setting in every condition, and the §12
  setting shows the same spread across seeds. The misses come from how much the outputs of each seed loop, not from
  the parameters: the saved share of a condition differs between seeds by up to 9.5 points under the §12 setting
  and 11.3 points under the chosen one.
- **Reading:** around the §12 setting, the loop rule's estimate depends little on its parameters, and in this grid it
  never changes a grade or stops a finished output. How much it saves is a property of each trained model and its
  samples, and varies between seeds by more than between parameter settings. All of this stays exploratory (see Limits).
- **Records:** `MyDrive/hpml_sched/analysis/early_stop_sensitivity.json` (all 81 settings, per cell, condition and
  seed), `early_stop_sensitivity.log` and `early_stop_sensitivity_pip_freeze.txt` (math-verify 0.9.0, Python 3.13).

## 13. Demo: the four conditions side by side (saved outputs, fixed selection)

The demo replays saved outputs of the final run (§9); nothing is generated. Script: `demo/demo_extract.py` on a Colab
CPU runtime (cell `demo/colab_demo_cell.py`), reading `MyDrive/hpml_final/evals` (outputs checked against their
SHA-256) and writing `MyDrive/hpml_demo/demo_data.json`.

- **Selection, fixed before any output is read:** in each tier the scored problem ids are sorted, and
  `random.Random(20261109 + tier number).sample()` draws 4 (tier 1), 4 (tier 2) and 2 (tier 3). No draw is replaced.
- **Outputs shown:** sample 0 of each condition: condition 1 from its one run, conditions 2–4 from training seed 1.
  The request seed of an item and sample is the same for every model (§9).
- **Shown next to each output:** tokens, finish reason, grade (§9), and for conditions 2–4 the point where the §12
  loop rule would stop it, with the grade of the stopped prefix. Each problem also shows its accuracy per condition
  over all seeds and samples, so a single sample can be read against the rate.
- **Playback:** the four texts are revealed at one constant rate in tokens, proportional to each output's length. This
  is not the recorded generation timing, which was not kept per output.
- **Not claimed:** the drawn problems illustrate the outputs; the results are those of §10.
