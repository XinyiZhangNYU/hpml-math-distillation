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

**Light cleaning** (groups 3 and 4), `light_clean.py`:

| Rule | What it deletes |
|---|---|
| L1 loops | Any run of ≥ 20 words repeated ≥ 3 times back to back. The first copy is kept. |
| L2 garbage | U+FFFD, private-use and unassigned code points, and control characters other than `\n` and `\t`. |
| L3 script switch | A sentence in a writing system the problem does not use. Greek letters and math symbols do not count. |

There is a safety valve: a trace that would lose more than 10% of its reasoning counts as an admission failure.

- "Content unrelated to the problem" is not a rule, because it has no mechanical definition.
- Result: 0 triggers, so the light-cleaned traces are byte-identical to the raw traces.

**Heavy cleaning** (group 4 only), `claude_label.py`:

- Scope: only traces with ≥ 300 reasoning tokens (797 traces). Shorter traces equal their light-cleaned version.
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
| 3 | LoRA on light-cleaned traces (main) | `final_traces.jsonl` (raw = light) |
| 4 | LoRA on heavy-cleaned traces | `group4_heavy.jsonl` |

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
| 3 (raw = light) | 2,420,158 | 1,227.9 | 534 | 23,538 | 44 / 8 / 0 |
| 4 (heavy) | 1,750,516 | 888.1 | 512 | 20,162 | 12 / 2 / 0 |

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
| 3 (light-cleaned traces) | 0.895 | 0.910 | 0.878 | **0.915** | 0.890 | 0.905 |
| 4 (heavy-cleaned traces) | 0.870 | 0.875 | 0.878 | **0.898** | 0.880 | 0.885 |

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
| 3 lightly cleaned traces | 0.948 [0.918, 0.977] | 0.393 [0.310, 0.476] | 0.907 [0.889, 0.925] |
| 4 heavily cleaned traces | 0.928 [0.887, 0.969] | 0.324 [0.249, 0.399] | 0.886 [0.867, 0.904] |

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
| 3 lightly cleaned | 2,420,158 | 937.3 (914.5–964.6) | 2,583 | 0.91 | 25.6 (max 27.3) | 28.0 (max 29.7) |
| 4 heavily cleaned | 1,750,516 | 668.0 (608.5–705.1) | 2,632 | 0.89 | 23.1 | 25.0 |

Training time follows the number of real tokens: group 4 has 0.72 times the tokens of group 3 and needs 0.71 times
the time. Group 2 has the lowest throughput and the most padding. The nine adapters took
5,607 s (1.56 h) of training in total; each merge took 62–132 s.

Evaluation (§9, all three tiers; mean per seed, condition 1 has one run):

| Condition | Generation time (h) | Output tokens (millions) | Output tokens/s |
|---|---|---|---|
| 1 base model | 2.36 | 6.61 | 777 |
| 2 human solutions | 1.82 | 5.63 | 859 |
| 3 lightly cleaned traces | 4.46 | 15.37 | 956 |
| 4 heavily cleaned traces | 3.49 | 12.50 | 995 |

The ten evaluations took 32.2 h of job time, about 20 times the training time, so generation dominates the cost of
this study. Condition 4 generates 19% fewer tokens than condition 3 and needs 22% less generation time; it is also
less accurate (§10).

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
