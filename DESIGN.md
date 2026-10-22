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

## 10. Not frozen yet

- The final evaluation script and the vLLM scheduling benchmark (they implement §9 and §7 without changing them).
