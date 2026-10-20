# Frozen design

Everything in sections 1–7 is frozen: the data, prompts, rules and test sets are fixed
and pinned by SHA-256, and they will not change once training starts. Section 8 lists what is still open.

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

- Grader: math-verify 0.9.0 through `validation/eval_common.py` (`37aa9b22…`), applied to the last `\boxed{}`.
- Generation budget: up to 32k tokens. The same outputs are also scored after truncation at 4k, 8k and 16k
  tokens.
- Reported per tier: accuracy, truncation rate and output length.
- Statistics: several samples per problem are averaged. The conditions are compared with paired, clustered
  confidence intervals following Miller (2024).
- Systems experiment: on vLLM, fixed-size request groups versus a continuously refilled queue. We measure
  throughput, latency, memory, and LoRA training cost.

## 8. Not frozen yet

- Training hyperparameters (learning rate, epochs, LoRA rank and targets, packing), tuned on dev data, and
  the number of training seeds.
- The number of samples per problem at evaluation, the cluster unit for the confidence intervals, and which
  budget is the headline (32k with prefix rescoring is the plan).
- The prompt for condition 1 (no training): the training chat template zero-shot, or the Minerva-style
  4-shot prompt used in the base probes.
- The training script, the final evaluation script, and the vLLM scheduling benchmark.
