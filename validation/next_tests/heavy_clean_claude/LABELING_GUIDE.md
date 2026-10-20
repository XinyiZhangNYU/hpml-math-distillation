# Heavy cleaning labeling guide (group 4, labeled by Claude)

The labeling convention is the one used for the human5 hand labels. The heldout5 reference labels
(`../heavy_clean_pilot/heldout_labels_r3.json`, labeler `claude`) follow the same convention.

The SHA-256 of the guide is recorded in every batch's label file and report. The labels were produced with the
Chinese original of this guide; this English version has the same content.

## 1. Units and format

- The reasoning (`reasoning_content`) is split into sentence units by `heavy_clean_v3.split_units`, numbered from 0.
  Batch files show these units with whitespace collapsed; the underlying text is unchanged.
- Only whole units are deleted; no text is rewritten. The final solution (`text`) is never touched.
- Traces with fewer than 300 reasoning tokens are not labeled; their heavy version equals the light version.

Each trace gets:

```json
{"main_derivation": {"first": i, "last": j},
 "delete": [
   {"first": a, "last": b, "label": "abandoned", "kind": "branch", "dropped_at": m, "why": "at most 15 words"},
   {"first": c, "last": d, "label": "repeat", "kind": "reverify", "same_as": {"first": k, "last": l}, "why": "..."}
 ]}
```

- Ranges are inclusive and deletion ranges do not overlap. If nothing is deleted, write `"delete": []`;
  `main_derivation` is still required.
- `kind` must be one of the four kinds in §2.

## 2. What to delete

Two labels, four kinds:

| label | kind | What is deleted | Required |
|---|---|---|---|
| abandoned | branch | An approach, guess or reading that the reasoning itself gives up, refutes or finds wrong, and that the final answer does not depend on. Examples: a guess refuted by a counterexample, a calculation found to be wrong, the part before "this does not work, try another way". | `dropped_at`: the unit where the reasoning gives it up or finds the error, inside the range or right after it |
| repeat | reverify | The second and later checks of a result that has already been obtained and checked (same method or a different one). | `same_as`: the earlier text it repeats |
| repeat | reread | Re-reading or restating an interpretation or condition that is already settled. | `same_as` |
| repeat | tail | Re-deriving or rewriting the solution steps at the end of the reasoning after the answer is already found. | `same_as` |

`same_as` must come before the deleted range and must not itself be deleted.

## 3. What not to delete

- **The main line**, including work whose results are used later. Even if a later passage redoes it with a cleaner
  method, the earlier passage is not deleted for being "replaced by a cleaner method": the earlier one is the main
  line and the later one is the repeat.
- **The first check.** The first verification of each result is kept; only the second and later ones are deleted.
- **The first reading of the problem.** Understanding the problem at the start, clarifying conditions, and
  comparing readings before choosing one are all kept. Only when the reasoning has worked under a reading and then
  abandons it as wrong is that worked part a branch.
- **Set-up used later**: coordinates, notation, variable definitions. Keep them if they are used later, even when
  they appear next to an abandoned approach.
- **Planning statements**, e.g. "first find x, then substitute back".
- **A short wrap-up at the end**, e.g. how to write the final answer or which format to use. Only re-deriving the
  steps at the end counts as tail.
- **When in doubt, do not delete.**

## 4. main_derivation

- The contiguous units that first reach the final answer; its last unit is the first one that states the final
  answer's value.
- Keep it short, with no re-derivations or re-checks inside. Earlier work that the answer uses is main line; it is
  not labeled and need not be inside main_derivation.
- These units are never deleted. Deletion ranges must not overlap it (the script reports an overlap as an error
  and does not trim it).

## 5. Per-batch procedure

1. Before each batch, if the context was compacted or this is a new session, re-read this guide and
   `examples_human5.md` (the human5 labels per unit).
2. Read only `batches/batch_NNN.md`. Do not read `batches/batch_NNN_key.json` or earlier batches' results. Each
   batch contains one calibration trace (one of heldout5, which has reference labels); it is not marked, so label
   it like the others.
3. Write the labels to `labels/batch_NNN.json`: `{"guide_sha256": "...", "labels": {"t01": {...}, ...}}`.
4. Run `python -I claude_label.py apply WORK_DIR NNN`. The script checks that:
   - every trace is labeled, all fields are present and every `kind` is valid;
   - no deletion touches main_derivation, every `same_as` comes earlier and is not deleted, and every
     `dropped_at` is in a valid position;
   - the result is a subsequence of the original and the final solution is unchanged;
   - a trace that would lose more than 80% is left undeleted and recorded.

   Any format or rule error fails the batch; fix the labels and rerun.
5. **Calibration gate:** the calibration trace is scored against its reference labels and passes at precision
   ≥ 0.70 and F1 ≥ 0.60. If it fails, stop; do not change the calibration trace's labels to pass. The differences
   are reviewed before the next batch.
