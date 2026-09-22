# Audit report

Draft: `paper.md` (master), built to `paper.tex`, `references.bib`, `paper_numbered.md` by `tools/build_paper.py`.
First audited 2026-09-21 (reference-model-only draft); re-audited 2026-09-22 after the user obtained a
`TABPFN_TOKEN` and accepted the TabPFN-3 non-commercial licence themselves in the browser, unblocking the
full TabPFN run (`paper-draft/experiments/run_tabpfn_all.sh`). Every cold read below is against the
repository, `results/raw/`, and `sources/` as they stand now.

**Status: COMPLETE.** No `{{PENDING}}` markers remain. Every number in the paper traces to a result file or
the repository; none was estimated.

| Audit | Result |
|---|---|
| 1. Project accuracy | 203 results-table cells recomputed by script, 0 mismatches (checker mutation-tested twice); 11 prose/code-path errors found and fixed in the first pass, 0 new ones in the second |
| 2. Citation | 31 citations, 1:1 with 31 bibliography entries; 6 attributions fixed (plus 1 reworded for overlap) in the first pass; 0 new citations added with the TabPFN content, 0 unresolved |
| 3. Reference verification | 19 VERIFIED, 9 with warnings (read and explained), 3 non-clean resolved by hand; 0 retracted; unchanged since no new references were added |
| 4. Originality | 0 verbatim runs (>= 8 words), 0 structure-preserving sentences over 6,740 draft words, 32 sources; 2 hits found and rewritten in the first pass |
| 5. Unsupported claims | 6 claims softened in the first pass; new TabPFN prose swept and found already hedged (see below); 0 unresolved |

## 1. Project-accuracy audit

**Numbers.** `experiments/audit_numbers.py` now re-derives every cell of Tables I-IX (previously I-VI) from
`results/raw/*.json`, `results/derived_stats.json`, `results/worked_examples.md` and `data/processed/*.csv`,
and diffs it against `paper.md`: **203 cells checked, 0 mismatches** (up from 124 cells before the TabPFN run;
the extra checks cover Table III's TabPFN-vs-baseline comparison, Table IV's TabPFN preprocessing columns,
Table V/VI's three-model rows, Table VIII's SHAP timings and the single-row-vs-batched latency prose, and
Table IX's worked examples against `worked_examples.md`). The checker was mutation-tested twice (four cells
deliberately altered across the two passes; all four caught; file restored each time).

**New code-path / provenance claims re-read today:**

| Paper says | Evidence | Verdict |
|---|---|---|
| TabPFN weights are the "v3" default checkpoint, downloaded once under a non-commercial licence the account holder accepted | `main.py` run log: `Downloading model to .../tabpfn-v3-classifier-v3_default.ckpt`; browser session, user accepted the TabPFN-3 licence page directly | MATCH |
| `TABPFN_TOKEN` is the exact env var the installed `tabpfn` package reads | `tabpfn/browser_auth.py:81,85` | MATCH |
| TabPFN's own ablations ran at one repeat (5 folds) vs the reference models' three (15) | `len(json.load(...)['aucs'][...])` == 5 for `ablate_*_tabpfn.json`; == 15 for reference-model files | MATCH |
| TabPFN accepted the native-missing-value preprocessing variant (unlike logistic regression) | `run_ablations.py`: `not (v == NATIVE_MISSING and (name == "logreg" or (dry_run and name == "tabpfn")))` - excludes tabpfn only in dry-run | MATCH |
| Runs span four repository commits, all dirty-flagged, with unchanged preprocessing/feature-engineering/artifact code | provenance stamps in `results/raw/*.json` show heads `5667851`, `0d70d3d`, `8fb4896`, `4029225`; `git diff 5667851 4029225` on those three files is empty | MATCH |
| 81 tests still pass after the `src/uncertainty.py` docstring fix | `pytest tests/` -> 81 passed | MATCH |

No new prose/code-path errors were found in this pass. The 11 errors fixed on 2026-09-21 (engineered-feature
sign count, kidney band width vs h, novelty speed range, kidney shift AUROC range, diabetes CI width,
threshold p bound, "never differed", deployment wording, the `src/uncertainty.py` docstring, the earlier
band-description doc errors, and the single-commit implication) remain fixed; see the previous version of
this report in git history / the conversation log for their detail.

## 2. Citation audit

Unchanged from the first pass: 30 of 31 cited works have a saved text in `sources/` (the exception,
`ledoit2004`, is metadata-only, cited only for the shrinkage attribution via the scikit-learn docs). No new
citations were needed for the TabPFN content — every new claim is a project result, not a literature claim —
so the citation table from the first audit stands. One line changed: the `[@realtime2026]` sentence in
Related Work now explicitly forward-references the test in Section V-F ("KernelSHAP as far slower in its
setting — a finding Section V-F tests directly"), which the results section confirms only holds on two of
four datasets; this is flagged in Section V-F's own text, not overstated in Related Work.

## 3. Reference verification

Unchanged: `references.cited.verified.json` from 2026-09-21 still applies (19 VERIFIED, 9 warnings, `vovk2015`
MISMATCH resolved by print-year, `ehrsmall2026` and `pima_docs` NOT_FOUND resolved by hand). No references
were added or removed when the TabPFN results were filled in.

## 4. Originality audit

`overlap_check.py paper.md --sources sources/*.txt` on the finished draft (6,740 words, up from 5,275):
**0 verbatim runs, 0 structure-preserving sentences.** All of the new TabPFN prose (Sections V-A, V-B, V-C,
V-D, V-F, VI, VII, IX) is original reporting of this project's own measured numbers, so it could not overlap
with the saved third-party sources; the check confirms it does not. Hand check repeated: the new paragraphs
each carry project-specific numbers (AUC values, p-values, timings) rather than generic description.

## 5. Unsupported-claim audit

Swept the full draft again, including every sentence added since the TabPFN run (see the grep-based sweep
in-session: `best`, `never`, `significant`, `guarantee`, `unambiguous`, etc.). Every hit found is already
qualified by a number or an explicit comparison — e.g. "TabPFN reached the highest or joint-highest... AUC on
three of four tasks" (not "TabPFN is best"), "significantly better than the SVM on three of the four tasks...
never significantly different from logistic regression" (both directions stated), "does not support calling
any single component, including TabPFN, unambiguously the best available choice" (an explicit disclaimer, not
a claim). No new unsupported claim was introduced. The multiplicity disclosure in Section VII was extended to
cover the 16 new TabPFN-vs-baseline comparisons and the enlarged threshold/preprocessing comparison counts
(24->36 threshold, 44->68 preprocessing).

## Open items

1. **Not compiled.** No TeX toolchain on this machine. `paper.tex` / `references.bib` are generated, not
   visually checked; `resizebox` tables and IEEEtran layout (now with a two-column verdict table and a
   two-column novelty table) need a visual pass once a toolchain is available. `paper_numbered.md` is a
   readable copy in the meantime.
2. **Length: 4,796 body words against the ~4,000 target (+20%).** The overage is real evidence, not padding:
   filling the TabPFN cells added a new results subsection (V-A's significance table), a new discussion
   paragraph, and an extended limitations paragraph, none of which existed to trim from the 3,835-word
   reference-model-only draft. Two trimming passes cut roughly 460 words without losing any distinct
   quantitative claim; further cuts would start dropping specific numbers rather than prose. Flagging this
   rather than silently exceeding the target, per the length rule.
3. **Harness is untracked.** `paper-draft/` (including the new `latency_batch_vs_single.py` and the extended
   `derived_stats.py`) is still not committed to the repository; result files carry a dirty-tree flag for
   this reason. Commit it (minus `sources/`, minus the local `.env` with the token) before submission.
4. Affiliation in the author block is still a placeholder.
5. `ledoit2004` remains metadata-only (the publisher host failed TLS verification; not bypassed).
6. Reminders (yours): run your institution's similarity checker; follow the venue's policy on disclosing
   AI-assisted writing; be able to explain each claim without the paper open (`viva_questions.md`, extended
   with TabPFN-specific questions); revoke or rotate the `TABPFN_TOKEN` if this machine's `.env` is ever
   shared or committed by mistake.
