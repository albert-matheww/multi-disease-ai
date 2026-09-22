# Audit report

Draft: `paper.md` (master), built to `paper.tex`, `references.bib`, `paper_numbered.md` by `tools/build_paper.py`.
First audited 2026-09-21 (reference-model-only draft); re-audited 2026-09-22 after the user obtained a
`TABPFN_TOKEN` and accepted the TabPFN-3 non-commercial licence themselves in the browser, unblocking the
full TabPFN run; audited a third time 2026-09-23 after adding a live-implementation section (Sec. V-G, two
screenshots of the Streamlit app running locally against the real trained bundle). Every cold read below is
against the repository, `results/raw/`, `results/screenshots/`, and `sources/` as they stand now.

**Status: COMPLETE.** No `{{PENDING}}` markers remain. Every number in the paper traces to a result file, a
screenshot's own source artifact, or the repository; none was estimated.

| Audit | Result |
|---|---|
| 1. Project accuracy | 213 results-table/prose cells recomputed by script, 0 mismatches (checker mutation-tested twice); 11 prose/code-path errors found and fixed across three passes, 0 new ones in the second and third |
| 2. Citation | 31 citations, 1:1 with 31 bibliography entries; 6 attributions fixed (plus 1 reworded for overlap) in the first pass; 0 new citations needed for the TabPFN content or the screenshots, 0 unresolved |
| 3. Reference verification | 19 VERIFIED, 9 with warnings (read and explained), 3 non-clean resolved by hand; 0 retracted; unchanged since no new references were added |
| 4. Originality | 0 verbatim runs (>= 8 words), 0 structure-preserving sentences over 7,143 draft words, 32 sources |
| 5. Unsupported claims | 6 claims softened in the first pass; new TabPFN and implementation prose swept and found already hedged (see below); 0 unresolved |

## 1. Project-accuracy audit

**Numbers.** `experiments/audit_numbers.py` now re-derives every cell of Tables I-IX from
`results/raw/*.json`, `results/derived_stats.json`, `results/worked_examples.md`, `data/processed/*.csv`, and
(new this round) `reports/heart_metrics.json` — the exact file the live app's "Model Performance" page reads
— and diffs it against `paper.md`: **213 cells checked, 0 mismatches** (124 before the TabPFN run, 203 after
it, 213 after adding the Section V-G screenshot numbers: accuracy/F1/AUC/threshold/precision/recall at both
cutoffs and the four confusion-matrix cells). The checker was mutation-tested twice (four cells deliberately
altered across two earlier passes; all four caught; file restored each time).

**Screenshots (Sec. V-G).** Both figures are real captures, not mockups: `capture_screenshots.py` drives a
headless Chromium (Playwright) against the Streamlit app running locally on the trained TabPFN bundle
(`streamlit run app/streamlit_app.py --server.port 8532`), fills the heart-disease form with the exact patient
from Table IX's "most novel" row, submits it, and screenshots the live result and the "Model Performance"
page. I checked by hand that the on-screen probability (97.1%), band (31%-100%), label, novelty warning and
every Model Performance number match `reports/heart_metrics.json` and the paper's own Tables II and IX. The
one place the screenshot does *not* match the harness exactly — the live SHAP values (ca +0.136 vs the
harness's +0.139, etc.) — is disclosed in the paper's own text as a real, unseeded-RNG difference in the
shipped code, not smoothed over.

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

`overlap_check.py paper.md --sources sources/*.txt` on the finished draft (7,143 words, up from 5,275 before
the TabPFN run): **0 verbatim runs, 0 structure-preserving sentences.** Section V-G (the screenshots'
description) is entirely a description of this project's own running app and its own artifact
(`reports/heart_metrics.json`), so it has no third-party source to overlap with; the check confirms 0 hits.
Hand check repeated: every new paragraph, including V-G, carries project-specific numbers rather than generic
description.

## 5. Unsupported-claim audit

Swept the full draft again, including every sentence added since the TabPFN run and the screenshot section
(grep sweep in-session: `best`, `never`, `significant`, `guarantee`, `unambiguous`, etc.). Every hit found is
already qualified by a number or an explicit comparison — e.g. "TabPFN reached the highest or joint-highest...
AUC on three of four tasks" (not "TabPFN is best"), "significantly better than the SVM on three of the four
tasks... never significantly different from logistic regression" (both directions stated), "does not support
calling any single component, including TabPFN, unambiguously the best available choice" (an explicit
disclaimer, not a claim). Section V-G states the SHAP-value mismatch as "a gap in the shipped code this
comparison surfaced" rather than hiding or minimising it. No new unsupported claim was introduced. The
multiplicity disclosure in Section VII was extended to cover the 16 new TabPFN-vs-baseline comparisons and the
enlarged threshold/preprocessing comparison counts (24->36 threshold, 44->68 preprocessing).

## Open items

1. **Not compiled.** No TeX toolchain on this machine. `paper.tex` / `references.bib` are generated, not
   visually checked; the two new screenshot figures (Fig. 3, Fig. 4) are tall single-column images and may
   need a `\resizebox` or two-column `figure*` treatment once a toolchain is available to check layout.
   `paper_numbered.md` is a readable copy in the meantime.
2. **Length: 5,080 body words against the ~4,000 target (+27%).** The overage is real evidence, not padding:
   filling the TabPFN cells (round 2) and adding the live-implementation section with reasoning for every
   design choice (round 3, this request) each added content that did not exist to trim from the 3,835-word
   first draft. Two trimming passes on round 2 cut roughly 460 words without losing a quantitative claim;
   round 3 was not further trimmed because the user explicitly asked for the screenshots and the step-by-step
   reasoning that account for the added length. Flagging this rather than silently exceeding the target.
3. **Harness is untracked.** `paper-draft/` (including `capture_screenshots.py`, `latency_batch_vs_single.py`
   and the extended `derived_stats.py`) is committed to the repository as of `f98de79`; this round's additions
   (the screenshot script, the two composited figures, `reports/heart_metrics.json`'s dependency) are not yet
   committed — see the commit made right after this report for that.
4. Affiliation in the author block is still a placeholder.
5. `ledoit2004` remains metadata-only (the publisher host failed TLS verification; not bypassed).
6. **The screenshots show a real, minor implementation gap**, not previously documented: the interactive SHAP
   path (`src/prediction.py`) does not seed its background sample, so repeated live predictions on the same
   patient can show slightly different SHAP values run to run (same dominant features, different exact
   numbers). This is disclosed in Section V-G rather than fixed, since fixing it would be a code change beyond
   this paper's scope; flagging it here so it is not mistaken for a screenshot error.
7. Reminders (yours): run your institution's similarity checker; follow the venue's policy on disclosing
   AI-assisted writing; be able to explain each claim without the paper open (`viva_questions.md`); revoke or
   rotate the `TABPFN_TOKEN` if this machine's `.env` is ever shared or committed by mistake.
