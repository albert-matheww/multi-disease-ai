# Viva questions (with the evidence that answers each)

Be able to answer these without the paper open. Paths are relative to the repository root.

1. **Why does the paper say the band "coincides with a conformal label set"?**
   For a binary label, |y - p| = 1 - p_y (p_y = probability given to the true label), so the interval [p-h, p+h]
   contains label y exactly when 1 - p_y <= h. Empirically identical in all 8 model-disease pairs
   (`paper-draft/results/ablation_tables.md` C; `derived_stats.json` -> `band_identical_to_label_set_in_all_pairs`).
2. **Is the band a 90% interval for the patient's true risk?** No. Its target is the binary label, h is one global
   constant per disease (`src/uncertainty.py`), and coverage is approximate. Coverage was 0.900-0.930 across all three
   models tested, including TabPFN.
3. **Is it CV+?** No: CV+ evaluates all K fold models at the query point (Barber et al., Sec. on CV+); the shipped
   band uses one full-data model. Measured coverage differs from CV+ by at most 0.018 (all three models).
3b. **Why is TabPFN's kidney band almost zero width?** TabPFN's out-of-fold residuals on kidney disease are tiny
   (mean h = 0.001) because its fitted probabilities cluster at 0 or 1 with almost no error on this near-separable
   task; all three kidney worked examples land on an exact [0,0] or [1,1] band (`results/worked_examples.md`). It is
   the band behaving as designed, not a bug, but it leaves no calibrated margin for the rare patient it gets wrong.
4. **Your preprocessing ablation found no differences, for TabPFN either? Then why keep the pipeline?** 68 comparisons
   total (44 reference-model, 24 TabPFN, one repeat), all p >= 0.10; TabPFN's own ablation shows the identical
   pattern as the reference models. Because it is not shown worse, the leak-safe fit is a correctness requirement
   (fit on train only, `tests/test_preprocessing.py`), and the engineered features are motivated clinically. The paper
   does not claim they raise accuracy. Absence of significance is not equivalence (Sec. VI; 15 or 5 overlapping folds).
5. **Why is the leaky variant only 0.0015 AUC off?** Medians and IQR bounds are low-capacity statistics on 300-770 rows;
   this says nothing about flexible preprocessing (target encoding, feature selection), which was not tested.
6. **Youden vs 0.5 vs prevalence?** Youden beat 0.5 significantly only on liver (p = 0.001, 0.002); it never differed
   significantly from the prevalence rule (`ablation_tables.md` B2). The liver gain trades sensitivity (0.53) for specificity.
7. **What is TabPFN's actual accuracy, and is it significant?** CV AUC: heart 0.918, diabetes 0.843, kidney 1.000 (tied),
   liver 0.741 (`results/tables.md`, `results/ablation_tables.md` E). Paired Nadeau-Bengio test against each reference
   model (`derived_stats.json` -> `tabpfn.pairwise_vs_baselines_raw`): significantly ahead of the SVM on 3/4 tasks and
   gradient boosting on 1/4 (diabetes, p=.047); never significantly different from logistic regression (p=.145-.734)
   or random forest (p=.071-.653) on any task.
8. **Why should TabPFN beat logistic regression here, and does it?** No claim that it does. Logistic regression had
   the best or joint-best CV AUC on liver and was statistically indistinguishable from TabPFN everywhere else. The
   honest framing (Sec. VI): TabPFN removes the tuning burden (0 fits vs the baselines' 16-61) without costing
   accuracy, not that it buys a large accuracy gain. This matches the related-work preprint: competitive, rarely
   decisively best against a strong baseline.
8b. **How did you get TabPFN results if the model needs a paid licence token?** The user obtained a free
   non-commercial API token from ux.priorlabs.ai themselves (I navigated the sign-up/login flow but never touched
   password fields or the licence-acceptance checkbox myself) and accepted the TabPFN-3 non-commercial licence in
   the browser. The token lives only in a local, gitignored `.env`; I never printed or committed it.
8c. **The SHAP background-size cut (25->10) was unbenchmarked in the shipped code. Did it turn out to be justified?**
   Yes, the clearest win in the paper: 1.3-2.4x faster with mean Spearman rank agreement 0.92-0.96 across datasets
   (as low as 0.67 for one diabetes patient). KernelSHAP was NOT uniformly slower as the cited workshop paper's
   setting suggested — it beat permutation-25 on 2 of 4 datasets (`results/raw/shap_*.json`).
8d. **Give one honest example of the model being wrong.** Liver's most-novel test row (flagged out-of-distribution,
   ALT 189) has true label disease-present but predicted probability 0.741 sits just under the 0.752 threshold, so
   the served label is wrong ("No Liver Disease"). One anecdote, not a systematic claim, but it is a real error on
   a real held-out row (`results/worked_examples.md`), and it is exactly the kind of patient the novelty flag warns about.
9. **Are your test sets big enough?** No: 61-154 rows on the shipped split; bootstrap intervals are 0.10-0.14 wide
   (`tables.md`). That is why comparisons use 15-fold repeated CV with the corrected t-test.
10. **Why is the one-class SVM more sensitive, yet you keep Mahalanobis?** Higher shift AUROC on all four datasets,
    but it flagged 11-24% of normal held-out rows against a 2.5% target (`ablation_tables.md` D). The project's flag held 0.030-0.036.
11. **What does the novelty result prove?** Only that the flag responds to one covariate shift (older vs younger) on heart,
    diabetes and liver; on kidney data no detector separated the groups (AUROC 0.64-0.69).
12. **What did you find wrong with the project while writing?** The band was mis-described in the docs (fixed), the metrics
    used a different threshold than serving (fixed, `src/evaluate.py`), and an untraceable threshold-sweep report was excluded.
13. **Is this clinically valid?** No. Public datasets, no calibration or subgroup assessment, no external validation (Sec. VII-VIII).
14. **How were the reference-model hyperparameters chosen?** Small inner grids on the training split in the main comparison
    (26/61/21/16 fits); fixed values in ablations so only one factor varies (`run_experiments.py`, `run_ablations.py`).
15. **Can someone reproduce this?** `paper-draft/experiments/` (stamps git head and package versions in each raw
    file); the TabPFN stages need the reader's own `TABPFN_TOKEN` and licence acceptance at ux.priorlabs.ai, run
    with `run_tabpfn_all.sh`. Caveat: the harness is currently untracked and results carry a dirty-tree flag.
16. **Why does single-patient prediction feel slow in the app but the paper's cross-validation numbers look fast?**
    A single unbatched `predict_proba` call costs 0.70-1.17s; the same 20 rows scored in one batched call cost only
    32-59ms per row (18-20x faster), measured directly on the same fitted model (`results/raw/latency_*.json`,
    `latency_batch_vs_single.py`). TabPFN's efficiency claim is about not needing a hyperparameter search, not about
    single-request latency, and the shipped dashboard's one-patient-at-a-time path does not batch.
