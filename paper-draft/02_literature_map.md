# 02 - Literature map

Reading depth: **full** = text read and saved in `sources/`; **abstract** = abstract only, so the
paper may claim only what the abstract states; **metadata** = existence only.
Notes are in my own words. Reference verification status: `references.verified.json`
(30/33 clean; the three others are resolved by hand, see the end of this file).

## A. Tabular foundation models and where they stand

| key | depth | note (own words) | use in paper |
|---|---|---|---|
| hollmann2025 (Nature) | full | Presents TabPFN as a transformer trained across millions of synthetic datasets, predicting in a single forward pass. Reports the largest advantage on datasets up to 10,000 samples and 500 features against baselines tuned for hours, on the authors' own benchmarks. States support for categorical features, missing values, and robustness to uninformative features and outliers. | What TabPFN is; all four datasets (303-768 rows, 10-25 features) fall inside the stated range. Note: the project imputes and winsorizes before TabPFN although the model claims to handle both natively - a design choice not tested here |
| hollmann2023 (ICLR) | full | The first TabPFN: a PFN trained offline on a causal-structure prior; evaluated on small numeric OpenML datasets (<=1,000 training rows, <=100 numeric features without missing values, <=10 classes), competitive with AutoML. | Lineage and the much narrower scope of the first version |
| muller2022 (ICLR) | full | Introduces prior-data fitted networks: train a set-input transformer on tasks sampled from a prior so that one forward pass approximates the posterior predictive. | Why "fitting" TabPFN stores the training table |
| grinsztajn2022 | full | Benchmark of 45 tabular datasets in which tree ensembles stayed ahead of neural networks at around 10K samples, plus an analysis of why. | Evidence that baselines matter; it predates the TabPFN generation, so it says nothing about TabPFN itself |
| closerlook2025 | abstract | Studies how TabPFN v2 copes with heterogeneous columns and how its limits (high dimension, many classes, large data) can be mitigated. | Only to characterise limits; no specific finding cited beyond the abstract |
| clinbench2026 | abstract, **preprint** | Twelve binary clinical tasks, TabPFN vs twelve established methods: generally competitive, best in 16.7% of tasks, most AUROC gaps within +/-0.01, about 5.5x longer runtime. Not peer reviewed. | Closest clinical evidence; the direct reason the paper includes baselines |
| ehrsmall2026 | abstract | The saved abstract says only that a fine-tuned TabPFN consistently outperformed conventional methods across tabular clinical prediction tasks. (An earlier note here named baselines and a dataset count that the saved abstract does not contain; removed in audit 2.) | Mixed evidence; this project does not fine-tune |

## B. Explaining TabPFN

| key | depth | note | use |
|---|---|---|---|
| lundberg2017 | full | Defines SHAP as a unified additive attribution framework. | Definition of SHAP |
| rundel2024 | full | Adapts interpretability methods (Shapley values, leave-one-covariate-out, data valuation) to TabPFN by exploiting in-context learning, avoiding retraining. | A TabPFN-specific alternative to the model-agnostic explainer this project uses |
| realtime2026 (workshop) | full | Proposes ShapPFN, a model that emits prediction and Shapley explanation in one pass; reports KernelSHAP as far slower in its setting. Workshop paper. | Motivation for caring about explanation latency; its numbers belong to its setting, not to this project |
| kumar2020 | full | Argues Shapley-value feature importance has mathematical problems whose mitigations bring extra complexity (e.g. causal reasoning). | Caution on reading SHAP output |

## C. Uncertainty, calibration, novelty

| key | depth | note | use |
|---|---|---|---|
| vovk2015 | full | Introduces cross-conformal prediction (inductive conformal + cross-validation) and studies validity empirically rather than proving it. | Lineage of the band |
| barber2021 | full | Jackknife interval = point prediction +/- quantile of leave-one-out residuals (their Eq. 7); it can fail badly. Jackknife+ / CV+ use the fold models' predictions at the test point and carry a worst-case 1-2a guarantee. | **Precise description of the project's band**: a K-fold analogue of the plain jackknife interval, not CV+ |
| angelopoulos2021 | full | Tutorial on conformal prediction: marginal coverage (averaged over calibration and test draws) is guaranteed; usefulness is set by the score function. | Definitions; validity is not informativeness |
| lee2018 | full | Mahalanobis-distance OOD score from class-conditional Gaussians fit on deep-network features. | Related idea; the project fits one shrunk Gaussian on encoded inputs instead |
| guo2017 | full | Shows modern networks are often poorly calibrated; temperature scaling is a simple fix. | Calibration of TabPFN probabilities is not assessed here |
| shan2015 | full | Explains the Youden index and its optimal cut-point; equal weight on sensitivity and specificity. | Context for the learned decision threshold |
| ledoit2004 | metadata | Shrinkage estimator for covariance. Full text not retrievable (403). Attribution taken from the scikit-learn documentation, which states that Ledoit and Wolf's 2004 paper gives the optimal shrinkage coefficient and that shrinkage avoids an unstable or non-invertible sample covariance. | Cited only for that attribution |

## D. Evaluation practice and leakage

| key | depth | note | use |
|---|---|---|---|
| kapoor2023 | full | Surveys leakage across 17 fields (329 papers) and gives an 8-type taxonomy; type L1 is lack of clean train/test separation during preprocessing, modelling and evaluation. | Motivates the leakage-safe design; the project addresses fit-on-train-only preprocessing, not every type |
| varoquaux2018 | full | Small samples give large error bars in cross-validation (about +/-10% at 100 samples in his setting); the spread across folds understates them. | Why bootstrap intervals and repeated CV are reported on 61-154-row test sets |
| nadeau2003 | full | Ordinary variance estimates for CV comparisons are biased low; proposes a corrected resampled t-test with variance factor 1/J + n2/n1. | Used for the paired TabPFN-vs-baseline comparison |
| brown2001 | abstract | Recommends the Wilson interval for small n. | Coverage intervals |
| diciccio1996 | abstract | Survey of improved bootstrap confidence intervals (BCa etc.). | The paper uses the simple percentile bootstrap and says so |
| pedregosa2011 | full | scikit-learn library paper. | Baselines and utilities |

## E. Data and disease-prediction literature

| key | depth | note | use |
|---|---|---|---|
| detrano1989 | abstract | The Cleveland-derived model (303 patients) was tested on patients from Hungary, Long Beach, and Switzerland. | Provenance of the heart data; the project uses the Cleveland subset only |
| uci_heart / uci_ckd / uci_ilpd | full (UCI pages) | Dataset pages: heart 303 instances, missing values yes, donated 1988-06-30; CKD 400 instances, missing values yes, donated 2015-07-02; ILPD 583 instances, no missing values listed, donated 2012-05-20; all CC BY 4.0. Suggested citation years are 1989, 2015, 2022. | Data section |
| pima_docs, smith1988 | full (docs) / metadata | Pima documentation: 768 instances, all patients female, at least 21, Pima heritage; lists the ADAP study of Smith et al. as past usage. | Data section and limitations |
| straw2022 | full | Recreates published ILPD classifiers and finds a higher false-negative rate for females across all classifiers. | Limitation: subgroup performance is unassessed here |
| islam2023 | abstract | CKD study comparing 12 classifiers; XGBoost reached accuracy 0.983 using about 30% of the variables. | Prior results on the CKD data sit near ceiling |
| bouqentar2024 | abstract | Heart-disease ML system using Cleveland and Statlog data, comparing several classical algorithms, with feature engineering. | Prior work on the same data |

**Considered and not used:** Shwartz-Ziv & Armon (fetched, not needed). Youden (1950): full text
blocked (403) and the registry abstract is unusable, so it is **not cited**; the Youden
index is described through Shan (2015). The deleted `threshold_sweep_results.md` numbers are
not evidence (see `01_project_understanding.md`).

## Positioning

**1. What already exists.** A large applied literature runs classical classifiers on these four UCI
datasets and reports accuracies from the 70s (liver) to the high 90s (CKD). TabPFN and its
successors are established as small-data tabular predictors, with mixed evidence in clinical
settings (a preprint finds it competitive but rarely best; a fine-tuning study finds gains).
Explanation methods tailored to TabPFN exist, and generic conformal and Mahalanobis tools are well
described.

**2. Limitations found.** (a) Clinical evidence for TabPFN is mixed and partly unrefereed.
(b) Shapley-based explanations are costly for foundation models (ShapPFN) and have conceptual
caveats (Kumar et al.). (c) Jackknife-type residual intervals lack the guarantee that CV+ has
(Barber et al.); a conformal guarantee says nothing about how informative the sets are
(Angelopoulos & Bates). (d) Published ILPD classifiers show a sex disparity (Straw & Wu).
(e) Leakage is a documented cause of overoptimistic results (Kapoor & Narayanan). *Own
observations, not attributed to sources:* I did not audit the application papers for confidence
intervals or leakage; the paper makes no claim about them.

**3. Where this project fits.** Between the model and the user: an integrated, reproducible
serving pipeline around off-the-shelf TabPFN for small clinical tables, with cheap precomputed
uncertainty add-ons, and an evaluation that is honest about small test sets and includes baselines.

**4. What it contributes (framed).** A system contribution (config-driven, leakage-safe pipeline
with a replayable transform and a single model bundle) plus an empirical characterisation of the
add-ons: whether the band attains its nominal coverage, whether it is informative, and whether the
novelty flag responds to shift. It does **not** contribute a new algorithm.

**5. Evidence.** Real, provenance-stamped results exist for the baselines, the band coverage
(baselines) and the novelty flag (`paper-draft/results/`); the TabPFN runs, the corrected paired
comparison against the baselines, and the SHAP timing are **pending the licence token**. 78 tests
pass.

## Manual verification notes (from `verify_refs.py`)

- `vovk2015`: Crossref lists online 2013 and print 2015 (vol. 74, issue 1-2, pp. 9-28); cited by
  print year. The script's year "mismatch" compares against the online date.
- `ehrsmall2026`: absent from Crossref/OpenAlex/arXiv; confirmed via Europe PMC record
  PMC13274287 (title, authors, AMIA Joint Summits proceedings, 2026, pp. 604-613).
- `pima_docs`: a documentation file (not scholarly, so unindexed); fetched and read.
- Venue strings for `hollmann2023`, `muller2022` (ICLR), `lundberg2017`, `lee2018` (NIPS), and
  `realtime2026` (FM4Science workshop) were taken from the text of the PDFs read; the year
  warnings are the usual preprint-vs-proceedings gap.
- `smith1988` was matched by title search to its PMC copy; citation details are those listed in
  the dataset documentation. Read depth is metadata only.
