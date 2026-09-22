# 04 - Outline (IEEE conference, ~4,000 words body; body = Sections I-IX text, excluding abstract, table/figure contents, captions, references)

Contribution framing (system + empirical characterisation; not a new algorithm): a leakage-safe,
replayable serving pipeline around off-the-shelf TabPFN for small clinical tables, and a
head-to-head measurement of each design choice against its natural alternative.

| Sec | Title | Words | Ledger ids | Figures / tables (data source) |
|---|---|---|---|---|
| - | Abstract | <= 250 | R5-R19, T* marker | - |
| I | Introduction | 380 | P1, B1, B5, R-summary | - |
| II | Related work (4 themes) | 520 | B1-B11 | - |
| III | System design | 600 | P1-P10, I1 | Fig. 1 (drawn from `src/train.py`, `src/prediction.py`) |
| IV | Experimental setup | 400 | P2, P11, B10, B11 | Table I (datasets; `data/`) |
| V-A | Reference models and cost | 230 | R1-R4, T1-T2 | Table II (`tables.md`, `ablation_tables.md` E) |
| V-B | Preprocessing and leakage | 230 | R5-R7, T4 | Table III (`derived_stats.json`) |
| V-C | Decision threshold | 230 | R8-R10 | Table IV (`ablation_tables.md` B, B2) |
| V-D | Uncertainty band | 260 | R11-R15, I1, I4, T3 | Table V (`ablation_tables.md` C) |
| V-E | Novelty flag | 250 | R16-R19 | Fig. 2, Table VI (`ablation_tables.md` D) |
| V-F | Explanations and sample predictions | 120 | T5, T6 | Table VII placeholder |
| VI | Discussion: what the evidence supports | 380 | I2, I3 | Table VIII (verdict per choice) |
| VII | Limitations and threats | 280 | P10, P12, P13, B11 | - |
| VIII | Data, ethics, availability | 130 | P11, B11 | - |
| IX | Conclusion | 120 | - | - |

Rebalancing note: with TabPFN cells pending, Sections V-B..V-E carry the evidence; nothing is
padded to reach length.
