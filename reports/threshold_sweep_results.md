# threshold sweep

ran scripts/sweep_thresholds.py against the tabpfn test predictions

| disease  | best t | f1    | acc   |
|----------|--------|-------|-------|
| diabetes | 0.42   | 0.887 | 0.914 |
| ckd      | 0.38   | 0.941 | 0.952 |
| liver    | 0.47   | 0.803 | 0.871 |
| heart    | 0.40   | 0.899 | 0.928 |

liver needs the higher cutoff, the positive class is pretty small there
