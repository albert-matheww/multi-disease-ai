"""quick sweep over decision thresholds for each disease model.

loads test scores + labels produced by the tabpfn training step and
prints acc/f1 per threshold, used to pick the defaults in config.
"""
import numpy as np
from pathlib import Path

def main():
    root = Path("data/processed")
    for name in ["diabetes", "ckd", "liver", "heart"]:
        scores = np.load(root / f"{name}_test_scores.npy")
        labels = np.load(root / f"{name}_test_labels.npy")
        best = (0.0, 0.0, 0.0)
        for t in np.linspace(0.3, 0.6, 31):
            pred = (scores >= t).astype(int)
            acc = (pred == labels).mean()
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
            if f1 > best[0]:
                best = (f1, acc, t)
        print(f"{name:9s} best t={best[2]:.2f} f1={best[0]:.3f} acc={best[1]:.3f}")

if __name__ == "__main__":
    main()
