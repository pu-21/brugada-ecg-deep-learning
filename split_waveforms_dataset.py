from pathlib import Path

import numpy as np
import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parent
    x_path = base_dir / "ecg_waveforms_1200.npy"
    y_path = base_dir / "ecg_labels.npy"
    index_path = base_dir / "ecg_waveforms_index.csv"

    x = np.load(x_path)
    y = np.load(y_path)
    index_df = pd.read_csv(index_path)

    if len(x) != len(y) or len(x) != len(index_df):
        raise ValueError("X, y, and index table must have the same number of rows.")

    positive_mask = y == 1
    negative_mask = y == 0

    print(f"Total samples: {len(y)}")
    print(f"Negative samples (0): {int(negative_mask.sum())}")
    print(f"Positive samples (1): {int(positive_mask.sum())}")
    if (y == 2).any():
        print(f"Other label samples (2): {int((y == 2).sum())}")

    print("\nRecommended next step: use only binary labels (0 vs 1) if you are training a binary classifier.")
    print("Current dataset already preserves the full 1200 time steps and 12 leads.")


if __name__ == "__main__":
    main()
