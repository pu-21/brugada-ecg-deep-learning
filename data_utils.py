from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_feature_csv(csv_path: str | Path, target_col: str = "brugada") -> Tuple[pd.DataFrame, np.ndarray]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        candidate_index_paths = [
            csv_path.resolve().with_name("ecg_waveforms_index.csv"),
            csv_path.parent / "ecg_waveforms_index.csv",
            Path("ecg_waveforms_index.csv"),
        ]
        for index_path in candidate_index_paths:
            if not index_path.exists():
                continue
            index_df = pd.read_csv(index_path)
            if target_col not in index_df.columns:
                continue
            if "record_id" in df.columns and "patient_id" in index_df.columns:
                df = df.merge(index_df[["patient_id", target_col]], left_on="record_id", right_on="patient_id", how="left")
            elif "record_id" in df.columns and "record_id" in index_df.columns:
                df = df.merge(index_df[["record_id", target_col]], on="record_id", how="left")
            elif "patient_id" in df.columns and "patient_id" in index_df.columns:
                df = df.merge(index_df[["patient_id", target_col]], on="patient_id", how="left")
            break
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    y = df[target_col].astype(int).clip(0, 1).to_numpy()
    drop_cols = [c for c in [target_col, "patient_id", "record_id"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    X = pd.get_dummies(X, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y


def make_split(
    X: pd.DataFrame,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=random_state
    )
    rel_test_size = test_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=rel_test_size, stratify=y_temp, random_state=random_state
    )
    return SplitData(x_train=x_train, x_val=x_val, x_test=x_test, y_train=y_train, y_val=y_val, y_test=y_test)


def load_waveform_csv(csv_path: str | Path, target_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    y = df[target_col].astype(int).clip(0, 1).to_numpy()
    if len(np.unique(y)) < 2:
        index_path = csv_path.parent.parent / "ecg_waveforms_index.csv"
        if index_path.exists() and "record_id" in df.columns:
            idx = pd.read_csv(index_path)
            if {"patient_id", "brugada"}.issubset(idx.columns):
                idx = idx[["patient_id", "brugada"]].rename(columns={"patient_id": "record_id"})
                merged = df.drop(columns=[target_col]).merge(idx, on="record_id", how="left", validate="one_to_one")
                if merged["brugada"].notna().all():
                    y = merged["brugada"].astype(int).clip(0, 1).to_numpy()
                    df = merged.drop(columns=["brugada"])

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Waveform labels in {csv_path} contain only one class after label recovery; "
            "cannot train/compare supervised deep learning models reliably."
        )

    drop_cols = [c for c in [target_col, "record_id"] if c in df.columns]
    X = df.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    n_leads = 12
    n_steps = X.shape[1] // n_leads
    X = X.reshape(len(df), n_steps, n_leads).transpose(0, 2, 1)
    return X, y
