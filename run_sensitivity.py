"""Phase 2 sensitivity analysis (BSPC revision).

Positive class redefined: only the 69 confirmed Brugada ECGs (index `brugada` == 1).
The 7 atypical ECGs (index `brugada` == 2) are excluded entirely -> N = 356 (287 neg / 69 pos).

Everything else is bit-for-bit the main-experiment protocol (train_dl.py):
  - patient-level stratified 5-fold CV (StratifiedKFold, shuffle=True, random_state=42)
  - inner 80/20 train/val split per fold (random_state = 42 + fold_idx)
  - Focal Loss (gamma=2) with class-frequency weights from the training fold
  - Adam lr=1e-3, batch=32, <=30 epochs, early stopping patience=6, seed=42
  - decision threshold = best-F1 threshold on the validation fold, applied to the test fold

Model classes / training helpers are imported from train_dl.py so the two runs share
the exact same code. Outputs go to ./outputs (sensitivity_results.csv,
sensitivity_fold_results.csv, sensitivity_predictions.csv) and never touch the
main-experiment output folders.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT = Path(__file__).resolve().parents[2]  # .../brugada-huca-...-1.0.0
sys.path.insert(0, str(PROJECT / "模型"))

from train_dl import (  # noqa: E402
    CSVLogger,
    CNN1D,
    LSTM1D,
    ResNet1D,
    SEResNet1D,
    TCN1D,
    TrainConfig,
    _make_criterion,
    _to_loader,
    set_seed,
    train_one,
)
from metrics import compute_classification_metrics, find_best_f1_threshold, save_json  # noqa: E402

CSV_PATH = PROJECT / "目前没用的" / "model_ready_ecg_waveforms.csv"
INDEX_PATH = PROJECT / "ecg_waveforms_index.csv"
OUTDIR = Path(__file__).resolve().parent / "outputs"
MODELS = ["cnn", "resnet", "seresnet", "lstm", "tcn"]
METRIC_KEYS = ["auroc", "auprc", "precision", "recall", "specificity", "f1", "brier_score"]


def summarize(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def build_model(name: str) -> torch.nn.Module:
    if name == "cnn":
        return CNN1D()
    if name == "resnet":
        return ResNet1D()
    if name == "seresnet":
        return SEResNet1D()
    if name == "lstm":
        return LSTM1D()
    if name == "tcn":
        return TCN1D()
    raise ValueError(f"Unknown model_name: {name}")


def load_sensitivity_data():
    """Same loading path as the main experiment (load_waveform_csv label recovery),
    but keeping the raw 3-class label so we can drop class 2 (atypical)."""
    df = pd.read_csv(CSV_PATH)
    idx = pd.read_csv(INDEX_PATH)[["patient_id", "brugada"]].rename(columns={"patient_id": "record_id"})
    merged = df.drop(columns=["label"]).merge(idx, on="record_id", how="left", validate="one_to_one")
    if not merged["brugada"].notna().all():
        raise ValueError("Label recovery from ecg_waveforms_index.csv failed")

    keep = merged["brugada"].isin([0, 1])  # 0 = negative, 1 = confirmed Brugada, 2 = atypical (excluded)
    n_excluded = int((merged["brugada"] == 2).sum())
    merged = merged.loc[keep].reset_index(drop=True)

    y = merged["brugada"].astype(int).to_numpy()
    record_ids = merged["record_id"].astype(str).to_numpy()
    X = merged.drop(columns=["brugada", "record_id"]).to_numpy(dtype=np.float32)
    n_leads = 12
    n_steps = X.shape[1] // n_leads
    X = X.reshape(len(merged), n_steps, n_leads).transpose(0, 2, 1)
    return X, y, record_ids, n_excluded


def run_cv_sens(model_name: str, x: np.ndarray, y: np.ndarray, record_ids: np.ndarray,
                outdir: Path, config: TrainConfig):
    """Fold loop copied 1:1 from train_dl.run_cv, plus OOF prediction collection;
    no checkpoint files, no attention figure (not needed for sensitivity)."""
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    logger = CSVLogger(outdir / f"{model_name}_cv_training_log.csv")
    fold_metrics = []
    oof_probs = np.zeros(len(y), dtype=float)
    oof_folds = np.full(len(y), -1, dtype=int)
    best_thresholds = []

    try:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
            x_train_full, x_test = x[train_idx], x[test_idx]
            y_train_full, y_test = y[train_idx], y[test_idx]
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_full,
                y_train_full,
                test_size=0.2,
                stratify=y_train_full,
                random_state=config.seed + fold_idx,
            )

            train_loader = _to_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
            val_loader = _to_loader(x_val, y_val, batch_size=64, shuffle=False)
            test_loader = _to_loader(x_test, y_test, batch_size=64, shuffle=False)
            criterion = _make_criterion(config, y_train, device)
            model = build_model(model_name)
            model, best_val_f1, best_epoch = train_one(
                model, train_loader, val_loader, device, config=config, criterion=criterion,
                logger=logger, fold_idx=fold_idx,
            )

            probs = []
            model.eval()
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            probs = np.concatenate(probs)
            oof_probs[test_idx] = probs
            oof_folds[test_idx] = fold_idx

            val_probs = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    val_probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            val_probs = np.concatenate(val_probs)
            threshold = find_best_f1_threshold(y_val, val_probs)
            best_thresholds.append(float(threshold))

            m = compute_classification_metrics(y_test, probs, threshold=threshold)
            m["fold"] = fold_idx
            m["model"] = model_name
            m["best_epoch"] = best_epoch
            m["val_f1"] = float(best_val_f1)
            m["threshold"] = float(threshold)
            fold_metrics.append(m)
            print(f"[{model_name}] fold {fold_idx}/5: auroc={m['auroc']:.3f} auprc={m['auprc']:.3f} "
                  f"f1={m['f1']:.3f} best_epoch={best_epoch}", flush=True)
    finally:
        logger.close()

    summary = {
        "model": model_name,
        "seed": config.seed,
        "n_splits": config.n_splits,
        "use_focal_loss": config.use_focal_loss,
        "gamma": config.gamma,
        "class_distribution": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "fold_metrics": fold_metrics,
        "threshold": summarize(best_thresholds),
        **{k: summarize([m[k] for m in fold_metrics]) for k in METRIC_KEYS},
        "mean_best_epoch": float(np.mean([m["best_epoch"] for m in fold_metrics])),
    }
    save_json(summary, outdir / f"{model_name}_cv_summary.json")
    return summary, fold_metrics, oof_probs, oof_folds, best_thresholds


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    x, y, record_ids, n_excluded = load_sensitivity_data()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    assert len(y) == 356 and n_pos == 69 and n_neg == 287, (len(y), n_pos, n_neg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainConfig()  # defaults identical to main-experiment run_manifest.json
    manifest = {
        "purpose": "Phase 2 sensitivity analysis: positive class = confirmed Brugada only",
        "csv": str(CSV_PATH),
        "index": str(INDEX_PATH),
        "seed": config.seed,
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "patience": config.patience,
        "n_splits": config.n_splits,
        "use_focal_loss": config.use_focal_loss,
        "gamma": config.gamma,
        "class_distribution": {"0": n_neg, "1": n_pos},
        "excluded_atypical": n_excluded,
        "prevalence": n_pos / len(y),
        "device": device,
    }
    save_json(manifest, OUTDIR / "run_manifest.json")
    print(f"manifest: {manifest}", flush=True)

    all_folds, all_preds, summaries = [], [], {}
    for name in MODELS:
        summary, fold_metrics, oof_probs, oof_folds, fold_thresholds = run_cv_sens(
            name, x, y, record_ids, OUTDIR, config
        )
        summaries[name] = summary
        all_folds.extend(fold_metrics)
        for i in range(len(y)):
            all_preds.append({
                "model": name,
                "record_id": record_ids[i],
                "fold": int(oof_folds[i]),
                "y_true": int(y[i]),
                "y_prob": float(oof_probs[i]),
                "y_pred": int(oof_probs[i] >= fold_thresholds[oof_folds[i] - 1]),
                "fold_threshold": float(fold_thresholds[oof_folds[i] - 1]),
            })
        print(f"[{name}] done: auroc={summary['auroc']['mean']:.3f}±{summary['auroc']['std']:.3f} "
              f"f1={summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f} "
              f"brier={summary['brier_score']['mean']:.3f}", flush=True)

    # ---- deliverable 1: per-model summary table ----
    with (OUTDIR / "sensitivity_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["model", "n_total", "n_negative", "n_positive", "prevalence"]
        for k in METRIC_KEYS + ["threshold"]:
            header += [f"{k}_mean", f"{k}_std"]
        header += ["mean_best_epoch"]
        w.writerow(header)
        for name in MODELS:
            s = summaries[name]
            row = [name, len(y), n_neg, n_pos, f"{n_pos / len(y):.4f}"]
            for k in METRIC_KEYS + ["threshold"]:
                row += [f"{s[k]['mean']:.4f}", f"{s[k]['std']:.4f}"]
            row += [f"{s['mean_best_epoch']:.1f}"]
            w.writerow(row)

    # ---- deliverable 2: per-fold metrics ----
    fold_cols = ["model", "fold", "best_epoch", "threshold", "val_f1", "prevalence",
                 "auroc", "auprc", "precision", "recall", "specificity", "f1", "brier_score",
                 "tp", "fp", "tn", "fn"]
    with (OUTDIR / "sensitivity_fold_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fold_cols, extrasaction="ignore")
        w.writeheader()
        for m in all_folds:
            w.writerow(m)

    # ---- deliverable 3: OOF predictions ----
    with (OUTDIR / "sensitivity_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "record_id", "fold", "y_true", "y_prob", "y_pred", "fold_threshold"])
        w.writeheader()
        w.writerows(all_preds)

    save_json(summaries, OUTDIR / "cv_all_models_summary.json")

    # ---- console comparison vs main experiment (76-positive primary analysis) ----
    main_dir = PROJECT / "模型" / "outputs" / "dl_cv_tcn"
    print("\n=== Sensitivity (69 confirmed only, N=356) vs Main (76 incl. atypical, N=363) ===", flush=True)
    print(f"{'model':10s} {'sens AUROC':>14s} {'main AUROC':>14s} {'sens F1':>13s} {'main F1':>13s}", flush=True)
    for name in MODELS:
        main_json = main_dir / f"{name}_cv_summary.json"
        if main_json.exists():
            main_s = json.loads(main_json.read_text(encoding="utf-8"))
            print(f"{name:10s} {summaries[name]['auroc']['mean']:.3f}±{summaries[name]['auroc']['std']:.3f} "
                  f"{main_s['auroc']['mean']:.3f}±{main_s['auroc']['std']:.3f} "
                  f"{summaries[name]['f1']['mean']:6.3f}    "
                  f"{main_s['f1']['mean']:6.3f}", flush=True)
        else:
            print(f"{name:10s} {summaries[name]['auroc']['mean']:.3f}  (main summary not found)", flush=True)

    print("\nAll outputs written to:", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
