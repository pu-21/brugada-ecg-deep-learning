from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_waveform_csv
from metrics import build_pr_curve, compute_classification_metrics, find_best_f1_threshold, save_json


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    patience: int = 6
    seed: int = 42
    n_splits: int = 5
    use_focal_loss: bool = True
    gamma: float = 2.0


class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=["fold", "epoch", "train_loss", "val_loss", "val_f1", "val_auprc", "threshold"],
        )
        self._writer.writeheader()
        self._fh.flush()

    def write(self, row: Dict[str, float]) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int = 12, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class SEResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels)
        self.act = nn.ReLU()

    def forward(self, x, return_attention: bool = False):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        scale = self.se.pool(out).squeeze(-1)
        scale = self.se.fc(scale)
        out = out * scale.unsqueeze(-1)
        out = self.act(out + x)
        if return_attention:
            return out, scale
        return out


class SEResNet1D(nn.Module):
    def __init__(self, in_channels: int = 12, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([SEResBlock(channels) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x, return_attention: bool = False):
        x = self.stem(x)
        attentions = []
        for block in self.blocks:
            if return_attention:
                x, scale = block(x, return_attention=True)
                attentions.append(scale)
            else:
                x = block(x)
        x = self.pool(x).squeeze(-1)
        logits = self.head(x).squeeze(-1)
        if return_attention:
            if attentions:
                attn = torch.stack(attentions).mean(dim=0)
            else:
                attn = None
            return logits, attn
        return logits


class LSTM1D(nn.Module):
    def __init__(self, in_channels: int = 12, hidden_size: int = 64, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def _crop(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        return x[..., -target_len:]

    def forward(self, x):
        out = self.conv1(x)
        out = self._crop(out, x.size(-1))
        out = self.act(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self._crop(out, x.size(-1))
        out = self.bn2(out)
        out = self.act(out + x)
        return self.dropout(out)


class TCN1D(nn.Module):
    def __init__(self, in_channels: int = 12, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[TCNBlock(channels, dilation=2**i) for i in range(blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = (1 - pt).pow(self.gamma)
        loss = focal * bce_loss
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha[0], self.alpha[1])
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _to_loader(x, y, batch_size: int, shuffle: bool):
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


def _compute_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    pos_w = neg / max(pos, 1.0)
    return torch.tensor([1.0, pos_w], dtype=torch.float32, device=device)


def _make_criterion(config: TrainConfig, y_train: np.ndarray, device: str):
    alpha = _compute_class_weights(y_train, device)
    if config.use_focal_loss:
        return FocalLoss(alpha=alpha, gamma=config.gamma)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([alpha[1]], dtype=torch.float32, device=device))


def _loss_fn(criterion, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return criterion(logits, targets)


def _eval_model(model, loader, device, criterion):
    model.eval()
    losses, probs, ys = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            losses.append(_loss_fn(criterion, logits, yb).item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probs), np.concatenate(ys)


def _predict_with_attention(model, xb: torch.Tensor):
    out = model(xb, return_attention=True)
    if isinstance(out, tuple) and len(out) == 2:
        logits, attn = out
        return logits, attn
    return out, None


def train_one(model, train_loader, val_loader, device, config: TrainConfig, criterion, logger: CSVLogger, fold_idx: int):
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.to(device)
    best_state, best_val_f1, best_epoch = None, -np.inf, -1
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = _loss_fn(criterion, model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        val_loss, val_probs, y_val = _eval_model(model, val_loader, device, criterion)
        threshold = find_best_f1_threshold(y_val, val_probs)
        val_metrics = compute_classification_metrics(y_val, val_probs, threshold=threshold)
        val_f1 = val_metrics["f1"]
        val_auprc = val_metrics["auprc"]
        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        logger.write(
            {
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_auprc": val_auprc,
                "threshold": threshold,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1, best_epoch


def _summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def _save_fold_metrics(path: Path, fold_metrics: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fold_metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_best_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _load_best_checkpoint(model: nn.Module, path: Path, device: str):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def _extract_seresnet_attention(model: SEResNet1D, sample: np.ndarray, device: str):
    model.eval()
    xb = torch.tensor(sample[None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, attn = model(xb, return_attention=True)
        prob = torch.sigmoid(logits).cpu().item()
        attn = attn.squeeze(0).cpu().numpy() if attn is not None else None
    return prob, attn


def _plot_waveform_attention(sample: np.ndarray, attention: np.ndarray, outpath: Path, lead_names: List[str]):
    fig, axes = plt.subplots(len(lead_names), 1, figsize=(14, 16), sharex=True, dpi=220, gridspec_kw={"height_ratios": [1] * len(lead_names)})
    time = np.arange(sample.shape[-1])
    if len(lead_names) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(time, sample[i], color="#D62728", lw=0.9)
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=25, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time index")
    fig.suptitle("Figure 3. Representative Brugada-positive waveform with lead attention", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 0.92, 0.98])

    heat_ax = fig.add_axes([0.93, 0.11, 0.03, 0.77])
    attn = attention.reshape(-1, 1)
    im = heat_ax.imshow(attn, aspect="auto", cmap="Blues")
    heat_ax.set_xticks([])
    heat_ax.set_yticks(np.arange(len(lead_names)))
    heat_ax.set_yticklabels(lead_names, fontsize=8)
    heat_ax.yaxis.tick_right()
    heat_ax.set_title("Lead\nattention", fontsize=9)
    fig.colorbar(im, ax=heat_ax, fraction=0.8, pad=0.02)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def run_cv(model_name: str, x: np.ndarray, y: np.ndarray, outdir: Path, config: TrainConfig):
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    logger = CSVLogger(outdir / f"{model_name}_cv_training_log.csv")
    fold_metrics: List[Dict[str, float]] = []
    oof_probs = np.zeros(len(y), dtype=float)
    best_thresholds: List[float] = []

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
            if model_name == "cnn":
                model = CNN1D()
            elif model_name == "resnet":
                model = ResNet1D()
            elif model_name == "seresnet":
                model = SEResNet1D()
            elif model_name == "lstm":
                model = LSTM1D()
            elif model_name == "tcn":
                model = TCN1D()
            else:
                raise ValueError(f"Unknown model_name: {model_name}")
            model, best_val_f1, best_epoch = train_one(
                model, train_loader, val_loader, device, config=config, criterion=criterion, logger=logger, fold_idx=fold_idx
            )

            ckpt_path = outdir / f"{model_name}_fold{fold_idx}_best.pt"
            _save_best_checkpoint(model, ckpt_path)
            if model_name == "seresnet":
                _load_best_checkpoint(model, ckpt_path, device)

            probs = []
            model.eval()
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    if model_name == "seresnet":
                        logits, _ = _predict_with_attention(model, xb)
                    else:
                        logits = model(xb)
                    probs.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.concatenate(probs)
            oof_probs[test_idx] = probs

            val_probs = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    if model_name == "seresnet":
                        logits, _ = _predict_with_attention(model, xb)
                    else:
                        logits = model(xb)
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())
            val_probs = np.concatenate(val_probs)
            threshold = find_best_f1_threshold(y_val, val_probs)
            best_thresholds.append(float(threshold))
            metrics = compute_classification_metrics(y_test, probs, threshold=threshold)
            metrics["fold"] = fold_idx
            metrics["model"] = model_name
            metrics["best_epoch"] = best_epoch
            metrics["val_f1"] = float(best_val_f1)
            metrics["threshold"] = float(threshold)
            fold_metrics.append(metrics)
            save_json(metrics, outdir / f"{model_name}_fold{fold_idx}_metrics.json")
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
        "threshold": _summarize(best_thresholds),
        "auroc": _summarize([m["auroc"] for m in fold_metrics]),
        "auprc": _summarize([m["auprc"] for m in fold_metrics]),
        "precision": _summarize([m["precision"] for m in fold_metrics]),
        "recall": _summarize([m["recall"] for m in fold_metrics]),
        "specificity": _summarize([m["specificity"] for m in fold_metrics]),
        "f1": _summarize([m["f1"] for m in fold_metrics]),
        "brier_score": _summarize([m["brier_score"] for m in fold_metrics]),
        "mean_best_epoch": float(np.mean([m["best_epoch"] for m in fold_metrics])),
    }

    save_json(summary, outdir / f"{model_name}_cv_summary.json")
    save_json(build_pr_curve(y, oof_probs), outdir / f"{model_name}_oof_pr_curve.json")

    if model_name == "seresnet":
        sample_idx = int(np.where(y == 1)[0][0])
        sample = x[sample_idx]
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        probe = SEResNet1D().to(device)
        probe = _load_best_checkpoint(probe, outdir / f"seresnet_fold1_best.pt", device) if (outdir / f"seresnet_fold1_best.pt").exists() else model
        prob, attn = _extract_seresnet_attention(probe, sample, device)
        if attn is None:
            attn = np.ones(12, dtype=float) / 12.0
        figdir = outdir.parent / "figures" / "final_visuals"
        figdir.mkdir(parents=True, exist_ok=True)
        _plot_waveform_attention(sample, attn, figdir / "figure3_waveform_attention.png", lead_names)
        save_json({"sample_idx": sample_idx, "probability": prob, "attention": attn.tolist()}, figdir / "figure3_waveform_attention.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../目前没用的/model_ready_ecg_waveforms.csv")
    parser.add_argument("--outdir", default="outputs/dl_cv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-focal-loss", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    x, y = load_waveform_csv(args.csv, target_col="label")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        n_splits=args.n_splits,
        use_focal_loss=not args.no_focal_loss,
        gamma=args.gamma,
    )

    manifest = {
        "csv": str(Path(args.csv)),
        "seed": config.seed,
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "patience": config.patience,
        "n_splits": config.n_splits,
        "use_focal_loss": config.use_focal_loss,
        "gamma": config.gamma,
        "class_distribution": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
    }
    save_json(manifest, outdir / "run_manifest.json")

    summaries = {}
    for model_name in ["cnn", "resnet", "seresnet", "lstm", "tcn"]:
        summaries[model_name] = run_cv(model_name, x, y, outdir, config)

    save_json(summaries, outdir / "cv_all_models_summary.json")


if __name__ == "__main__":
    main()
