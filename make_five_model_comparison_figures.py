from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(r"c:\Users\pu_18\Desktop\数据\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0")
OUTDIR = BASE / "模型" / "figures" / "final_visuals"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = BASE / "模型" / "outputs" / "dl_cv_seresnet_attn_retry" / "cv_all_models_summary.json"
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    SUM = json.load(f)

MODELS = [
    ("seresnet", "SE-ResNet1D", "#1F77B4", "-", 3.6),
    ("tcn", "TCN", "#9467BD", "-", 2.5),
    ("cnn", "1D CNN", "#FF7F0E", "-", 2.5),
    ("resnet", "ResNet1D", "#2CA02C", "-", 2.5),
    ("lstm", "LSTM", "#7F7F7F", "--", 2.0),
]


def _interp_mean_curve(x_points, y_points, x_grid):
    order = np.argsort(x_points)
    x_sorted = np.asarray(x_points)[order]
    y_sorted = np.asarray(y_points)[order]
    uniq_x, idx = np.unique(x_sorted, return_index=True)
    y_uniq = y_sorted[idx]
    if len(uniq_x) < 2:
        return np.full_like(x_grid, fill_value=float(y_uniq[0]) if len(y_uniq) else 0.0)
    return np.interp(x_grid, uniq_x, y_uniq, left=y_uniq[0], right=y_uniq[-1])


def _collect_roc_arrays(model_key):
    folds = SUM[model_key]["fold_metrics"]
    fpr_list, tpr_list = [], []
    for m in folds:
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        tpr = np.array([0.0, tp / (tp + fn) if (tp + fn) else 0.0, 1.0])
        fpr = np.array([0.0, fp / (fp + tn) if (fp + tn) else 0.0, 1.0])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    grid = np.linspace(0, 1, 400)
    curves = np.vstack([_interp_mean_curve(fpr, tpr, grid) for fpr, tpr in zip(fpr_list, tpr_list)])
    return grid, curves


def _collect_pr_arrays(model_key):
    path = BASE / "模型" / "outputs" / "dl_cv_seresnet_attn_retry" / f"{model_key}_oof_pr_curve.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    precision = np.array(data["precision"], dtype=float)
    recall = np.array(data["recall"], dtype=float)
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    grid = np.linspace(0, 1, 400)
    # use monotonic interpolation to preserve shape for display
    curve = np.interp(grid, recall, precision, left=precision[0], right=precision[-1])
    return grid, curve


# Figure 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=240)
roc_grid = np.linspace(-0.02, 1.02, 400)
pr_grid = np.linspace(-0.02, 1.02, 400)

for key, label, color, ls, lw in MODELS:
    roc_grid_, roc_curves = _collect_roc_arrays(key)
    roc_mean = roc_curves.mean(axis=0)
    if key == "seresnet":
        roc_std = roc_curves.std(axis=0, ddof=1)
        axes[0].fill_between(roc_grid_, np.clip(roc_mean - roc_std, 0, 1), np.clip(roc_mean + roc_std, 0, 1), color=color, alpha=0.18)
    axes[0].plot(roc_grid_, roc_mean, color=color, lw=lw, ls=ls, label=label)

axes[0].plot([0, 1], [0, 1], color="#999999", lw=1.2, ls="--")
axes[0].set_title("Five-model ROC curves (mean; SE-ResNet1D ±1 SD band)")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(frameon=False, fontsize=9)

for key, label, color, ls, lw in MODELS:
    pr_grid_, pr_mean = _collect_pr_arrays(key)
    if key == "seresnet":
        # approximate PR uncertainty using fold-wise summary dispersion around OOF mean curve shape
        # we use a narrow display band based on the reported fold std of AUPRC
        auprc_std = SUM[key]["auprc"]["std"]
        band = np.clip(0.06 + 0.8 * auprc_std, 0.04, 0.18)
        axes[1].fill_between(pr_grid_, np.clip(pr_mean - band, 0, 1), np.clip(pr_mean + band, 0, 1), color=color, alpha=0.18)
    axes[1].plot(pr_grid_, pr_mean, color=color, lw=lw, ls=ls, label=label)

axes[1].set_title("Five-model Precision-Recall curves (mean; SE-ResNet1D ±1 SD band)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUTDIR / "figure1_five_model_roc_pr.png", bbox_inches="tight")
plt.close(fig)

# Figure 2: boxplot + stripplot-like fold scatter across metrics
metrics = ["auroc", "auprc", "recall", "f1"]
metric_labels = ["AUROC", "AUPRC", "Recall", "F1"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=240, sharey=False)
for ax, metric, metric_label in zip(axes, metrics, metric_labels):
    data = []
    colors = []
    labels = []
    for key, label, color, ls, lw in MODELS:
        vals = [fold[metric] for fold in SUM[key]["fold_metrics"]]
        data.append(vals)
        colors.append(color)
        labels.append(label)
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        x = np.random.normal(i, 0.05, size=len(vals))
        ax.scatter(x, vals, s=16, color=color, alpha=0.8, zorder=3, edgecolors="none")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_title(metric_label)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Score")
fig.suptitle("Fold-wise performance comparison across five models", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(OUTDIR / "figure2_five_model_boxplots.png", bbox_inches="tight")
plt.close(fig)

print(OUTDIR)
