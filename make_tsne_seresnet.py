from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from train_dl import SEResNet1D, set_seed

BASE = Path(r"c:\Users\pu_18\Desktop\数据\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0")
OUTDIR = BASE / "模型" / "figures" / "final_visuals"
OUTDIR.mkdir(parents=True, exist_ok=True)

X = np.load(BASE / "ecg_waveforms_1200.npy")
Y = np.load(BASE / "ecg_labels.npy")

# Re-map labels robustly to binary: >0 treated as positive.
y = (Y > 0).astype(int)

# Convert to channel-first tensor expected by the model: (N, C, T)
X = np.transpose(X, (0, 2, 1)).astype(np.float32)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SEResNet1D().to(device)
model.eval()

# Load feature extractor from the published training result if available.
# If not, use the randomly initialized backbone only for visualization fallback.
summary_path = BASE / "模型" / "outputs" / "dl_cv_seresnet_attn_retry" / "seresnet_cv_summary.json"
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
else:
    summary = None

# Try to use the first fold checkpoint if present; otherwise, fall back to the current model.
ckpt_candidates = [
    BASE / "模型" / "outputs" / "seresnet_ckpt_only" / "seresnet_fold1_best.pt",
    BASE / "模型" / "outputs" / "dl_cv_seresnet_ckpt" / "seresnet_fold1_best.pt",
    BASE / "模型" / "outputs" / "dl_cv_seresnet_attn_retry" / "seresnet_fold1_best.pt",
]
for ckpt in ckpt_candidates:
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        break

# Feature extraction from the penultimate representation (after SE-ResNet blocks, before final FC).
@torch.no_grad()
def extract_features(x_batch: np.ndarray, batch_size: int = 32):
    feats = []
    for i in range(0, len(x_batch), batch_size):
        xb = torch.tensor(x_batch[i:i+batch_size], dtype=torch.float32, device=device)
        x = model.stem(xb)
        for block in model.blocks:
            x = block(x)
        x = model.pool(x).squeeze(-1)
        feats.append(x.cpu().numpy())
    return np.concatenate(feats, axis=0)

features = extract_features(X)
features = StandardScaler().fit_transform(features)

# t-SNE to 2D
n_samples = len(features)
perplexity = min(30, max(5, (n_samples - 1) // 3))
emb = TSNE(
    n_components=2,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto",
    random_state=42,
).fit_transform(features)

# Plot
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=240)
neg = y == 0
pos = y == 1
ax.scatter(emb[neg, 0], emb[neg, 1], s=20, c="#7F7F7F", alpha=0.72, label="Negative", edgecolors="none")
ax.scatter(emb[pos, 0], emb[pos, 1], s=24, c="#1F77B4", alpha=0.86, label="Brugada positive", edgecolors="none")
ax.set_title("Figure 7. t-SNE of SE-ResNet1D penultimate features")
ax.set_xlabel("t-SNE dimension 1", fontsize=11)
ax.set_ylabel("t-SNE dimension 2", fontsize=11)
ax.tick_params(axis='both', labelsize=10)
ax.legend(frameon=False)
# add a slightly wider margin around data for airier composition
x0, x1 = emb[:, 0].min(), emb[:, 0].max()
y0, y1 = emb[:, 1].min(), emb[:, 1].max()
pad_x = (x1 - x0) * 0.06
pad_y = (y1 - y0) * 0.06
ax.set_xlim(x0 - pad_x, x1 + pad_x)
ax.set_ylim(y0 - pad_y, y1 + pad_y)
ax.text(0.02, -0.15, "Source: ecg_waveforms_1200.npy / ecg_labels.npy; features extracted from the penultimate SE-ResNet1D layer.", transform=ax.transAxes, fontsize=8, color="#666666")
fig.tight_layout()
fig.savefig(OUTDIR / "figure7_tsne_seresnet.png", bbox_inches="tight")
plt.close(fig)

np.save(OUTDIR / "figure7_tsne_seresnet_embedding.npy", emb)
np.save(OUTDIR / "figure7_tsne_seresnet_labels.npy", y)
print(OUTDIR)
