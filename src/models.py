"""
Model zoo: the reference architectures, strong tabular baselines, and the
biologically-blocked attention network introduced by this project.

Everything here exposes the scikit-learn estimator API (``fit`` / ``predict_proba``)
so that a single cross-validation and calibration harness can drive all of them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _seed_everything(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------
def make_preprocessor() -> Pipeline:
    """Median imputation + z-scoring, fitted inside each CV fold.

    Fitting the imputer and the scaler on the whole dataset before splitting is
    a subtle leak that the reference notebook commits; keeping them inside a
    Pipeline means the harness refits them per fold automatically.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017), as used by the reference notebook.

    ``alpha`` up-weights the positive class and ``gamma`` down-weights examples
    the model already classifies confidently, concentrating gradient on the
    hard inhibitor-positive patients.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        ce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        p_t = target * p + (1 - target) * (1 - p)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Torch architectures
# ---------------------------------------------------------------------------
class DeepMLP(nn.Module):
    """Self-normalising MLP: SELU activations with AlphaDropout."""

    def __init__(self, n_features: int, widths=(256, 128, 64, 32), dropout=0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for w in widths:
            layers += [nn.Linear(prev, w), nn.SELU(), nn.AlphaDropout(dropout)]
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class ResidualMLP(nn.Module):
    """Deep MLP with skip connections, per the reference architecture."""

    def __init__(self, n_features: int, dim: int = 128, n_blocks: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_features, dim), nn.BatchNorm1d(dim),
                                  nn.GELU())
        self.blocks = nn.Sequential(*[_ResidualBlock(dim, dropout)
                                      for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


class MultiScaleCNN1D(nn.Module):
    """Treats the feature vector as a 1-D signal, per the reference 1D-CNN."""

    def __init__(self, n_features: int, channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.b3 = nn.Conv1d(1, channels, 3, padding=1)
        self.b5 = nn.Conv1d(1, channels, 5, padding=2)
        self.b7 = nn.Conv1d(1, channels, 7, padding=3)
        self.mix = nn.Sequential(
            nn.BatchNorm1d(channels * 3), nn.ReLU(),
            nn.Conv1d(channels * 3, channels, 3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(channels, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        z = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return self.head(self.mix(z)).squeeze(-1)


class TabTransformer(nn.Module):
    """Feature-token transformer: every feature becomes a token that attends
    to every other, per the reference TabTransformer arm."""

    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        # value-scaled feature embeddings: token_i = x_i * e_i + b_i
        tok = x.unsqueeze(-1) * self.embed.unsqueeze(0) + self.bias.unsqueeze(0)
        z = self.encoder(tok).mean(dim=1)
        return self.head(z).squeeze(-1)


class BioBlockAttentionNet(nn.Module):
    """The architecture this project contributes.

    The 126 engineered features are not exchangeable: they fall into seven
    biological axes (molecular consequence, position in the FVIII molecule,
    truncation severity, residue chemistry, nucleotide context, splicing,
    clinical phenotype). A generic tabular network has to rediscover that
    grouping from ~2,300 rows, which it cannot reliably do.

    Here the grouping is supplied as structure: each block is encoded by its
    own small subnetwork, and a gated attention layer learns how much each
    biological axis should contribute -- both globally and per patient. The
    attention weights are a first-class output, so every prediction comes with
    an inherent "which kind of biology drove this" explanation that does not
    depend on a post-hoc approximation such as SHAP or LIME.
    """

    def __init__(self, block_indices: dict[str, list[int]], d_block: int = 24,
                 dropout: float = 0.3):
        super().__init__()
        self.block_names = [b for b, idx in block_indices.items() if idx]
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self.index_lists = [torch.tensor(block_indices[b], dtype=torch.long)
                            for b in self.block_names]
        for i, idx in enumerate(self.index_lists):
            self.register_buffer(f"idx_{i}", idx, persistent=True)

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(block_indices[b]), d_block * 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_block * 2, d_block), nn.LayerNorm(d_block), nn.GELU(),
            )
            for b in self.block_names
        ])
        # gated attention over blocks (Ilse et al. style), producing one weight
        # per biological axis per patient
        self.attn_v = nn.Linear(d_block, d_block)
        self.attn_u = nn.Linear(d_block, d_block)
        self.attn_w = nn.Linear(d_block, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_block), nn.Dropout(dropout),
                                  nn.Linear(d_block, 32), nn.GELU(),
                                  nn.Linear(32, 1))
        self.last_attention_: torch.Tensor | None = None

    def forward(self, x):
        embs = []
        for i, enc in enumerate(self.encoders):
            idx = getattr(self, f"idx_{i}")
            embs.append(enc(x.index_select(1, idx)))
        h = torch.stack(embs, dim=1)                      # (B, n_blocks, d)
        gate = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        a = torch.softmax(self.attn_w(gate).squeeze(-1), dim=1)   # (B, n_blocks)
        self.last_attention_ = a.detach()
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# sklearn adapter for the torch models
# ---------------------------------------------------------------------------
class TorchClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a torch module so it behaves like a scikit-learn classifier."""

    def __init__(self, builder=None, epochs: int = 200, batch_size: int = 64,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 25, alpha: float = 0.75, gamma: float = 2.0,
                 val_fraction: float = 0.15, random_state: int = RANDOM_STATE,
                 verbose: bool = False):
        self.builder = builder
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.alpha = alpha
        self.gamma = gamma
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        _seed_everything(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # hold out a slice for early stopping, stratified on the outcome
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(y))
        val_idx = np.concatenate([
            rng.choice(idx[y == c],
                       size=max(1, int(round(self.val_fraction * (y == c).sum()))),
                       replace=False)
            for c in (0, 1) if (y == c).sum() > 0
        ])
        tr_idx = np.setdiff1d(idx, val_idx)

        self.model_ = self.builder(self.n_features_in_)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = FocalLoss(self.alpha, self.gamma)

        Xtr = torch.from_numpy(X[tr_idx]); ytr = torch.from_numpy(y[tr_idx])
        Xva = torch.from_numpy(X[val_idx]); yva = torch.from_numpy(y[val_idx])

        best, best_state, bad = math.inf, None, 0
        n = len(tr_idx)
        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                if len(b) < 2:            # BatchNorm needs >1 row
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr[b]), ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
            sched.step()

            self.model_.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.model_(Xva), yva))
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:3d}  val_loss {vloss:.5f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
