"""PyTorch MLP + sklearn preprocessing for Adult Income (binary classification).

This is the ONLY place model code lives. It is used by the training CLI
(``api/train.py``) and by the FastAPI service on Render (to load a stored model
and predict). It is never imported by Streamlit.

Pipeline, in order:
  1. a scikit-learn Pipeline imputes missing values, scales numeric columns, and
     one-hot encodes categoricals (fit on the TRAIN split only);
  2. a PyTorch MLP (>= 2 hidden layers) is trained with Adam + BCE loss, and the
     epoch with the lowest VALIDATION loss is kept as the best checkpoint;
  3. temperature scaling is fit on the validation logits to calibrate
     probabilities;
  4. metrics, confusion matrix, reliability bins, and permutation importance are
     computed on the held-out TEST split.

The fitted artifact (preprocessor + weights + temperature) is pickled and
base64-encoded so it can live in Supabase and be reloaded at /predict time.
"""
from __future__ import annotations

import base64
import copy
import pickle
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from shared.data import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET_CLASSES,
    TARGET_NAME,
)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

Records = Union[pd.DataFrame, Sequence[Dict[str, object]]]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Feed-forward net: [Linear -> activation -> Dropout] x N, then Linear -> 1 logit."""

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("hidden_sizes needs at least two hidden layers")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")
        act = ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        prev = in_dim
        for width in hidden_sizes:
            layers += [nn.Linear(prev, width), act(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# preprocessing (scikit-learn)
# ---------------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    """Impute, scale numerics, one-hot categoricals. Fit on TRAIN data only."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            # Missing categoricals become their own "Unknown" level.
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        [("num", numeric, NUMERIC_COLS), ("cat", categorical, CATEGORICAL_COLS)]
    )
    return Pipeline([("columns", columns)])


def to_frame(records: Records) -> pd.DataFrame:
    """Records (or a DataFrame) -> model-ready frame: right columns, NaN for missing."""
    df = pd.DataFrame(records).reindex(columns=FEATURE_COLS)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(object)
    # None -> NaN so the imputers recognise it.
    return df.where(df.notna(), np.nan)


# ---------------------------------------------------------------------------
# prediction wrapper (used by the API and by training for evaluation)
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class Predictor:
    """A fitted preprocessor + MLP + calibration temperature."""

    def __init__(self, pre: Pipeline, model: MLP, temperature: float,
                 categories: Dict[str, List[str]], numeric_stats: Dict[str, Dict[str, float]]):
        self.pre = pre
        self.model = model.eval()
        self.temperature = float(temperature)
        self.categories = categories
        self.numeric_stats = numeric_stats

    def logits(self, records: Records) -> np.ndarray:
        x = self.pre.transform(to_frame(records)).astype(np.float32)
        with torch.no_grad():
            return self.model(torch.from_numpy(x)).squeeze(1).numpy()

    def predict_proba(self, records: Records) -> np.ndarray:
        """Calibrated P(income > 50K)."""
        return _sigmoid(self.logits(records) / self.temperature)

    def predict(self, records: Records) -> List[Tuple[int, float]]:
        """Return (label, calibrated probability) per record; threshold 0.5."""
        proba = self.predict_proba(records)
        return [(int(p >= 0.5), float(p)) for p in proba]

    def schema_info(self) -> Dict[str, Any]:
        """What /schema needs: feature names, dtypes, allowed values, numeric ranges."""
        return {
            "numeric_features": NUMERIC_COLS,
            "categorical_features": CATEGORICAL_COLS,
            "categories": self.categories,
            "numeric_stats": self.numeric_stats,
            "target_name": TARGET_NAME,
            "target_classes": TARGET_CLASSES,
        }


def serialize_artifact(artifact: Dict[str, Any]) -> bytes:
    return pickle.dumps(artifact)


def load_predictor(model_b64: str) -> Predictor:
    """Rebuild a Predictor from the base64 pickle stored in Supabase."""
    obj = pickle.loads(base64.b64decode(model_b64))
    model = MLP(obj["in_dim"], obj["hidden_sizes"], obj["activation"], obj["dropout"])
    model.load_state_dict(obj["state_dict"])
    return Predictor(
        obj["pre"], model, obj["temperature"], obj["categories"], obj["numeric_stats"]
    )


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
              loss_fn: nn.Module) -> Tuple[float, float]:
    """(loss, accuracy) on a full tensor, in eval mode (dropout off)."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        acc = ((logits > 0).float() == y).float().mean().item()
    return loss, acc


def _fit_temperature(val_logits: np.ndarray, y_val: np.ndarray) -> float:
    """Temperature scaling: find T > 0 minimising BCE of sigmoid(logit / T) on val."""
    logits = torch.from_numpy(val_logits).float()
    y = torch.from_numpy(y_val).float()
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = bce(logits / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def _reliability(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Tuple[float, List[dict]]:
    """Expected calibration error and per-bin reliability data."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob, edges[1:-1])  # 0 .. n_bins-1
    ece = 0.0
    bins: List[dict] = []
    for b in range(n_bins):
        mask = ids == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(prob[mask].mean())
        frac_pos = float(y[mask].mean())
        ece += abs(mean_pred - frac_pos) * n / len(y)
        bins.append(
            {"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
             "mean_pred": mean_pred, "frac_pos": frac_pos, "count": n}
        )
    return float(ece), bins


def _headline_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)  # threshold function -> class labels
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def _permutation_importance(predictor: Predictor, df_test: pd.DataFrame, y: np.ndarray,
                            n_repeats: int = 5, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Drop in test ROC-AUC when one feature's column is shuffled (bigger = more important)."""
    rng = np.random.default_rng(seed)
    frame = to_frame(df_test)
    baseline = roc_auc_score(y, predictor.predict_proba(frame))
    out: Dict[str, Dict[str, float]] = {}
    for col in FEATURE_COLS:
        drops = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(baseline - roc_auc_score(y, predictor.predict_proba(shuffled)))
        out[col] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_model(
    config: Dict[str, Any],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    """Train one configuration.

    Each frame holds the FEATURE_COLS plus the ``income`` label (0/1).
    Returns ``(run_fields, artifact, test_proba)``:
        run_fields  -> a dict that matches the ``runs`` table columns
        artifact    -> everything needed to serve the model (pickled by the caller)
        test_proba  -> calibrated test-set probabilities, in df_test row order
    """
    seed = int(config.get("seed", 42))
    hidden_sizes = [int(h) for h in config["hidden_sizes"]]
    activation = str(config.get("activation", "relu"))
    dropout = float(config.get("dropout", 0.0))
    lr = float(config.get("lr", 0.001))
    weight_decay = float(config.get("weight_decay", 0.0))
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 60))
    patience = int(config.get("patience", 8))

    torch.manual_seed(seed)
    np.random.seed(seed)

    y_train = df_train[TARGET_NAME].to_numpy(dtype=np.float32)
    y_val = df_val[TARGET_NAME].to_numpy(dtype=np.float32)
    y_test = df_test[TARGET_NAME].to_numpy(dtype=np.float32)

    pre = build_preprocessor()
    x_train = pre.fit_transform(to_frame(df_train)).astype(np.float32)  # fit on train only
    x_val = pre.transform(to_frame(df_val)).astype(np.float32)
    x_train_t, x_val_t = torch.from_numpy(x_train), torch.from_numpy(x_val)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    in_dim = x_train.shape[1]
    model = MLP(in_dim, hidden_sizes, activation, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy (the cost function)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val, best_epoch, best_state, stale = float("inf"), 0, None, 0
    n = x_train_t.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)            # forward propagation
            loss = loss_fn(logits, yb)    # error from the cost function
            loss.backward()               # backpropagation: gradients (derivatives) via autograd
            optimizer.step()              # update the weights (Adam)

        tr_loss, tr_acc = _evaluate(model, x_train_t, y_train_t, loss_fn)
        va_loss, va_acc = _evaluate(model, x_val_t, y_val_t, loss_fn)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        # Best checkpoint = lowest VALIDATION loss; stop after `patience` epochs without it.
        if va_loss < best_val - 1e-5:
            best_val, best_epoch, stale = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # --- calibration: temperature scaling fit on validation logits ----------
    with torch.no_grad():
        val_logits = model(x_val_t).squeeze(1).numpy()
    temperature = _fit_temperature(val_logits, y_val)

    categories = {
        col: [str(c) for c in cats]
        for col, cats in zip(
            CATEGORICAL_COLS,
            pre.named_steps["columns"].named_transformers_["cat"]
            .named_steps["onehot"].categories_,
        )
    }
    train_frame = to_frame(df_train)
    numeric_stats = {
        col: {
            "min": float(train_frame[col].min()),
            "max": float(train_frame[col].max()),
            "median": float(train_frame[col].median()),
        }
        for col in NUMERIC_COLS
    }
    predictor = Predictor(pre, model, temperature, categories, numeric_stats)

    # --- held-out TEST evaluation ------------------------------------------
    test_logits = predictor.logits(df_test)
    prob_raw = _sigmoid(test_logits)
    prob = _sigmoid(test_logits / temperature)  # calibrated
    y_int = y_test.astype(int)
    pred = (prob >= 0.5).astype(int)

    metrics = _headline_metrics(y_int, prob)
    ece_before, bins_before = _reliability(y_int, prob_raw)
    ece_after, bins_after = _reliability(y_int, prob)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_int, pred, labels=[0, 1], zero_division=0
    )
    per_class = {
        TARGET_CLASSES[i]: {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i in (0, 1)
    }

    run_fields = {
        "name": str(config.get("name", "unnamed")),
        "hidden_sizes": hidden_sizes,
        "activation": activation,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "brier": metrics["brier"],
        "ece": ece_after,
        "ece_uncalibrated": ece_before,
        "temperature": temperature,
        "confusion_matrix": confusion_matrix(y_int, pred, labels=[0, 1]).tolist(),
        "per_class": per_class,
        "history": history,
        "calibration": {"before": bins_before, "after": bins_after},
        "permutation_importance": _permutation_importance(predictor, df_test, y_int),
        "config": config,
        "is_active": False,
    }

    artifact = {
        "pre": pre,
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden_sizes": hidden_sizes,
        "activation": activation,
        "dropout": dropout,
        "temperature": temperature,
        "categories": categories,
        "numeric_stats": numeric_stats,
    }
    return run_fields, artifact, prob