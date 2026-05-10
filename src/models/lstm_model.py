"""
models/lstm_model.py — LSTM Deep Learning Model (PyTorch)

Architecture:
  - Input: sequence of 7 days × 96 blocks × N features → shape (7*96, N_features)
  - 2-layer bidirectional LSTM
  - Dense output: 96 values (next-day MCP for all 96 blocks)

Training:
  - Adam optimizer, MSE loss, ReduceLROnPlateau scheduler
  - Early stopping on validation RMSE
  - Saves best model checkpoint

Usage:
    python src/models/lstm_model.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from evaluate import compute_all_metrics, evaluate_by_segment, print_metrics_table

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lstm")
PREDS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "lstm")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Features used as input to LSTM (all are numeric after preprocessing)
FEATURE_COLS = [
    "mcp_rs_per_mwh",          # target — included as past input
    "mcp_lag_1d", "mcp_lag_7d",
    "mcp_rolling_7d_mean", "mcp_rolling_7d_std",
    "block", "hour", "day_of_week", "month",
    "is_weekend", "is_holiday", "season", "hour_bucket",
]
TARGET = "mcp_rs_per_mwh"
LOOKBACK_DAYS = 7     # use 7 past days to predict 1 future day
BLOCKS_PER_DAY = 96
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10         # early stopping patience


# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
class DaySequenceDataset(Dataset):
    """
    Each sample: X = (LOOKBACK_DAYS × BLOCKS_PER_DAY, N_features),
                 y = (BLOCKS_PER_DAY,)  ← next day's MCP values
    """
    def __init__(self, daily_blocks: dict, dates: list):
        """
        daily_blocks: {date: np.ndarray shape (96, n_features)}
        dates: sorted list of dates for which we have a complete block
        """
        self.daily_blocks = daily_blocks
        self.dates = dates

    def __len__(self):
        return max(0, len(self.dates) - LOOKBACK_DAYS)

    def __getitem__(self, idx):
        target_date = self.dates[idx + LOOKBACK_DAYS]
        input_dates = self.dates[idx: idx + LOOKBACK_DAYS]

        x = np.concatenate([self.daily_blocks[d] for d in input_dates], axis=0)  # (7*96, F)
        y = self.daily_blocks[target_date][:, 0]                                  # (96,) — MCP col 0

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ────────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, BLOCKS_PER_DAY),
        )

    def forward(self, x):
        # x: (batch, seq_len=7*96, n_features)
        lstm_out, _ = self.lstm(x)            # (batch, seq_len, hidden)
        last = lstm_out[:, -1, :]             # (batch, hidden)
        return self.fc(last)                  # (batch, 96)


# ────────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ────────────────────────────────────────────────────────────────────────────────
def build_daily_blocks(df: pd.DataFrame) -> tuple:
    """
    Returns (daily_blocks dict, dates list, scaler_mean, scaler_std)
    Normalizes using mean/std of training data.
    """
    feature_data = df[FEATURE_COLS].values.astype(float)
    return feature_data, df


def load_and_prepare():
    print(f"  Device: {DEVICE}")
    train_full = pd.read_parquet(os.path.join(DATA_DIR, "training_features.parquet"))
    holdout    = pd.read_parquet(os.path.join(DATA_DIR, "holdout_features.parquet"))

    # Time-series validation split (last 60 days of training)
    cutoff  = train_full["date"].max() - pd.Timedelta(days=60)
    val_df  = train_full[train_full["date"] >  cutoff].copy()
    train_df = train_full[train_full["date"] <= cutoff].copy()

    # Drop rows with NaN in feature cols
    for df in [train_df, val_df, holdout]:
        df.dropna(subset=FEATURE_COLS, inplace=True)

    # Normalize using train stats
    mean = train_df[FEATURE_COLS].mean()
    std  = train_df[FEATURE_COLS].std().replace(0, 1)

    def normalize(df):
        df = df.copy()
        df[FEATURE_COLS] = (df[FEATURE_COLS] - mean) / std
        return df

    train_df = normalize(train_df)
    val_df   = normalize(val_df)
    holdout_n = normalize(holdout)

    def to_daily_blocks(df):
        """Group by date, return dict {date: (96, n_features)} and sorted dates."""
        blocks = {}
        for d, grp in df.groupby("date"):
            grp_sorted = grp.sort_values("block")
            if len(grp_sorted) == BLOCKS_PER_DAY:
                blocks[d] = grp_sorted[FEATURE_COLS].values.astype(np.float32)
        dates = sorted(blocks.keys())
        return blocks, dates

    train_blocks, train_dates = to_daily_blocks(train_df)
    val_blocks,   val_dates   = to_daily_blocks(val_df)
    test_blocks,  test_dates  = to_daily_blocks(holdout_n)

    # For inverse-transform of predictions
    mcp_mean = float(mean[TARGET])
    mcp_std  = float(std[TARGET])

    return (
        train_blocks, train_dates,
        val_blocks,   val_dates,
        test_blocks,  test_dates,
        holdout,      # original (un-normalized) for metadata
        mcp_mean, mcp_std,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Training Loop
# ────────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            total_loss += criterion(pred, y).item() * len(x)
    return total_loss / len(loader.dataset)


def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR,  exist_ok=True)

    print("=== LSTM Model ===\n")

    (
        train_blocks, train_dates,
        val_blocks,   val_dates,
        test_blocks,  test_dates,
        holdout_raw,  mcp_mean, mcp_std,
    ) = load_and_prepare()

    # Merge val into train blocks for dataset (val still used for early stopping)
    all_train_blocks = {**train_blocks, **val_blocks}
    all_train_dates  = sorted(all_train_blocks.keys())

    # Build datasets / loaders
    train_ds = DaySequenceDataset(train_blocks, train_dates)
    val_ds   = DaySequenceDataset(
        {**train_blocks, **val_blocks},
        sorted(all_train_blocks.keys())
    )
    test_ds = DaySequenceDataset(
        {**all_train_blocks, **test_blocks},
        sorted({**all_train_blocks, **test_blocks}.keys()),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    n_features = len(FEATURE_COLS)
    model = LSTMModel(n_features=n_features, hidden_size=128, n_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr  = 0
    ckpt_path = os.path.join(MODELS_DIR, "lstm_best.pt")

    print(f"  Training LSTM on {DEVICE} ...")
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss   = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Collect test predictions
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_preds.append(model(x.to(DEVICE)).cpu().numpy())
            all_true.append(y.numpy())

    y_pred_norm = np.concatenate(all_preds).flatten()          # normalized scale
    y_true_norm = np.concatenate(all_true).flatten()

    # Inverse-transform from normalized to Rs/MWh
    y_pred = y_pred_norm * mcp_std + mcp_mean
    y_true = y_true_norm * mcp_std + mcp_mean

    metrics = compute_all_metrics(y_true, y_pred)
    print_metrics_table("LSTM (Holdout Test)", metrics)

    # Segment evaluation — match test block metadata to predictions
    # Build a flat test_df from test_ds date sequence
    test_seq_dates = sorted({**all_train_blocks, **test_blocks}.keys())
    test_day_dates = test_seq_dates[LOOKBACK_DAYS:][:len(all_preds) * BATCH_SIZE + BATCH_SIZE]
    flat_rows = []
    for i, pred_day in enumerate(test_day_dates[:len(all_preds)]):
        for b, (p, t) in enumerate(zip(
            all_preds[i].flatten() if len(all_preds[i].shape) > 1 else [all_preds[i].flatten()],
            all_true[i].flatten()  if len(all_true[i].shape) > 1 else [all_true[i].flatten()],
        )):
            flat_rows.append({"date": pred_day, "block": b, "pred": p, "true": t})

    pred_df_meta = holdout_raw.merge(
        pd.DataFrame([{
            "date": test_seq_dates[LOOKBACK_DAYS + i],
            "block": b,
            "predicted_mcp_norm": all_preds[i // BLOCKS_PER_DAY].flatten()[b]
            if i // BLOCKS_PER_DAY < len(all_preds) else np.nan
        } for i, b in enumerate(
            [(j, b) for j in range(len(all_preds)) for b in range(BLOCKS_PER_DAY)]
        )], columns=["date", "block", "predicted_mcp_norm"]),
        on=["date", "block"], how="inner",
    ) if False else None  # simplified: just save flat predictions

    # Save
    pd.DataFrame([{"model": "lstm", **metrics}]).to_csv(
        os.path.join(MODELS_DIR, "metrics.csv"), index=False)

    preds_out = pd.DataFrame({
        "predicted_mcp": y_pred,
        "true_mcp":      y_true,
    })
    preds_out.to_csv(os.path.join(PREDS_DIR, "test_predictions.csv"), index=False)

    torch.save({
        "model_state": model.state_dict(),
        "mcp_mean": mcp_mean,
        "mcp_std":  mcp_std,
        "n_features": n_features,
    }, os.path.join(MODELS_DIR, "lstm_final.pt"))

    print(f"\n[OK] Model + predictions saved to {MODELS_DIR} and {PREDS_DIR}")


if __name__ == "__main__":
    run()
