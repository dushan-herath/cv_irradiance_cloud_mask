import os
import json
import csv
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_new import IrradianceForecastDataset
from model_new import CloudMaskAblationForecaster


# ============================================================
# USER CONFIGURATION
# Change all experiment settings here.
# No command-line arguments are used.
# ============================================================

# -------------------------------
# Dataset paths
# -------------------------------
CSV_PATH = "dataset_full_1M.csv"
ROOT_DIR = ""

# -------------------------------
# Experiment mode
# -------------------------------
# "cloud_mask"     -> cloud-mask images + time-series
# "original_image" -> original sky images + time-series
VISION_INPUT_MODE = "cloud_mask"

# Recommended output directories:
# Cloud mask     -> "runs/cloud_mask_time_series_target_norm"
# Original image -> "runs/original_image_time_series_target_norm"
OUTPUT_DIR = f"runs/{VISION_INPUT_MODE}_time_series_target_norm"

# -------------------------------
# Dataset split
# -------------------------------
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# -------------------------------
# Sequence settings
# -------------------------------
IMG_SEQ_LEN = 5
TS_SEQ_LEN = 30
HORIZON = 20

# -------------------------------
# Input features
# Only these time-series parameters are used.
# temp, pressure, and delta_ghi are ignored.
# -------------------------------
FEATURE_COLS = ["ghi", "dni", "dhi"]
TARGET_COLS = ["ghi"]
NORMALIZE_TARGETS = True

# -------------------------------
# Vision image settings
# -------------------------------
IMG_SIZE = 224

# Cloud-mask convention:
# R = low cloud
# G = mid cloud
# B = high cloud
# black = sky

# -------------------------------
# Model settings
# -------------------------------
VISION_MODEL_NAME = "vit_base_patch16_224"
VISION_IN_CHANS = 3
PRETRAINED = True
FREEZE_VISION = False

D_MODEL = 128
FUSED_DIM = 128
TARGET_DIM = 1

# -------------------------------
# Training settings
# -------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 1
NUM_WORKERS = 4

LR_VISION = 1e-5
LR_OTHER = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 10
MAX_GRAD_NORM = 1.0

SEED = 42
RESUME = False

# -------------------------------
# Augmentation
# -------------------------------
APPLY_ROTATION = False

# -------------------------------
# Saved file names
# -------------------------------
CHECKPOINT_NAME = "checkpoint.pth"
BEST_MODEL_NAME = "best_model.pth"
HISTORY_NAME = "history.csv"
LOSS_CURVE_NAME = "training_curve.png"
NORM_STATS_NAME = "norm_stats.json"
FINAL_RESULTS_NAME = "final_results.json"


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# DataParallel helpers
# ============================================================
def get_core_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def get_model_state(model):
    return get_core_model(model).state_dict()


def load_model_state(model, state_dict):
    get_core_model(model).load_state_dict(state_dict)


# ============================================================
# Metrics
# ============================================================
def compute_forecasting_metrics(preds, targets):
    """
    preds:
        numpy array with shape (N, horizon, 1)

    targets:
        numpy array with shape (N, horizon, 1)

    Metrics are computed in whatever scale preds/targets use. During training
    this is usually normalized target scale; helper code below also reports
    W/m^2 when target normalization stats are available.
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    errors = preds - targets

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    mbe = np.mean(errors)

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mbe": float(mbe),
        "r2": float(r2),
    }


def can_inverse_normalize_targets(norm_stats):
    return (
        norm_stats is not None
        and bool(norm_stats.get("normalize_targets", False))
        and "target_mean" in norm_stats
        and "target_std" in norm_stats
    )


def inverse_normalize_targets(values, norm_stats):
    mean = np.array(
        [float(norm_stats["target_mean"][col]) for col in TARGET_COLS],
        dtype=np.float32,
    ).reshape(1, 1, -1)
    std = np.array(
        [float(norm_stats["target_std"][col]) for col in TARGET_COLS],
        dtype=np.float32,
    ).reshape(1, 1, -1)
    return values * std + mean


def attach_physical_metrics(metrics, preds, targets, norm_stats):
    if not can_inverse_normalize_targets(norm_stats):
        return metrics

    preds_wm2 = inverse_normalize_targets(preds, norm_stats)
    targets_wm2 = inverse_normalize_targets(targets, norm_stats)
    metrics_wm2 = compute_forecasting_metrics(preds_wm2, targets_wm2)

    for key, value in metrics_wm2.items():
        metrics[f"{key}_wm2"] = value

    return metrics


def format_metrics_line(label, metrics):
    line = (
        f"{label} | "
        f"Loss: {metrics['loss']:.5f} | "
        f"RMSE(norm): {metrics['rmse']:.3f} | "
        f"MAE(norm): {metrics['mae']:.3f} | "
        f"MBE(norm): {metrics['mbe']:.3f}"
    )

    if "rmse_wm2" in metrics:
        line += (
            f" | RMSE: {metrics['rmse_wm2']:.2f} W/m^2 | "
            f"MAE: {metrics['mae_wm2']:.2f} W/m^2 | "
            f"MBE: {metrics['mbe_wm2']:.2f} W/m^2"
        )

    return line + f" | R2: {metrics['r2']:.4f}"


# ============================================================
# Train one epoch
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    norm_stats=None,
    max_grad_norm=None,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    loop = tqdm(loader, total=len(loader), desc="Training", leave=True)

    for i, batch in enumerate(loop):
        vision_seq, ts_seq, targets, *_ = batch

        vision_seq = vision_seq.to(device, non_blocking=True)
        ts_seq = ts_seq.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = vision_seq.size(0)

        optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(
                vision=vision_seq,
                ts=ts_seq,
            )

            loss = criterion(preds, targets)

        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        avg_loss = total_loss / total_samples

        loop.set_postfix(
            {
                "avg_loss": f"{avg_loss:.5f}",
                "batch_loss": f"{loss.item():.5f}",
            },
            refresh=True,
        )

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_forecasting_metrics(all_preds, all_targets)
    metrics["loss"] = float(total_loss / total_samples)
    metrics = attach_physical_metrics(metrics, all_preds, all_targets, norm_stats)

    return metrics


# ============================================================
# Validation / test
# ============================================================
def evaluate_one_epoch(
    model,
    loader,
    criterion,
    device,
    norm_stats=None,
    desc="Validation",
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    loop = tqdm(loader, total=len(loader), desc=desc, leave=True)

    with torch.no_grad():
        for i, batch in enumerate(loop):
            vision_seq, ts_seq, targets, *_ = batch

            vision_seq = vision_seq.to(device, non_blocking=True)
            ts_seq = ts_seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = vision_seq.size(0)

            preds = model(
                vision=vision_seq,
                ts=ts_seq,
            )

            loss = criterion(preds, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            avg_loss = total_loss / total_samples

            loop.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.5f}",
                    "batch_loss": f"{loss.item():.5f}",
                },
                refresh=True,
            )

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_forecasting_metrics(all_preds, all_targets)
    metrics["loss"] = float(total_loss / total_samples)
    metrics = attach_physical_metrics(metrics, all_preds, all_targets, norm_stats)

    return metrics


# ============================================================
# Plot training curve
# ============================================================
def plot_losses(history, save_path):
    if len(history) == 0:
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved loss curve to: {save_path}")


# ============================================================
# Save history CSV
# ============================================================
def save_history_csv(history, save_path):
    if len(history) == 0:
        return

    fieldnames = list(history[0].keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved history CSV to: {save_path}")


# ============================================================
# Save normalization statistics
# ============================================================
def save_norm_stats(norm_stats, save_path):
    serializable = {}

    for key, value in norm_stats.items():
        if hasattr(value, "to_dict"):
            serializable[key] = value.to_dict()
        else:
            serializable[key] = value

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=4)

    print(f"Saved normalization statistics to: {save_path}")


# ============================================================
# Checkpoint handling
# ============================================================
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_rmse,
    best_val_loss,
    history,
    filename,
):
    state = {
        "epoch": epoch,
        "model_state": get_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_rmse": best_val_rmse,
        "best_val_loss": best_val_loss,
        "history": history,
        "config": {
            "csv_path": CSV_PATH,
            "root_dir": ROOT_DIR,
            "output_dir": OUTPUT_DIR,
            "vision_input_mode": VISION_INPUT_MODE,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "img_seq_len": IMG_SEQ_LEN,
            "ts_seq_len": TS_SEQ_LEN,
            "horizon": HORIZON,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "normalize_targets": NORMALIZE_TARGETS,
            "img_size": IMG_SIZE,
            "vision_model_name": VISION_MODEL_NAME,
            "vision_in_chans": VISION_IN_CHANS,
            "pretrained": PRETRAINED,
            "freeze_vision": FREEZE_VISION,
            "d_model": D_MODEL,
            "fused_dim": FUSED_DIM,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr_vision": LR_VISION,
            "lr_other": LR_OTHER,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "seed": SEED,
        },
    }

    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer, scheduler, device):
    checkpoint = torch.load(filename, map_location=device)

    load_model_state(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_rmse = checkpoint.get("best_val_rmse", float("inf"))
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    history = checkpoint.get("history", [])

    print(
        f"Resumed from checkpoint: epoch {start_epoch} | "
        f"best val RMSE (normalized) = {best_val_rmse:.5f}"
    )

    return start_epoch, best_val_rmse, best_val_loss, history


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print("Solar Irradiance Forecasting")
    print("==============================")
    print(f"Device: {device}")
    print(f"CSV path: {CSV_PATH}")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Vision input mode: {VISION_INPUT_MODE}")
    print(f"Vision sequence length: {IMG_SEQ_LEN}")
    print(f"Time-series length: {TS_SEQ_LEN}")
    print(f"Forecast horizon: {HORIZON} minutes")
    print(f"Time-series features: {FEATURE_COLS}")
    print(f"Target columns: {TARGET_COLS}")
    print(f"Normalize targets: {NORMALIZE_TARGETS}")
    print("Ignored columns: raw_image_path, optical_flow_image_path, temp, pressure, delta_ghi")
    print("Cloud-mask convention: R=low cloud, G=mid cloud, B=high cloud, black=sky")

    if VISION_INPUT_MODE == "cloud_mask":
        print("Experiment: CLOUD MASK + time-series")
    elif VISION_INPUT_MODE == "original_image":
        print("Experiment: ORIGINAL IMAGE + time-series")
    else:
        raise ValueError(
            "VISION_INPUT_MODE must be either 'cloud_mask' or 'original_image'"
        )

    # -------------------------------
    # Dataset
    # -------------------------------
    train_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        img_size=IMG_SIZE,
        root_dir=ROOT_DIR,
        vision_input_mode=VISION_INPUT_MODE,
        apply_rotation=APPLY_ROTATION,
        normalize_targets=NORMALIZE_TARGETS,
    )

    val_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        img_size=IMG_SIZE,
        root_dir=ROOT_DIR,
        vision_input_mode=VISION_INPUT_MODE,
        apply_rotation=False,
        normalization_stats=train_ds.normalization_stats,
        normalize_targets=NORMALIZE_TARGETS,
    )

    test_ds = None

    if TEST_RATIO > 0:
        test_ds = IrradianceForecastDataset(
            csv_path=CSV_PATH,
            split="test",
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            img_seq_len=IMG_SEQ_LEN,
            ts_seq_len=TS_SEQ_LEN,
            horizon=HORIZON,
            feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS,
            img_size=IMG_SIZE,
            root_dir=ROOT_DIR,
            vision_input_mode=VISION_INPUT_MODE,
            apply_rotation=False,
            normalization_stats=train_ds.normalization_stats,
            normalize_targets=NORMALIZE_TARGETS,
        )

    save_norm_stats(
        train_ds.normalization_stats,
        os.path.join(OUTPUT_DIR, NORM_STATS_NAME),
    )

    # -------------------------------
    # DataLoaders
    # -------------------------------
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = None

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # -------------------------------
    # Model
    # -------------------------------
    model = CloudMaskAblationForecaster(
        ts_feat_dim=len(FEATURE_COLS),
        img_size=IMG_SIZE,
        vision_model_name=VISION_MODEL_NAME,
        vision_in_chans=VISION_IN_CHANS,
        vision_input_mode=VISION_INPUT_MODE,
        pretrained=PRETRAINED,
        freeze_vision=FREEZE_VISION,
        d_model=D_MODEL,
        fused_dim=FUSED_DIM,
        horizon=HORIZON,
        target_dim=TARGET_DIM,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    core_model = get_core_model(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"Model ready | "
        f"Total parameters: {total_params / 1e6:.2f}M | "
        f"Trainable parameters: {trainable_params / 1e6:.2f}M"
    )

    # -------------------------------
    # Loss, optimizer, scheduler
    # -------------------------------
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": core_model.vision_encoder.parameters(),
                "lr": LR_VISION,
            },
            {
                "params": core_model.vision_temporal.parameters(),
                "lr": LR_OTHER,
            },
            {
                "params": core_model.ts_encoder.parameters(),
                "lr": LR_OTHER,
            },
            {
                "params": core_model.fusion.parameters(),
                "lr": LR_OTHER,
            },
            {
                "params": core_model.final_temporal.parameters(),
                "lr": LR_OTHER,
            },
            {
                "params": core_model.head.parameters(),
                "lr": LR_OTHER,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # -------------------------------
    # Resume setup
    # -------------------------------
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
    best_model_path = os.path.join(OUTPUT_DIR, BEST_MODEL_NAME)
    history_path = os.path.join(OUTPUT_DIR, HISTORY_NAME)
    loss_curve_path = os.path.join(OUTPUT_DIR, LOSS_CURVE_NAME)
    final_results_path = os.path.join(OUTPUT_DIR, FINAL_RESULTS_NAME)

    start_epoch = 0
    best_val_rmse = float("inf")
    best_val_loss = float("inf")
    history = []

    if RESUME and os.path.exists(checkpoint_path):
        start_epoch, best_val_rmse, best_val_loss, history = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )
    else:
        print("Starting new training session")

    # -------------------------------
    # Training loop
    # -------------------------------
    epochs_without_improvement = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            norm_stats=train_ds.normalization_stats,
            max_grad_norm=MAX_GRAD_NORM,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            norm_stats=train_ds.normalization_stats,
            desc="Validation",
        )

        scheduler.step(val_metrics["rmse"])

        row = {
            "epoch": epoch + 1,

            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_mbe": train_metrics["mbe"],
            "train_r2": train_metrics["r2"],
            "train_rmse_wm2": train_metrics.get("rmse_wm2"),
            "train_mae_wm2": train_metrics.get("mae_wm2"),
            "train_mbe_wm2": train_metrics.get("mbe_wm2"),
            "train_r2_wm2": train_metrics.get("r2_wm2"),

            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_mbe": val_metrics["mbe"],
            "val_r2": val_metrics["r2"],
            "val_rmse_wm2": val_metrics.get("rmse_wm2"),
            "val_mae_wm2": val_metrics.get("mae_wm2"),
            "val_mbe_wm2": val_metrics.get("mbe_wm2"),
            "val_r2_wm2": val_metrics.get("r2_wm2"),

            "lr_vision": optimizer.param_groups[0]["lr"],
            "lr_other": optimizer.param_groups[1]["lr"],
        }

        history.append(row)

        print(format_metrics_line("Train", train_metrics))
        print(format_metrics_line("Val  ", val_metrics))

        save_history_csv(history, history_path)
        plot_losses(history, loss_curve_path)

        # -------------------------------
        # Save best model by normalized validation RMSE
        # -------------------------------
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": get_model_state(model),
                    "best_val_rmse": best_val_rmse,
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "vision_input_mode": VISION_INPUT_MODE,
                    "vision_in_chans": VISION_IN_CHANS,
                    "feature_cols": FEATURE_COLS,
                    "target_cols": TARGET_COLS,
                    "normalize_targets": NORMALIZE_TARGETS,
                    "horizon": HORIZON,
                },
                best_model_path,
            )

            print(f"Best model updated: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{PATIENCE} epochs"
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_rmse=best_val_rmse,
            best_val_loss=best_val_loss,
            history=history,
            filename=checkpoint_path,
        )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered")
            break

    # -------------------------------
    # Final evaluation using best model
    # -------------------------------
    print("\nLoading best model for final evaluation")

    best_checkpoint = torch.load(best_model_path, map_location=device)
    load_model_state(model, best_checkpoint["model_state"])

    final_val_metrics = evaluate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        norm_stats=train_ds.normalization_stats,
        desc="Final Validation",
    )

    final_results = {
        "experiment": f"{VISION_INPUT_MODE}_time_series",
        "vision_input_mode": VISION_INPUT_MODE,
        "csv_path": CSV_PATH,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "normalize_targets": NORMALIZE_TARGETS,
        "ignored_columns": [
            "raw_image_path",
            "optical_flow_image_path",
            "temp",
            "pressure",
            "delta_ghi",
        ],
        "img_seq_len": IMG_SEQ_LEN,
        "ts_seq_len": TS_SEQ_LEN,
        "horizon": HORIZON,
        "vision_in_chans": VISION_IN_CHANS,
        "best_val_rmse": best_val_rmse,
        "best_val_rmse_wm2": (
            final_val_metrics.get("rmse_wm2")
            if final_val_metrics is not None
            else None
        ),
        "best_val_loss": best_val_loss,
        "final_val_metrics": final_val_metrics,
    }

    if test_loader is not None:
        final_test_metrics = evaluate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            norm_stats=train_ds.normalization_stats,
            desc="Final Test",
        )

        final_results["final_test_metrics"] = final_test_metrics

    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nSaved final results to: {final_results_path}")

    print("\nTraining complete")
    print(f"Best validation RMSE (normalized): {best_val_rmse:.3f}")
    print(f"Best validation loss: {best_val_loss:.5f}")

    print("\nFinal validation metrics:")
    for k, v in final_val_metrics.items():
        print(f"  {k}: {v:.5f}")

    if test_loader is not None:
        print("\nFinal test metrics:")
        for k, v in final_results["final_test_metrics"].items():
            print(f"  {k}: {v:.5f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
