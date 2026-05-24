import os
import json
import csv
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_new import IrradianceForecastDataset
from model_new import CloudMaskAblationForecaster


# ============================================================
# USER CONFIGURATION
# Change everything here.
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
# Use the same mode as the model you trained.
#
# "cloud_mask"     -> evaluate cloud-mask images + time-series
# "original_image" -> evaluate original sky images + time-series
VISION_INPUT_MODE = "original_image"

# Must match the folder used during training.
# Cloud mask     -> "runs/cloud_mask_time_series_target_norm"
# Original image -> "runs/original_image_time_series_target_norm"
OUTPUT_DIR = f"runs/{VISION_INPUT_MODE}_time_series_target_norm"

# -------------------------------
# Checkpoint to evaluate
# -------------------------------
BEST_MODEL_NAME = "best_model.pth"
NORM_STATS_NAME = "norm_stats.json"

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, BEST_MODEL_NAME)
NORM_STATS_PATH = os.path.join(OUTPUT_DIR, NORM_STATS_NAME)

# Evaluation results will be saved here.
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluation")

# -------------------------------
# Dataset split
# Must match training.
# -------------------------------
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# -------------------------------
# Sequence settings
# Must match training.
# -------------------------------
IMG_SEQ_LEN = 5
TS_SEQ_LEN = 30
HORIZON = 20

# -------------------------------
# Input features
# Only these time-series parameters are used.
# temp, pressure, delta_ghi, and optical flow are ignored.
# -------------------------------
FEATURE_COLS = ["ghi", "dni", "dhi"]
TARGET_COLS = ["ghi"]
NORMALIZE_TARGETS = True

# -------------------------------
# Vision image settings
# Must match training.
# -------------------------------
IMG_SIZE = 224

# Cloud-mask convention:
# R = low cloud
# G = mid cloud
# B = high cloud
# black = sky

# -------------------------------
# Model settings
# Must match training.
# -------------------------------
VISION_MODEL_NAME = "vit_tiny_patch16_224"
VISION_IN_CHANS = 3

# During evaluation we do not need to load ImageNet weights,
# because we load your trained checkpoint.
PRETRAINED = False

FREEZE_VISION = False
D_MODEL = 128
FUSED_DIM = 128
TARGET_DIM = 1

# -------------------------------
# Evaluation settings
# -------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

# Choose which splits to evaluate.
EVAL_SPLITS = ["train", "val", "test"]

# Save per-sample predictions to CSV.
# This can create large files, especially for the train split.
SAVE_PREDICTIONS = True

# Use mixed precision during evaluation if CUDA is available.
USE_AMP = True


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# DataParallel helpers
# ============================================================
def get_core_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def load_model_state(model, state_dict):
    """
    Loads a state dict into the model.

    This also handles the case where the checkpoint keys start with 'module.'
    from DataParallel.
    """
    cleaned_state = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_key = key[len("module."):]
        else:
            cleaned_key = key

        cleaned_state[cleaned_key] = value

    get_core_model(model).load_state_dict(cleaned_state, strict=True)


# ============================================================
# Load normalization statistics
# ============================================================
def load_norm_stats(path):
    """
    Loads mean/std statistics saved during training.

    The dataset expects pandas Series objects for mean and std.
    """
    if not os.path.exists(path):
        print(f"Normalization file not found: {path}")
        print("Validation/test normalization will use train dataset stats.")
        return None

    with open(path, "r") as f:
        stats = json.load(f)

    feature_mean = pd.Series(
        stats.get("feature_mean", stats.get("mean")),
        dtype="float32",
    )
    feature_std = pd.Series(
        stats.get("feature_std", stats.get("std")),
        dtype="float32",
    )

    target_mean_data = stats.get("target_mean", stats.get("mean"))
    target_std_data = stats.get("target_std", stats.get("std"))
    target_mean = pd.Series(target_mean_data, dtype="float32")
    target_std = pd.Series(target_std_data, dtype="float32")
    normalize_targets = bool(stats.get("normalize_targets", NORMALIZE_TARGETS))

    print(f"Loaded normalization statistics from: {path}")

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "normalize_targets": normalize_targets,
        # Legacy aliases expected by older dataset/helper code.
        "mean": feature_mean,
        "std": feature_std,
    }


def inverse_normalize_ghi(values, norm_stats):
    """
    Converts normalized GHI values back to original W/m^2 scale.

    values:
        numpy array with shape (N, horizon, 1)

    norm_stats:
        dictionary containing mean and std for GHI
    """
    ghi_mean = float(norm_stats["target_mean"]["ghi"])
    ghi_std = float(norm_stats["target_std"]["ghi"])

    return values * ghi_std + ghi_mean


def can_inverse_normalize_ghi(norm_stats):
    return (
        norm_stats is not None
        and bool(norm_stats.get("normalize_targets", False))
        and TARGET_COLS == ["ghi"]
        and "ghi" in norm_stats["target_mean"]
        and "ghi" in norm_stats["target_std"]
    )


# ============================================================
# Metrics
# ============================================================
def compute_forecasting_metrics(preds, targets):
    """
    Overall metrics across all samples and all forecast horizons.

    preds:
        shape = (N, horizon, 1)

    targets:
        shape = (N, horizon, 1)
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


def compute_per_horizon_metrics(preds, targets):
    """
    Computes metrics separately for each forecast minute.

    Example:
        horizon_step = 1 means t+1 minute
        horizon_step = 20 means t+20 minutes
    """
    horizon = preds.shape[1]
    rows = []

    for h in range(horizon):
        p = preds[:, h, :].reshape(-1)
        y = targets[:, h, :].reshape(-1)

        errors = p - y

        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mbe = np.mean(errors)

        ss_res = np.sum((y - p) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        rows.append(
            {
                "horizon_step": h + 1,
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mbe": float(mbe),
                "r2": float(r2),
            }
        )

    return rows


# ============================================================
# Timestamp helpers
# ============================================================
def get_collated_time(collated_times, sample_idx, time_idx):
    """
    Handles the way PyTorch DataLoader collates list-of-string timestamps.

    Dataset returns:
        ts_time  = list of length T_ts
        tgt_time = list of length horizon

    After batching, PyTorch usually converts this into:
        list length T, each element containing batch_size timestamps.
    """
    if collated_times is None:
        return ""

    try:
        if len(collated_times) == 0:
            return ""
    except Exception:
        return ""

    # Most common case:
    # collated_times[time_idx][sample_idx]
    try:
        return str(collated_times[time_idx][sample_idx])
    except Exception:
        pass

    # Alternative case:
    # collated_times[sample_idx][time_idx]
    try:
        return str(collated_times[sample_idx][time_idx])
    except Exception:
        pass

    # Fallback for batch size 1 or unusual collation.
    try:
        return str(collated_times[time_idx])
    except Exception:
        return ""


# ============================================================
# Save CSV utilities
# ============================================================
def save_dict_rows_csv(rows, save_path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {save_path}")


def save_json(data, save_path):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved: {save_path}")


# ============================================================
# Evaluation loop
# ============================================================
def evaluate_split(
    model,
    loader,
    device,
    split_name,
    norm_stats=None,
    save_predictions_path=None,
):
    model.eval()

    criterion = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    prediction_rows = []
    global_sample_index = 0
    has_physical_scale = can_inverse_normalize_ghi(norm_stats)

    loop = tqdm(loader, total=len(loader), desc=f"Evaluating {split_name}", leave=True)

    with torch.no_grad():
        for batch in loop:
            vision_seq, ts_seq, targets, ts_time, tgt_time = batch

            vision_seq = vision_seq.to(device, non_blocking=True)
            ts_seq = ts_seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = vision_seq.size(0)

            use_amp_now = device.type == "cuda" and USE_AMP

            with torch.cuda.amp.autocast(enabled=use_amp_now):
                preds = model(
                    vision=vision_seq,
                    ts=ts_seq,
                )

                loss = criterion(preds, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            preds_wm2_np = None
            targets_wm2_np = None

            if has_physical_scale:
                preds_wm2_np = inverse_normalize_ghi(preds_np, norm_stats)
                targets_wm2_np = inverse_normalize_ghi(targets_np, norm_stats)

            all_preds.append(preds_np)
            all_targets.append(targets_np)

            avg_loss = total_loss / total_samples

            loop.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.5f}",
                },
                refresh=True,
            )

            if save_predictions_path is not None:
                for b in range(batch_size):
                    input_start_time = get_collated_time(ts_time, b, 0)
                    input_end_time = get_collated_time(ts_time, b, TS_SEQ_LEN - 1)

                    for h in range(HORIZON):
                        target_time = get_collated_time(tgt_time, b, h)

                        pred_value = float(preds_np[b, h, 0])
                        target_value = float(targets_np[b, h, 0])
                        error = pred_value - target_value
                        pred_wm2 = None
                        target_wm2 = None
                        error_wm2 = None

                        if has_physical_scale:
                            pred_wm2 = float(preds_wm2_np[b, h, 0])
                            target_wm2 = float(targets_wm2_np[b, h, 0])
                            error_wm2 = pred_wm2 - target_wm2

                        prediction_rows.append(
                            {
                                "split": split_name,
                                "sample_index": global_sample_index + b,
                                "input_start_time": input_start_time,
                                "input_end_time": input_end_time,
                                "target_time": target_time,
                                "horizon_step": h + 1,
                                "value_scale": (
                                    "wm2" if has_physical_scale else "normalized"
                                ),
                                "predicted_ghi": (
                                    pred_wm2
                                    if pred_wm2 is not None
                                    else pred_value
                                ),
                                "target_ghi": (
                                    target_wm2
                                    if target_wm2 is not None
                                    else target_value
                                ),
                                "error": (
                                    error_wm2
                                    if error_wm2 is not None
                                    else error
                                ),
                                "absolute_error": (
                                    abs(error_wm2)
                                    if error_wm2 is not None
                                    else abs(error)
                                ),
                                "squared_error": (
                                    error_wm2 ** 2
                                    if error_wm2 is not None
                                    else error ** 2
                                ),
                                "predicted_ghi_normalized": pred_value,
                                "target_ghi_normalized": target_value,
                                "error_normalized": error,
                                "absolute_error_normalized": abs(error),
                                "squared_error_normalized": error ** 2,
                                "predicted_ghi_wm2": pred_wm2,
                                "target_ghi_wm2": target_wm2,
                                "error_wm2": error_wm2,
                                "absolute_error_wm2": (
                                    abs(error_wm2)
                                    if error_wm2 is not None
                                    else None
                                ),
                                "squared_error_wm2": (
                                    error_wm2 ** 2
                                    if error_wm2 is not None
                                    else None
                                ),
                            }
                        )

            global_sample_index += batch_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    overall_metrics = compute_forecasting_metrics(all_preds, all_targets)
    overall_metrics["loss"] = float(total_loss / total_samples)
    overall_metrics["num_samples"] = int(all_preds.shape[0])
    overall_metrics["num_forecast_points"] = int(all_preds.shape[0] * all_preds.shape[1])

    per_horizon_rows = compute_per_horizon_metrics(all_preds, all_targets)

    overall_metrics_wm2 = None
    per_horizon_rows_wm2 = None

    if has_physical_scale:
        all_preds_wm2 = inverse_normalize_ghi(all_preds, norm_stats)
        all_targets_wm2 = inverse_normalize_ghi(all_targets, norm_stats)

        overall_metrics_wm2 = compute_forecasting_metrics(
            all_preds_wm2,
            all_targets_wm2,
        )
        overall_metrics_wm2["loss_normalized"] = float(total_loss / total_samples)
        overall_metrics_wm2["num_samples"] = int(all_preds_wm2.shape[0])
        overall_metrics_wm2["num_forecast_points"] = int(
            all_preds_wm2.shape[0] * all_preds_wm2.shape[1]
        )

        per_horizon_rows_wm2 = compute_per_horizon_metrics(
            all_preds_wm2,
            all_targets_wm2,
        )

    if save_predictions_path is not None:
        save_dict_rows_csv(prediction_rows, save_predictions_path)

    return overall_metrics, per_horizon_rows, overall_metrics_wm2, per_horizon_rows_wm2


# ============================================================
# Build datasets
# ============================================================
def build_datasets(norm_stats):
    """
    Creates train, val, and test datasets.

    Train dataset computes its own normalization stats, but val/test use the
    saved training stats if available.
    """
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
        apply_rotation=False,
        normalize_targets=NORMALIZE_TARGETS,
    )

    if norm_stats is None:
        norm_stats = train_ds.normalization_stats

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
        normalization_stats=norm_stats,
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
            normalization_stats=norm_stats,
            normalize_targets=NORMALIZE_TARGETS,
        )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }


# ============================================================
# Build model
# ============================================================
def build_model(device):
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

    return model


# ============================================================
# Load checkpoint
# ============================================================
def load_trained_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        print(f"Loaded checkpoint dictionary from: {checkpoint_path}")

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")

        if "best_val_rmse" in checkpoint:
            print(f"Best validation RMSE in checkpoint: {checkpoint['best_val_rmse']:.5f}")

        if "best_val_loss" in checkpoint:
            print(f"Best validation loss in checkpoint: {checkpoint['best_val_loss']:.5f}")
    else:
        state_dict = checkpoint
        print(f"Loaded raw state_dict from: {checkpoint_path}")

    load_model_state(model, state_dict)

    print("Model weights loaded successfully")


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print("Solar Irradiance Forecasting Evaluation")
    print("==============================")
    print(f"Device: {device}")
    print(f"CSV path: {CSV_PATH}")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Evaluation output directory: {EVAL_OUTPUT_DIR}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
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
    # Load normalization stats
    # -------------------------------
    norm_stats = load_norm_stats(NORM_STATS_PATH)

    if norm_stats is not None:
        print(f"GHI input mean from training stats: {float(norm_stats['feature_mean']['ghi']):.5f}")
        print(f"GHI input std from training stats: {float(norm_stats['feature_std']['ghi']):.5f}")
        print(f"GHI target mean from training stats: {float(norm_stats['target_mean']['ghi']):.5f}")
        print(f"GHI target std from training stats: {float(norm_stats['target_std']['ghi']):.5f}")

    # -------------------------------
    # Build datasets
    # -------------------------------
    datasets = build_datasets(norm_stats)

    if norm_stats is None:
        norm_stats = datasets["train"].normalization_stats
        print("Using train dataset normalization stats for input features.")
        print(f"GHI input mean from training stats: {float(norm_stats['feature_mean']['ghi']):.5f}")
        print(f"GHI input std from training stats: {float(norm_stats['feature_std']['ghi']):.5f}")
        print(f"GHI target mean from training stats: {float(norm_stats['target_mean']['ghi']):.5f}")
        print(f"GHI target std from training stats: {float(norm_stats['target_std']['ghi']):.5f}")

    # -------------------------------
    # DataLoaders
    # -------------------------------
    pin_memory = device.type == "cuda"

    loaders = {}

    for split_name in EVAL_SPLITS:
        ds = datasets.get(split_name)

        if ds is None:
            print(f"Skipping split '{split_name}' because it is not available.")
            continue

        loaders[split_name] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # -------------------------------
    # Build and load model
    # -------------------------------
    model = build_model(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"Model ready | "
        f"Total parameters: {total_params / 1e6:.2f}M | "
        f"Trainable parameters: {trainable_params / 1e6:.2f}M"
    )

    load_trained_checkpoint(model, CHECKPOINT_PATH, device)

    # -------------------------------
    # Evaluate all splits
    # -------------------------------
    summary_rows = []

    all_results = {
        "experiment": f"{VISION_INPUT_MODE}_time_series",
        "vision_input_mode": VISION_INPUT_MODE,
        "checkpoint_path": CHECKPOINT_PATH,
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
        "splits": {},
    }

    for split_name, loader in loaders.items():
        print(f"\nEvaluating split: {split_name.upper()}")

        prediction_path = None

        if SAVE_PREDICTIONS:
            prediction_path = os.path.join(
                EVAL_OUTPUT_DIR,
                f"predictions_{split_name}.csv",
            )

        (
            metrics_norm,
            per_horizon_rows_norm,
            metrics_wm2,
            per_horizon_rows_wm2,
        ) = evaluate_split(
            model=model,
            loader=loader,
            device=device,
            split_name=split_name,
            norm_stats=norm_stats,
            save_predictions_path=prediction_path,
        )

        if metrics_wm2 is not None:
            print(
                f"{split_name.upper()} W/m^2 | "
                f"Samples: {metrics_wm2['num_samples']} | "
                f"Loss(norm): {metrics_norm['loss']:.5f} | "
                f"RMSE: {metrics_wm2['rmse']:.3f} | "
                f"MAE: {metrics_wm2['mae']:.3f} | "
                f"MBE: {metrics_wm2['mbe']:.3f} | "
                f"R2: {metrics_wm2['r2']:.4f}"
            )
            print(
                f"{split_name.upper()} normalized | "
                f"RMSE: {metrics_norm['rmse']:.5f} | "
                f"MAE: {metrics_norm['mae']:.5f} | "
                f"MBE: {metrics_norm['mbe']:.5f}"
            )
        else:
            print(
                f"{split_name.upper()} normalized | "
                f"Samples: {metrics_norm['num_samples']} | "
                f"Loss: {metrics_norm['loss']:.5f} | "
                f"RMSE: {metrics_norm['rmse']:.5f} | "
                f"MAE: {metrics_norm['mae']:.5f} | "
                f"MBE: {metrics_norm['mbe']:.5f} | "
                f"R2: {metrics_norm['r2']:.4f}"
            )

        summary_row = {
            "split": split_name,
            "num_samples": metrics_norm["num_samples"],
            "num_forecast_points": metrics_norm["num_forecast_points"],
            "loss_normalized": metrics_norm["loss"],
            "mse_normalized": metrics_norm["mse"],
            "rmse_normalized": metrics_norm["rmse"],
            "mae_normalized": metrics_norm["mae"],
            "mbe_normalized": metrics_norm["mbe"],
            "r2_normalized": metrics_norm["r2"],
            "mse_wm2": metrics_wm2["mse"] if metrics_wm2 is not None else None,
            "rmse_wm2": metrics_wm2["rmse"] if metrics_wm2 is not None else None,
            "mae_wm2": metrics_wm2["mae"] if metrics_wm2 is not None else None,
            "mbe_wm2": metrics_wm2["mbe"] if metrics_wm2 is not None else None,
            "r2_wm2": metrics_wm2["r2"] if metrics_wm2 is not None else None,
        }

        summary_rows.append(summary_row)

        per_horizon_norm_path = os.path.join(
            EVAL_OUTPUT_DIR,
            f"per_horizon_metrics_normalized_{split_name}.csv",
        )

        for row in per_horizon_rows_norm:
            row["split"] = split_name

        save_dict_rows_csv(per_horizon_rows_norm, per_horizon_norm_path)

        per_horizon_wm2_path = None

        if per_horizon_rows_wm2 is not None:
            per_horizon_wm2_path = os.path.join(
                EVAL_OUTPUT_DIR,
                f"per_horizon_metrics_wm2_{split_name}.csv",
            )

            for row in per_horizon_rows_wm2:
                row["split"] = split_name

            save_dict_rows_csv(per_horizon_rows_wm2, per_horizon_wm2_path)

        all_results["splits"][split_name] = {
            "overall_metrics_normalized": metrics_norm,
            "overall_metrics_wm2": metrics_wm2,
            "per_horizon_metrics_normalized": per_horizon_rows_norm,
            "per_horizon_metrics_wm2": per_horizon_rows_wm2,
            "predictions_csv": prediction_path,
            "per_horizon_normalized_csv": per_horizon_norm_path,
            "per_horizon_wm2_csv": per_horizon_wm2_path,
        }

    # -------------------------------
    # Save summary results
    # -------------------------------
    summary_csv_path = os.path.join(EVAL_OUTPUT_DIR, "evaluation_summary.csv")
    summary_json_path = os.path.join(EVAL_OUTPUT_DIR, "evaluation_results.json")

    save_dict_rows_csv(summary_rows, summary_csv_path)
    save_json(all_results, summary_json_path)

    # -------------------------------
    # Print final table
    # -------------------------------
    print("\n================ EVALUATION SUMMARY ================")
    print("Physical metrics are in W/m^2. Loss is normalized MSE.")
    print(
        f"{'split':<10} "
        f"{'samples':>10} "
        f"{'loss_norm':>12} "
        f"{'rmse_wm2':>12} "
        f"{'mae_wm2':>12} "
        f"{'mbe_wm2':>12} "
        f"{'r2':>10}"
    )

    for row in summary_rows:
        rmse = row["rmse_wm2"]
        mae = row["mae_wm2"]
        mbe = row["mbe_wm2"]
        r2 = row["r2_wm2"]

        if rmse is None:
            rmse = row["rmse_normalized"]
            mae = row["mae_normalized"]
            mbe = row["mbe_normalized"]
            r2 = row["r2_normalized"]

        print(
            f"{row['split']:<10} "
            f"{row['num_samples']:>10} "
            f"{row['loss_normalized']:>12.5f} "
            f"{rmse:>12.3f} "
            f"{mae:>12.3f} "
            f"{mbe:>12.3f} "
            f"{r2:>10.4f}"
        )

    print("\nEvaluation complete")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")

    if SAVE_PREDICTIONS:
        print(f"Prediction CSV files saved inside: {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
