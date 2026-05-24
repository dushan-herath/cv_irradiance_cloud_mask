import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import torchvision.transforms.functional as TF


class IrradianceForecastDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        img_seq_len: int = 5,
        ts_seq_len: int = 30,
        horizon: int = 20,
        feature_cols=None,
        target_cols=None,
        img_size: int = 224,
        time_col: str = "timestamp",
        normalization_stats: dict = None,
        root_dir: str = "",
        vision_input_mode: str = "cloud_mask",
        apply_rotation: bool = True,
        normalize_targets: bool = True,
    ):
        """
        Dataset for solar irradiance forecasting.

        Inputs:
            Vision image sequence:
                shape = (T_img, 3, H, W)
                cloud_mask     -> RGB cloud-type mask
                original_image -> RGB sky image

            Time-series sequence:
                shape = (T_ts, 3)
                features = ghi, dni, dhi

        Target:
            Future GHI sequence:
                shape = (horizon, 1)

        vision_input_mode:
            "cloud_mask"     -> load cloud-mask images
            "original_image" -> load original sky images

        normalize_targets:
            True -> train on normalized target GHI and invert back to W/m^2
                    during evaluation.
        """

        # -------------------------------
        # Load full dataset
        # -------------------------------
        df = pd.read_csv(csv_path)

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)

        n = len(df)

        # -------------------------------
        # Train / validation / test split
        # -------------------------------
        if test_ratio > 0:
            train_end = int(n * (1.0 - val_ratio - test_ratio))
            val_end = int(n * (1.0 - test_ratio))

            if split == "train":
                self.df = df.iloc[:train_end].reset_index(drop=True)
            elif split == "val":
                self.df = df.iloc[train_end:val_end].reset_index(drop=True)
            elif split == "test":
                self.df = df.iloc[val_end:].reset_index(drop=True)
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")
        else:
            split_idx = int(n * (1.0 - val_ratio))

            if split == "train":
                self.df = df.iloc[:split_idx].reset_index(drop=True)
            elif split == "val":
                self.df = df.iloc[split_idx:].reset_index(drop=True)
            else:
                raise ValueError("split must be 'train' or 'val' when test_ratio=0")

        # -------------------------------
        # Configuration
        # -------------------------------
        self.split = split
        self.img_seq_len = img_seq_len
        self.ts_seq_len = ts_seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.time_col = time_col
        self.root_dir = root_dir
        self.vision_input_mode = vision_input_mode
        self.apply_rotation = apply_rotation
        self.normalize_targets = normalize_targets

        self.feature_cols = feature_cols or ["ghi", "dni", "dhi"]
        self.target_cols = target_cols or ["ghi"]

        self.image_col = "raw_image_path"
        self.mask_col = "cloud_mask_image_path"

        self.max_lookback = max(img_seq_len, ts_seq_len)

        if self.vision_input_mode not in ["cloud_mask", "original_image"]:
            raise ValueError(
                "vision_input_mode must be either 'cloud_mask' or 'original_image'"
            )

        # -------------------------------
        # Check required columns
        # -------------------------------
        vision_col = (
            self.mask_col
            if self.vision_input_mode == "cloud_mask"
            else self.image_col
        )

        required_cols = [vision_col, self.time_col] + self.feature_cols + self.target_cols

        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        # -------------------------------
        # Feature and target normalization
        #
        # Keep the raw DataFrame unchanged. GHI is both an input feature and
        # the forecast target in the default config, so input and target values
        # must be normalized into separate tables instead of mutating self.df.
        # -------------------------------
        if split == "train":
            feature_mean = self.df[self.feature_cols].mean()
            feature_std = self.df[self.feature_cols].std()
            target_mean = self.df[self.target_cols].mean()
            target_std = self.df[self.target_cols].std()

            # Avoid division by zero if any column is constant.
            feature_std = feature_std.replace(0, 1.0)
            target_std = target_std.replace(0, 1.0)

            self.normalization_stats = {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "normalize_targets": self.normalize_targets,
                # Legacy aliases for older helper code.
                "mean": feature_mean,
                "std": feature_std,
            }
        else:
            if normalization_stats is None:
                raise ValueError(
                    "Validation/test split requires normalization_stats from the train dataset"
                )

            self.normalization_stats = normalization_stats
            self.normalize_targets = bool(
                normalization_stats.get("normalize_targets", self.normalize_targets)
            )
            feature_mean = normalization_stats.get(
                "feature_mean",
                normalization_stats.get("mean"),
            )
            feature_std = normalization_stats.get(
                "feature_std",
                normalization_stats.get("std"),
            )
            target_mean = normalization_stats.get("target_mean")
            target_std = normalization_stats.get("target_std")

            if self.normalize_targets and (target_mean is None or target_std is None):
                target_mean = normalization_stats.get("mean")
                target_std = normalization_stats.get("std")

        if feature_mean is None or feature_std is None:
            raise ValueError("normalization_stats must include feature mean/std values")

        if self.normalize_targets and (target_mean is None or target_std is None):
            raise ValueError("normalize_targets=True requires target mean/std values")

        feature_mean = pd.Series(feature_mean, dtype="float32").reindex(self.feature_cols)
        feature_std = pd.Series(feature_std, dtype="float32").reindex(self.feature_cols)

        if feature_mean.isna().any() or feature_std.isna().any():
            raise ValueError("Feature normalization stats do not match feature_cols")

        if target_mean is not None and target_std is not None:
            target_mean = pd.Series(target_mean, dtype="float32").reindex(self.target_cols)
            target_std = pd.Series(target_std, dtype="float32").reindex(self.target_cols)

            if target_mean.isna().any() or target_std.isna().any():
                raise ValueError("Target normalization stats do not match target_cols")

        self.feature_df = self.df[self.feature_cols].astype("float32").copy()
        self.feature_df = (self.feature_df - feature_mean) / feature_std

        self.target_df = self.df[self.target_cols].astype("float32").copy()

        if self.normalize_targets:
            self.target_df = (self.target_df - target_mean) / target_std

        # -------------------------------
        # Vision image preprocessing
        # -------------------------------
        self.vision_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        self.image_resize = transforms.Resize((img_size, img_size))
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # -------------------------------
        # Logging
        # -------------------------------
        print(f"\nDataset initialized ({split.upper()}): {len(self)} samples")
        print(f"CSV path: {csv_path}")
        print(f"Vision sequence length: {img_seq_len}")
        print(f"Time-series length: {ts_seq_len}")
        print(f"Forecast horizon: {horizon}")
        print(f"Time-series features: {self.feature_cols}")
        print(f"Target columns: {self.target_cols}")
        print(f"Normalize targets: {self.normalize_targets}")
        print(f"Vision input mode: {self.vision_input_mode}")
        print(f"Vision column: {vision_col}")
        print(f"Mask column: {self.mask_col}")
        print(f"Image column: {self.image_col}")
        print("Mask convention: R=low cloud, G=mid cloud, B=high cloud, black=sky")

    def __len__(self):
        return len(self.df) - self.max_lookback - self.horizon

    def _resolve_path(self, path_value):
        """
        Handles both absolute paths and paths relative to root_dir.
        """
        path_value = str(path_value)

        if os.path.isabs(path_value):
            return path_value

        return os.path.join(self.root_dir, path_value)

    def _load_cloud_mask(self, path):
        """
        Loads a cloud-mask image.

        Important:
            R = low cloud
            G = mid cloud
            B = high cloud
            black = clear sky

        The mask is loaded as RGB, resized using nearest-neighbor, and
        converted into a binary 3-channel tensor.
        """
        mask = Image.open(path).convert("RGB")
        mask = self.vision_resize(mask)
        return mask

    def _load_original_image(self, path):
        image = Image.open(path).convert("RGB")
        image = self.image_resize(image)
        return image

    def __getitem__(self, idx):
        # -------------------------------
        # Select windows
        # -------------------------------
        vision_window = self.df.iloc[
            idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len
        ]

        ts_window = self.df.iloc[
            idx : idx + self.ts_seq_len
        ]

        target_window = self.df.iloc[
            idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon
        ]

        # -------------------------------
        # One random rotation per sequence. Masks use nearest-neighbor
        # interpolation; original images use bilinear interpolation.
        # -------------------------------
        if self.split == "train" and self.apply_rotation:
            angle = random.uniform(-180, 180)
        else:
            angle = 0.0

        vision_seq = []
        vision_col = (
            self.mask_col
            if self.vision_input_mode == "cloud_mask"
            else self.image_col
        )

        for vision_p in vision_window[vision_col].values:
            vision_path = self._resolve_path(vision_p)

            if self.vision_input_mode == "cloud_mask":
                vision_img = self._load_cloud_mask(vision_path)
                rotation_interpolation = TF.InterpolationMode.NEAREST
            else:
                vision_img = self._load_original_image(vision_path)
                rotation_interpolation = TF.InterpolationMode.BILINEAR

            # ---- Augmentation ----
            if angle != 0.0:
                vision_img = TF.rotate(
                    vision_img,
                    angle=angle,
                    interpolation=rotation_interpolation,
                    fill=0,
                )

            if self.vision_input_mode == "cloud_mask":
                vision_tensor = TF.to_tensor(vision_img)
                # Preserve the RGB cloud-type channels as binary indicators.
                vision_tensor = (vision_tensor > 0.5).float()
            else:
                vision_tensor = TF.to_tensor(vision_img)
                vision_tensor = self.image_normalize(vision_tensor)

            vision_seq.append(vision_tensor)

        vision_seq = torch.stack(vision_seq)    # (T_img, 3, H, W)

        # -------------------------------
        # Time-series inputs
        # Only GHI, DNI, DHI are used.
        # -------------------------------
        ts_seq = torch.tensor(
            self.feature_df.iloc[idx : idx + self.ts_seq_len].values,
            dtype=torch.float32,
        )  # (T_ts, 3)

        # -------------------------------
        # Target sequence
        # Future GHI values for 20 minutes.
        # -------------------------------
        target_seq = torch.tensor(
            self.target_df.iloc[
                idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon
            ].values,
            dtype=torch.float32,
        )  # (horizon, 1)

        # -------------------------------
        # Timestamps
        # -------------------------------
        if self.time_col in self.df.columns:
            ts_time = (
                ts_window[self.time_col]
                .dt.floor("s")
                .astype(str)
                .tolist()
            )

            tgt_time = (
                target_window[self.time_col]
                .dt.floor("s")
                .astype(str)
                .tolist()
            )
        else:
            ts_time = []
            tgt_time = []

        return vision_seq, ts_seq, target_seq, ts_time, tgt_time


# ======================================================================
# Debug / Visualization
# ======================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    CSV_PATH = "dataset_full_1M.csv"

    train_dataset = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        val_ratio=0.2,
        test_ratio=0.0,
        img_seq_len=5,
        ts_seq_len=30,
        horizon=20,
        feature_cols=["ghi", "dni", "dhi"],
        target_cols=["ghi"],
        img_size=224,
        root_dir="",
        vision_input_mode="cloud_mask",
    )

    val_dataset = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        val_ratio=0.2,
        test_ratio=0.0,
        img_seq_len=5,
        ts_seq_len=30,
        horizon=20,
        feature_cols=["ghi", "dni", "dhi"],
        target_cols=["ghi"],
        img_size=224,
        root_dir="",
        vision_input_mode="cloud_mask",
        normalization_stats=train_dataset.normalization_stats,
    )

    vision_seq, ts_seq, target_seq, ts_time, tgt_time = train_dataset[2800]

    print("Vision sequence shape:", vision_seq.shape)    # (T_img, 3, H, W)
    print("TS sequence shape:", ts_seq.shape)            # (T_ts, 3)
    print("Target sequence shape:", target_seq.shape)    # (20, 1)

    print("\nFirst TS timestamp:", ts_time[0])
    print("Last TS timestamp:", ts_time[-1])
    print("First target timestamp:", tgt_time[0])
    print("Last target timestamp:", tgt_time[-1])

    # -------------------------------
    # Visualize cloud masks
    # -------------------------------
    T = vision_seq.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))

    if T == 1:
        axes = [axes]

    for i in range(T):
        axes[i].imshow(vision_seq[i].permute(1, 2, 0).clamp(0, 1))
        axes[i].set_title("Vision")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
