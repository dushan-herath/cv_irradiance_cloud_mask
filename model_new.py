import math
import torch
from torch import nn
import timm


# -------------------------------------------------------
# Utility: causal mask
# -------------------------------------------------------
def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


# -------------------------------------------------------
# Positional encoding
# -------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -------------------------------------------------------
# Mask-aware vision encoder
# -------------------------------------------------------
class MaskAwareVisionEncoder(nn.Module):
    """
    Input:
        RGB image       : 3 channels
        Cloud mask      : 1 channel

    Final input:
        RGB + mask      : 4 channels

    Mask convention:
        white = cloud = 1
        black = sky   = 0
    """

    def __init__(
        self,
        model_name="vit_base_patch16_224",
        img_size=224,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            in_chans=4,
            global_pool="avg",
        )

        self.out_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, rgb, cloud_mask=None, use_cloud_mask=True):
        """
        rgb:
            (B, T_img, 3, H, W)

        cloud_mask:
            (B, T_img, 1, H, W)

        use_cloud_mask:
            True  -> use real cloud mask
            False -> replace cloud mask with zeros
        """

        B, T, C, H, W = rgb.shape

        if use_cloud_mask and cloud_mask is not None:
            mask = cloud_mask.float()

            # Convert 0-255 mask into 0-1 mask if needed.
            if mask.max() > 1.0:
                mask = mask / 255.0

            # White = cloud, black = sky.
            # So we do NOT invert the mask.
            mask = (mask > 0.5).float()

        else:
            # Without-cloud-mask experiment.
            # Same architecture, but the mask channel contains no information.
            mask = torch.zeros(B, T, 1, H, W, device=rgb.device, dtype=rgb.dtype)

        # RGB + cloud mask = 4-channel image.
        x = torch.cat([rgb, mask], dim=2)  # (B, T, 4, H, W)

        x = x.view(B * T, 4, H, W)
        feat = self.backbone(x)
        feat = feat.view(B, T, -1)

        # Mask-derived cloud statistics.
        # This is not from the CSV time-series columns.
        # It is computed only from the segmentation mask.
        cloud_coverage = mask.mean(dim=[2, 3, 4]).unsqueeze(-1)  # (B, T, 1)

        delta_cloud_coverage = torch.zeros_like(cloud_coverage)
        delta_cloud_coverage[:, 1:] = cloud_coverage[:, 1:] - cloud_coverage[:, :-1]

        cloud_stats = torch.cat(
            [cloud_coverage, delta_cloud_coverage],
            dim=-1
        )  # (B, T, 2)

        return feat, cloud_stats


# -------------------------------------------------------
# Time-series transformer encoder
# -------------------------------------------------------
class TimeSeriesTransformerEncoder(nn.Module):
    """
    Time-series input uses only:
        ghi, dni, dhi

    Therefore:
        ts_feat_dim = 3
    """

    def __init__(
        self,
        in_dim=3,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x):
        """
        x:
            (B, T_ts, 3)

        Feature order:
            ghi, dni, dhi
        """

        x = self.input_proj(x)
        x = self.pos_enc(x)

        mask = causal_mask(x.size(1), x.device)
        x = self.encoder(x, mask)

        return x


# -------------------------------------------------------
# Temporal encoder for image features
# -------------------------------------------------------
class VisionTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x):
        """
        x:
            (B, T_img, image_feature_dim)
        """

        x = self.input_proj(x)
        x = self.pos_enc(x)

        mask = causal_mask(x.size(1), x.device)
        x = self.encoder(x, mask)

        return x


# -------------------------------------------------------
# Gated fusion
# -------------------------------------------------------
class MaskAwareGatedFusion(nn.Module):
    def __init__(self, vision_dim, ts_dim, cloud_stat_dim=2, fused_dim=128):
        super().__init__()

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

        self.ts_proj = nn.Sequential(
            nn.Linear(ts_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

        self.cloud_proj = nn.Sequential(
            nn.Linear(cloud_stat_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, vision_feat, ts_feat, cloud_stats):
        v = self.vision_proj(vision_feat)
        t = self.ts_proj(ts_feat)
        c = self.cloud_proj(cloud_stats)

        gate = self.gate(torch.cat([v, t, c], dim=-1))

        fused = gate * t + (1.0 - gate) * v

        # Add mask-derived cloud information directly.
        fused = fused + c

        return self.out_proj(fused)


# -------------------------------------------------------
# Final forecasting model
# -------------------------------------------------------
class CloudMaskAblationForecaster(nn.Module):
    """
    Same model for both studies:

    1. Without cloud segmentation mask:
        use_cloud_mask=False

    2. With cloud segmentation mask:
        use_cloud_mask=True

    Time-series input:
        ghi, dni, dhi only

    Forecast horizon:
        20 minutes
    """

    def __init__(
        self,
        ts_feat_dim=3,
        img_size=224,
        vision_model_name="vit_base_patch16_224",
        pretrained=True,
        freeze_vision=False,
        d_model=128,
        fused_dim=128,
        horizon=20,
        target_dim=1,
    ):
        super().__init__()

        self.vision_encoder = MaskAwareVisionEncoder(
            model_name=vision_model_name,
            img_size=img_size,
            pretrained=pretrained,
            freeze_backbone=freeze_vision,
        )

        self.vision_temporal = VisionTemporalEncoder(
            in_dim=self.vision_encoder.out_dim,
            d_model=d_model,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )

        self.ts_encoder = TimeSeriesTransformerEncoder(
            in_dim=ts_feat_dim,
            d_model=d_model,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
        )

        self.fusion = MaskAwareGatedFusion(
            vision_dim=d_model,
            ts_dim=d_model,
            cloud_stat_dim=2,
            fused_dim=fused_dim,
        )

        self.final_pos_enc = PositionalEncoding(fused_dim)

        final_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.final_temporal = nn.TransformerEncoder(final_layer, num_layers=2)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, horizon * target_dim),
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, rgb, ts, cloud_mask=None, use_cloud_mask=True):
        """
        rgb:
            (B, T_img, 3, H, W)

        ts:
            (B, T_ts, 3)
            feature order: ghi, dni, dhi

        cloud_mask:
            (B, T_img, 1, H, W)
            white = cloud, black = sky

        output:
            (B, 20, 1)
            next 20 minutes of GHI
        """

        B, T_img = rgb.shape[:2]

        # Image + optional cloud mask encoding.
        vision_feat, cloud_stats = self.vision_encoder(
            rgb=rgb,
            cloud_mask=cloud_mask,
            use_cloud_mask=use_cloud_mask,
        )

        # Temporal modeling of image sequence.
        vision_feat = self.vision_temporal(vision_feat)

        # Time-series modeling using only GHI, DNI, DHI.
        ts_feat = self.ts_encoder(ts)

        # Align time-series tokens with image frames.
        ts_aligned = ts_feat[:, -T_img:]

        # Fuse vision, time-series, and mask-derived cloud statistics.
        fused = self.fusion(
            vision_feat=vision_feat,
            ts_feat=ts_aligned,
            cloud_stats=cloud_stats,
        )

        # Final temporal reasoning after fusion.
        fused = self.final_pos_enc(fused)

        mask = causal_mask(fused.size(1), fused.device)
        fused = self.final_temporal(fused, mask)

        # Forecast from last timestep.
        context = fused[:, -1]
        out = self.head(context)

        return out.view(B, self.horizon, self.target_dim)


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CloudMaskAblationForecaster(
        ts_feat_dim=3,          # ghi, dni, dhi only
        img_size=224,
        vision_model_name="vit_base_patch16_224",
        pretrained=True,
        freeze_vision=False,
        d_model=128,
        fused_dim=128,
        horizon=20,            # 20-minute forecast horizon
        target_dim=1,
    ).to(device)

    B = 2
    T_img = 5
    T_ts = 30
    H = 224
    W = 224

    rgb = torch.randn(B, T_img, 3, H, W).to(device)

    # White = cloud, black = sky.
    # Example mask should be in shape (B, T_img, 1, H, W).
    cloud_mask = torch.randint(0, 2, (B, T_img, 1, H, W)).float().to(device)

    # Only GHI, DNI, DHI.
    ts = torch.randn(B, T_ts, 3).to(device)

    # Experiment 1: without cloud segmentation mask
    y_without_mask = model(
        rgb=rgb,
        ts=ts,
        cloud_mask=cloud_mask,
        use_cloud_mask=False,
    )

    # Experiment 2: with cloud segmentation mask
    y_with_mask = model(
        rgb=rgb,
        ts=ts,
        cloud_mask=cloud_mask,
        use_cloud_mask=True,
    )

    print("Without mask output:", y_without_mask.shape)
    print("With mask output:", y_with_mask.shape)