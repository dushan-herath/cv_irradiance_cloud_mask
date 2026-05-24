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
# Ablation vision encoder
# -------------------------------------------------------
class AblationVisionEncoder(nn.Module):
    """
    Input:
        3-channel vision image sequence

    Supported modes:
        cloud_mask:
            R = low cloud
            G = mid cloud
            B = high cloud
            black = clear sky

        original_image:
            normalized RGB sky image

    Channel convention for generated RGB masks:
        R = low cloud
        G = mid cloud
        B = high cloud
        black = clear sky
    """

    def __init__(
        self,
        model_name="vit_base_patch16_224",
        img_size=224,
        in_chans=3,
        vision_input_mode="cloud_mask",
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()

        if vision_input_mode not in ["cloud_mask", "original_image"]:
            raise ValueError(
                "vision_input_mode must be either 'cloud_mask' or 'original_image'"
            )

        self.in_chans = in_chans
        self.vision_input_mode = vision_input_mode
        self.stat_dim = in_chans * 2

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            in_chans=in_chans,
            global_pool="avg",
        )

        self.out_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, vision):
        """
        vision:
            (B, T_img, 3, H, W)
        """

        if vision is None:
            raise ValueError("vision input must be provided")

        B, T, C, H, W = vision.shape

        x = vision.float()

        if C != self.in_chans:
            if C == 1 and self.in_chans == 3:
                x = x.repeat(1, 1, 3, 1, 1)
            else:
                raise ValueError(
                    f"Expected {self.in_chans} vision channel(s), got {C}"
                )

        if self.vision_input_mode == "cloud_mask":
            # Convert possible 0-255 masks into 0-1 masks if needed.
            if x.max() > 1.0:
                x = x / 255.0

            x = (x > 0.5).float()

            coverage = x.mean(dim=[3, 4])  # (B, T, C)
            delta_coverage = torch.zeros_like(coverage)
            delta_coverage[:, 1:] = coverage[:, 1:] - coverage[:, :-1]
            vision_stats = torch.cat([coverage, delta_coverage], dim=-1)
        else:
            # Keep the same fusion shape for both ablation arms without adding
            # hand-crafted cloud-mask statistics to the original-image run.
            vision_stats = torch.zeros(
                B,
                T,
                self.stat_dim,
                device=x.device,
                dtype=x.dtype,
            )

        x = x.reshape(B * T, self.in_chans, H, W)
        feat = self.backbone(x)
        feat = feat.view(B, T, -1)

        return feat, vision_stats


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
class AblationGatedFusion(nn.Module):
    def __init__(self, vision_dim, ts_dim, vision_stat_dim=6, fused_dim=128):
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
            nn.Linear(vision_stat_dim, fused_dim),
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

    def forward(self, vision_feat, ts_feat, vision_stats):
        v = self.vision_proj(vision_feat)
        t = self.ts_proj(ts_feat)
        c = self.cloud_proj(vision_stats)

        gate = self.gate(torch.cat([v, t, c], dim=-1))

        fused = gate * t + (1.0 - gate) * v

        # Add auxiliary vision statistics. For original-image runs this is zero.
        fused = fused + c

        return self.out_proj(fused)


# -------------------------------------------------------
# Final forecasting model
# -------------------------------------------------------
class CloudMaskAblationForecaster(nn.Module):
    """
    Vision-input ablation forecasting model.

    Vision input:
        cloud_mask or original_image

    Time-series input:
        ghi, dni, dhi only

    Forecast horizon:
        20 minutes

    Both ablation arms use the same model shape and time-series pathway.
    """

    def __init__(
        self,
        ts_feat_dim=3,
        img_size=224,
        vision_model_name="vit_base_patch16_224",
        vision_in_chans=3,
        vision_input_mode="cloud_mask",
        pretrained=True,
        freeze_vision=False,
        d_model=128,
        fused_dim=128,
        horizon=20,
        target_dim=1,
    ):
        super().__init__()

        self.vision_encoder = AblationVisionEncoder(
            model_name=vision_model_name,
            img_size=img_size,
            in_chans=vision_in_chans,
            vision_input_mode=vision_input_mode,
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

        self.fusion = AblationGatedFusion(
            vision_dim=d_model,
            ts_dim=d_model,
            vision_stat_dim=self.vision_encoder.stat_dim,
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

    def forward(self, vision, ts):
        """
        vision:
            (B, T_img, 3, H, W)

        ts:
            (B, T_ts, 3)
            feature order: ghi, dni, dhi

        output:
            (B, 20, 1)
            next 20 minutes of GHI
        """

        B, T_img = vision.shape[:2]

        vision_feat, vision_stats = self.vision_encoder(vision=vision)

        # Temporal modeling of cloud-mask sequence.
        vision_feat = self.vision_temporal(vision_feat)

        # Time-series modeling using only GHI, DNI, DHI.
        ts_feat = self.ts_encoder(ts)

        # Align time-series tokens with image frames.
        ts_aligned = ts_feat[:, -T_img:]

        # Fuse vision, time-series, and mask-derived cloud statistics.
        fused = self.fusion(
            vision_feat=vision_feat,
            ts_feat=ts_aligned,
            vision_stats=vision_stats,
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
        vision_in_chans=3,
        vision_input_mode="cloud_mask",
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

    # Vision sequence. For cloud-mask mode: R=low, G=mid, B=high.
    vision = torch.randint(0, 2, (B, T_img, 3, H, W)).float().to(device)

    # Only GHI, DNI, DHI.
    ts = torch.randn(B, T_ts, 3).to(device)

    y = model(
        vision=vision,
        ts=ts,
    )

    print("Output:", y.shape)
