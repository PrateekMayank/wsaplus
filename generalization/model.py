import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation,
            padding_mode='circular'
        )
        self.norm = nn.GroupNorm(8, channels)
        self.act  = nn.PReLU()

    def forward(self, x):
        y = self.act(self.norm(self.conv(x)))
        return y + x

class ASPPModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, padding=0, padding_mode='circular'),
                nn.GroupNorm(8, channels),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=6, dilation=6, padding_mode='circular'),
                nn.GroupNorm(8, channels),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=12, dilation=12, padding_mode='circular'),
                nn.GroupNorm(8, channels),
                nn.PReLU()
            ),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1, padding=0, padding_mode='circular'),
            nn.GroupNorm(8, channels),
            nn.PReLU()
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))

class RefineHeadV2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res1  = ResidualBlock(in_channels, dilation=1)
        self.res2  = ResidualBlock(in_channels, dilation=2)
        self.aspp  = ASPPModule(in_channels)
        self.res3  = ResidualBlock(in_channels, dilation=1)
        self.final = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, v):
        v = self.res1(v)
        v = self.res2(v)
        v = self.aspp(v)
        v = self.res3(v)
        return self.final(v)


class WSASurrogateModel(nn.Module):
    """
    WSA surrogate model predicting 0.1 AU speed map from 2-channel input:
      Channels: [exp_factors, min_distances, magnetogram]

    Input:
      x: [B, 3, 360, 180]  # batch size B, 3 feature maps, lonxlat

    Output:
      output_0p1AU: [B, 1, 360, 180]  # predicted speed map (200-1500 km/s)
    """
    def __init__(self, img_size=(360, 180), vmin=200, vmax=1200):
        super(WSASurrogateModel, self).__init__()

        self.vmin = vmin
        self.vmax = vmax

        #----------------------------------------
        # Encoder: Swin Transformer backbone
        #----------------------------------------
        self.backbone = timm.create_model(
            'swin_small_patch4_window7_224',
            pretrained=True,
            img_size=img_size,
            in_chans=4,
            features_only=True,
            drop_rate=0.2,
            attn_drop_rate=0.2,
            drop_path_rate=0.2,
            out_indices=(0, 1, 2, 3)
        )

        # Feature channel counts for skip connections
        c1, c2, c3, c4 = self.backbone.feature_info.channels()

        # Activation
        self.activation       = nn.ReLU(inplace=False)
        self.decoder_dropout  = nn.Dropout2d(p=0.5)
        self.last_activation  = nn.PReLU(num_parameters=1, init=0.25)  # Final activation

        #----------------------------------------
        # Decoder: Interpolation + Conv2d upsampling
        #----------------------------------------

        # --- Three learned upsampling stages + skip‐merge convs ---
        self.up4_to_3 = nn.ConvTranspose2d(c4, c3, 4, 2, 1)
        self.merge3 = nn.Conv2d(c3*2, c3, kernel_size=3, padding=1, padding_mode='circular')

        self.up3_to_2 = nn.ConvTranspose2d(c3, c2, 4, 2, 1)
        self.merge2 = nn.Conv2d(c2*2, c2, kernel_size=3, padding=1, padding_mode='circular')

        self.up2_to_1 = nn.ConvTranspose2d(c2, c1, 4, 2, 1)
        self.merge1 = nn.Conv2d(c1*2, c1, kernel_size=3, padding=1, padding_mode='circular')

        # --- Final projection to 1 channel ---
        self.refine = RefineHeadV2(c1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 2, 360, 180] → add lon/lat channels → [B, 4, 360, 180]

        B, C, H, W = x.shape
        device = x.device

        # Add normalized lon/lat channels for coordinate awareness
        lon = torch.linspace(0, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        lat = torch.linspace(0, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, lon, lat], dim=1)

        if not torch.isfinite(x).all():
            raise RuntimeError("x contains NaN or Inf")
        
        #----------------------------------------
        # Encoder: extract multi-scale features
        #----------------------------------------
        f1, f2, f3, f4 = self.backbone(x.contiguous())

        f1 = f1.permute(0, 3, 1, 2)      # [B, H, W, C] -> [B, C, H, W] shape = [16, 96, 90, 45]
        f2 = f2.permute(0, 3, 1, 2)      # [16, 192, 45, 23]
        f3 = f3.permute(0, 3, 1, 2)      # [16, 384, 23, 12]
        f4 = f4.permute(0, 3, 1, 2)      # [16, 768, 12, 6]

        # Check if Nan in any f1, f2, f3 or f4
        if torch.isnan(f1).any():
            raise RuntimeError("Found NaN in f1 of model.py!")

        #----------------------------------------
        # Decoder Stage 4 -> Stage 3
        #----------------------------------------
        #u3 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        # u3: [B, c4, 24, 12] (12*2=24, 6*2=12)
        u3 = self.activation(self.up4_to_3(f4))                      # [B, c3, 24, 12]
        u3 = self.decoder_dropout(u3)
        #print(f3.shape, u3.shape)
        u3 = u3[:, :, :f3.shape[2], :f3.shape[3]]   # crop to [B, c3, 23, 12]
        #print(f3.shape, u3.shape)
        x3 = torch.cat([u3, f3], dim=1)             # [B, c4+c3, H, W]
        x3 = self.activation(self.merge3(x3))       # [B, c3, H, W]

        #----------------------------------------
        # Decoder Stage 3 -> Stage 2
        #----------------------------------------
        #u2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        # u2: [B, c3, 46, 24]
        u2 = self.activation(self.up3_to_2(x3))     # [B, c2, 46, 24]
        u2 = self.decoder_dropout(u2)
        u2 = u2[:, :, :f2.shape[2], :f2.shape[3]]   # crop to [B, c2, 45, 23]
        x2 = torch.cat([u2, f2], dim=1)
        x2 = self.activation(self.merge2(x2))

        #----------------------------------------
        # Decoder Stage 2 -> Stage 1
        #----------------------------------------
        #u1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        # u1: [B, c2, 90, 46]
        u1 = self.activation(self.up2_to_1(x2))     # [B, c1, 90, 46]
        u1 = self.activation(u1)
        u1 = self.decoder_dropout(u1)
        u1 = u1[:, :, :f1.shape[2], :f1.shape[3]]   # crop to [B, c1, 90, 45]
        x1 = self.last_activation(self.merge1(torch.cat([u1, f1], dim=1)))

        #----------------------------------------
        # Final upsampling to full resolution
        #----------------------------------------
        u0 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        # u0: [B, c1, 360, 180]
        output_raw = self.refine(u0)  # [B, 1, 360, 180]

        # Debugging for Nan in output
        if torch.isnan(output_raw).any():
            print("Found NaN in model output_raw in model.py!")
        
        # Scale to wind speed range
        scale = (self.vmax - self.vmin) / 2
        mid = (self.vmax + self.vmin) / 2
        output_0p1AU = mid + scale * torch.tanh(output_raw)

        return output_0p1AU