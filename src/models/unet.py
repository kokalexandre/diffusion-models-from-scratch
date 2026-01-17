from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] int64 -> float
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(_pick_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(_pick_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        chs = [base_channels * m for m in channel_mults]
        self.downs = nn.ModuleList()
        cur = base_channels

        # Down path
        for li, ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur, ch, time_dim=time_emb_dim, dropout=dropout))
                cur = ch
            downsample = Downsample(cur) if li != len(chs) - 1 else None
            self.downs.append(nn.ModuleDict({"blocks": blocks, "downsample": downsample}))

        # Middle
        self.mid1 = ResBlock(cur, cur, time_dim=time_emb_dim, dropout=dropout)
        self.mid2 = ResBlock(cur, cur, time_dim=time_emb_dim, dropout=dropout)

        # Up path
        self.ups = nn.ModuleList()
        for li, ch in enumerate(reversed(chs)):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur + ch, ch, time_dim=time_emb_dim, dropout=dropout))
                cur = ch
            upsample = Upsample(cur) if li != len(chs) - 1 else None
            self.ups.append(nn.ModuleDict({"blocks": blocks, "upsample": upsample}))

        self.out_norm = nn.GroupNorm(_pick_groups(cur), cur)
        self.out_conv = nn.Conv2d(cur, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h = self.init_conv(x)
        skips = []

        for stage in self.downs:
            for block in stage["blocks"]:
                h = block(h, t_emb)
                skips.append(h)
            if stage["downsample"] is not None:
                h = stage["downsample"](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for stage in self.ups:
            for block in stage["blocks"]:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            if stage["upsample"] is not None:
                h = stage["upsample"](h)

        return self.out_conv(F.silu(self.out_norm(h)))
