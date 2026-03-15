from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be rank-1, got shape={tuple(timesteps.shape)}")

    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    ) / max(half_dim - 1, 1)
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = bool(with_conv)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1) if self.with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = bool(with_conv)
        if self.with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.norm1 = group_norm(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = group_norm(self.out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1))

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = h + self.temb_proj(swish(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.norm = group_norm(self.channels)
        self.q = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = zero_module(nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, height, width = q.shape
        q = q.reshape(b, c, height * width).permute(0, 2, 1)
        k = k.reshape(b, c, height * width)
        w = torch.bmm(q, k) * (c ** -0.5)
        w = torch.softmax(w, dim=2)

        v = v.reshape(b, c, height * width)
        h = torch.bmm(v, w.permute(0, 2, 1)).reshape(b, c, height, width)
        h = self.proj_out(h)
        return x + h


@dataclass
class UNetCIFAR10Config:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.1
    resolution: int = 32
    resamp_with_conv: bool = True


class UNetCIFAR10(nn.Module):
    def __init__(self, cfg: UNetCIFAR10Config):
        super().__init__()
        self.cfg = cfg

        self.in_channels = int(cfg.in_channels)
        self.out_channels = int(cfg.out_channels)
        self.ch = int(cfg.base_channels)
        self.ch_mult = tuple(int(v) for v in cfg.channel_mults)
        self.num_res_blocks = int(cfg.num_res_blocks)
        self.attn_resolutions = set(int(v) for v in cfg.attn_resolutions)
        self.dropout = float(cfg.dropout)
        self.resolution = int(cfg.resolution)
        self.temb_ch = self.ch * 4

        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        self.temb_dense = nn.ModuleList(
            [
                nn.Linear(self.ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

        curr_res = self.resolution
        in_ch = self.ch
        self.down = nn.ModuleList()
        self.skip_channels: list[int] = [in_ch]

        for level, mult in enumerate(self.ch_mult):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = self.ch * mult
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch, temb_channels=self.temb_ch, dropout=self.dropout))
                in_ch = out_ch
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(in_ch))
                self.skip_channels.append(in_ch)
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != len(self.ch_mult) - 1:
                down.downsample = Downsample(in_ch, with_conv=cfg.resamp_with_conv)
                curr_res //= 2
                self.skip_channels.append(in_ch)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch, temb_channels=self.temb_ch, dropout=self.dropout)
        self.mid.attn_1 = AttnBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch, temb_channels=self.temb_ch, dropout=self.dropout)

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.ch_mult))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = self.ch * mult
            for _ in range(self.num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                block.append(ResnetBlock(in_ch + skip_ch, out_ch, temb_channels=self.temb_ch, dropout=self.dropout))
                in_ch = out_ch
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(in_ch))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(in_ch, with_conv=cfg.resamp_with_conv)
                curr_res *= 2
            self.up.append(up)

        if self.skip_channels:
            raise RuntimeError(f"Internal skip-channel bookkeeping error: leftover={self.skip_channels}")

        self.norm_out = group_norm(in_ch)
        self.conv_out = zero_module(nn.Conv2d(in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dtype != torch.long:
            timesteps = timesteps.long()

        temb = get_timestep_embedding(timesteps, self.ch)
        temb = self.temb_dense[0](temb)
        temb = swish(temb)
        temb = self.temb_dense[1](temb)

        hs = [self.conv_in(x)]
        h = hs[-1]

        for level in self.down:
            for i, block in enumerate(level.block):
                h = block(h, temb)
                if i < len(level.attn):
                    h = level.attn[i](h)
                hs.append(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
                hs.append(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for level in self.up:
            for i, block in enumerate(level.block):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, temb)
                if i < len(level.attn):
                    h = level.attn[i](h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)

        if hs:
            raise RuntimeError(f"Skip-connection stack not exhausted: remaining={len(hs)}")

        h = self.conv_out(swish(self.norm_out(h)))
        return h