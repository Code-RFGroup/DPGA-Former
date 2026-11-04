from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except Exception:
    # Backward compatibility with older timm
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # type: ignore


# -----------------------------
# Basic MLP blocks
# -----------------------------
class Mlp(nn.Module):
    """Token MLP used inside Swin blocks (Linear -> GELU -> Dropout -> Linear -> Dropout)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpDecoder(nn.Module):
    """Channel MLP for decoder feature maps (1x1 convs)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------
# Small conv helpers
# -----------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )


# -----------------------------
# Swin window utils
# -----------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct feature map from windows.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
    Returns: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# -----------------------------
# Swin attention blocks
# -----------------------------
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.
    x: (nW*B, N, C)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # merge
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        # if odd sizes, pad
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


# -----------------------------
# HiLo detail-preserving stem
# -----------------------------
class HiLoStem(nn.Module):
    def __init__(self, in_ch=3, embed=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, embed, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed, embed, 7, 1, 3, groups=embed, bias=False),  # depthwise large-kernel
            nn.SiLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed, embed // 4, 1), nn.SiLU(inplace=True),
            nn.Conv2d(embed // 4, embed, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        f = self.conv(x)
        return f * self.se(f)


class SwinTransformer(nn.Module):
    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=128,
                 depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.3, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False,
                 use_hilo: bool = True, hilo_gamma: float = 0.1):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_hilo = use_hilo

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm else None)
        if self.use_hilo:
            self.hilo = HiLoStem(in_chans, embed_dim)
            # Stabilize fusion so HiLo doesn't dominate early training
            self.hilo_gamma = nn.Parameter(torch.tensor(hilo_gamma))

        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            self.add_module(f"norm{i_layer}", layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        # Patch embedding
        pe = self.patch_embed(x)  # (B, C, H/4, W/4) if norm is None; else normalized map
        Wh, Ww = pe.size(2), pe.size(3)

        if self.use_hilo:
            hi = self.hilo(x)                  # (B, C, H, W)
            hi_ds = F.avg_pool2d(hi, 4, 4)     # -> (B, C, H/4, W/4)
            pe = pe + self.hilo_gamma * hi_ds  # gated fusion for stability

        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x_tokens = (pe + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x_tokens = pe.flatten(2).transpose(1, 2)
        x_tokens = self.pos_drop(x_tokens)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x_tokens, Wh, Ww = layer(x_tokens, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)  # res1..res4


# -----------------------------
# Decoder blocks
# -----------------------------
class ReGLA(nn.Module):
    """
    ReGLA = Resolution-preserving Global-Local Attention
    """
    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    @staticmethod
    def _pad_windows(x: torch.Tensor, ws: int):
        B, C, H, W = x.shape
        pad_r = (ws - W % ws) if W % ws != 0 else 0
        pad_b = (ws - H % ws) if H % ws != 0 else 0
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b), mode='reflect')
        return x, H, W

    @staticmethod
    def _pad_out(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)

        # pad for windowing
        x_pad, H0, W0 = self._pad_windows(x, self.ws)
        Bp, Cp, Hp, Wp = x_pad.shape

        # q, k, v: (B, C, Hp, Wp)
        qkv = self.qkv(x_pad)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        heads = self.num_heads
        d = C // heads

        def reshape_to_windows(t: torch.Tensor) -> torch.Tensor:
            # (B, C, Hp, Wp) -> (B, heads, Hp, Wp, d)
            t = t.view(Bp, heads, d, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()
            # unfold spatial dims into windows: (B, heads, hh, ww, ws, ws, d)
            t = t.unfold(2, self.ws, self.ws).unfold(3, self.ws, self.ws)
            B_, h_win, w_win = t.shape[0], t.shape[2], t.shape[3]
            # -> (B*h_win*w_win, heads, ws*ws, d)
            t = t.contiguous().view(B_, heads, h_win * w_win, self.ws * self.ws, d)
            t = t.permute(0, 2, 1, 3, 4).contiguous().view(-1, heads, self.ws * self.ws, d)
            return t

        q = reshape_to_windows(q)
        k = reshape_to_windows(k)
        v = reshape_to_windows(v)

        attn = (q * (d ** -0.5)) @ k.transpose(-2, -1)

        if self.relative_pos_embedding:
            rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            rpb = rpb.view(self.ws * self.ws, self.ws * self.ws, -1).permute(2, 0, 1).contiguous()
            attn = attn + rpb.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B*nW, heads, ws*ws, d)

        # restore windows back to feature map
        nW = (Hp // self.ws) * (Wp // self.ws)
        out = out.view(Bp, nW, heads, self.ws, self.ws, d)
        out = out.view(Bp, Hp // self.ws, Wp // self.ws, heads, self.ws, self.ws, d)
        out = out.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(Bp, heads * d, Hp, Wp)
        out = out[:, :, :H0, :W0]

        out = (
            self.attn_x(F.pad(out, pad=(0, 0, 0, 1), mode='reflect')) +
            self.attn_y(F.pad(out, pad=(0, 1, 0, 0), mode='reflect'))
        )

        out = out + local
        out = self._pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return out


class DecoderBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16, window_size=8, drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = ReGLA(dim, num_heads=num_heads, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MlpDecoder(in_features=dim, hidden_features=int(dim * 4), out_features=dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AxialLite(nn.Module):
    """A light state-space style mixer (axis-wise depthwise convs) for long-range mixing."""
    def __init__(self, dim: int, d_expand: int = 2, k: int = 7):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim * d_expand, 1, bias=False)
        self.out_proj = nn.Conv2d(dim * d_expand, dim, 1, bias=False)
        self.ssm_x = nn.Conv2d(dim * d_expand, dim * d_expand, (1, k), padding=(0, k//2), groups=dim * d_expand, bias=False)
        self.ssm_y = nn.Conv2d(dim * d_expand, dim * d_expand, (k, 1), padding=(k//2, 0), groups=dim * d_expand, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = self.act(self.ssm_x(z)) + self.act(self.ssm_y(z))
        z = self.out_proj(z)
        return x + z


class WF_R(nn.Module):
    """WF_R = Weighted Fusion – Revised (renamed from WF)."""
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FRH_E(nn.Module):
    """FRH_E = Feature Refinement Head – Edge-aware (renamed from FeatureRefinementHead)."""
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x


class BDHead(nn.Module):
    """Boundary + Distance auxiliary prediction head (optional outputs)."""
    # CHANGED: add fuse_k (supports 1x1 or 3x3) and keep mid configurable
    def __init__(self, in_ch: int, mid: int = 64, fuse_k: int = 1):
        super().__init__()
        self.fuse = nn.Conv2d(in_ch, mid, fuse_k, padding=fuse_k // 2, bias=False)
        self.boundary = nn.Conv2d(mid, 1, 1, bias=True)
        self.distance = nn.Conv2d(mid, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]):
        z = F.relu(self.fuse(x), inplace=True)
        b = F.interpolate(self.boundary(z), size=size_hw, mode='bilinear', align_corners=False)
        d = F.interpolate(self.distance(z), size=size_hw, mode='bilinear', align_corners=False)
        return b, d


class Decoder(nn.Module):
    def __init__(self, encoder_channels: Tuple[int, int, int, int], decode_channels: int = 256,
                 dropout: float = 0.2, window_size: int = 8, num_classes: int = 6,
                 use_axial_at: str = 'b3', return_aux: bool = False,
                 axial_kernel: int = 7):  # ← add this parameter
        super().__init__()
        self.return_aux = return_aux

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = DecoderBlock(dim=decode_channels, num_heads=16, window_size=window_size)

        # middle stage: choose between ReGLA block or AxialLite
        if use_axial_at == 'b3':
            self.b3 = AxialLite(decode_channels, k=axial_kernel)  # ← use the parameter
        else:
            self.b3 = DecoderBlock(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p3 = WF_R(encoder_channels[-2], decode_channels)

        # next stage: keep a standard DecoderBlock by default
        if use_axial_at == 'b2':
            self.b2 = AxialLite(decode_channels, k=axial_kernel)  # ← use the parameter
        else:
            self.b2 = DecoderBlock(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p2 = WF_R(encoder_channels[-3], decode_channels)

        self.p1 = FRH_E(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1),
        )

        self.bd_head = BDHead(decode_channels, mid=64, fuse_k=1) if self.return_aux else None
        self._init_weight()

    def _init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, res1: torch.Tensor, res2: torch.Tensor, res3: torch.Tensor, res4: torch.Tensor,
                h: int, w: int):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.p1(x, res1)

        if self.return_aux:
            b_map, d_map = self.bd_head(x, (h, w))
        else:
            b_map, d_map = None, None

        seg = self.segmentation_head(x)
        seg = F.interpolate(seg, size=(h, w), mode='bilinear', align_corners=False)

        if self.return_aux:
            return seg, b_map, d_map
        else:
            return seg


# -----------------------------
# Top-level model
# -----------------------------
class DPGAFormer(nn.Module):
    def __init__(self, decode_channels: int = 256, dropout: float = 0.2,
                 embed_dim: int = 128, depths: Tuple[int, int, int, int] = (2, 2, 18, 2),
                 num_heads: Tuple[int, int, int, int] = (4, 8, 16, 32), freeze_stages: int = -1,
                 window_size: int = 8, num_classes: int = 6, use_hilo: bool = True,
                 use_axial_at: str = 'b3', return_aux: bool = False, hilo_gamma: float = 0.1, axial_kernel: int = 7):
        super().__init__()
        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                        frozen_stages=freeze_stages, use_hilo=use_hilo, hilo_gamma=hilo_gamma)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.decoder = Decoder(tuple(encoder_channels), decode_channels, dropout, window_size,
                               num_classes, use_axial_at=use_axial_at, return_aux=return_aux, axial_kernel=axial_kernel)

    def forward(self, x: torch.Tensor):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        return self.decoder(res1, res2, res3, res4, h, w)


def dpga_former(
    num_classes: int = 6,
    decoder_channels: int = 256,
    use_hilo: bool = True,
    use_axial_at: str = "b3",
    return_aux: bool = False,
    pretrained: bool = True,
    weight_path: Optional[str] = 'weights/stseg_base.pth',
    hilo_gamma: float = 0.1,
    axial_kernel: int = 7,
) -> DPGAFormer:
    model = DPGAFormer(
        num_classes=num_classes,
        decode_channels=decoder_channels,
        use_hilo=use_hilo,
        use_axial_at=use_axial_at,
        return_aux=return_aux,
        hilo_gamma=hilo_gamma,
        axial_kernel=axial_kernel,
    )

    if pretrained and weight_path is not None:
        try:
            ckpt = torch.load(weight_path, map_location='cpu')
            old_sd = ckpt.get('state_dict', ckpt)
            new_sd = model.state_dict()
            filtered = {k: v for k, v in old_sd.items() if k in new_sd and v.shape == new_sd[k].shape}
            new_sd.update(filtered)
            model.load_state_dict(new_sd, strict=False)
        except Exception as e:
            print(f"[DPGAFormer] Warning: failed to load weights from {weight_path}: {e}")

    return model


__all__ = [
    'DPGAFormer',
    'dpga_former',
    'ReGLA',
    'WF_R',
    'FRH_E',
]
