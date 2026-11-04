# geoseg/models/backbones/NAViT.py
# NAViT: Navigation-enhanced Vision Transformer Backbone (segmentation-centric)
# Produces a 4-level feature pyramid (C1..C4) compatible with decoders.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------- small ops ----------

class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, bias=False):
        p = ((s - 1) + d * (k - 1)) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, bias=False):
        p = ((s - 1) + d * (k - 1)) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

class MLP2d(nn.Module):
    def __init__(self, c, expansion=4, drop=0.0):
        super().__init__()
        hid = int(c * expansion)
        self.block = nn.Sequential(
            nn.Conv2d(c, hid, 1, 1, 0, bias=True),
            nn.ReLU6(inplace=True),
            nn.Dropout(drop, inplace=True),
            nn.Conv2d(hid, c, 1, 1, 0, bias=True),
            nn.Dropout(drop, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# ---------- Linear attention (2D, O(N)) ----------
# Positive feature map trick: phi(x) = elu(x) + 1 (use F.elu)

class LinearAttention2D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, qkv_bias=False, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        d_head = dim // heads if dim_head is None else dim_head
        self.d_head = d_head
        inner = d_head * heads

        self.to_q = nn.Conv2d(dim, inner, 1, bias=qkv_bias)
        self.to_k = nn.Conv2d(dim, inner, 1, bias=qkv_bias)
        self.to_v = nn.Conv2d(dim, inner, 1, bias=qkv_bias)
        self.out = nn.Conv2d(inner, dim, 1, bias=True)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def phi(x):
        return F.elu(x, alpha=1.0) + 1.0

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b (h d) hh ww -> b h d (hh ww)', h=self.heads)
        k = rearrange(k, 'b (h d) hh ww -> b h d (hh ww)', h=self.heads)
        v = rearrange(v, 'b (h d) hh ww -> b h d (hh ww)', h=self.heads)

        q = self.phi(q)
        k = self.phi(k)

        # k_sum: [B, H, D]
        k_sum = k.sum(dim=-1)
        # kv: [B, H, D, D]
        kv = torch.einsum('bhdn,bhen->bhde', k, v)
        # numerator: [B, H, D, N]
        num = torch.einsum('bhdn,bhde->bhen', q, kv)
        # denominator: [B, H, 1, N]
        den = torch.einsum('bhdn,bhd->bhn', q, k_sum).unsqueeze(2).clamp_min(1e-6)

        out = num / den  # [B, H, D, N]
        out = rearrange(out, 'b h d n -> b (h d) n', h=self.heads, d=self.d_head)
        out = out.view(B, self.heads * self.d_head, H, W)
        out = self.drop(self.out(out))
        return out

# ---------- Local window path (cheap) ----------
# Ensure SAME spatial size for any kernel (odd/even) by cropping if needed.

class LocalWindowConv(nn.Module):
    def __init__(self, dim, window=7):
        super().__init__()
        self.window = window
        self.dw = nn.Conv2d(dim, dim, kernel_size=window, padding=window//2, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.dw(x)
        # If window is even, padding=window//2 can produce +1 size -> crop back
        if y.size(-2) != H or y.size(-1) != W:
            y = y[:, :, :H, :W]
        y = self.pw(y)
        y = self.bn(y)
        return self.act(y)

# ---------- AFB: Gated cross-modal fusion (RGB + optional modality M) ----------

class AFB(nn.Module):
    def __init__(self, c, heads=4, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.lin = LinearAttention2D(c, heads=heads, qkv_bias=qkv_bias, dropout=dropout)
        self.proj = nn.Conv2d(c, c, 1, bias=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c*2, c//4, 1, bias=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(c//4, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, R, M=None, meta=None):
        if M is None:
            return R
        g_in = torch.cat([F.adaptive_avg_pool2d(R, 1), F.adaptive_avg_pool2d(M, 1)], dim=1)
        g = self.gate(g_in)  # [B,C,1,1]
        fused = R + g * M
        out = self.lin(fused)
        return R + self.proj(out)

# ---------- MSNA block: local + global with dynamic gating ----------

class MSNABlock(nn.Module):
    def __init__(self, dim, heads=8, window=7, mlp_ratio=4.0, drop=0.0, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.local = LocalWindowConv(dim, window=window)
        self.global_lin = LinearAttention2D(dim, heads=heads, qkv_bias=qkv_bias, dropout=drop)
        self.gater = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//4, 1, bias=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim//4, 2, 1, bias=True),
            nn.Softmax(dim=1)
        )
        self.drop_path = nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP2d(dim, expansion=mlp_ratio, drop=drop)

    def forward(self, x):
        idn = x
        x = self.norm1(x)
        loc = self.local(x)      # guaranteed to be HxW
        glob = self.global_lin(x)
        w = self.gater(x)        # [B,2,1,1]
        wl, wg = w[:, :1], w[:, 1:]
        fused = wl * loc + wg * glob
        x = idn + fused
        x = x + self.mlp(self.norm2(x))
        return x

# ---------- Strong Saliency-Guided Downsample ----------

class SaliencyDownsample(nn.Module):
    """
    Stronger learnable downsample:
      - saliency reweight
      - depthwise separable conv with stride=2 (better gradients vs. pooling)
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(c_in, c_in // 4, 1, bias=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(c_in // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=2,
                            padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        s = self.score(x)
        x = x * (1.0 + s)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# ---------- Patch embedding ----------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=96, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(dim)
    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        return F.relu6(x, inplace=True)

# ---------- Stage (stack of MSNA blocks) ----------

class Stage(nn.Module):
    def __init__(self, dim, depth, heads, window=7, mlp_ratio=4.0, drop=0.0, qkv_bias=False,
                 downsample_out=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSNABlock(dim, heads=heads, window=window, mlp_ratio=mlp_ratio, drop=drop, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.down = None
        if downsample_out is not None:
            self.down = SaliencyDownsample(dim, downsample_out)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x_out = x
        if self.down is not None:
            x = self.down(x)
            return x_out, x
        return x_out, x_out

# ---------- NAViT backbone ----------

class NAViT(nn.Module):
    """
    Returns a feature pyramid (C1, C2, C3, C4) for segmentation decoders.
    """
    def __init__(
        self,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        window=7,
        mlp_ratio=4.0,
        drop=0.0,
        qkv_bias=False,
        use_afb=False
    ):
        super().__init__()
        self.use_afb = use_afb
        self.patch = PatchEmbed(in_ch=in_chans, dim=embed_dim, patch=4)

        c1, c2, c3, c4 = embed_dim, embed_dim*2, embed_dim*4, embed_dim*8

        self.stage1 = Stage(c1, depths[0], heads[0], window=window, mlp_ratio=mlp_ratio, drop=drop,
                            qkv_bias=qkv_bias, downsample_out=c2)
        self.stage2 = Stage(c2, depths[1], heads[1], window=window, mlp_ratio=mlp_ratio, drop=drop,
                            qkv_bias=qkv_bias, downsample_out=c3)
        self.stage3 = Stage(c3, depths[2], heads[2], window=window, mlp_ratio=mlp_ratio, drop=drop,
                            qkv_bias=qkv_bias, downsample_out=c4)
        self.stage4 = Stage(c4, depths[3], heads[3], window=window, mlp_ratio=mlp_ratio, drop=drop,
                            qkv_bias=qkv_bias, downsample_out=None)

        if use_afb:
            self.afb1 = AFB(c1, heads=4)
            self.afb2 = AFB(c2, heads=6)
            self.afb3 = AFB(c3, heads=8)
            self.afb4 = AFB(c4, heads=8)

        self._enc_channels = (c1, c2, c3, c4)

    @property
    def out_channels(self):
        return self._enc_channels

    def forward(self, x, aux=None, meta=None):
        x = self.patch(x)        # 1/4

        # Stage 1
        s1_top, s1_down = self.stage1(x)
        if hasattr(self, 'afb1') and aux is not None:
            s1_top = self.afb1(s1_top, F.interpolate(aux, size=s1_top.shape[-2:], mode='bilinear', align_corners=False))
        # Stage 2
        s2_top, s2_down = self.stage2(s1_down)
        if hasattr(self, 'afb2') and aux is not None:
            s2_top = self.afb2(s2_top, F.interpolate(aux, size=s2_top.shape[-2:], mode='bilinear', align_corners=False))
        # Stage 3
        s3_top, s3_down = self.stage3(s2_down)
        if hasattr(self, 'afb3') and aux is not None:
            s3_top = self.afb3(s3_top, F.interpolate(aux, size=s3_top.shape[-2:], mode='bilinear', align_corners=False))
        # Stage 4 (no downsample after)
        s4_top, _ = self.stage4(s3_down)
        if hasattr(self, 'afb4') and aux is not None:
            s4_top = self.afb4(s4_top, F.interpolate(aux, size=s4_top.shape[-2:], mode='bilinear', align_corners=False))

        return (s1_top, s2_top, s3_top, s4_top)

def build_navit(
    in_chans=3,
    embed_dim=128,
    depths=(2, 2, 12, 2),
    heads=(4, 8, 16, 32),
    window=8,  # even window OK: LocalWindowConv crops correctly
    mlp_ratio=4.0,
    drop=0.0,
    qkv_bias=True,
    use_afb=False
):
    return NAViT(
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        heads=heads,
        window=window,
        mlp_ratio=mlp_ratio,
        drop=drop,
        qkv_bias=qkv_bias,
        use_afb=use_afb
    )
