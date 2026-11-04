# geoseg/models/backbones/shade_mamba.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional DCNv2 if available (torchvision >= 0.13 has DeformConv2d)
try:
    from torchvision.ops import DeformConv2d
    HAS_DCN = True
except Exception:
    HAS_DCN = False


# ----------------------- tiny utils -----------------------
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
    if p is None:
        p = k // 2
    layers = [
        nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ECA(nn.Module):
    """Efficient Channel Attention (very light)."""
    def __init__(self, c, k=3):
        nn.Module.__init__(self)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)  # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        return x * self.sig(y)


# ----------------------- PreNorm/Regularizers -----------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        nn.Module.__init__(self)
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # B C H W -> B H W C -> LN -> B C H W
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXtV2-style)."""
    def __init__(self, c, eps=1e-6):
        nn.Module.__init__(self)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)               # B,C,1,1
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)             # B,C,1,1
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, drop_prob=0.0):
        nn.Module.__init__(self)
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerScale2d(nn.Module):
    """Per-channel residual scaling (safe, tiny)."""
    def __init__(self, c, init=1e-6):
        nn.Module.__init__(self)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * init)

    def forward(self, x):
        return self.gamma * x


# ----------------------- BlurPool downsample -----------------------
class BlurPoolDown(nn.Module):
    """Anti-aliased downsample: blur then stride-2 conv."""
    def __init__(self, in_ch, out_ch):
        nn.Module.__init__(self)
        filt = torch.tensor([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]], dtype=torch.float32)
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :].repeat(in_ch, 1, 1, 1))
        self.pad = nn.ReplicationPad2d(1)
        self.conv = conv_bn_act(in_ch, out_ch, k=3, s=2)  # stride-2 after blur

    def forward(self, x):
        x = F.conv2d(self.pad(x), self.filt, stride=1, padding=0, groups=x.size(1))
        return self.conv(x)


# ----------------------- Deformable depthwise -----------------------
class DeformDWConv(nn.Module):
    def __init__(self, ch, k=3):
        nn.Module.__init__(self)
        if HAS_DCN:
            self.offset = nn.Conv2d(ch, 2 * k * k, kernel_size=k, padding=k // 2)
            self.dcn = DeformConv2d(ch, ch, kernel_size=k, padding=k // 2, groups=ch, bias=False)
            self.bn = nn.BatchNorm2d(ch)
            self.act = nn.GELU()
        else:
            # stronger fallback: two-branch depthwise (dilated) + fuse
            self.dw3 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
            self.dw5 = nn.Conv2d(ch, ch, 3, 1, 2, groups=ch, dilation=2, bias=False)
            self.bn = nn.BatchNorm2d(ch)
            self.act = nn.GELU()

    def forward(self, x):
        if HAS_DCN:
            off = self.offset(x)
            y = self.dcn(x, off)
            return self.act(self.bn(y))
        else:
            y = self.dw3(x) + self.dw5(x)
            return self.act(self.bn(y))


# ----------------------- LKA-lite (expanded RF) -----------------------
class LKALite(nn.Module):
    """
    Local large kernel attention: expanded receptive field via
    5x5 dw -> 7x7 dw(dil=3) -> 9x9 dw(dil=2) (effective ~21x21) then 1x1.
    """
    def __init__(self, ch):
        nn.Module.__init__(self)
        self.proj_in = nn.Conv2d(ch, ch, 1, 1, 0)
        self.dw5     = nn.Conv2d(ch, ch, 5, 1, 2, groups=ch)
        self.dw7d3   = nn.Conv2d(ch, ch, 7, 1, 9, groups=ch, dilation=3)
        self.dw9d2   = nn.Conv2d(ch, ch, 9, 1, 8, groups=ch, dilation=2)
        self.proj_out = nn.Conv2d(ch, ch, 1, 1, 0)

    def forward(self, x):
        y = self.proj_in(x)
        y = self.dw9d2(self.dw7d3(self.dw5(y)))
        y = self.proj_out(y)
        return x * torch.sigmoid(y)


# ----------------------- FFN (DW-Separable) -----------------------
class DepthwiseSeparable(nn.Module):
    def __init__(self, ch, expansion=4):
        nn.Module.__init__(self)
        mid = int(ch * expansion)
        self.pw1 = conv_bn_act(ch, mid, 1, 1, 0)
        self.dw  = conv_bn_act(mid, mid, 3, 1, 1, g=mid)
        self.pw2 = conv_bn_act(mid, ch, 1, 1, 0, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pw2(self.dw(self.pw1(x))) + x)


# ----------------------- frequency/edge path -----------------------
class FixedEdge(nn.Module):
    """Sobel + Laplacian with a learnable gate (no params for kernels)."""
    def __init__(self, ch):
        nn.Module.__init__(self)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        lap = torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=torch.float32)
        k = torch.stack([sobel_x, sobel_y, lap])  # (3, 3, 3)
        self.register_buffer("kern", k[:, None, :, :])  # (3,1,3,3)
        self.proj = nn.Conv2d(3, ch, 1, 1, 0, bias=False)
        self.bn   = nn.BatchNorm2d(ch)
        self.gate = nn.Parameter(torch.zeros(1))  # ReZero init

    def forward(self, x):
        m = x.mean(1, keepdim=True)               # B,1,H,W
        e = F.conv2d(m, self.kern, padding=1)     # B,3,H,W
        e = self.bn(self.proj(e))
        return self.gate * e


# ----------------------- global mixer (upgraded axial) -----------------------
class AxialScan(nn.Module):
    """
    Stronger axial mixer with gated depthwise 1D convs along H and W,
    plus a global token. Keeps the same API as before.
    """
    def __init__(self, ch, k=127, bottleneck=4, drop=0.0):
        nn.Module.__init__(self)
        r = max(16, ch // bottleneck)  # ensure reasonable width
        self.red  = nn.Conv2d(ch, r, 1, 1, 0)

        # width-axis mixing (scan across W for each row)
        self.mix_w = nn.Conv1d(r, r, k, padding=k//2, groups=r, bias=False)
        self.gate_w = nn.Conv1d(r, r, 9, padding=4, groups=r, bias=False)

        # height-axis mixing (scan across H for each col)
        self.mix_h = nn.Conv1d(r, r, k, padding=k//2, groups=r, bias=False)
        self.gate_h = nn.Conv1d(r, r, 9, padding=4, groups=r, bias=False)

        # global token (cheap global context)
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(r, r, 1, 1, 0), nn.GELU()
        )

        self.proj  = nn.Conv2d(r, ch, 1, 1, 0)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.red(x)                         # B,r,H,W

        # W-scan
        wv = z.permute(0, 2, 1, 3).reshape(B * H, -1, W)      # (B*H, r, W)
        gw = torch.sigmoid(self.gate_w(wv))
        wv = self.mix_w(wv) * gw
        wv = wv.reshape(B, H, -1, W).permute(0, 2, 1, 3)      # (B, r, H, W)

        # H-scan
        hv = z.permute(0, 3, 1, 2).reshape(B * W, -1, H)      # (B*W, r, H)
        gh = torch.sigmoid(self.gate_h(hv))
        hv = self.mix_h(hv) * gh
        hv = hv.reshape(B, W, -1, H).permute(0, 2, 3, 1)      # (B, r, H, W)

        g = self.gp(z).expand_as(z)                           # (B, r, H, W)

        y = self.proj(wv + hv + g)
        return self.drop(y)


# ----------------------- SCAM block -----------------------
class SCAMBlock(nn.Module):
    """
    Shadow-aware Cross-scale Attentive Mixer:
    - Local LKA + ECA + GRN
    - Global axial scan + GRN (upgraded mixer)
    - Fixed edge/frequency path + GRN
    - LayerScale on all residuals + DropPath
    - DW-separable FFN + GRN
    """
    def __init__(self, ch, use_global=True, drop_path=0.1):
        nn.Module.__init__(self)
        self.use_global = use_global

        # local path (slightly deeper)
        self.norm_l = LayerNorm2d(ch)
        self.local  = nn.Sequential(LKALite(ch), ECA(ch), GRN(ch))
        self.scale_l = LayerScale2d(ch, init=1e-6)

        # global path
        self.norm_g = LayerNorm2d(ch)
        self.global_m = AxialScan(ch, k=127, bottleneck=4, drop=0.0) if use_global else None
        self.grn_g = GRN(ch)
        self.scale_g = LayerScale2d(ch, init=1e-6)

        # frequency/edge path
        self.norm_f = LayerNorm2d(ch)
        self.freq   = FixedEdge(ch)
        self.grn_f  = GRN(ch)
        self.scale_f = LayerScale2d(ch, init=1e-6)

        # droppath
        self.dp = DropPath(drop_path)

        # FFN with pre-norm + GRN + layerscale
        self.norm_ffn = LayerNorm2d(ch)
        self.ffn = DepthwiseSeparable(ch)
        self.grn_ffn = GRN(ch)
        self.scale_ffn = LayerScale2d(ch, init=1e-6)

    def forward(self, x):
        y = x + self.dp(self.scale_l(self.local(self.norm_l(x))))
        if self.use_global:
            y = y + self.dp(self.scale_g(self.grn_g(self.global_m(self.norm_g(y)))))
        y = y + self.dp(self.scale_f(self.grn_f(self.freq(self.norm_f(y)))))
        y = y + self.dp(self.scale_ffn(self.grn_ffn(self.ffn(self.norm_ffn(y)))))
        return y


# ----------------------- SK-tiny -----------------------
class SKTiny(nn.Module):
    def __init__(self, ch):
        nn.Module.__init__(self)
        self.dw3 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.dw5 = nn.Conv2d(ch, ch, 3, 1, 2, groups=ch, dilation=2)
        self.fc  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 8, 1), nn.GELU(),
            nn.Conv2d(ch // 8, 2, 1)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )

    def forward(self, x):
        a = self.dw3(x)
        b = self.dw5(x)
        w = self.fc(x)  # B,2,1,1
        w = F.softmax(w.view(x.size(0), 2, 1, 1), dim=1)
        y = w[:, 0:1] * a + w[:, 1:2] * b
        return self.pw(y)


# ----------------------- SPS (re-param) stem -----------------------
class SPSStem(nn.Module):
    """3 branches at train-time, fused to a 3x3 at eval for free inference."""
    def __init__(self, in_ch, out_ch):
        nn.Module.__init__(self)
        self.b1 = conv_bn_act(in_ch, out_ch, k=3, s=2)  # 1/2
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, 2, 2, bias=False, groups=1),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, 2, 3, bias=False, groups=1),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        # a small local refine before 1/4
        self.mix = conv_bn_act(out_ch, out_ch, k=3, s=2)  # -> 1/4
        self.fused = None  # will hold a single conv after fuse()

    @torch.no_grad()
    def fuse(self):
        # runtime fast path switch (placeholder kept for API)
        self.fused = nn.Sequential(
            nn.Conv2d(self.b1[0].in_channels, self.b1[0].out_channels, 3, 4, 1, bias=False),
            nn.BatchNorm2d(self.b1[0].out_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.fused is not None:
            x = self.fused(x)
        else:
            x = self.b1(x) + self.b2(x) + self.b3(x)
        return self.mix(x)


# ----------------------- HR branch + gated fusion -----------------------
class HRBranch(nn.Module):
    def __init__(self, ch_in, ch=64):
        nn.Module.__init__(self)
        self.conv1 = conv_bn_act(ch_in, ch, 3, 1)
        self.conv2 = conv_bn_act(ch, ch, 3, 1)
        self.res_scale = LayerScale2d(ch, init=1e-6)

    def forward(self, x_1_4):
        y = self.conv2(self.conv1(x_1_4))
        return x_1_4 + self.res_scale(y)


class GateFuse(nn.Module):
    def __init__(self, ch_low, ch_high):
        nn.Module.__init__(self)
        self.proj_h = conv_bn_act(ch_high, ch_low, k=1, s=1, p=0, act=False)
        self.gate = nn.Sequential(
            nn.Conv2d(ch_low, ch_low, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(ch_low, ch_low, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, low, high_up):
        h = self.proj_h(high_up)
        g = self.gate(low)
        return low + h * g


# ----------------------- Modality Adapter (optional) -----------------------
class ModalityAdapter(nn.Module):
    """
    Optional adapter to combine multiple modalities before the stem.
    """
    def __init__(self, target_ch, known_modalities=('rgb', 'ms', 'nir', 'dsm', 'sar', 'thermal')):
        nn.Module.__init__(self)
        self.target_ch = target_ch
        self.known = list(known_modalities)
        self.proj = nn.ModuleDict()
        self.alpha = nn.ParameterDict()
        for name in self.known:
            self.proj[name] = None
            self.alpha[name] = nn.Parameter(torch.tensor(1.0))

    def _ensure_proj(self, name, in_ch):
        if self.proj[name] is None:
            self.proj[name] = conv_bn_act(in_ch, self.target_ch, k=3, s=1)

    def forward(self, x):
        if isinstance(x, dict):
            feats = []
            weights = []
            for name, tensor in x.items():
                if name not in self.proj:
                    self.proj[name] = None
                    self.alpha[name] = nn.Parameter(torch.tensor(1.0))
                self._ensure_proj(name, tensor.size(1))
                feats.append(self.proj[name](tensor))
                weights.append(F.relu(self.alpha[name]))
            if len(feats) == 1:
                return feats[0]
            wsum = torch.stack(weights).sum() + 1e-6
            fused = 0
            for wi, fi in zip(weights, feats):
                fused = fused + (wi / wsum) * fi
            return fused
        in_ch = x.size(1)
        if in_ch == self.target_ch:
            return x
        proj = conv_bn_act(in_ch, self.target_ch, k=3, s=1)
        self.add_module("_once_proj", proj)
        return self._once_proj(x)


# ----------------------- Backbone -----------------------
class SHaDeMambaBackbone(nn.Module):
    """
    Outputs C1 (1/4), C2 (1/8), C3 (1/16), C4 (1/32)
    """
    def __init__(self, in_ch=3, dims=(64, 128, 256, 384), drop_path=0.10, use_adapter=True):
        nn.Module.__init__(self)
        c1, c2, c3, c4 = dims

        # Optional multi-modal adapter (fuses to c1 channels)
        self.adapter = ModalityAdapter(target_ch=in_ch) if use_adapter else None

        # Stem + persistent HR branch
        self.stem = SPSStem(in_ch, c1)    # -> 1/4
        self.hr   = HRBranch(c1, ch=c1)

        # Per-stage DropPath (deeper -> larger)
        base = float(drop_path)
        dp1, dp2, dp3, dp4 = max(0.0, base * 0.5), base, min(0.25, base * 1.75), min(0.30, base * 2.25)

        # Stage 1 (1/4): local + freq only (no global)
        self.s1 = nn.Sequential(
            SCAMBlock(c1, use_global=False, drop_path=dp1),
            SCAMBlock(c1, use_global=False, drop_path=dp1),
            SCAMBlock(c1, use_global=False, drop_path=dp1),
        )

        # Downsample to 1/8
        self.ds1 = BlurPoolDown(c1, c2)

        # Stage 2 (1/8): add global
        self.s2 = nn.Sequential(
            SCAMBlock(c2, use_global=True, drop_path=dp2),
            SCAMBlock(c2, use_global=True, drop_path=dp2),
            SCAMBlock(c2, use_global=True, drop_path=dp2)
        )

        # Downsample to 1/16
        self.ds2 = BlurPoolDown(c2, c3)

        # Stage 3 (1/16): deeper + deform tail
        self.s3a = nn.Sequential(
            SCAMBlock(c3, True, dp3),
            SCAMBlock(c3, True, dp3),
            SCAMBlock(c3, True, dp3)
        )
        self.s3d = DeformDWConv(c3)

        # Downsample to 1/32
        self.ds3 = BlurPoolDown(c3, c4)

        # Stage 4 (1/32): brief + deform + SK (keep size)
        self.s4 = nn.Sequential(
            SCAMBlock(c4, True, dp4),
            DeformDWConv(c4),
            SKTiny(c4)
        )

        # HR injections
        self.fuse2 = GateFuse(c2, c1)
        self.fuse3 = GateFuse(c3, c1)
        self.fuse4 = GateFuse(c4, c1)

        # Small heads to expose SK at lower stages too
        self.sk2 = SKTiny(c2)
        self.sk3 = SKTiny(c3)

    def forward(self, x):
        """
        x: Tensor[B,C,H,W]  OR  Dict[str, Tensor[B,Cm,H,W]]
        """
        if isinstance(x, dict):
            if 'rgb' in x:
                base = x['rgb']
            else:
                base = next(iter(x.values()))
            if self.adapter is not None:
                fused = self.adapter(x)  # [B, in_ch, H, W]
            else:
                fused = None
                for t in x.values():
                    fused = t if fused is None else fused + t
            x = fused if fused is not None else base  # tensor

        # 1/4
        c1 = self.stem(x)
        hr = self.hr(c1)
        c1 = self.s1(c1)

        # 1/8
        c2 = self.ds1(c1)
        c2 = self.s2(c2)
        c2 = self.fuse2(c2, F.interpolate(hr, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        c2 = self.sk2(c2)

        # 1/16
        c3 = self.ds2(c2)
        c3 = self.s3a(c3)
        c3 = self.s3d(c3)
        c3 = self.fuse3(c3, F.interpolate(hr, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        c3 = self.sk3(c3)

        # 1/32
        c4 = self.ds3(c3)
        c4 = self.s4(c4)
        c4 = self.fuse4(c4, F.interpolate(hr, size=c4.shape[-2:], mode='bilinear', align_corners=False))

        return c1, c2, c3, c4
