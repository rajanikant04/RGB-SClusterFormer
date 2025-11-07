# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ======================================================================
# --- Deformable Convolution (from deform_conv.py) ---
# ======================================================================

class DeformConv2d(nn.Module):
    """
    Deformable Convolution v2.
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                     g_rb.unsqueeze(dim=1) * x_q_rb + \
                     g_lb.unsqueeze(dim=1) * x_q_lb + \
                     g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            indexing='ij'
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0.to(offset.device) + p_n.to(offset.device) + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset


# ======================================================================
# --- Model Components (from SClusterFormer.py) ---
# ======================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def pairwise_euclidean_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    em = torch.norm(x1.unsqueeze(-2) - x2.unsqueeze(-3), dim=-1)
    sim = torch.exp(-em)
    return sim


class Cluster2D(nn.Module):
    def __init__(self, patch_size, dim=768, out_dim=768, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4,
                 head_dim=24, return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.4))

    def forward(self, x):
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, f"Feature map size ({w0}*{h0}) not divisible by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')
        b, c, ww, hh = centers.shape
        sim = self.rule2(self.sim_beta + self.sim_alpha * pairwise_euclidean_sim(centers.reshape(b, c, -1).permute(0, 2, 1), x.reshape(b, c, -1).permute(0, 2, 1)))
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out


class PixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, i=0):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1 if i == 0 else 2, padding=1 if i == 0 else (3 // 2))
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))
        after_feature_map_size = x.shape[2]
        x = x.flatten(2).transpose(1, 2)
        return x, after_feature_map_size


class Block2D(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster2D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1, fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleDeformConv2D(nn.Module):
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.fuse = nn.Conv2d(deform_conv.outc * 3, deform_conv.outc, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for k in self.kernel_sizes:
            scale = k / self.kernel_sizes[0]
            if scale != 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            feat = self.deform_conv(x_scaled)
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(feat)
        out = torch.cat(feats, dim=1)
        out = self.fuse(out) + feats[1]
        return out


# ======================================================================
# --- Main Model: RGB_SClusterFormer ---
# ======================================================================

class RGB_SClusterFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_stages=3,
                 embed_dims=[256, 128, 64], num_heads=[8, 8, 8], mlp_ratios=[1, 1, 1],
                 depths=[2, 2, 2]):
        super().__init__()
        self.num_stages = num_stages
        
        deform_conv_shared_rgb = DeformConv2d(inc=in_chans, outc=30, kernel_size=9, padding=1, bias=False, modulation=True)
        self.deform_conv_layer_rgb = MultiScaleDeformConv2D(deform_conv_shared_rgb)
        
        stem_out_channels = 30 
        self.embed_img = [img_size, math.ceil(img_size / 2), math.ceil(math.ceil(img_size / 2) / 2)]
        
        for i in range(num_stages):
            patch_embed2d = PixelEmbedding(
                in_feature_map_size=img_size if i == 0 else self.embed_img[i - 1],
                in_chans=stem_out_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                i=i
            )
            block2d = nn.ModuleList([Block2D(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                drop=0., patch_size=self.embed_img[i]) for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            setattr(self, f"patch_embed2d{i + 1}", patch_embed2d)
            setattr(self, f"block2d{i + 1}", block2d)
            setattr(self, f"norm2d{i + 1}", norm)
            
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed2d{i + 1}")
            block = getattr(self, f"block2d{i + 1}")
            norm = getattr(self, f"norm2d{i + 1}")
            x, s = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.deform_conv_layer_rgb(x)
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x