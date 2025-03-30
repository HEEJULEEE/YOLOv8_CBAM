import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import CBAM

__all__ = "FusionNeck"


class FusionNeck(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.cbam_blocks = nn.ModuleList([
            CBAM(ch * 2) for ch in channels_list
        ])
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, kernel_size=1) for ch in channels_list
        ])

        # ✅ 추가: 후처리용 1x1 conv + BN + ReLU
        self.out_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1),
                #nn.BatchNorm2d(ch),
                nn.ReLU()
            ) for ch in channels_list
        ])

    def forward(self, rgb_feats, thermal_feats, weights):
        fused_feats = []
        B = rgb_feats[0].shape[0]
        
        w = torch.stack([weights[:, 0], weights[:, 1]], dim=1)  # [B, 2]
        w = torch.softmax(w, dim=1)  # Normalize across modalities

        for i in range(len(rgb_feats)):
            fr = rgb_feats[i]
            ft = thermal_feats[i]
            
            wr = weights[:, 0].view(B, 1, 1, 1).to(fr.device)
            wt = weights[:, 1].view(B, 1, 1, 1).to(ft.device)

            fr_weighted = fr * wr
            ft_weighted = ft * wt

            f_cat = torch.cat([fr_weighted, ft_weighted], dim=1)
            f_cbam = self.cbam_blocks[i](f_cat)
            f_reduced = self.reduce_layers[i](f_cbam)

            # ✅ 후처리 (선택지 A: out_conv 통과)
            f_out = self.out_conv[i](f_reduced)

            # ✅ (선택지 B: 단순 scaling 하고 싶다면)
            # f_out = f_reduced * 2

            fused_feats.append(f_out)
            #print(f"🔎 [Level {i}] RGB mean: {fr.mean().item():.4f}, Thermal mean: {ft.mean().item():.4f}")
            #print(f"⚖️ [Level {i}] Weighted RGB mean: {fr_weighted.mean().item():.4f}, Weighted Thermal mean: {ft_weighted.mean().item():.4f}")
            #print(f"🧠 [Level {i}] Fused feature mean: {f_out.mean().item():.4f}")
        return fused_feats

'''import torch
import torch.nn as nn

__all__ = "FusionNeck"


class FusionNeck(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        
        # CBAM 제거, reduce_layers 제거
        self.out_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU()
            ) for ch in channels_list
        ])

    def forward(self, rgb_feats, thermal_feats, weights):
        fused_feats = []
        B = rgb_feats[0].shape[0]

        # ✅ Softmax 기반 정규화된 가중치
        w = torch.stack([weights[:, 0], weights[:, 1]], dim=1)  # [B, 2]
        w = torch.softmax(w, dim=1)
        wr = w[:, 0].view(B, 1, 1, 1).to(rgb_feats[0].device)
        wt = w[:, 1].view(B, 1, 1, 1).to(thermal_feats[0].device)

        for i in range(len(rgb_feats)):
            fr = rgb_feats[i]
            ft = thermal_feats[i]

            # ✅ Softmax 정규화된 weighted fusion
            f_fused = fr * wr + ft * wt

            # ✅ 후처리 conv
            f_out = self.out_conv[i](f_fused)

            fused_feats.append(f_out)

            # Debug (원하면 주석 해제)
            # print(f"[Level {i}] fr_mean={fr.mean().item():.4f}, ft_mean={ft.mean().item():.4f}")
            # print(f"wr_mean={wr.mean().item():.4f}, wt_mean={wt.mean().item():.4f}")
            # print(f"Fused mean={f_out.mean().item():.4f}")

        return fused_feats
'''