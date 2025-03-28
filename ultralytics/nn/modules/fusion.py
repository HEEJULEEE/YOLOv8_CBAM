import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import CBAM 
from ultralytics.utils import LOGGER

__all__ = "FusionNeck"

class FusionNeck(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.cbam_blocks = nn.ModuleList([CBAM(ch * 2) for ch in channels_list])
        self.reduce_layers = nn.ModuleList([nn.Conv2d(ch * 2, ch, kernel_size=1) for ch in channels_list])

    def forward(self, rgb_feats, thermal_feats, weights):
        #LOGGER.info("🚀 FusionNeck forward called!")
        fused_feats = []
        B = rgb_feats[0].shape[0]

        for i in range(len(rgb_feats)):
            fr = rgb_feats[i]
            ft = thermal_feats[i]
            wr = weights[:, 0].view(B, 1, 1, 1).to(fr.device)
            wt = weights[:, 1].view(B, 1, 1, 1).to(ft.device)

            fr_weighted = fr * wr
            ft_weighted = ft * wt

            f_cat = torch.cat([fr_weighted, ft_weighted], dim=1)
            #print(f"[FusionNeck] f_cat shape: {f_cat.shape}")
            f_cbam = self.cbam_blocks[i](f_cat)
            #print(f"[FusionNeck] f_cbam shape: {f_cbam.shape}")
            f_reduced = self.reduce_layers[i](f_cbam)
            fused_feats.append(f_reduced)

            #LOGGER.info(f"[FusionNeck] Layer {i} - RGB weights: {wr.view(-1)}, Thermal weights: {wt.view(-1)}")
        #print(f"[FusionNeck] Final fused_feats shape: {[f.shape for f in fused_feats]}")  # 최종 출력 확인
        return fused_feats