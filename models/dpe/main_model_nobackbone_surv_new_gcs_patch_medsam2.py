import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "./")  # noqa: E402
# import models.backbones.resnet3D as resnet3D  # noqa: E402
# from models.dpe.dpe import DPENet  # noqa: E402

# from data.multimodal_features import MultimodalCTWSIDataset  # noqa: E402
# from torch.utils.data import DataLoader  # noqa: E402

from collections import OrderedDict


class DynamicPositionalEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, topk_pos=16, topk_neg=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels * 2, out_channels))
        # Learnable token for missing modality
        self.missing_modality_token = nn.Parameter(torch.zeros(1, out_channels))
        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, f_rad, f_histo, rad_mask, histo_mask):
        """
        Now expects:
          f_rad:   [B, 131, token_dim]
          f_histo: [B, 131, token_dim]
        """
        pred_pos, pred_neg = self._correlation(f_rad, f_histo)
        # pred_pos, pred_neg are each [B, 131]

        # Concatenate along “feature” dimension → [B, 132].
        class_layers = torch.cat([pred_pos, pred_neg], dim=1)

        # modality_flag: 1 if *either* is missing
        modality_flags = ~torch.min(rad_mask, histo_mask)  # shape [B]

        # Flatten to feed MLP
        pos_input_flat = class_layers.view(class_layers.size(0), -1)  # [B, 132]
        pos_embedding = self.mlp(pos_input_flat)                      # [B, token_dim]

        adjusted_embedding = (
            pos_embedding
            + self.missing_modality_token * modality_flags.unsqueeze(1)
        )
        return adjusted_embedding  # [B, token_dim]

    def _correlation(self, f_rad, f_histo):
        """
        f_rad:   [B, 131, token_dim]
        f_histo: [B, 131, token_dim]
        Returns:
          pos_map: [B, 131]
          neg_map: [B, 131]
        """

        # 1) L2‐normalize along the feature dimension (dim=2)
        f_rad_norm   = F.normalize(f_rad,   p=2, dim=2)  # [B, 131, token_dim]
        f_histo_norm = F.normalize(f_histo, p=2, dim=2)  # [B, 131, token_dim]

        # 2) Compute full [B, 131, 131] similarity
        #    sim[b, i, j] = dot( f_rad_norm[b,i,:], f_histo_norm[b,j,:] )
        sim = torch.einsum("bnd,bmd->bnm", f_rad_norm, f_histo_norm)  # [B, 131, 131]

        # 3) For each “i” in 131, take top‐k positives across dim=2, then mean
        pos_map = torch.mean(
            torch.topk(sim,     self.topk_pos, dim=-1).values,  # [B, 131, topk_pos]
            dim=-1                                              # → [B, 131]
        )
        neg_map = torch.mean(
            torch.topk(-sim,    self.topk_neg, dim=-1).values,  # [B, 131, topk_neg]
            dim=-1                                              # → [B, 131]
        )

        return pos_map, neg_map


class HistoAdapter(nn.Module):
    """
    Now: accepts x of shape [batch, 131, input_dim],
         outputs [batch, 131, inter_dim].
    """

    def __init__(self, input_dim, inter_dim):
        super().__init__()
        # Project each of the 131 “slots” from input_dim → inter_dim
        self.fc_in = nn.Linear(input_dim, inter_dim)

        # Two residual blocks (operating on last dimension = inter_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(inter_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, inter_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(inter_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, inter_dim),
        )

        # Final normalization per slot
        self.final_norm = nn.LayerNorm(inter_dim, eps=1e-5)

    def forward(self, x):
        """
        x: [batch, 131, input_dim]
        returns: [batch, 131, inter_dim]
        """
        # 1) project each slot
        x = self.fc_in(x)  # [batch, 131, inter_dim]

        # 2) residual blocks (still per slot)
        x = x + self.block1(x)  # [batch, 131, inter_dim]
        x = x + self.block2(x)  # [batch, 131, inter_dim]

        # 3) final norm
        x = self.final_norm(x)  # [batch, 131, inter_dim]
        return x
# attention based feature fusion network
class fusion_layer(nn.Module):
    def __init__(self, d_model=64, dim_hider=256, nhead=2, dropout=0.1):
        super().__init__()
        self.cross_att1 = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.cross_att2 = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.cross_att3 = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

    def forward(self, f1, f2, pos):
        fv, fv_weights = self.cross_att1(f1, f2, f2 + pos)
        fk, fk_weights = self.cross_att1(f1, f2, f2)
        fq, fq_weights = self.cross_att2(f2, f1, f1)
        f22, f22_weights = self.cross_att3(fq, fk, fv)

        return f22

class MADPENetNoBackbonesSurv(nn.Module):
    def __init__(
        self,
        rad_input_dim=1024,
        histo_input_dim=1024,
        inter_dim=512,
        token_dim=64,
        dim_hider=256,
        num_classes=3,
        n_patch = 131
    ):
        super().__init__()
        self.rad_input_dim = rad_input_dim
        self.histo_input_dim = histo_input_dim
        self.inter_dim = inter_dim
        self.token_dim = token_dim
        self.num_classes = num_classes
        self.dim_hider = dim_hider
        self.n_patch = n_patch
        #
        # 1) RAD_ADAPTER:   [B, 131, rad_input_dim] → [B, 131, inter_dim]
        #
        # We use two Conv1d layers (kernel_size=1) over the “channel=rad_input_dim” dimension,
        # keeping sequence_length=131 intact.  After each conv, we still have shape [B, inter_dim, 131].
        # Finally we permute back to [B, 131, inter_dim].
        #
        self.rad_adapter = nn.Sequential(
            nn.Conv1d(
                in_channels=self.rad_input_dim,   # 1024
                out_channels=self.inter_dim,      # 512
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # → [B, 512, 131]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=self.inter_dim,      # 512
                out_channels=self.inter_dim,      # 512
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # → [B, 512, 131]
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        #
        # 2) HISTO_ADAPTER: [B, 131, histo_input_dim] → [B, 131, inter_dim]
        #
        self.histo_adapter = HistoAdapter(
            input_dim=self.histo_input_dim,  # e.g. 768
            inter_dim=self.inter_dim,        # 512
        )

        # Missing‐modality tokens (now each must be repeated 131 times later)
        self.missing_rad_token   = nn.Parameter(torch.randn(1, 1, self.inter_dim))   # [1,1,512]
        self.missing_histo_token = nn.Parameter(torch.randn(1, 1, self.inter_dim))   # [1,1,512]

        #
        # 3) TOKEN_ADAPT_RAD (unchanged): still takes raw [b1, 131, rad_input_dim] →
        #    conv → [b1, token_dim] (via average‐pooling over the 131 slots at the end).
        #
        self.token_adapt_rad = nn.Sequential(
            nn.Conv1d(
                in_channels=self.rad_input_dim,   # 1024
                out_channels=self.inter_dim,      # 512
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.Conv1d(
                in_channels=self.inter_dim,      # 512
                out_channels=self.inter_dim,      # 512
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.AdaptiveAvgPool1d(output_size=1),  # → [b1, 512, 1]
            nn.Flatten(start_dim=1),              # → [b1, 512]
            nn.Linear(self.inter_dim, self.inter_dim),  # [b1, 512]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.inter_dim, self.token_dim),  # [b1, 64]
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        #
        # 4) TOKEN_ADAPT_HISTO: [B, 131, histo_input_dim] → [B, token_dim]
        #
        #   (a) First project each of the 131 slots to inter_dim via same blocks as HistoAdapter
        #   (b) Then Conv1d→ token_dim over channel dimension, keep sequence length=131
        #   (c) AdaptiveAvgPool over the 131 “slots” to get a single [B, token_dim].
        #
        self.token_adapt_histo = nn.Sequential(
            # Project input_dim → inter_dim, per slot
            nn.Conv1d(
                in_channels=self.histo_input_dim,  # e.g. 768
                out_channels=self.inter_dim,       # 512
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # → [B, 512, 131]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=self.inter_dim,       # 512
                out_channels=self.inter_dim,       # 512
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # → [B, 512, 131]
            nn.ReLU(),
            nn.Dropout(0.1),

            # Project inter_dim → token_dim, still keeping length=131
            nn.Conv1d(
                in_channels=self.inter_dim,       # 512
                out_channels=self.token_dim,       # 64
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # → [B, 64, 131]
            nn.AdaptiveAvgPool1d(output_size=1),   # → [B, 64, 1]
            nn.Flatten(start_dim=1),               # → [B, 64]
            nn.LayerNorm(self.token_dim, eps=1e-5),# → [B, 64]
        )

        #
        # 5) TOKEN_ADAPT_RAD_PE  & TOKEN_ADAPT_HISTO_PE:
        #    Each now maps [B, 131, inter_dim] → [B, 131, token_dim], i.e. applies per‐slot Linear.
        #
        self.token_adapt_rad_pe = nn.Sequential(
            nn.Linear(self.inter_dim, self.token_dim),  # [B, 131, 512] → [B, 131, 64]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim),  # [B, 131, 64] → [B, 131, 64]
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.token_adapt_histo_pe = nn.Sequential(
            nn.Linear(self.inter_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        #
        # 6) DPE, FUSION, HAZARD_NET, etc. remain unchanged
        #
        self.dpe = DynamicPositionalEmbedding(
            in_channels=self.n_patch, out_channels=self.token_dim
        )

        self.fusion = fusion_layer(
            d_model=self.token_dim, dim_hider=self.dim_hider, nhead=4, dropout=0.25
        )

        self.hazard_net = nn.Sequential(
            nn.Linear(self.token_dim, self.dim_hider),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim_hider, self.token_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, 1),
        )

        self.norm_pe  = nn.LayerNorm(self.token_dim, eps=1e-5)
        self.norm_att = nn.LayerNorm(self.token_dim, eps=1e-5)
        self.gamma    = nn.Parameter(torch.randn(1), requires_grad=True)
        self.missing_rad_token_fusion = nn.Parameter(
            torch.randn(
                1,
                self.token_dim,
            )
        )
        self.missing_histo_token_fusion = nn.Parameter(
            torch.randn(
                1,
                self.token_dim,
            )
        )
    def forward(
        self, rad_feature, histo_feature, modality_flag=None, output_layers=["hazard"]
    ):
        """
        rad_feature:   [B, 131, rad_input_dim]
        histo_feature: [B, 131, histo_input_dim]
        modality_flag: [B, 2] boolean mask

        We will produce:
          f_adapted_rad   → [B, 131, inter_dim]
          f_adapted_histo → [B, 131, inter_dim]
        """

        outputs = OrderedDict()
        rad_mask   = modality_flag[:, 0].bool().to(rad_feature.device)   # [B]
        histo_mask = modality_flag[:, 1].bool().to(histo_feature.device) # [B]
        batch_size = rad_mask.shape[0]

        # ----------------------------
        # 1) Adapt RAD features (only where rad_mask == True)
        # ----------------------------
        # rad_feature[rad_mask] has shape [b1, 131, rad_input_dim]
        # Permute to [b1, rad_input_dim, 131] for Conv1d
        if rad_mask.any():
            _rad_seq = rad_feature[rad_mask].permute(0, 2, 1)  # [b1, 1024, 131]
            _rad_conv = self.rad_adapter(_rad_seq)             # [b1, 512, 131]
            adapted_rad_seq = _rad_conv.permute(0, 2, 1)        # [b1, 131, 512]
        else:
            adapted_rad_seq = torch.zeros((0, 131, self.inter_dim), device=rad_feature.device)

        # ----------------------------
        # 2) Adapt HISTO features (only where histo_mask == True)
        # ----------------------------
        # histo_feature[histo_mask] has shape [b2, 131, histo_input_dim]
        if histo_mask.any():
            adapted_histo_seq = self.histo_adapter(histo_feature[histo_mask])  # [b2, 131, 512]
        else:
            adapted_histo_seq = torch.zeros((0, 131, self.inter_dim), device=histo_feature.device)

        # Prepare placeholders for the full batch: [B, 131, inter_dim]
        f_adapted_rad = torch.empty(batch_size, 131, self.inter_dim, device=adapted_rad_seq.device)
        f_adapted_histo = torch.empty(batch_size, 131, self.inter_dim, device=adapted_histo_seq.device)

        # Put adapted where modality is present
        if rad_mask.any():
            f_adapted_rad[rad_mask] = adapted_rad_seq
        if histo_mask.any():
            f_adapted_histo[histo_mask] = adapted_histo_seq

        # Fill missing with a learned token repeated 131 times
        if (~rad_mask).any():
            n_miss = (~rad_mask).sum()
            # missing_rad_token is [1,1,512] → repeat to [n_miss,131,512]
            f_adapted_rad[~rad_mask] = self.missing_rad_token.repeat(n_miss, 131, 1)
        if (~histo_mask).any():
            n_miss = (~histo_mask).sum()
            f_adapted_histo[~histo_mask] = self.missing_histo_token.repeat(n_miss, 131, 1)

        # If user just wants adapted features, return them
        if self._add_output_and_check(
            "adapted_features",
            torch.cat(
                [f_adapted_rad.view(batch_size, -1), f_adapted_histo.view(batch_size, -1)],
                dim=1,
            ),  # [B, 131*512 * 2]
            outputs,
            output_layers,
        ):
            return outputs

        if self._add_output_and_check("adapted_histo", f_adapted_histo, outputs, output_layers):
            return outputs
        if self._add_output_and_check("adapted_rad", f_adapted_rad, outputs, output_layers):
            return outputs

        # ----------------------------
        # 3) Compute Positional Embedding via DPE
        #    We first map each of the 131 slots from inter_dim → token_dim
        #    so that we end up with [B, 131, token_dim] for both rad & histo.
        # ----------------------------
        f_rad_pe   = self.token_adapt_rad_pe(f_adapted_rad)    # [B, 131, token_dim]
        f_histo_pe = self.token_adapt_histo_pe(f_adapted_histo)# [B, 131, token_dim]
        pe = self.dpe(
            f_rad_pe,        # [B, 131, token_dim]
            f_histo_pe,      # [B, 131, token_dim]
            rad_mask,
            histo_mask,
        )  # → [B, token_dim]

        if self._add_output_and_check("positional_embeddings", pe, outputs, output_layers):
            return outputs

        # ----------------------------
        # 4) Token‐level adaptation for fusion
        #
        #    RAD tokens: same as before (operates on raw rad_feature)
        #    HISTO tokens: now we take [b2, 131, histo_input_dim] → [b2, token_dim]
        # ----------------------------
        if rad_mask.any():
            _rad_t = rad_feature[rad_mask].permute(0, 2, 1)    # [b1, 1024, 131]
            rad_tokens_pre = self.token_adapt_rad(_rad_t)     # [b1, 64]
        else:
            rad_tokens_pre = torch.zeros((0, self.token_dim), device=rad_feature.device)

        if histo_mask.any():
            # histo_feature[histo_mask]: [b2, 131, histo_input_dim]
            # Permute so Conv1d sees [b2, histo_input_dim, 131]
            _his_t = histo_feature[histo_mask].permute(0, 2, 1)  # [b2, 768, 131]
            histo_tokens_pre = self.token_adapt_histo(_his_t)    # [b2, 64]
        else:
            histo_tokens_pre = torch.zeros((0, self.token_dim), device=histo_feature.device)

        # Place them into full‐batch [B, 64]
        rad_tokens   = torch.empty(batch_size, self.token_dim, device=rad_tokens_pre.device)
        histo_tokens = torch.empty(batch_size, self.token_dim, device=histo_tokens_pre.device)

        if rad_mask.any():
            rad_tokens[rad_mask] = rad_tokens_pre
        if histo_mask.any():
            histo_tokens[histo_mask] = histo_tokens_pre

        # Fill missing slots with learned single‐vector tokens
        if (~rad_mask).any():
            rad_tokens[~rad_mask] = self.missing_rad_token_fusion.repeat((~rad_mask).sum(), 1)
        if (~histo_mask).any():
            histo_tokens[~histo_mask] = self.missing_histo_token_fusion.repeat((~histo_mask).sum(), 1)

        # ----------------------------
        # 5) Attention‐based fusion
        # ----------------------------
        f_att = self.fusion(
            rad_tokens,         # [B, 64]
            histo_tokens,       # [B, 64]
            pe.sigmoid(),       # [B, 64]
        )

        # Skip connection with layer‐norms
        pe_norm    = self.norm_pe(pe)    # [B, 64]
        f_att_norm = self.norm_att(f_att)# [B, 64]
        out        = f_att_norm + pe_norm

        if self._add_output_and_check("fused_features", out, outputs, output_layers):
            return outputs

        # ----------------------------
        # 6) Final hazard network
        # ----------------------------
        out = self.hazard_net(out)  # [B, 1]
        outputs["hazard"] = out
        return outputs

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)


def madpe_nobackbone(
    rad_input_dim=1024,
    histo_input_dim=768,
    inter_dim=512,
    token_dim=64,
):
    model = MADPENetNoBackbonesSurv(
        rad_input_dim,
        histo_input_dim,
        inter_dim,
        token_dim,
    )
    return model
