import sys

import torch
import torch.nn as nn

sys.path.insert(0, "./")  # noqa: E402
import models.backbones.resnet3D as resnet3D  # noqa: E402
from models.dpe.dpe import DPENet  # noqa: E402

# from data.unimodal3D import UnimodalCTDataset3D  # noqa: E402
# from data.unimodal_wsi3D import UnimodalWSIDataset3D  # noqa: E402
from data.multimodal_features import MultimodalCTWSIDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from collections import OrderedDict
import torch.nn.functional as F


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
        Args:
            pos_input (torch.Tensor): Input feature to generate positional embedding [B, C, H, W].
            modality_flag (torch.Tensor): Flag indicating if a modality is missing (1 if missing, 0 if present)
        Returns:
            torch.Tensor: Adjusted positional embedding [B, C, H, W].
        """

        pred_pos, pred_neg = self._correlation(f_rad, f_histo)
        # concatenate maps
        class_layers = torch.cat([pred_pos, pred_neg], dim=1)

        modality_flags = ~torch.min(
            rad_mask, histo_mask
        )  # 1 when sample contains missing modality

        # Flatten input for MLP
        pos_input_flat = class_layers.view(class_layers.size(0), -1)
        pos_embedding = self.mlp(pos_input_flat)
        # Adjust positional embedding with the missing modality token
        adjusted_embedding = (
            pos_embedding + self.missing_modality_token * modality_flags.unsqueeze(1)
        )

        # Reshape back to spatial dimensions
        return adjusted_embedding

    # correlation operation
    def _correlation(self, f_rad, f_histo):

        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one

        # klv - mnw is the pair of coordinates of the volume sections
        # on which the similarity is calculated

        sim = torch.einsum(
            "bi,bj->bij",
            F.normalize(f_rad, p=2, dim=1),
            F.normalize(f_histo, p=2, dim=1),
        )

        # TODO: Patho fusion

        sim = sim.unsqueeze(1)
        # sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2]*sim.shape[3])
        # print(sim_resh.shape)

        pos_map = torch.mean(torch.topk(sim, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(torch.topk(-sim, self.topk_neg, dim=-1).values, dim=-1)
        return pos_map, neg_map


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


class HistoAdapter(nn.Module):
    def __init__(self, input_dim, inter_dim, token_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, inter_dim)
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
        self.block3 = nn.Sequential(
            nn.LayerNorm(inter_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, token_dim),
        )
        self.block4 = nn.Sequential(
            nn.LayerNorm(token_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim, token_dim),
        )
        self.final_norm = nn.LayerNorm(token_dim, eps=1e-5)

    def forward(self, x):
        x = self.fc_in(x)
        x = x + self.block1(x)  # Residual connection
        x = x + self.block2(x)

        x = self.block3(x)  # Projection to token_dim
        x = x + self.block4(x)  # Residual connection
        x = self.final_norm(x)
        return x


class MADPENetNoBackbonesSurv(nn.Module):  # ModalityAwareDPENet da decidere nome
    """Classification network module"""

    def __init__(
        self,
        rad_input_dim=1024,
        histo_input_dim=512,
        inter_dim=512,
        token_dim=256,
        dim_hider=256,  # For the attention fusion
        num_classes=3,
    ):
        """
        args:
            backbone - backbone feature extractor
            class_predictor - classification module
            backbone_layers - List containing the name of the layers from
                feature_extractor, which are used in segm_predictor
            extractor_grad - Bool indicating whether backbone
                feature extractor requires gradients
        """
        super().__init__()
        self.rad_input_dim = rad_input_dim
        self.histo_input_dim = histo_input_dim
        self.inter_dim = inter_dim
        self.token_dim = token_dim
        self.num_classes = num_classes
        self.dim_hider = dim_hider

        self.rad_adapter = nn.Sequential(
            nn.Conv1d(
                in_channels=self.rad_input_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.Conv1d(
                in_channels=self.inter_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.AdaptiveAvgPool1d(output_size=1),  #
            nn.Flatten(start_dim=1),
        )

        self.histo_adapter = HistoAdapter(
            self.histo_input_dim, self.inter_dim, self.inter_dim
        )

        # self.histo_adapter = nn.Sequential(
        #     nn.Linear(self.histo_input_dim, self.inter_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.inter_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.inter_dim),
        #
        #     nn.GELU(), # Layer in più nell'adaptation dell'istologia...
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.inter_dim),
        #     nn.GELU(), # Layer in più nell'adaptation dell'istologia...
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.inter_dim),
        #     nn.LayerNorm(self.inter_dim),
        # )

        self.missing_rad_token = nn.Parameter(
            torch.randn(
                1,
                self.inter_dim,
            )
        )
        self.missing_histo_token = nn.Parameter(
            torch.randn(
                1,
                self.inter_dim,
            )
        )

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

        self.token_adapt_rad = nn.Sequential(
            nn.Conv1d(
                in_channels=self.rad_input_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.Conv1d(
                in_channels=self.inter_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.inter_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.token_adapt_histo = HistoAdapter(
            self.histo_input_dim, self.inter_dim, self.token_dim
        )

        # self.token_adapt_histo = nn.Sequential(
        #     nn.Linear(self.histo_input_dim, self.inter_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.inter_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.inter_dim), # Aggiunto layer nell'adaptation della istologia
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.inter_dim, self.token_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.token_dim, self.token_dim),
        #
        #     nn.LayerNorm(self.token_dim),
        # )

        self.token_adapt_rad_pe = nn.Sequential(
            nn.Linear(self.inter_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.token_adapt_histo_pe = nn.Sequential(
            nn.Linear(self.inter_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.dpe = DynamicPositionalEmbedding(
            in_channels=self.token_dim, out_channels=self.token_dim
        )

        self.fusion = fusion_layer(
            d_model=self.token_dim, dim_hider=self.dim_hider, nhead=4, dropout=0.25
        )

        self.hazard_net = nn.Sequential(
            nn.Linear(self.token_dim, self.dim_hider),  # First hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout for regularization
            nn.Linear(self.dim_hider, self.token_dim),  # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, 1),  # Output layer
        )

        self.norm_pe = nn.LayerNorm(self.token_dim, eps=1e-5)
        self.norm_att = nn.LayerNorm(self.token_dim, eps=1e-5)
        # self.act = nn.Sigmoid() #
        # self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False) #
        # self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False) #
        self.gamma = nn.Parameter(torch.randn(1), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d((2,2))

    def forward(
        self, rad_feature, histo_feature, modality_flag=None, output_layers=["hazard"]
    ):
        """Forward pass:
        rad_feature (batch, slices, feature) [B,66,1024]
        histo_feature (batch, feature) [B,768]
        """
        B, S, C, H, W = rad_feature.shape
        
        outputs = OrderedDict()
        # if self._add_output_and_check("features",torch.cat([rad_feature,histo_feature], dim = 1), outputs, output_layers):
        #    return outputs

        rad_mask = modality_flag[:, 0].bool().to(rad_feature.device)
        histo_mask = modality_flag[:, 1].bool().to(histo_feature.device)
        batch_size = rad_mask.shape[0]
        rad_feature = rad_feature.view(-1,C,H,W) #[B *66,256,32,32]
        rad_feature = self.pool(rad_feature) # [B*66, 256, 2, 2]
        rad_feature = rad_feature.view(-1,1024) # [B*66, 1024]
        rad_feature = rad_feature.view(B,S,1024) # [B, 66, 1024]
        # Adapt rad_features to (,512) (adapt only the available modality)
        adapted_rad = self.rad_adapter(
            rad_feature[rad_mask].permute(0, 2, 1)
        )  # (.,512)

        # Adapt histo_features to (,512) (only the available modality)
        adapted_histo = self.histo_adapter(histo_feature[histo_mask])  # (.,512)
        # Return the concatenated adapted features if chosen

        f_adapted_rad = torch.empty(batch_size, adapted_rad.shape[1]).to(
            self.missing_rad_token.device
        )

        f_adapted_histo = torch.empty(
            batch_size,
            adapted_histo.shape[1],
        ).to(self.missing_rad_token.device)

        f_adapted_rad[rad_mask] = adapted_rad
        f_adapted_histo[histo_mask] = adapted_histo

        f_adapted_rad[~rad_mask] = self.missing_rad_token.repeat((~rad_mask).sum(), 1)
        f_adapted_histo[~histo_mask] = self.missing_histo_token.repeat(
            (~histo_mask).sum(), 1
        )

        modality_flags = ~torch.min(rad_mask, histo_mask)

        # Return the concatenated adapted features if chosen
        if self._add_output_and_check(
            "adapted_features",
            torch.cat([f_adapted_rad, f_adapted_histo], dim=1),
            outputs,
            output_layers,
        ):
            return outputs
        if self._add_output_and_check(
            "adapted_histo", f_adapted_histo, outputs, output_layers
        ):
            return outputs
        if self._add_output_and_check(
            "adapted_rad", f_adapted_rad, outputs, output_layers
        ):
            return outputs

        # Calculate Positional Embedding
        pe = self.dpe(
            self.token_adapt_rad_pe(f_adapted_rad),  # (B,64)
            self.token_adapt_histo_pe(f_adapted_histo),  # (B,64)
            rad_mask,
            histo_mask,
        )  # (B, 64)
        # Return the positional embeddings if chosen
        if self._add_output_and_check(
            "positional_embeddings", pe, outputs, output_layers
        ):
            return outputs

        # Adapt for tokenization and inject missing tokens for missing modalities
        rad_tokens_pre = self.token_adapt_rad(
            rad_feature[rad_mask].permute(0, 2, 1)
        )  # (, 64)
        histo_tokens_pre = self.token_adapt_histo(histo_feature[histo_mask])  # (,64)
        rad_tokens = torch.empty(
            batch_size,
            rad_tokens_pre.shape[1],
        ).to(self.missing_rad_token.device)

        histo_tokens = torch.empty(
            batch_size,
            histo_tokens_pre.shape[1],
        ).to(self.missing_histo_token.device)

        rad_tokens[rad_mask] = rad_tokens_pre
        histo_tokens[histo_mask] = histo_tokens_pre

        rad_tokens[~rad_mask] = self.missing_rad_token_fusion.repeat(
            (~rad_mask).sum(), 1
        )
        histo_tokens[~histo_mask] = self.missing_histo_token_fusion.repeat(
            (~histo_mask).sum(), 1
        )

        # Attention-based fusion
        f_att = self.fusion(
            rad_tokens,
            histo_tokens,
            pe.sigmoid(),
        )

        # Skip connection
        pe_norm = self.norm_pe(pe)
        f_att_norm = self.norm_att(f_att)
        out = f_att_norm + pe_norm

        # Return the fused features if chosen
        if self._add_output_and_check("fused_features", out, outputs, output_layers):
            return outputs

        out = self.hazard_net(out)

        # hazard = self.act(out) #
        # outputs["hazard"] = self.output_range * hazard + self.output_shift #
        # outputs["hazard"] = 3.0 * torch.tanh(out) #
        outputs["hazard"] = out
        return outputs

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)


# @model_constructor
# def dpet_resnet18(segm_input_dim=(256,256), segm_inter_dim=(256,256),
# backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
#    # backbone
#    backbone_net = backbones.resnet3D(pretrained=backbone_pretrained)
#
#    # segmentation dimensions
#    segm_input_dim = (64, 64, 128, 256)
#    segm_inter_dim = (4, 16, 32, 64)
#    segm_dim = (64, 64)  # convolutions before cosine similarity
#
#    # segmentation
#    segm_predictor = segmmodels.DPETNet(segm_input_dim=segm_input_dim,
#                                            segm_inter_dim=segm_inter_dim,
#                                            segm_dim=segm_dim,
#                                            topk_pos=topk_pos,
#                                            topk_neg=topk_neg,
#                                            mixer_channels=mixer_channels)
#
#    net = DPETNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
#                      segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)
#
#    return net
#
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


# @model_constructor
# def dpet_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
# backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
#    # backbone
#    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
#
#    # segmentation dimensions
#    segm_input_dim = (64, 256, 512, 1024)
#    segm_inter_dim = (4, 16, 32, 64, 128)
#    segm_dim = (64, 64)  # convolutions before cosine similarity
#
#    # segmentation
#    segm_predictor = segmmodels.DPETNet(segm_input_dim=segm_input_dim,
#                                            segm_inter_dim=segm_inter_dim,
#                                            segm_dim=segm_dim,
#                                            topk_pos=topk_pos,
#                                            topk_neg=topk_neg,
#                                            mixer_channels=mixer_channels)
#
#    net = DPETNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
#                      segm_layers=['conv1', 'layer1', 'layer2', 'layer3'],
# extractor_grad=False)  # extractor_grad=False
#
#    return net


if __name__ == "__main__":
    madpe = madpe_nobackbone()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    madpe.to(device)
    train_multimode_dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path="./data/features/TITAN_MedImageInsights",
        ct_path="../MedImageInsights/embeddings_output_cptacpda",
        wsi_path="../trident/trident_processed/20x_224px_0px_overlap/slide_features_titan",
        missing_modality_prob=0.0,  # chance of each modality being missing
        require_both_modalities=True,
        pairing_mode="all_combinations",
        allow_repeats=True,
        pairs_per_patient=None,
        downsample=False,
    )

    # Create dataloaders
    # We use the same sample for train and validation
    train_multimode_loader = DataLoader(
        train_multimode_dataset, batch_size=2, shuffle=False
    )
    val_multimode_loader = DataLoader(
        train_multimode_dataset, batch_size=2, shuffle=False
    )
    train_sample = next(iter(train_multimode_loader))

    train_ct_vol = train_sample["ct_feature"].float().to(device)
    train_wsi_vol = train_sample["wsi_feature"].float().to(device)
    train_label = train_sample["label"].to(device)
    modality_mask = train_sample["modality_mask"].to(device)
    madpe.eval()
    with torch.no_grad():
        out = madpe(train_ct_vol, train_wsi_vol, modality_flag=modality_mask)
        print(out["classification"])
    # madpe = madpe_resnet34()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # madpe.to(device)
