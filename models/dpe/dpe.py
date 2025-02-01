import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv_no_relu3D(
    in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1
):
    return nn.Sequential(
        nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.BatchNorm3d(out_planes),
    )


# multihead attention network
class MultiheadAtt(nn.Module):
    def __init__(self, d_model=64, dim_hider=256, nhead=8, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.dk = dim_hider // nhead
        self.fcq = nn.Linear(d_model, dim_hider)
        self.fck = nn.Linear(d_model, dim_hider)
        self.fcv = nn.Linear(d_model, dim_hider)
        self.fco = nn.Linear(dim_hider, d_model)

        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_hider)
        self.linear2 = nn.Linear(dim_hider, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # multihead attention
        dim, nhd, bsz, qsz = self.dk, self.nhead, q.size()[0], q.size()[1]
        qc = q
        q = (
            self.fcq(q)
            .view(bsz, qsz, nhd, dim)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, qsz, dim)
        )
        k = (
            self.fck(k)
            .view(bsz, qsz, nhd, dim)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, qsz, dim)
        )
        v = (
            self.fcv(v)
            .view(bsz, qsz, nhd, dim)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, qsz, dim)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.dk))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = self.fco(
            torch.matmul(attn, v)
            .view(nhd, bsz, qsz, dim)
            .permute(1, 2, 0, 3)
            .contiguous()
            .view(bsz, qsz, -1)
        )

        # feedfoward network
        qc = qc + self.dropout1(out)
        qc = self.norm1(qc)
        out = self.linear2(self.dropout2(F.relu(self.linear1(qc))))
        qc = qc + self.dropout3(out)
        qc = self.norm2(qc)

        return qc


# attention based feature fusion network
class fusion_layer(nn.Module):
    def __init__(self, d_model=64, dim_hider=256, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_att1 = MultiheadAtt(d_model, dim_hider, nhead, dropout)
        self.cross_att2 = MultiheadAtt(d_model, dim_hider, nhead, dropout)
        self.cross_att3 = MultiheadAtt(d_model, dim_hider, nhead, dropout)

    def forward(self, f1, f2, pos):
        bsz, chl, z, wh = f1.size()[0], f1.size()[1], f1.size()[2], f1.size()[3]
        f1 = f1.view(bsz, chl, -1).permute(0, 2, 1).contiguous()
        f2 = f2.view(bsz, chl, -1).permute(0, 2, 1).contiguous()
        pos = pos.view(bsz, chl, -1).permute(0, 2, 1).contiguous()
        fv = self.cross_att1(f1, f2, f2 + pos)
        fk = self.cross_att1(f1, f2, f2)
        fq = self.cross_att2(f2, f1, f1)
        f22 = self.cross_att3(fq, fk, fv)

        return f22.permute(0, 2, 1).contiguous().view(bsz, chl, z, wh, wh)


class PositionalEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, out_channels))
        # Learnable token for missing modality
        self.missing_modality_token = nn.Parameter(torch.zeros(1, out_channels))

    def forward(self, pos_input, modality_flags):
        """
        Args:
            pos_input (torch.Tensor): Input feature to generate positional embedding [B, C, H, W].
            modality_flag (torch.Tensor): Flag indicating if a modality is missing (1 if missing, 0 if present)
        Returns:
            torch.Tensor: Adjusted positional embedding [B, C, H, W].
        """
        # Flatten input for MLP
        pos_input_flat = pos_input.view(pos_input.size(0), -1)
        print("flattened_dim:", pos_input_flat.size())
        pos_embedding = self.mlp(pos_input_flat)
        print("pos_embedding_size", pos_embedding.size())
        # Adjust positional embedding with the missing modality token
        adjusted_embedding = (
            pos_embedding + self.missing_modality_token * modality_flags.unsqueeze(1)
        )

        # Reshape back to spatial dimensions
        return adjusted_embedding.view(
            pos_input.size(0),
            -1,
            pos_input.size(2),
            pos_input.size(3),
            pos_input.size(4),
        )


class DPENet(nn.Module):
    def __init__(
        self,
        input_dim=256,
        inter_dim=64,
        topk_pos=3,
        topk_neg=3,
        mixer_channels=2,
        num_classes=3,
        vol_depth=66,
        vol_wh=224,
    ):
        # segm_dim=(64, 64), mixer_channels=2, topk_pos=3, topk_neg=3
        super().__init__()
        # attention based fusion net
        self.fusion = fusion_layer(d_model=64, dim_hider=256, nhead=8, dropout=0.1)
        # 1x1,3x3 conv for correlation net
        self.cor_conv0 = conv3D(input_dim, inter_dim, kernel_size=1, padding=0)
        self.cor_conv1 = conv_no_relu3D(inter_dim, inter_dim)

        self.att_conv0 = conv3D(
            input_dim, inter_dim, kernel_size=1, padding=0
        )  # 1x1,3x3 conv for attention net
        self.att_conv1 = conv_no_relu3D(inter_dim, inter_dim)
        # self.att_conv2 = conv(segm_inter_dim[3], segm_inter_dim[3])
        # self.att_conv3 = conv_no_relu(segm_inter_dim[3], segm_inter_dim[3])

        # self.mixer0 = conv3D(mixer_channels, inter_dim)  # DPE-Net
        # self.mixer1 = conv_no_relu3D(inter_dim, inter_dim)

        # Init weights with He initialization
        for m in self.modules():
            if (
                isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose3d)
                or isinstance(m, nn.Linear)
            ):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

        # Classification head layers
        self.conv3d_1 = conv3D(inter_dim, 128, kernel_size=3, stride=2, padding=1)
        self.conv3d_2 = conv3D(128, 256, kernel_size=3, stride=2, padding=1)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully Connected Layers
        self.fc = nn.Linear(256, num_classes)

        # Learnable missing modality's tokens for correlation map
        self.missing_rad_token = nn.Parameter(
            torch.randn(
                1,
                inter_dim,
                math.ceil(vol_depth / 8),
                math.ceil(vol_wh / 16),
                math.ceil(vol_wh / 16),
            )
        )
        self.missing_histo_token = nn.Parameter(
            torch.randn(
                1,
                inter_dim,
                math.ceil(vol_depth / 8),
                math.ceil(vol_wh / 16),
                math.ceil(vol_wh / 16),
            )
        )

        # Positional Embedding network for modality awareness
        flattened_dim = 2 * math.ceil(vol_depth / 8) * math.ceil(vol_wh / 16) ** 2
        flattened_dim_out = 64 * math.ceil(vol_depth / 8) * math.ceil(vol_wh / 16) ** 2
        # print("flattened_dim:", flattened_dim)
        # print("dimensions:", math.ceil(vol_depth / 8), math.ceil(vol_wh / 16))
        self.positional_embedding = PositionalEmbedding(
            flattened_dim, flattened_dim_out
        )

    #   mask_train, test_dist=None, feat_ups=None, up_masks=None, segm_update_flag=False):
    def forward(self, feat_rad, feat_histo, rad_mask, histo_mask):

        # if modality_flag.data[0].item() < 1.: # radiology missing
        #    modality_flag = torch.tensor(1)
        #    f_rad = None
        # else:
        #    f_rad = self.cor_conv1(self.cor_conv0(feat_rad[3]))
        #
        # if modality_flag.data[1].item() < 1.: # histology missing
        #    print("ci passo")
        #    modality_flag = torch.tensor(1)
        #    f_histo = None
        # else:
        #    f_histo = self.cor_conv1(self.cor_conv0(feat_histo[3]))
        batch_size = rad_mask.shape[0]
        f_rad_present = self.cor_conv1(self.cor_conv0(feat_rad[3]))
        f_histo_present = self.cor_conv1(self.cor_conv0(feat_histo[3]))

        f_rad = torch.empty(
            batch_size,
            f_rad_present.shape[1],
            f_rad_present.shape[2],
            f_rad_present.shape[3],
            f_rad_present.shape[4],
        ).to(self.missing_rad_token.device)
        f_histo = torch.empty(
            batch_size,
            f_histo_present.shape[1],
            f_histo_present.shape[2],
            f_histo_present.shape[3],
            f_histo_present.shape[4],
        ).to(self.missing_histo_token.device)

        f_rad[rad_mask] = f_rad_present
        f_histo[histo_mask] = f_histo_present
        # substitute missing tokens with optimizable parameters
        f_rad[~rad_mask] = self.missing_rad_token.repeat((~rad_mask).sum(), 1, 1, 1, 1)
        f_histo[~histo_mask] = self.missing_histo_token.repeat(
            (~histo_mask).sum(), 1, 1, 1, 1
        )

        # compute similarity maps
        pred_pos, pred_neg = self.correlation(f_rad, f_histo)

        # concatenate maps
        class_layers = torch.cat(
            (torch.unsqueeze(pred_pos, dim=1), torch.unsqueeze(pred_neg, dim=1)), dim=1
        )

        modality_flags = ~torch.min(
            rad_mask, histo_mask
        )  # 1 when sample contains missing modality

        pe = self.positional_embedding(class_layers, modality_flags)

        # print("out:", out.size())
        # pe = self.mixer1(self.mixer0(class_layers))

        # Calculate tokens for feature fusion
        rad_token_pres = self.att_conv1(self.att_conv0(feat_rad[3]))
        histo_token_pres = self.att_conv1(self.att_conv0(feat_histo[3]))

        rad_tokens = torch.empty(
            batch_size,
            f_rad_present.shape[1],
            f_rad_present.shape[2],
            f_rad_present.shape[3],
            f_rad_present.shape[4],
        ).to(self.missing_rad_token.device)
        histo_tokens = torch.empty(
            batch_size,
            f_histo_present.shape[1],
            f_histo_present.shape[2],
            f_histo_present.shape[3],
            f_histo_present.shape[4],
        ).to(self.missing_histo_token.device)

        rad_tokens[rad_mask] = rad_token_pres
        histo_tokens[histo_mask] = histo_token_pres
        # substitute missing modalities with optimizable missing token parameters
        # (for now we use the same ones used for pe)
        rad_tokens[~rad_mask] = self.missing_rad_token.repeat(
            (~rad_mask).sum(), 1, 1, 1, 1
        )
        histo_tokens[~histo_mask] = self.missing_histo_token.repeat(
            (~histo_mask).sum(), 1, 1, 1, 1
        )

        # Attention-based fusion for classification
        f_att = self.fusion(
            rad_tokens,
            histo_tokens,
            pe.sigmoid(),
        )
        #
        # Classification net
        out = self.conv3d_1(f_att)  # [bsize, 128, depth/2, h/2, w/2]
        out = self.conv3d_2(out)  # [bsize, 256, depth/4, h/4, w/4]
        out = self.global_avg_pool(out)  # [bsize, 256, 1, 1, 1]
        out = torch.flatten(out, 1)  # [bsize, 256]
        out = self.fc(out)  # [bsize, num_classes]
        return out

    # correlation operation
    def correlation(self, f_rad, f_histo, modality_flag=[1, 1]):

        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one

        # klv - mnw is the pair of coordinates of the volume sections
        # on which the similarity is calculated
        sim = torch.einsum(
            "ijklv,ijmnw->iklvmnw",
            F.normalize(f_rad, p=2, dim=1),
            F.normalize(f_histo, p=2, dim=1),
        )

        sim_resh = sim.view(
            sim.shape[0],
            sim.shape[1],
            sim.shape[2],
            sim.shape[3],
            sim.shape[4] * sim.shape[5] * sim.shape[6],
        )

        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
        # re-weight samples (take out positive ang negative samples)
        # sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)
        # sim_neg = sim_resh * mask_neg.view(mask_neg.shape[0], 1, 1, -1)

        # take top k positive and negative examples
        # mean over the top positive and negative examples
        pos_map = torch.mean(torch.topk(sim_resh, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(
            torch.topk(-sim_resh, self.topk_neg, dim=-1).values, dim=-1
        )

        return pos_map, neg_map
