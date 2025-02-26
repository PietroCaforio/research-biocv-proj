import sys

import torch
import torch.nn as nn

sys.path.insert(0, "./")  # noqa: E402
import models.backbones.resnet3D as resnet3D  # noqa: E402
from models.dpe.dpe import DPENet  # noqa: E402

# from data.unimodal3D import UnimodalCTDataset3D  # noqa: E402
# from data.unimodal_wsi3D import UnimodalWSIDataset3D  # noqa: E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class MADPENet(nn.Module):  # ModalityAwareDPENet da decidere nome
    """Classification network module"""

    def __init__(
        self,
        rad_backbone,
        histo_backbone,
        class_predictor,
        backbone_layers,
        backbone_grad=False,
        backbone_unfreeze_layers=None,
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

        self.rad_backbone = rad_backbone
        self.histo_backbone = histo_backbone
        self.class_predictor = class_predictor
        self.backbone_layers = backbone_layers

        if not backbone_grad:  # Freeze the backbones
            for p in self.rad_backbone.parameters():
                p.requires_grad_(False)
            for p in self.histo_backbone.parameters():
                p.requires_grad_(False)
        if backbone_unfreeze_layers is not None:
            for name, p in self.rad_backbone.named_parameters():
                if name.split(".")[0] in backbone_unfreeze_layers:
                    p.requires_grad_(True)
                    print(name, "requires grad in rad backbone")
            for name, p in self.histo_backbone.named_parameters():
                if name.split(".")[0] in backbone_unfreeze_layers:
                    p.requires_grad_(True)
                    print(name, "requires grad in histo backbone")

    def forward(self, rad_vols, histo_vols, modality_flag=None):
        """Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5,
        then the batch dimension corresponds to the first dimensions.
        test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        rad_mask = modality_flag[:, 0].bool().to(rad_vols.device)
        histo_mask = modality_flag[:, 1].bool().to(histo_vols.device)

        # extract backbone features only for available modalities
        rad_feat = self.extract_rad_backbone_features(rad_vols[rad_mask])
        rad_feat = [feat for feat in rad_feat.values()]
        histo_feat = self.extract_histo_backbone_features(histo_vols[histo_mask])
        histo_feat = [feat for feat in histo_feat.values()]

        class_prediction = self.class_predictor(
            rad_feat, histo_feat, rad_mask, histo_mask
        )

        return class_prediction

    def extract_rad_backbone_features(self, vol, layers=None):
        if layers is None:
            layers = self.backbone_layers
        return self.rad_backbone(vol, layers)

    def extract_histo_backbone_features(self, vol, layers=None):
        if layers is None:
            layers = self.backbone_layers
        return self.histo_backbone(vol, layers)

    # def extract_features(self, vol, layers):
    #    return self.backbone(vol, layers)


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

# @model_constructor
def madpe_resnet34(
    backbone_pretrained=True,
    vol_depth=66,
    vol_wh=224,
    pretrained_rad_path="./models/pretrain_weights/r3d34_K_200ep.pth",
    pretrained_histo_path="./models/pretrain_weights/r3d34_K_200ep.pth",
    backbone_grad=False,
    backbone_unfreeze_layers=None,
):
    # radiology backbone
    rad_backbone_net = resnet3D.generate_model(34, n_input_channels=3, n_classes=700)
    histo_backbone_net = resnet3D.generate_model(34, n_input_channels=3, n_classes=700)
    if backbone_pretrained:
        # !conda install -y gdown
        # !gdown --id 1fFN5J2He6eTqMPRl_M9gFtFfpUmhtQc9
        pretrain_histo = torch.load(
            pretrained_histo_path,
            map_location="cuda:0",
            weights_only=True,
        )
        pretrain_rad = torch.load(
            pretrained_rad_path,
            map_location="cuda:0",
            weights_only=True,
        )
        rad_backbone_net.load_state_dict(pretrain_rad["state_dict"])
        histo_backbone_net.load_state_dict(pretrain_histo["state_dict"])
        # block_inplanes = get_inplanes()
        # rad_backbone_net.conv1 = nn.Conv3d(1,
        #                        block_inplanes[0],
        #                        kernel_size=(7, 7, 7),
        #                        stride=(1, 2, 2),
        #                        padding=(7 // 2, 3, 3),
        #                        bias=False)
        # rad_backbone_net.conv1.weight = torch.nn.Parameter(pretrain['state_dict']
        # ["conv1.weight"].mean(dim=1, keepdim=True))
        # rad_backbone_net.fc = nn.Linear(rad_backbone_net.fc.in_features,3)

    # classification
    class_predictor = DPENet(vol_depth=vol_depth, vol_wh=vol_wh)

    net = MADPENet(
        rad_backbone=rad_backbone_net,
        histo_backbone=histo_backbone_net,
        class_predictor=class_predictor,
        backbone_layers=["conv1", "layer1", "layer2", "layer3"],
        backbone_grad=backbone_grad,
        backbone_unfreeze_layers=backbone_unfreeze_layers,
    )
    return net


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
    madpe = madpe_resnet34()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    madpe.to(device)
    train_multimode_dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
        missing_modality_prob=1.0,
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

    train_ct_vol = (
        train_sample["ct_volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
    )
    train_wsi_vol = train_sample["wsi_volume"].float().to(device)
    train_label = train_sample["label"].to(device)
    modality_mask = train_sample["modality_mask"].to(device)
    madpe.eval()
    with torch.no_grad():
        madpe(train_ct_vol, train_wsi_vol, modality_flag=modality_mask)
    # madpe = madpe_resnet34()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # madpe.to(device)
#
# train_dataset = UnimodalCTDataset3D(
#     split="train", dataset_path="./data/processed/processed_CPTAC_PDA_71_3D"
# )
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#
# # sample a single volume from the dataset
# data_iter = iter(train_loader)
# sample = next(data_iter)
# # print(sample["volume"])
# volume = sample["volume"].float().to(device)
#
# wsi_train_dataset = UnimodalWSIDataset3D(
#     split="train",
#     dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
#     patches_per_wsi=66,
#     sampling_strategy="consecutive",
# )
# wsi_train_loader = DataLoader(wsi_train_dataset, batch_size=1, shuffle=True)
#
# # sample a single volume from the dataset
# wsi_data_iter = iter(wsi_train_loader)
# wsi_sample = next(wsi_data_iter)
# # print(sample["volume"])
# wsi_volume = wsi_sample["volume"].float().to(device)
#
# madpe.eval()
# with torch.no_grad():
#     # print(wsi_volume.unsqueeze(1).repeat(1, 3, 1, 1, 1).size())
#     madpe(volume.unsqueeze(1).repeat(1, 3, 1, 1, 1), wsi_volume)
#
