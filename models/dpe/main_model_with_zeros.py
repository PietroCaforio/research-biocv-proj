import sys

import torch

from models.dpe.main_model import MADPENet

sys.path.insert(0, "./")  # noqa: E402

import models.backbones.resnet3D as resnet3D  # noqa: E402

# from models.dpe.dpe import DPENet  # noqa: E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class MADPE_zeros(MADPENet):
    def forward(
        self, rad_vols, histo_vols, test_dist=None, modality_flag=torch.tensor([1, 1])
    ):
        """Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5,
        then the batch dimension corresponds to the first dimensions.
        test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        rad_mask = modality_flag[:, 0].bool().to(rad_vols.device)
        histo_mask = modality_flag[:, 1].bool().to(histo_vols.device)

        # Apply zero mask to missing modalities
        rad_vols[~rad_mask] = 0
        histo_vols[~histo_mask] = 0

        print(rad_vols[~rad_mask].size())
        # extract backbone features only for available modalities
        rad_feat = self.extract_rad_backbone_features(rad_vols)
        rad_feat = [feat for feat in rad_feat.values()]
        histo_feat = self.extract_histo_backbone_features(histo_vols)
        histo_feat = [feat for feat in histo_feat.values()]

        # class_prediction = self.class_predictor(
        #    rad_feat, histo_feat, rad_mask, histo_mask
        # )

        # return class_prediction


def madpe_resnet34_zeros(
    backbone_pretrained=True,
    check_point_path="./models/pretrain_weights/r3d34_K_200ep.pth",
):
    # radiology backbone
    rad_backbone_net = resnet3D.generate_model(34, n_input_channels=3, n_classes=700)
    histo_backbone_net = resnet3D.generate_model(34, n_input_channels=3, n_classes=700)
    if backbone_pretrained:
        # !conda install -y gdown
        # !gdown --id 1fFN5J2He6eTqMPRl_M9gFtFfpUmhtQc9
        pretrain = torch.load(
            check_point_path,
            map_location="cpu",
            weights_only=True,
        )

        rad_backbone_net.load_state_dict(pretrain["state_dict"])
        histo_backbone_net.load_state_dict(pretrain["state_dict"])
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
    # class_predictor = DPENet(vol_depth=66, vol_wh=224)
    class_predictor = None

    net = MADPE_zeros(
        rad_backbone=rad_backbone_net,
        histo_backbone=histo_backbone_net,
        class_predictor=class_predictor,
        backbone_layers=["conv1", "layer1", "layer2", "layer3"],
        backbone_grad=False,
    )
    return net


if __name__ == "__main__":
    madpe = madpe_resnet34_zeros()
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
    print(train_sample["modality_mask"])
    modality_mask = train_sample["modality_mask"].to(device)
    madpe.eval()
    with torch.no_grad():
        madpe(train_ct_vol, train_wsi_vol, modality_flag=modality_mask)
