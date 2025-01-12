import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.unimodal3D import UnimodalCTDataset3D
from data.unimodal_wsi3D import UnimodalWSIDataset3D

sys.path.insert(0, "../../")  # noqa: E402
sys.path.insert(0, "../")  # noqa: E402
sys.path.insert(0, "./")  # noqa: E402


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
        output_layers=["conv1", "layer1", "layer2", "layer3", "layer4", "fc"],
    ):
        super().__init__()

        self.output_layers = output_layers

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check("conv1", x, outputs, output_layers):
            return outputs

        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)

        if self._add_output_and_check("layer1", x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check("layer2", x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check("layer3", x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check("layer4", x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check("fc", x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == "default":
            return x

        raise ValueError(
            "output_layer is wrong. Choose the layers from the "
            + "following list \n [conv1,layer1,layer2,layer3,layer4,fc] "
        )


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


# -----------------------------------------------------------------------------
def main():
    model = generate_model(34, n_input_channels=3, n_classes=700)
    # !conda install -y gdown
    # !gdown --id 1fFN5J2He6eTqMPRl_M9gFtFfpUmhtQc9
    pretrain = torch.load(
        "./models/pretrain_weights/r3d34_K_200ep.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(pretrain["state_dict"])
    # block_inplanes = get_inplanes()
    # model.conv1 = nn.Conv3d(1,
    #                        block_inplanes[0],
    #                        kernel_size=(7, 7, 7),
    #                        stride=(1, 2, 2),
    #                        padding=(7 // 2, 3, 3),
    #                        bias=False)
    # model.conv1.weight = torch.nn.Parameter(pretrain
    # ['state_dict']["conv1.weight"].mean(dim=1, keepdim=True))
    # model.fc = nn.Linear(model.fc.in_features,3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Let's try to generate some embeddings from this pretrained network
    train_dataset = UnimodalCTDataset3D(
        split="train", dataset_path="./data/processed/processed_CPTAC_PDA_71_3D"
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # sample a single volume from the dataset
    data_iter = iter(train_loader)
    sample = next(data_iter)
    # print(sample["volume"])
    volume = sample["volume"].float().to(device)

    wsi_train_dataset = UnimodalWSIDataset3D(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
    )
    wsi_train_loader = DataLoader(wsi_train_dataset, batch_size=1, shuffle=True)

    # sample a single volume from the dataset
    wsi_data_iter = iter(wsi_train_loader)
    wsi_sample = next(wsi_data_iter)
    # print(sample["volume"])
    wsi_volume = wsi_sample["volume"].float().to(device)

    model.eval()
    with torch.no_grad():
        # print(wsi_volume.unsqueeze(1).repeat(1, 3, 1, 1, 1).size())
        wsi_output = model(
            wsi_volume, ["conv1", "layer1", "layer2", "layer3", "layer4"]
        )
        ct_output = model(
            volume.unsqueeze(1).repeat(1, 3, 1, 1, 1),
            ["conv1", "layer1", "layer2", "layer3", "layer4"],
        )
    print(wsi_output["layer4"].size())
    print(ct_output["layer4"].size())


if __name__ == "__main__":
    main()
