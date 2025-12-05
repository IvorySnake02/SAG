# ---- Imports needed for resnet18 ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

# ---- Pretrained model URLs ----
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=dilation, groups=groups,
        bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=1,
        stride=stride, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2   = norm_layer(planes)

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# ---- Bottleneck (needed for architecture completeness, not used in resnet18) ----
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Bottleneck is not used for resnet18.")

class ResNet2D1(nn.Module):
    def __init__(self, block, layers, num_classes=20,
                 zero_init_residual=False, groups=1,
                 input_size=3, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv1d(input_size, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=8)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(512 * block.expansion, 256)

        self.hash_fc = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64, momentum=0.1)
        )
        self.fcn = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample,
                  self.groups, self.base_width, self.dilation, norm_layer)
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))
        x = F.relu(self.hash_fc(x))
        logits = self.fcn(x)

        return F.softmax(logits, dim=1), logits

# ---- Internal builder ----
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet2D1(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],
                   pretrained, progress, **kwargs)
