from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ResNet(nn.Module):
    def __init__(self, version, num_classes, pretrained=False, layers_to_unfreeze=0, expand=False):
        super().__init__()
        
        self.pretrained = pretrained
        self.name = f"resnet{version}{'_ex' if expand else ''}"

        if version == 18:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif version == 50:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Invalid version. Use 18 for resnet18 or 50 for resnet50.")

        # Works better for CIFAR-100 (less aggressive downsampling, since images are smaller)
        self.model.conv1.stride = (1, 1)
        self.model.maxpool = nn.Identity()
        
        if expand:
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.model.fc.in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if pretrained and layers_to_unfreeze > 0:
            self._unfreeze_layers(layers_to_unfreeze)

    def _unfreeze_layers(self, layers_to_unfreeze):
        # First freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        if layers_to_unfreeze < 1 or layers_to_unfreeze > 5:
            raise ValueError("layers_to_unfreeze must be between 1 and 5")

        # Unfreeze selected layers from the end
        layers = []
        if layers_to_unfreeze >= 1:
            layers.append("fc")
        if layers_to_unfreeze >= 2:
            layers.append("layer4")
        if layers_to_unfreeze >= 3:
            layers.append("layer3")
        if layers_to_unfreeze >= 4:
            layers.append("layer2")
        if layers_to_unfreeze == 5:
            layers.append("layer1")

        for layer_name in layers:
            for param in getattr(self.model, layer_name).parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super().__init__()
        
        self.name = f"wide_resnet{depth}_{widen_factor}"
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        
        self.apply(self._init_weights)

    def _wide_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _init_weights(self, m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        if type(m) == nn.BatchNorm2d:
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)