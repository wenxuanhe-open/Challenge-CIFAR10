import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block using Inception-style convolutions."""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2, bias=False)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_multiscale=False):
        super(ResidualBlock, self).__init__()
        self.use_multiscale = use_multiscale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.use_multiscale:
            self.multiscale = MultiScaleBlock(out_channels, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_multiscale:
            out = self.multiscale(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_multiscale=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_multiscale=use_multiscale)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_multiscale=use_multiscale)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_multiscale=use_multiscale)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_multiscale=use_multiscale)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, use_multiscale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=strides[i], use_multiscale=use_multiscale))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18WithMS(use_multiscale=True):
    return ResNet(ResidualBlock, [2, 2, 2, 2], use_multiscale=use_multiscale)

def ResNet34WithMS(use_multiscale=True):
    return ResNet(ResidualBlock, [3, 4, 6, 3], use_multiscale=use_multiscale)

if __name__ == "__main__":
    model = ResNet18WithMS(use_multiscale=True)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())
