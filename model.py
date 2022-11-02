import torch
from torch import nn
from torch.nn import functional as F


class Reshape(nn.Module):
    """docstring for Reshape"""

    def forward(self, x):
        return x.view(-1, 3, 32, 32)


class LeNet5(nn.Module):
    """docstring for LeNet5"""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=16*6*6, out_features=120),
            nn.ReLU(),

            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),

            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(
            ResidualBlock, 64,  num_blocks[0], stride=1)
        self.layer2 = self.make_layer(
            ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(
            ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(
            ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock, [2, 2, 2, 2])


if __name__ == '__main__':
    model = LeNet5()
    X = torch.rand(size=(256, 3, 32, 32), dtype=torch.float32)
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__, '\toutput: \t', X.shape)
