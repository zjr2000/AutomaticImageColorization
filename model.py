import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LowLevelFeature(nn.Module):
    def __init__(self):
        super(LowLevelFeature, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return F.relu(self.conv6(x))


class MidLevelFeature(nn.Module):
    def __init__(self):
        super(MidLevelFeature, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        ) 

    def forawrd(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class GlobalFeature(nn.Module):
    def __init__(self):
        super(GlobalFeature, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.fc_layer1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc_layer2 = nn.Linear(1024, 512)
        self.fc_layer3 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = rearrange(x, 'B C H W -> B (C H W)')
        x = F.relu(self.fc_layer1(x))
        x = F.relu(self.fc_layer2(x))
        return self.fc_layer3(x)