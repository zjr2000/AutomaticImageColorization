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

        self.conv6 = nn.Conv2d(
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

    def forward(self, x):
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
        return F.relu(self.fc_layer3(x)), x


class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()
        self.projection = nn.Linear(512, 256)

    def forward(self, global_feat, mid_level_feat):
        mid_level_feat = rearrange(mid_level_feat, 'B C H W -> B H W C')
        global_feat = global_feat.unsqueeze(1).unsqueeze(1)
        global_feat = global_feat.repeat(1, mid_level_feat.size(1), mid_level_feat.size(2), 1)
        fusion = torch.cat([mid_level_feat, global_feat], dim=-1)
        fusion = torch.sigmoid(self.projection(fusion))
        fusion = rearrange(fusion, 'B H W C -> B C H W')
        return fusion


class UpsamplingNet(nn.Module):
    def __init__(self, out_dims=[128, 64, 64]):
        super(UpsamplingNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=out_dims[0],
            out_channels=out_dims[1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_dims[1],
            out_channels=out_dims[2],
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv1(x))
        return torch.sigmoid(self.conv2(x))


class Classifier(nn.Module):
    def __init__(self, class_num, input_dim=512, hidden_size=256):
        super(Classifier, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, class_num)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        return F.softmax(self.fc(x), dim=-1)
