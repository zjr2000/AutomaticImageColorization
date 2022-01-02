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


class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()
        self.projection = nn.Linear(512, 256)

    def forward(self, global_feat, mid_level_feat):
        global_feat = global_feat.unsqueeze(1).unsqueeze(1)
        global_feat = global_feat.repeat(1, mid_level_feat.size(1), mid_level_feat.size(2), 1)
        fusion = torch.cat([mid_level_feat, global_feat], dim=-1)
        return torch.sigmoid(self.projection(fusion))


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
        return torch.sigmoid(self.con2(x))


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.low_level_net = LowLevelFeature()
        self.mid_level_net = MidLevelFeature()
        self.global_feat_net = GlobalFeature()
        self.fusion_layer = FusionLayer()
        self.conv_fusion = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.upsampling1 = UpsamplingNet(out_dims=[128, 64, 64])
        self.upsampling2 = UpsamplingNet(out_dims=[64, 32, 2])

    def forward(self, x):
        x = self.low_level_net(x)
        mid_level_feature = self.mid_level_net(x)
        global_feature = self.global_feat_net(x)
        
        fused_feature = self.fusion_layer(
            global_feat=F.relu(global_feature),
            mid_level_feat=mid_level_feature
        )
        fused_feature = self.conv_fusion(fused_feature)
        out = self.upsampling1(fused_feature)
        out = self.upsampling2(out)