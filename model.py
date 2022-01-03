from modules import *
from data_loader import *
import torch.optim as optim

class ColorizationNet(nn.Module):
    def __init__(self, cfgs, isTrain):
        super(ColorizationNet, self).__init__()
        # self.cfgs = cfgs
        # self.isTrain = isTrain

        self.build_network(cfgs['class_num'])
        self.init_parameters()

        if isTrain:
            self.train_loader = get_data_loader('./data', cfgs['batch_size'], True)
        if not isTrain or cfgs['with_test']:
            self.test_loader = get_data_loader('./data', cfgs['batch_size'], False)

        if isTrain:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **cfgs["optimizer_cfg"])
            self.scheduler = optim.lr_scheduler.MultiStepLR(**cfgs["scheduler_cfg"])
        self.train(isTrain)

    def init_parameters(self):
        for m in self.modules():  # 深度优先遍历
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def build_network(self, class_num):
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
        self.cls = Classifier(class_num=class_num)

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
        return out, self.cls(global_feature)

    @ staticmethod
    def run_train(model):
        for ipts in model.train_loader:
            retval = model(ipts)