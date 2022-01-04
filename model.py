from numpy import mod
from modules import *
from data_loader import *
import torch.optim as optim
import torch.nn.functional as F

class ColorizationNet(nn.Module):
    def __init__(self, cfgs, isTrain):
        super(ColorizationNet, self).__init__()
        self.cfgs = cfgs
        self.iteration = 0
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
            self.mse = nn.MSELoss(reduction='sum')
            self.ce = nn.CrossEntropyLoss()
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
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        return out, self.cls(global_feature)

    def loss(self, ab, ab_out, lab, lab_out):
        loss_col = self.mse(ab, ab_out)
        loss_class = self.ce(lab, lab_out)
        return loss_col * self.cfgs['loss_weight'][0] + loss_class * self.cfgs['loss_weight'][1]
    
    def train_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iteration += 1

    @ staticmethod
    def run_train(model):
        for ipts in model.train_loader:
            L, ab, lab = ipts # ipts[3]: L[b,1,h,w], ab[b,2,h,w], lab[b]
            ab_out, lab_out = model(L) # ab_out[b,2,h,w], lab_out[b, class_nums]
            loss = model.loss(ab, ab_out, lab, lab_out)
            model.train_step(loss)
