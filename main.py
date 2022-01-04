import argparse
import logging
import numpy as np
import yaml
import random
from models.colorization_net import *

# Log Settings
logging.basicConfig(level = logging.INFO,format = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Argument
parser = argparse.ArgumentParser(description='Main program.')
parser.add_argument('--cfgs', type=str,
                    default='./config.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()

def config_loader(path="./config.yaml"):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def train(cfgs):
    # Fix random seed
    init_seeds(0)
    # Init model
    model = ColorizationNet(cfgs)
    model.train()
    # Init data loader
    train_loader = get_data_loader('./data', cfgs['batch_size'], True)
    # Define train step
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **cfgs["optimizer_cfg"])
    scheduler = optim.lr_scheduler.MultiStepLR(**cfgs["scheduler_cfg"])
    def train_step(loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    # Define loss
    mse = nn.MSELoss(reduction='sum')
    ce = nn.CrossEntropyLoss()
    def loss(ab, ab_out, cls_gt, cls_out):
        colorization_loss = mse(ab, ab_out)
        classification_loss = ce(cls_gt, cls_out)
        total_loss = colorization_loss * cfgs['loss_weight'][0] + classification_loss * cfgs['loss_weight'][1]
        return total_loss, colorization_loss, classification_loss
    # Start training
    max_epoch = cfgs['epoch']
    log_step = cfgs['log_step']
    for epoch in range(max_epoch):
        total_step = len(model.train_loader)
        loss_cnt = col_loss_cnt = cls_loss_cnt = 0
        for i, ipts in enumerate(model.train_loader, start=1):
            L, ab, cls_gt = ipts # ipts[3]: L[b,1,h,w], ab[b,2,h,w], lab[b]
            ab_out, cls_out = model(L) # ab_out[b,2,h,w], lab_out[b, class_nums]
            loss, colorization_loss, classification_loss = loss(ab, ab_out, cls_gt, cls_out)
            loss_cnt += loss.item()
            col_loss_cnt += colorization_loss.item()
            cls_loss_cnt += classification_loss.item()
            train_step(loss)
            if i % log_step == 0:
                msg = 'Epoach [%d/%d] Step [%d/%d] cls_loss=%.3f col_loss=%.3f total=%.3f' % (epoch+1,
                 max_epoch, i, total_step, cls_loss_cnt/log_step, col_loss_cnt/log_step ,loss_cnt/log_step)
                logger.info(msg) 
                loss_cnt = cls_loss_cnt = col_loss_cnt = 0


if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    logger.info(cfgs)
    isTrain = (opt.phase == 'train')
    if isTrain:
        train(cfgs)


