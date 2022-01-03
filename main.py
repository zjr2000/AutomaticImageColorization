import argparse
import logging
import numpy as np
import yaml
import random
from model import *

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

if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    isTrain = (opt.phase == 'train')
    init_seeds(0)
    model = ColorizationNet(cfgs, isTrain)
    if isTrain:
        ColorizationNet.run_train(model)
    else:
        ColorizationNet.run_test(model)


