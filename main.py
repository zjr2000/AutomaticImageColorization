import argparse
import logging
import numpy as np
import yaml
import random
from models.colorization_net import *
import torch
from tqdm import tqdm
from tensorboard_logger import configure, log_value, log_images
import colorize
import cv2
import os

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log Settings
logging.basicConfig(level = logging.INFO,format = '[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Argument
parser = argparse.ArgumentParser(description='Main program.')
parser.add_argument('--cfgs', type=str,
                    default='./config.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--local_rank', default=0, help="debug")
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

def init_model(cfgs, is_train=True):
    model = ColorizationNet(cfgs)
    model.train(is_train)
    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    return model

def train(cfgs):
    # log
    configure('./logs', flush_secs=10)
    # Fix random seed
    init_seeds(0)
    # Init model
    model = init_model(cfgs)
    # Init data loader
    train_loader = get_data_loader('./data', cfgs['batch_size'], True)
    # Define train step
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **cfgs["optimizer_cfg"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **cfgs["scheduler_cfg"])
    def train_step(loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Define loss
    mse = nn.MSELoss(reduction='mean')
    ce = nn.CrossEntropyLoss()
    def loss_cal(ab, ab_out, cls_gt, cls_out):
        colorization_loss = mse(ab_out, ab)
        classification_loss = ce(cls_out, cls_gt)
        total_loss = colorization_loss * cfgs['loss_weight'][0] + classification_loss * cfgs['loss_weight'][1]
        return total_loss, colorization_loss, classification_loss
    # Start training
    max_epoch = cfgs['epoch']
    log_step = cfgs['log_step']
    save_per_epoch = cfgs['save_per_epoch']
    total_step = len(train_loader)
    count = 0
    best_raw_acc = 100000
    best_cls_acc = 0
    save_metric = cfgs['save_metric']
    best_model_path = cfgs['best_model_path']
    saving_schedule = [int(x * total_step / save_per_epoch) for x in list(range(1, save_per_epoch + 1))]
    logger.info('Saving schedule:')
    logger.info(saving_schedule)
    for epoch in range(max_epoch):
        loss_cnt = col_loss_cnt = cls_loss_cnt = 0
        for i, ipts in enumerate(train_loader, start=1):
            L, ab, cls_gt = ipts # ipts[3]: L[b,1,h,w], ab[b,2,h,w], lab[b]
            
            with torch.no_grad():
                for j in range(len(L)):
            #         file_name = 'img_train_gt' + str(i) +'.png'
                    image = colorize.net_out2rgb(L[j], ab[j])
                    image = image[np.newaxis,:]
            #         file_name = os.path.join("./results/train_gt/", file_name)
            #         cv2.imwrite(file_name, image)
                    log_images("train_gt", image, epoch * total_step + i)
                    break

            L = L.to(DEVICE)
            ab = ab.to(DEVICE)
            cls_gt = cls_gt.to(DEVICE)
            ab_out, cls_out = model(L) # ab_out[b,2,h,w], lab_out[b, class_nums]
            
            with torch.no_grad():
                for j in range(len(L)):
                    # file_name = 'img_train_pre' + str(i) +'.png'
                    image = colorize.net_out2rgb(L[j], ab_out[j])
                    image = image[np.newaxis,:]
                    # file_name = os.path.join("./results/train_pre/", file_name)
                    # cv2.imwrite(file_name, image)
                    log_images("train_pre", image, epoch * total_step + i)
                    break

            loss, colorization_loss, classification_loss = loss_cal(ab, ab_out, cls_gt, cls_out)
            loss_cnt += loss.item()
            log_value('total_loss', loss.item(), epoch * total_step + i)
            col_loss_cnt += colorization_loss.item()
            cls_loss_cnt += classification_loss.item()
            train_step(loss)
            if i % log_step == 0:
                msg = 'Epoach [%d/%d] Step [%d/%d] cls_loss=%.5f col_loss=%.5f total=%.5f' % (epoch+1,
                 max_epoch, i, total_step, cls_loss_cnt/log_step, col_loss_cnt/log_step ,loss_cnt/log_step)
                logger.info(msg) 
                loss_cnt = cls_loss_cnt = col_loss_cnt = 0

            if i in saving_schedule:
                model.eval()
                scores = _evaluate(cfgs, model)
                for name, val in scores.items():
                    log_value(name, val, count)
                count += 1
                logger.info('Evaluate results: raw_acc: %.4f cls_acc %.4f' % (scores['raw_acc'], scores['cls_acc']))
                if save_metric == 'raw_acc':
                    if scores['raw_acc'] <= best_raw_acc:
                        torch.save(model.state_dict(), best_model_path)
                elif save_metric == 'cls_acc':
                    if scores['cls_acc'] >= best_cls_acc:
                        torch.save(model.state_dict(), best_model_path)    
                model.train()
        scheduler.step()


def _evaluate(cfgs, model):
    eval_loader = get_data_loader('./data', cfgs['batch_size'], False)
    total_step = len(eval_loader)
    total_l2_dist = 0.
    correct_num = 0.
    total_cnt = 0.
    with torch.no_grad():
        for i, ipts in tqdm(enumerate(eval_loader, start=1)):
            L, ab, cls_gt = ipts
            total_cnt += L.size(0)
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)
            cls_gt = cls_gt.to(DEVICE)
            ab_out, cls_out = model(L)
            # L2 distance calculate
            for i in range(len(ab)): total_l2_dist += torch.dist(ab[i], ab_out[i], 2)
            # correct num
            correct_num += ((cls_out.max(1)[1] == cls_gt).sum())
        
    avg_l2_dist = total_l2_dist / total_cnt
    cls_acc = correct_num / total_cnt
    # TODO add more evaluate metrics
    scores = {'raw_acc': avg_l2_dist, 'cls_acc': cls_acc}
    return scores  
         

if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    logger.info(cfgs)
    isTrain = (opt.phase == 'train')
    if isTrain:
        train(cfgs)


