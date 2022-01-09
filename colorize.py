from os import path
from typing import Coroutine
import yaml
from models.colorization_net import ColorizationNet
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_loader(path="./config.yaml"):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def load_model(cfg_path="./config.yaml"):
    cfgs = config_loader(path=cfg_path)
    model = ColorizationNet(cfgs)
    model.eval()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(cfgs['best_model_path']))
    model = model.to(DEVICE)
    return model


def colorize_test_set():
    pass


def colorize_single_image(image_path):
    pass