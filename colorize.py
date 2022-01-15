import yaml
from models.colorization_net import ColorizationNet
import torch
from tqdm import tqdm
from data_loader import get_data_loader
import numpy as np
from skimage import color
import cv2
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_loader(path="./config.yaml"):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def load_model(cfgs):
    model = ColorizationNet(cfgs)
    model.eval()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(cfgs['best_model_path']))
    model = model.to(DEVICE)
    return model


def net_out2rgb(L, ab_out):
    """Translates the net output back to an image.
    More specifically: unnormalizes both L and ab_out channels, stacks them
    into an image in LAB color space and converts back to RGB.
  
    Args:
        L: original L channel of an image
        ab_out: ab channel which was learnt by the network
    
    Retruns: 
        3 channel RGB image
    """
    # Convert to numpy and unnnormalize
    Lab = torch.cat([L, ab_out]).cpu().numpy()
    
    Lab = Lab.transpose((1, 2, 0))
    Lab[:,:,:1] = Lab[:,:,:1] * 100.0
    Lab[:,:,1:] = Lab[:,:,1:] * 256.0 - 127.0

    Lab = Lab.astype(np.float64)	
    return  color.lab2rgb(Lab) * 255.0


def colorize_test_set(cfgs):
    pred_dir = cfgs['pred_dir']
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    model = load_model(cfgs)
    eval_loader = get_data_loader('./data/places10', 1, False, shuffle=False)
    with torch.no_grad():
        for i, ipts in tqdm(enumerate(eval_loader)):
            file_name = 'img_pred' + str(i) +'.png'
            L, ab, _ = ipts
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)
            ab_out, _ = model(L)
            image = net_out2rgb(L.squeeze(0), ab_out.squeeze(0))
            # image = cv2.resize(image, (32, 32))
            file_name = os.path.join(pred_dir, file_name)
            cv2.imwrite(file_name, image)



def colorize_single_image(image_path):
    pass


if __name__ == '__main__':
    cfgs = config_loader(path="./config.yaml")
    colorize_test_set(cfgs)