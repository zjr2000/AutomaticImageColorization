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
    L = L.numpy() * 100.0
    ab_out = ab_out.numpy() * 254.0 - 127.0
    
    
    # L and ab_out are tenosr i.e. are of shape of
    # Height x Width x Channels
    # We need to transpose axis back to HxWxC
    L = L.transpose((1, 2, 0))
    ab_out = ab_out.transpose((1, 2, 0))

    # Stack layers  
    img_stack = np.dstack((L, ab_out))
    
    # This line is CRUCIALL
    #   - torch requires tensors to be float32
    #   - thus all above (L, ab) are float32
    #   - scikit image floats are in range -1 to 1
    #   - http://scikit-image.org/docs/dev/user_guide/data_types.html
    #   - idk what's next, but converting image to float64 somehow
    #     does the job. scikit automagically converts those values.
    img_stack = img_stack.astype(np.float64)	
    return  color.lab2rgb(img_stack)


def colorize_test_set(cfgs):
    pred_dir = cfgs['pred_dir']
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    model = load_model(cfgs)
    eval_loader = get_data_loader('./data', 1, False)
    with torch.no_grad():
        for i, ipts in tqdm(enumerate(eval_loader, start=1)):
            file_name = 'img_pred' + str(i) +'.png'
            L, ab, _ = ipts
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)
            ab_out, _ = model(L)
            image = net_out2rgb(L.squeeze(0), ab_out.squeeze(0))
            image = cv2.resize(image, (32, 32))
            file_name = os.path.join(pred_dir, file_name)
            cv2.imwrite(image, file_name)



def colorize_single_image(image_path):
    pass


if __name__ == '__main__':
    cfgs = config_loader(path="./config.yaml")
    colorize_test_set()