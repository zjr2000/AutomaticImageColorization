import pickle
import numpy as np
import cv2
from tqdm import tqdm
import os

test_data_path = './data/cifar-100-python/test'
target_size = (224, 224)
target_root_dir = './data/test_data/'

with open(test_data_path, 'rb') as f:
    entry = pickle.load(f, encoding='latin1')

data = [entry['data']]
data = np.vstack(data).reshape(-1, 3, 32, 32)
data = data.transpose((0, 2, 3, 1))

if not os.path.exists(target_root_dir):
    os.mkdir(target_root_dir)

for i, img in tqdm(enumerate(data)):
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    file_name = 'img' + str(i) + '.png'
    file_name = os.path.join(target_root_dir, file_name)
    cv2.imwrite(file_name, img)


