import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from skimage import color, io
from skimage.transform import resize
import numpy as np
import os

class HandleGray(object):
    def __call__(self, image):
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        return image


class Rgb2Lab(object):
    def __call__(self, image):
        assert image.shape == (224, 224, 3)
        img_lab = color.rgb2lab(image)
        img_lab[:,:,:1] = img_lab[:,:,:1] / 100.0
        img_lab[:,:,1:] = (img_lab[:,:,1:] + 128.0) / 256.0
        return img_lab


class SplitLab(object):
    def __call__(self, image):
        assert image.shape == (3, 224, 224)
        L  = image[:1,:,:]
        ab = image[1:,:,:]
        return (L, ab)


class Resize(object):
    def __init__(self, size):
        self.size = (size, size)
    
    def __call__(self, image):
        return resize(image, self.size)


class DataSet(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=io.imread)
        self.composed = transforms.Compose(
            [HandleGray(), Resize(224), Rgb2Lab(), transforms.ToTensor(), SplitLab()]
        )
    def __getitem__(self, index):
        image, label = super().__getitem__(index) # image: [c, h, w]
        L, ab = self.composed(image)
        return L, ab, label


def get_data_loader(root, batch_size, isTrain, num_workers=1, shuffle=True):
    image_dir = 'train' if isTrain else 'test'
    root = os.path.join(root, image_dir)
    dataset = DataSet(root)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

if __name__ == '__main__':
    loader = get_data_loader('./data/places10', 1, True)
    d = next(iter(loader))
    print(d[0])
    print(d[1])
    print(len(d))
    print(d[2])
    print(d[0].size(), d[1].size(), d[2].size())