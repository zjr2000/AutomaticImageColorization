import skimage
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100
import torch
from skimage import color, io
import numpy as np

class Rgb2Lab(object):
    def __call__(self, image):
        image = np.asarray(image, dtype='float32')
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

class DataSet(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.composed = transforms.Compose(
            [transforms.Resize(224), Rgb2Lab(), transforms.ToTensor(), SplitLab()]
        )
    def __getitem__(self, index):
        image, label = super().__getitem__(index) # image: [c, h, w]
        L, ab = self.composed(image)
        return L, ab, label


def get_data_loader(root, batch_size, isTrain, num_workers=1, shuffle=True):
    dataset = DataSet(root, train=isTrain, download=False)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

if __name__ == '__main__':
    loader = get_data_loader('./data', 1, 1)
    d = next(iter(loader))
    print(len(d))
    print(d[2])
    print(d[0].size(), d[1].size(), d[2].size())