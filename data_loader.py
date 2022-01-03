import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100
import torch

def get_data_loader(root, batch_size, num_workers, shuffle=True):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    dataset = CIFAR100(root, train=True, transform=transform, target_transform=None, download=False)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

if __name__ == '__main__':
    loader = get_data_loader('./data', 1, 1)
    d = next(iter(loader))
    print(d[0].size(), d[1].size())