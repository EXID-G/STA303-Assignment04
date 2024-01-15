
import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision import transforms


def load_data(dataset_name,seed = 42):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "fashionmnist":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])        
        data_dir = './data/fashionmnist'
        train_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "cifar100":
        data_dir = './data/cifar100'
        train_data = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    # Split the original test_data into calibration and test sets
    cal_size = int(0.2 * len(test_data))
    test_size = len(test_data) - cal_size

    torch.manual_seed(seed)
    cal_data, test_data = random_split(test_data, [cal_size, test_size])

    return train_data, cal_data, test_data
