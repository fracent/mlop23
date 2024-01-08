"""This file was created x mlop dtu course 2023,
francesco centomo developed this code with help from
chat gpt4 and mlop material"""


import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
def load_data():
    train_images = torch.cat([torch.load(f'train_images_{i}.pt') for i in range(6)], dim=0)
    train_targets = torch.cat([torch.load(f'train_target_{i}.pt') for i in range(6)], dim=0)
    test_images = torch.load('test_images.pt')
    test_targets = torch.load('test_target.pt')

    train_dataset = TensorDataset(train_images, train_targets)
    test_dataset = TensorDataset(test_images, test_targets)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return trainloader, testloader
