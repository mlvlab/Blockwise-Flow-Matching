import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_cifar10_dataloader(batch_size=128, num_workers=4, train=True):
    """
    Create a DataLoader for CIFAR-10 dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses for data loading
        train (bool): If True, creates dataset from training set, otherwise creates from test set
        
    Returns:
        DataLoader: PyTorch DataLoader for CIFAR-10
    """
    
    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor and scale to [0,1]
    ])
    
    # If training, add data augmentation
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    # Create the dataset
    dataset = datasets.CIFAR10(
        root='./datasets',
        train=train,
        download=True,
        transform=transform
    )
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle only if training
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train  # Drop last incomplete batch only during training
    )
    
    return dataloader