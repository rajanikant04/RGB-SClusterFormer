# utils.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_loaders(data_dir, img_size, batch_size):
    """
    Creates training, validation, and testing DataLoaders.
    The 'train' folder is split into 80% train and 20% validation.
    The 'test' folder is used for the final test set.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    try:
        # Create dataset instances for train and validation (with different transforms)
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['val'])
        
        # Create dataset instance for test
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['val'])

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Could not find 'train' or 'test' folders inside '{data_dir}'")
        print("Please ensure your dataset is structured correctly.")
        exit()

    class_names = train_dataset.classes

    # Get targets for stratified split
    targets = train_dataset.targets
    
    # Create stratified 80/20 split
    train_indices, val_indices = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets
    )

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    dataset_sizes = {
        'train': len(train_indices),
        'val': len(val_indices),
        'test': len(test_dataset)
    }

    print(f"Total images in 'train' folder: {len(train_dataset)}")
    print(f"  -> Training set size:   {dataset_sizes['train']} (80%)")
    print(f"  -> Validation set size: {dataset_sizes['val']} (20%)")
    print(f"Total images in 'test' folder:  {dataset_sizes['test']}")
    print(f"Found {len(class_names)} classes: {class_names}")

    return dataloaders, dataset_sizes, class_names