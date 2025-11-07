# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse

# Import from our other files
from model import RGB_SClusterFormer
from utils import get_loaders

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset.indices)
    epoch_acc = running_corrects.double() / len(dataloader.dataset.indices)
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # Determine dataset size (works for both Subset and full Dataset)
    if hasattr(dataloader.dataset, 'indices'):
        dataset_size = len(dataloader.dataset.indices)
    else:
        dataset_size = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    return epoch_loss, epoch_acc.item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    dataloaders, dataset_sizes, class_names = get_loaders(args.data_path, args.img_size, args.batch_size)
    num_classes = len(class_names)

    # Initialize Model, Loss, and Optimizer
    model = RGB_SClusterFormer(
        img_size=args.img_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dims=[64, 128, 256],
        num_heads=[2, 4, 8],
        mlp_ratios=[4, 4, 4],
        depths=[2, 2, 2],
        num_stages=3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Starting training for {args.epochs} epochs...")

    best_acc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train on the 80% training split
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # Validate on the 20% validation split
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Time: {epoch_time:.0f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # --- Final Test ---
    print("\nLoading best model and running on the test set...")
    model.load_state_dict(torch.load("best_model.pth"))
    
    test_loss, test_acc = validate(model, dataloaders['test'], criterion, device)
    
    print(f"Final Test Set Performance:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RGB-SClusterFormer model.')
    parser.add_argument('--data_path', type=str, default="/kaggle/input/apple-disease-dataset/datasets",
                        help='Path to the dataset directory (must contain train/ and test/ folders).')
    parser.add_l_argument('--img_size', type=int, default=224, help='Input image size.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    
    args = parser.parse_args()
    main(args)