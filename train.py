import os
import logging
from pathlib import Path
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Results dir
results_dir = Path('training_results')
results_dir.mkdir(exist_ok=True)

# Logging
class TrainingLogger:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.start_time = datetime.now()
        
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
        
        # Save to CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.results_dir / 'training_metrics.csv', index=False)
        
        # Plot metrics
        self.plot_metrics()
        
    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        ax1.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        ax1.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        
        # Plot accuracies
        ax2.plot(self.metrics['epoch'], self.metrics['train_acc'], label='Train Acc')
        ax2.plot(self.metrics['epoch'], self.metrics['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_metrics.png')
        plt.close()

# Using Datasets
class IDC_HistopathologyDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples_per_patient=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.patient_ids = []
        self.max_samples_per_patient = max_samples_per_patient

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {root_dir}")

        # Process each patient
        patient_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        logging.info(f"Found {len(patient_dirs)} patients")

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            patient_images = []
            patient_labels = [] 

            for label_str in ['0', '1']:
                label_path = patient_dir / label_str
                if not label_path.is_dir():
                    continue

                images = list(label_path.glob('*.png'))
                if self.max_samples_per_patient and len(images) > self.max_samples_per_patient:
                    images = np.random.choice(images, self.max_samples_per_patient, replace=False)
                
                patient_images.extend(images)
                patient_labels.extend([int(label_str)] * len(images))

            self.image_paths.extend(patient_images)
            self.labels.extend(patient_labels)
            self.patient_ids.extend([patient_id] * len(patient_images))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        logging.info(f"Loaded {len(self.image_paths)} images from {len(patient_dirs)} patients")
        logging.info(f"Class distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
# Splitting the dataset
def create_patient_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    patient_ids = np.unique(dataset.patient_ids)
    np.random.shuffle(patient_ids)
    
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]
    
    train_mask = np.isin(dataset.patient_ids, train_patients)
    val_mask = np.isin(dataset.patient_ids, val_patients)
    test_mask = np.isin(dataset.patient_ids, test_patients)
    
    return train_mask, val_mask, test_mask

# Transforming for Resnet18 format
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Traing the model
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total


# For validation
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

def main():
    try:
        # Set up GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("Using CPU")

        # Initialize logger
        logger = TrainingLogger(results_dir)

        # Calling dataset class
        root_dir = "./Breast Histopathology Images"
        dataset = IDC_HistopathologyDataset(
            root_dir=root_dir, 
            transform=train_transform,
            max_samples_per_patient=1000
        )

        train_mask, val_mask, test_mask = create_patient_split(dataset)
        
        train_set = torch.utils.data.Subset(dataset, np.where(train_mask)[0])
        val_set = torch.utils.data.Subset(dataset, np.where(val_mask)[0])
        test_set = torch.utils.data.Subset(dataset, np.where(test_mask)[0])
        
        val_set.dataset.transform = val_transform
        test_set.dataset.transform = val_transform

        num_workers = min(4, mp.cpu_count())
        batch_size = 32 if torch.cuda.is_available() else 16

        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        logging.info(f"Training set size: {len(train_set)}")
        logging.info(f"Validation set size: {len(val_set)}")
        logging.info(f"Test set size: {len(test_set)}")

        # Using ResNet18-
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)

        # For Multiprocessing
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # Calculating loss and optimizing 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()

        # Training cycles
        epochs = 20
        best_val_acc = 0.0

        for epoch in range(epochs):
            logging.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            logging.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            # Validation
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.log_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, results_dir / 'best_model.pth')
                logging.info(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

        # Final evaluation on test set
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        logging.info(f"Final Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
        }, results_dir / 'final_model.pth')
        
        logging.info("Training completed!")

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()