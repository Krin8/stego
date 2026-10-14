import numpy as np
import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from runtime import load_image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'audit/naming'

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=1, border_mode=0),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        ToTensorV2()
    ]
    return A.Compose(train_transform)

class CTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are (1, 256, 256), need to convert to (256, 256, 1) for albumentations
        image = self.images[idx].transpose(1, 2, 0)
        mask = self.labels[idx]
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).long()
            
        return image, mask

def train():
    print("Loading datasets...")
    # Load the pre-extracted 256x256 images and K-means pseudo-labels
    images = np.load(HERE / 'train_images.npy', mmap_mode='r')
    labels = np.load(HERE / 'train_targets.npy', mmap_mode='r')
    
    train_dataset = CTDataset(images, labels, transforms=get_training_augmentation())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize a powerful ResNet50 U-Net
    print("Initializing ResNet50 U-Net (Zero-Shot / No ImageNet)...")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None, 
        in_channels=1, 
        classes=8
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Ignore class 8 (outside/background padding)
    criterion = nn.CrossEntropyLoss(ignore_index=8)
    
    epochs = 50
    print(f"Starting training for {epochs} epochs on GPU...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start = time.time()
        
        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Time: {time.time()-start:.1f}s | Avg Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), HERE / 'fcn_resnet50.pt')
    print("Training finished!")

if __name__ == '__main__':
    train()
