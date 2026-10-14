import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
import sys

# Add runtime directory to path
sys.path.append('experiments/self_training')
import runtime
from runtime import inverse_grid
runtime.STAGE = 'evaluation'

def get_dice_loss(pred, target, smooth=1e-5):
    # pred: (B, C, H, W) logits
    # target: (B, H, W) class indices
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=8).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class FeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 8, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

def train_and_eval():
    print("Loading data...")
    features = np.load('audit/improve/B2_448_features.npy', mmap_mode='r')
    centers = np.load('audit/naming/inductive_s0_centers.npy')
    
    rr = json.loads(Path('audit/naming/image_records.json').read_text())
    train_ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
    val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']
    
    # Generate pseudo-labels for training
    print("Generating pseudo-labels from K-means...")
    X_train = []
    Y_train = []
    for idx in train_ids:
        f = features[idx] # (784, 768)
        f = f.reshape(28, 28, 768).transpose(2, 0, 1) # (768, 28, 28)
        
        # Hard assignment
        f_flat = f.reshape(768, -1).T # (784, 768)
        d = np.linalg.norm(f_flat[:, None] - centers, axis=-1)
        assignment = d.argmin(axis=-1).reshape(28, 28)
        
        X_train.append(f)
        Y_train.append(assignment)
        
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.long)
    
    print(f"Train data: {X_train.shape}, Labels: {Y_train.shape}")
    
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = FeatureCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    
    epochs = 15
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = ce_loss(logits, y) + 0.5 * get_dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
        
    print("Evaluating on validation set...")
    model.eval()
    gt = np.load('audit/improve/scoring_masks_native.npy', mmap_mode='r')
    
    tp = fp = fn = 0
    slice_ious = []
    
    with torch.no_grad():
        for j, idx in enumerate(val_ids):
            f = features[idx].reshape(28, 28, 768).transpose(2, 0, 1)
            x = torch.tensor(f, dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            pred = logits.argmax(dim=1).squeeze(0).numpy() # (28, 28)
            
            # Upscale to native resolution
            native = inverse_grid(pred, rr[idx], outside=8)
            
            # Cluster 2 is liver
            gt_liver = (gt[idx] == 1)
            pred_liver = (native == 2)
            
            inter = np.logical_and(gt_liver, pred_liver).sum()
            s_tp = inter
            s_fp = pred_liver.sum() - inter
            s_fn = gt_liver.sum() - inter
            
            tp += s_tp
            fp += s_fp
            fn += s_fn
            
            s_iou = s_tp / (s_tp + s_fp + s_fn) if (s_tp + s_fp + s_fn) > 0 else 0
            slice_ious.append(s_iou)
            
    vol_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    mean_slice_iou = np.mean(slice_ious)
    
    print("\n--- Results ---")
    print(f"Volumetric IoU: {vol_iou*100:.2f}%")
    print(f"Mean Slice IoU: {mean_slice_iou*100:.2f}%")
    print(f"Precision: {tp/(tp+fp)*100:.2f}%")
    print(f"Recall: {tp/(tp+fn)*100:.2f}%")

if __name__ == '__main__':
    train_and_eval()
