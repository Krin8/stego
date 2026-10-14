import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
from pathlib import Path
from skimage.transform import resize
from runtime import load_image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'audit/naming'

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.down4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up3 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        
        self.out = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4)
        u1 = self.conv_up1(torch.cat([u1, d3], dim=1))
        
        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, d2], dim=1))
        
        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, d1], dim=1))
        
        return self.out(u3)

def train():
    print("Loading data...")
    images = np.load(HERE / 'train_images.npy', mmap_mode='r')
    labels = np.load(HERE / 'train_targets.npy', mmap_mode='r')
    N = len(images)
    
    model = MiniUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=8)
    
    epochs = 3
    batch_size = 16
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(N)
        epoch_loss = 0
        start = time.time()
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = torch.from_numpy(np.array(images[idx])).contiguous()
            y = torch.from_numpy(np.array(labels[idx])).contiguous()
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % (batch_size * 10) == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} done in {time.time()-start:.1f}s. Avg Loss: {epoch_loss / (N/batch_size):.4f}")
        
    torch.save(model.state_dict(), HERE / 'fcn.pt')
    print("Training finished!")

def predict():
    import runtime
    runtime.STAGE = 'evaluation'
    model = MiniUNet()
    model.load_state_dict(torch.load(HERE / 'fcn.pt'))
    model.eval()
    
    rr = json.loads((BASE / 'image_records.json').read_text())
    val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']
    
    (HERE / 'predictions').mkdir(exist_ok=True)
    out_native = np.lib.format.open_memmap(HERE / 'predictions/raw_clusters_native.npy', mode='w+', dtype=np.uint8, shape=(len(val_ids), 512, 512))
    
    print("Predicting validation set...")
    with torch.no_grad():
        for j, idx in enumerate(val_ids):
            hu = load_image(rr[idx])
            resized = resize(hu, (256, 256), order=1, preserve_range=True, anti_aliasing=False)
            x = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).contiguous()
            logits = model(x)
            pred = logits.argmax(dim=1).squeeze().numpy()
            out_native[j] = resize(pred, (512, 512), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            
            if j % 50 == 0:
                print(f"Predicted {j}/{len(val_ids)}")
                
    out_native.flush()
    print("Prediction finished!")
    
    # Save a dummy prediction_manifest so seal_all.py or evaluate.py can work
    manifest = {
        'indices': val_ids,
        'labels_read': False,
        'selected': json.loads((BASE / 'inductive_s0_naming.json').read_text())['selected']
    }
    (HERE / 'predictions/prediction_manifest.json').write_text(json.dumps(manifest))

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'predict':
        predict()
