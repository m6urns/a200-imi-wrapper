import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json

class DualStreamKnotClassifier(nn.Module):
    """Dual-stream classifier for RGB and depth data"""
    
    def __init__(self, num_classes=4, weights='DEFAULT'):
        super().__init__()
        
        # RGB stream - Using EfficientNet-B0 for good performance/size trade-off
        self.rgb_backbone = models.efficientnet_b0(weights=weights)
        # Store feature dimension before replacing classifier
        rgb_features = self.rgb_backbone.classifier[1].in_features
        self.rgb_backbone.classifier = nn.Identity()
        
        # Depth stream - Same architecture but modified for single-channel input
        self.depth_backbone = models.efficientnet_b0(weights=weights)
        # Store feature dimension
        depth_features = self.depth_backbone.classifier[1].in_features
        
        # Modify first conv layer for single-channel input while keeping pretrained weights
        original_conv = self.depth_backbone.features[0][0]
        self.depth_backbone.features[0][0] = nn.Conv2d(1, original_conv.out_channels,
                                                      kernel_size=original_conv.kernel_size,
                                                      stride=original_conv.stride,
                                                      padding=original_conv.padding,
                                                      bias=original_conv.bias is not None)
        # Initialize the new conv layer
        with torch.no_grad():
            new_weight = original_conv.weight.mean(dim=1, keepdim=True)
            self.depth_backbone.features[0][0].weight.copy_(new_weight)
            
        self.depth_backbone.classifier = nn.Identity()
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(rgb_features + depth_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone layers initially
        self._freeze_backbone()
        
    def _freeze_backbone(self):
        """Freeze backbone layers"""
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.depth_backbone.parameters():
            param.requires_grad = False
            
    def _unfreeze_backbone(self):
        """Unfreeze backbone layers for fine-tuning"""
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True
        for param in self.depth_backbone.parameters():
            param.requires_grad = True
            
    def forward(self, rgb, depth):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        combined = torch.cat((rgb_features, depth_features), dim=1)
        return self.classifier(combined)

class KnotDataset(Dataset):
    """Dataset for knot RGB-D data"""
    
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.rgb_transform = transform or self._default_rgb_transform()
        self.depth_transform = self._default_depth_transform()
        self.samples = self._load_samples()
        
    def _default_rgb_transform(self):
        """Default RGB data transformations"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _default_depth_transform(self):
        """Default depth data transformations"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def _load_samples(self):
        """Load all samples from the dataset"""
        samples = []
        stages = ['loose', 'loop', 'complete', 'tightened']
        
        for stage_idx, stage in enumerate(stages):
            stage_path = self.data_path / stage
            if not stage_path.exists():
                continue
                
            for sample_dir in stage_path.iterdir():
                if not sample_dir.is_dir():
                    continue
                    
                rgb_path = sample_dir / "rgb.png"
                depth_path = sample_dir / "depth.npy"
                metadata_path = sample_dir / "metadata.json"
                
                if rgb_path.exists() and depth_path.exists():
                    samples.append({
                        'rgb_path': str(rgb_path),
                        'depth_path': str(depth_path),
                        'metadata_path': str(metadata_path),
                        'stage': stage,
                        'label': stage_idx
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        rgb = cv2.imread(sample['rgb_path'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth data
        depth = np.load(sample['depth_path'])
        
        # Normalize depth to 0-255 range
        depth_min = depth[depth > 0].min() if np.any(depth > 0) else 0
        depth_max = depth.max()
        depth_normalized = np.zeros_like(depth, dtype=np.uint8)
        if depth_max > depth_min:
            valid_mask = depth > 0
            depth_normalized[valid_mask] = ((depth[valid_mask] - depth_min) * 255 / 
                                          (depth_max - depth_min))
        
        # Ensure depth is single channel
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply transforms
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        if self.depth_transform:
            depth = self.depth_transform(depth_normalized)
            
        return {
            'rgb': rgb,
            'depth': depth,
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, num_epochs=20, 
                device='cuda', unfreeze_epoch=10, early_stopping_patience=5):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    model = model.to(device)
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Unfreeze backbone for fine-tuning after specified epoch
        if epoch == unfreeze_epoch:
            model._unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(rgb, depth)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100 * correct / total
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.3f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model

def main():
    # Setup data
    dataset = KnotDataset("overhand_knot_dataset")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Use smaller batch size for small dataset
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    print(f"Dataset sizes:")
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("\nClass distribution:")
    stage_counts = {}
    for sample in dataset.samples:
        stage = sample['stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count} samples")
        
    # Check first batch to verify shapes
    first_batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"RGB: {first_batch['rgb'].shape}")
    print(f"Depth: {first_batch['depth'].shape}")
    print(f"Labels: {first_batch['label'].shape}")
    
    # Create and train model
    
    # Create and train model
    model = DualStreamKnotClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(model, train_loader, val_loader, device=device)

if __name__ == '__main__':
    main()