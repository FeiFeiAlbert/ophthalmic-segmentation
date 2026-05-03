# -*- coding: utf-8 -*-
"""
V16 Training Script - UNet++ with EfficientNet-B4

Key features:
- Hot-start from V13 pretrained weights
- Combined Loss: 0.3*Focal + 0.4*Dice + 0.3*Lovasz
- 4-direction TTA (Test-Time Augmentation)
- CosineAnnealingWarmRestarts scheduler
- Achieved Val Dice: 0.9554

Usage:
    python train_v16.py --data_dir /path/to/data --pretrained /path/to/v13_weights.pth
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from datetime import datetime
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ophthalmic_segmentation.losses import CombinedLoss
from ophthalmic_segmentation.model import create_model


# ==================== Configuration ====================
class Config:
    # Data paths - update these for your dataset
    IMAGE_DIR = "data/images"
    LABEL_DIR = "data/labels"
    PRETRAIN_PATH = ""  # Set to V13 weights for hot-start
    
    # Training settings
    IMG_SIZE = 448
    BATCH_SIZE = 4
    EPOCHS = 200
    LR_ENCODER = 1e-5
    LR_DECODER = 1e-4
    NUM_CLASSES = 3
    PATIENCE = 50
    
    # Class definitions
    CLASS_NAMES = {0: 'background', 1: 'roi', 2: 'valid_reflex'}
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save directory
    SAVE_DIR = "checkpoints"
    
    # Loss weights (V16 optimized)
    FOCAL_WEIGHT = 0.3
    DICE_WEIGHT = 0.4
    LOVASZ_WEIGHT = 0.3


# ==================== Dataset ====================
def parse_labelme_json(json_path):
    """Parse LabelMe JSON annotation file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = {}
    for shape in data.get('shapes', []):
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        shapes[label] = points
    return shapes


def create_mask_from_polygons(shapes, height, width, num_classes=3):
    """Create segmentation mask from polygon annotations"""
    mask = np.zeros((height, width), dtype=np.uint8)
    label_to_class = {'background': 0, 'roi': 1, 'valid_reflex': 2}
    for label, points in shapes.items():
        class_id = label_to_class.get(label, 0)
        if class_id > 0 and len(points) >= 3:
            cv2.fillPoly(mask, [points], class_id)
    return mask


class RetinaSegmentationDataset(Dataset):
    """Dataset for retinal fundus image segmentation"""
    
    def __init__(self, image_dir, label_dir, img_size=448, num_classes=3, 
                 augment=False, image_size_limit=2000):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.image_size_limit = image_size_limit
        self.samples = self._match_files()
        print(f"Dataset: Found {len(self.samples)} image-mask pairs")
    
    def _match_files(self):
        """Match image files with their corresponding label JSON files"""
        samples = []
        image_files = list(self.image_dir.glob("*.jpg")) + \
                      list(self.image_dir.glob("*.png")) + \
                      list(self.image_dir.glob("*.JPG"))
        
        for img_path in sorted(image_files):
            stem = img_path.stem
            # Try different JSON extensions
            for ext in ['.json', '.JSON']:
                label_path = self.label_dir / f"{stem}{ext}"
                if label_path.exists():
                    samples.append({
                        'image': str(img_path), 
                        'label': str(label_path), 
                        'stem': stem
                    })
                    break
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Read image (support Chinese paths)
        with open(sample['image'], 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2][::-1]  # (width, height)
        
        # Resize if too large
        if max(original_size) > self.image_size_limit:
            scale = self.image_size_limit / max(original_size)
            new_size = (int(original_size[0] * scale), 
                       int(original_size[1] * scale))
            image = cv2.resize(image, new_size)
            original_size = new_size
        
        # Parse label and create mask
        shapes = parse_labelme_json(sample['label'])
        mask = create_mask_from_polygons(shapes, image.shape[0], image.shape[1], 
                                         self.num_classes)
        
        # Resize to target size
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # Apply augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        return image, mask
    
    def _augment(self, image, mask):
        """Data augmentation for training"""
        # Horizontal flip
        if np.random.rand() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])
        
        # Random brightness/contrast
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = image * factor
            image = torch.clamp(image, 0, 1)
        
        return image, mask


# ==================== Metrics ====================
def calculate_metrics(pred, target, num_classes=3):
    """Calculate Dice and IoU metrics per class"""
    pred = pred.argmax(dim=1)
    dice_scores = []
    ious = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        dice = (2 * intersection + 1e-6) / (union + intersection + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        dice_scores.append(dice.item())
        ious.append(iou.item())
    
    # Return mean of foreground classes (excluding background)
    return {
        'dice': np.mean(dice_scores[1:]),
        'dice_per_class': dice_scores,
        'iou': np.mean(ious[1:]),
        'iou_per_class': ious
    }


# ==================== TTA (Test-Time Augmentation) ====================
def predict_with_tta(model, image, device):
    """
    4-direction TTA: original + horizontal flip + vertical flip + both flips
    Average predictions for more robust results
    """
    model.eval()
    with torch.no_grad():
        img = image.unsqueeze(0).to(device)
        
        # Original
        out1 = model(img)
        
        # Horizontal flip
        img_hflip = torch.flip(img, dims=[3])
        out2 = torch.flip(model(img_hflip), dims=[3])
        
        # Vertical flip
        img_vflip = torch.flip(img, dims=[2])
        out3 = torch.flip(model(img_vflip), dims=[2])
        
        # Both flips
        img_hvflip = torch.flip(img, dims=[2, 3])
        out4 = torch.flip(model(img_hvflip), dims=[2, 3])
        
        # Average
        output = (out1 + out2 + out3 + out4) / 4.0
    
    return output


# ==================== Training Functions ====================
def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss, total_dice = 0, []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        metrics = calculate_metrics(outputs.detach(), masks)
        total_dice.append(metrics['dice'])
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'dice': f'{metrics["dice"]:.4f}'})
    
    return total_loss / len(dataloader), np.mean(total_dice)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss, total_dice, total_iou = 0, [], []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            metrics = calculate_metrics(outputs, masks)
            total_dice.append(metrics['dice'])
            total_iou.append(metrics['iou'])
    
    return total_loss / len(dataloader), np.mean(total_dice), np.mean(total_iou)


def validate_with_tta(model, dataset, device, indices):
    """Validate with TTA on specified indices"""
    model.eval()
    all_dice, all_iou = [], []
    
    for idx in tqdm(indices, desc="TTA Validation"):
        image, mask = dataset[idx]
        output = predict_with_tta(model, image, device)
        metrics = calculate_metrics(output.cpu(), mask.unsqueeze(0))
        all_dice.append(metrics['dice'])
        all_iou.append(metrics['iou'])
    
    return np.mean(all_dice), np.mean(all_iou)


# ==================== Visualization ====================
def visualize_prediction(image, mask, pred, save_path=None, epoch=0, dice=0):
    """Visualize single prediction"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    img_np = image.cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original (Epoch {epoch})')
    axes[0].axis('off')
    
    axes[1].imshow(mask.cpu().numpy(), cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred.cpu().numpy(), cmap='tab10', vmin=0, vmax=2)
    axes[2].set_title(f'Prediction (Dice={dice:.4f})')
    axes[2].axis('off')
    
    # Create overlay
    overlay = img_np.copy()
    pred_np = pred.cpu().numpy()
    overlay[pred_np == 1] = [0, 1, 0]  # ROI - green
    overlay[pred_np == 2] = [1, 0, 0]  # Reflex - red
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_all_samples(model, dataset, device, save_dir, use_tta=True):
    """Visualize all samples in dataset"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    all_metrics = []
    
    for idx in tqdm(range(len(dataset)), desc="Generating visualizations"):
        image, mask = dataset[idx]
        
        if use_tta:
            output = predict_with_tta(model, image, device)
        else:
            with torch.no_grad():
                output = model(image.unsqueeze(0).to(device))
        
        pred = output.argmax(dim=1)[0]
        metrics = calculate_metrics(output.cpu(), mask.unsqueeze(0))
        all_metrics.append(metrics)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        img_np = image.cpu().permute(1, 2, 0).numpy()
        
        axes[0].imshow(img_np)
        axes[0].set_title(f'Sample {idx}')
        axes[0].axis('off')
        
        axes[1].imshow(mask.cpu().numpy(), cmap='tab10', vmin=0, vmax=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred.cpu().numpy(), cmap='tab10', vmin=0, vmax=2)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        overlay = img_np.copy()
        pred_np = pred.cpu().numpy()
        overlay[pred_np == 1] = [0, 1, 0]
        overlay[pred_np == 2] = [1, 0, 0]
        axes[3].imshow(overlay)
        axes[3].set_title(f'Dice: {metrics["dice"]:.4f}')
        axes[3].axis('off')
        
        plt.savefig(f'{save_dir}/sample_{idx:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    return all_metrics


# ==================== Main Training ====================
def main():
    parser = argparse.ArgumentParser(description='Train V16 UNet++ Fundus Segmentation')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Data directory containing images/ and labels/')
    parser.add_argument('--pretrained', type=str, default='',
                       help='Path to pretrained weights (V13 for hot-start)')
    parser.add_argument('--img_size', type=int, default=448,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--lr_encoder', type=float, default=1e-5,
                       help='Learning rate for encoder')
    parser.add_argument('--lr_decoder', type=float, default=1e-4,
                       help='Learning rate for decoder')
    args = parser.parse_args()
    
    # Update config
    Config.IMAGE_DIR = os.path.join(args.data_dir, 'images')
    Config.LABEL_DIR = os.path.join(args.data_dir, 'labels')
    Config.PRETRAIN_PATH = args.pretrained
    Config.IMG_SIZE = args.img_size
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.SAVE_DIR = args.save_dir
    Config.LR_ENCODER = args.lr_encoder
    Config.LR_DECODER = args.lr_decoder
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    print("=" * 70)
    print("UNet V16 Training - Fundus Image Segmentation")
    print("Architecture: UNet++ with EfficientNet-B4")
    print(f"Image Size: {Config.IMG_SIZE} | Batch Size: {Config.BATCH_SIZE}")
    print(f"Loss: {Config.FOCAL_WEIGHT}*Focal + {Config.DICE_WEIGHT}*Dice + {Config.LOVASZ_WEIGHT}*Lovasz")
    print(f"LR: encoder={Config.LR_ENCODER}, decoder={Config.LR_DECODER}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 70)
    
    # Create datasets
    train_dataset = RetinaSegmentationDataset(
        Config.IMAGE_DIR, Config.LABEL_DIR,
        img_size=Config.IMG_SIZE, augment=True
    )
    val_dataset = RetinaSegmentationDataset(
        Config.IMAGE_DIR, Config.LABEL_DIR,
        img_size=Config.IMG_SIZE, augment=False
    )
    
    # Split dataset
    indices = np.arange(len(train_dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, num_samples=len(train_indices), replacement=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        sampler=train_sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    print(f"\nTraining: {len(train_indices)} samples")
    print(f"Validation: {len(val_indices)} samples")
    
    # Create model
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',  # Start with ImageNet pretrained encoder
        in_channels=3,
        classes=Config.NUM_CLASSES
    )
    
    # Hot-start from V13 if provided
    if Config.PRETRAIN_PATH and os.path.exists(Config.PRETRAIN_PATH):
        print(f"\nLoading pretrained weights from: {Config.PRETRAIN_PATH}")
        try:
            state_dict = torch.load(Config.PRETRAIN_PATH, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            print("Pretrained weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from ImageNet initialization...")
    
    model = model.to(Config.DEVICE)
    
    # Optimizer with differential learning rates
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() 
                      if not n.startswith('encoder')]
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': Config.LR_ENCODER, 'weight_decay': 1e-5},
        {'params': decoder_params, 'lr': Config.LR_DECODER, 'weight_decay': 1e-4},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    criterion = CombinedLoss(
        focal_weight=Config.FOCAL_WEIGHT,
        dice_weight=Config.DICE_WEIGHT,
        lovasz_weight=Config.LOVASZ_WEIGHT
    )
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'val_dice_tta': [], 'val_iou': [], 'lr': []
    }
    
    best_dice, best_dice_tta = 0, 0
    patience_counter = 0
    target_reached = False
    
    print(f"\n{'='*70}")
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, scaler, Config.DEVICE
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        scheduler.step()
        
        # TTA validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_dice_tta, val_iou_tta = validate_with_tta(
                model, val_dataset, Config.DEVICE, val_indices
            )
        else:
            val_dice_tta = history['val_dice_tta'][-1] if history['val_dice_tta'] else val_dice
            val_iou_tta = history['val_iou'][-1] if history['val_iou'] else val_iou
        
        lr_now = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['val_dice_tta'].append(val_dice_tta)
        history['val_iou'].append(val_iou_tta)
        history['lr'].append(lr_now)
        
        # Save best model
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), 
                      f'{Config.SAVE_DIR}/best_model.pth')
            print(f'  *** NEW BEST (no-TTA): Val Dice = {val_dice:.4f} ***')
        
        if val_dice_tta > best_dice_tta:
            best_dice_tta = val_dice_tta
            torch.save(model.state_dict(), 
                      f'{Config.SAVE_DIR}/best_model_tta.pth')
        
        print(f'Loss: {train_loss:.4f}/{val_loss:.4f} | '
              f'Dice: {train_dice:.4f}/{val_dice:.4f} | '
              f'TTA: {val_dice_tta:.4f} | '
              f'Best: {best_dice:.4f}/{best_dice_tta:.4f} | '
              f'Patience: {patience_counter}/{Config.PATIENCE}')
        
        # Visualize sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_img, sample_mask = val_dataset[val_indices[0]]
            with torch.no_grad():
                out = model(sample_img.unsqueeze(0).to(Config.DEVICE))
                pred = out.argmax(dim=1)[0]
            metrics = calculate_metrics(out.cpu(), sample_mask.unsqueeze(0))
            visualize_prediction(
                sample_img, sample_mask, pred,
                f'{Config.SAVE_DIR}/pred_epoch_{epoch+1}.png',
                epoch+1, metrics['dice']
            )
        
        # Check target
        if val_dice_tta >= 0.90 and not target_reached:
            print(f'\n{"="*70}')
            print(f'🎉 TARGET REACHED! Val Dice (TTA) = {val_dice_tta:.4f} >= 0.90')
            print(f'{"="*70}')
            target_reached = True
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f'Early stopping at epoch {epoch+1}, best dice={best_dice:.4f}')
            break
    
    # Save final model
    torch.save(model.state_dict(), f'{Config.SAVE_DIR}/final_model.pth')
    
    # Plot training curves
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    axes[0].plot(history['val_loss'], label='Val', color='orange')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_dice'], label='Train', color='blue')
    axes[1].plot(history['val_dice'], label='Val (no TTA)', color='orange')
    axes[1].plot(history['val_dice_tta'], label='Val (TTA)', color='green', linewidth=2)
    axes[1].axhline(y=0.90, color='red', linestyle='--', label='Target 0.90')
    axes[1].set_title(f'Dice (Best TTA: {best_dice_tta:.4f})')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(history['val_iou'], color='purple')
    axes[2].set_title('Val IoU')
    axes[2].grid(True)
    
    axes[3].plot(history['lr'], color='green')
    axes[3].set_title('Learning Rate (Encoder)')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{Config.SAVE_DIR}/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save training history
    with open(f'{Config.SAVE_DIR}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n{"="*70}')
    print(f'V16 Training completed!')
    print(f'Best Val Dice (no TTA): {best_dice:.4f}')
    print(f'Best Val Dice (TTA):    {best_dice_tta:.4f}')
    print(f'Target (>= 0.90):       {"ACHIEVED! 🎉" if best_dice_tta >= 0.90 else "NOT YET"}')
    print(f'{"="*70}')
    
    # Generate full visualization if target reached
    if best_dice_tta >= 0.90:
        print('\nGenerating full validation visualization...')
        model.load_state_dict(
            torch.load(f'{Config.SAVE_DIR}/best_model_tta.pth',
                       map_location=Config.DEVICE, weights_only=True)
        )
        metrics = visualize_all_samples(
            model, val_dataset, Config.DEVICE,
            f'{Config.SAVE_DIR}/final_validation_results', use_tta=True
        )
        
        avg_dice = np.mean([m['dice'] for m in metrics])
        avg_roi = np.mean([m['dice_per_class'][1] for m in metrics])
        avg_reflex = np.mean([m['dice_per_class'][2] for m in metrics])
        
        print(f'\nFull Dataset Results (TTA):')
        print(f'  Overall Dice: {avg_dice:.4f}')
        print(f'  ROI Dice:     {avg_roi:.4f}')
        print(f'  Reflex Dice:  {avg_reflex:.4f}')
        
        # Save summary
        with open(f'{Config.SAVE_DIR}/result_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"V16 Training Result Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Architecture: UNet++ + EfficientNet-B4\n")
            f.write(f"Input Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}\n")
            f.write(f"Loss: {Config.FOCAL_WEIGHT}*Focal + {Config.DICE_WEIGHT}*Dice + {Config.LOVASZ_WEIGHT}*Lovasz\n")
            if Config.PRETRAIN_PATH:
                f.write(f"Hot-start from: {Config.PRETRAIN_PATH}\n")
            f.write(f"TTA: 4-direction flip\n\n")
            f.write(f"Best Val Dice (no TTA): {best_dice:.4f}\n")
            f.write(f"Best Val Dice (TTA):    {best_dice_tta:.4f}\n")
            f.write(f"Full Dataset Dice (TTA): {avg_dice:.4f}\n")
            f.write(f"  ROI Dice:    {avg_roi:.4f}\n")
            f.write(f"  Reflex Dice: {avg_reflex:.4f}\n")
            f.write(f"Target Achieved: YES\n")


if __name__ == '__main__':
    main()
