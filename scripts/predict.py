# -*- coding: utf-8 -*-
"""
V16 Inference Script - Real model inference with TTA

Features:
- Load trained V16 model
- 4-direction TTA for robust predictions
- Support for batch processing
- Visualize results

Usage:
    python predict.py --model checkpoints/best_model.pth --input image.jpg --output result.png
    python predict.py --model checkpoints/best_model.pth --input data/images --output results/
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ophthalmic_segmentation.model import create_model


# ==================== Configuration ====================
class Config:
    NUM_CLASSES = 3
    IMG_SIZE = 448
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = {0: 'background', 1: 'roi', 2: 'valid_reflex'}
    CLASS_COLORS = {
        0: [0, 0, 0],        # Black - Background
        1: [0, 255, 0],       # Green - ROI
        2: [255, 0, 0]        # Red - Valid Reflex
    }


# ==================== TTA Prediction ====================
def predict_with_tta(model, image_tensor, device):
    """
    4-direction TTA: average predictions from original and flipped versions
    """
    model.eval()
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)
        
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


def preprocess_image(image_path, img_size=448):
    """Load and preprocess image"""
    # Support Chinese paths
    with open(image_path, 'rb') as f:
        img_array = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_size = image.shape[:2][::-1]  # (width, height)
    
    # Resize to input size
    image_resized = cv2.resize(image, (img_size, img_size))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    
    return image_tensor, image, original_size


def postprocess_mask(mask, original_size):
    """Resize mask back to original size"""
    mask_resized = cv2.resize(
        mask.astype(np.uint8), 
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_resized


def create_colored_mask(mask, num_classes=3):
    """Create colored visualization of segmentation mask"""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        colored_mask[mask == class_id] = Config.CLASS_COLORS[class_id]
    
    return colored_mask


def create_overlay(image, mask, alpha=0.5):
    """Create overlay of image and segmentation mask"""
    overlay = image.copy()
    colored_mask = create_colored_mask(mask)
    
    # Add mask with transparency
    overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay


def visualize_results(image, mask, save_path=None):
    """Visualize original image, mask, and overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    cmap = plt.cm.colors.ListedColormap(['black', 'green', 'red'])
    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    colored_mask = create_colored_mask(mask)
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Results saved to: {save_path}")
    else:
        plt.show()


# ==================== Inference ====================
def inference_single(model, image_path, output_path=None, use_tta=True, img_size=448):
    """
    Run inference on a single image
    
    Args:
        model: trained segmentation model
        image_path: path to input image
        output_path: path to save result (optional)
        use_tta: whether to use TTA
        img_size: input image size
    
    Returns:
        mask: segmentation mask (numpy array)
    """
    # Preprocess
    image_tensor, image, original_size = preprocess_image(image_path, img_size)
    
    # Predict
    if use_tta:
        output = predict_with_tta(model, image_tensor, Config.DEVICE)
    else:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).to(Config.DEVICE))
    
    # Get mask
    mask = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Resize mask back to original size
    mask = postprocess_mask(mask, original_size)
    
    # Visualize and save
    if output_path:
        # Resize original for visualization
        image_vis = cv2.resize(image, (original_size[0], original_size[1]))
        visualize_results(image_vis, mask, output_path)
    
    return mask


def inference_batch(model, input_dir, output_dir, use_tta=True, img_size=448):
    """Run inference on all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images")
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}")
        
        output_path = os.path.join(output_dir, f'{image_path.stem}_result.png')
        
        try:
            inference_single(model, str(image_path), output_path, use_tta, img_size)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    print(f"\nAll results saved to: {output_dir}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description='V16 Segmentation Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='result.png',
                       help='Path to save result')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable TTA (Test-Time Augmentation)')
    parser.add_argument('--img-size', type=int, default=448,
                       help='Input image size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V16 UNet++ Segmentation Inference")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"TTA: {'Enabled' if not args.no_tta else 'Disabled'}")
    print("=" * 60)
    
    # Create model
    model = create_model('unetpp', 'efficientnet-b4', Config.NUM_CLASSES)
    
    # Load weights
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=Config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(Config.DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
    # Inference
    if os.path.isdir(args.input):
        inference_batch(model, args.input, args.output, 
                       use_tta=not args.no_tta, 
                       img_size=args.img_size)
    else:
        mask = inference_single(model, args.input, args.output,
                               use_tta=not args.no_tta,
                               img_size=args.img_size)
        print(f"\nSegmentation completed!")
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
