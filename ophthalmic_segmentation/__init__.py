# -*- coding: utf-8 -*-
"""
Ophthalmic Fundus Image Segmentation - Core Module
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2


__version__ = "1.0.0"


class FundusSegmenter:
    """Fundus image segmentation model wrapper"""

    CLASS_NAMES = {0: 'background', 1: 'roi', 2: 'valid_reflex'}
    CLASS_COLORS = {
        0: [0, 0, 0],        # Black - Background
        1: [0, 255, 0],       # Green - ROI
        2: [255, 0, 0]        # Red - Valid Reflex
    }

    def __init__(self, model=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        if model:
            self.model.to(self.device)
            self.model.eval()

    @classmethod
    def create_model(cls, architecture='unetpp', encoder='efficientnet-b4', num_classes=3):
        """Create segmentation model"""
        if architecture == 'unetpp':
            model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=num_classes
            )
        elif architecture == 'unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=num_classes
            )
        elif architecture == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=num_classes
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return cls(model)

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        return self

    def predict(self, image, input_size=448, apply_tta=False):
        """
        Predict segmentation mask for an image

        Args:
            image: PIL Image or numpy array (H, W, 3)
            input_size: Target input size
            apply_tta: Whether to apply test-time augmentation

        Returns:
            mask: numpy array (H, W) with class labels
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        original_size = image.shape[:2]

        # Preprocess
        image_resized = cv2.resize(image, (input_size, input_size))
        image_tensor = self._to_tensor(image_resized)

        # Predict
        if apply_tta:
            mask = self._predict_tta(image_tensor)
        else:
            with torch.no_grad():
                output = self.model(image_tensor)
                mask = output.argmax(dim=1).squeeze().cpu().numpy()

        # Resize back to original size
        mask = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]))
        return mask

    def _to_tensor(self, image):
        """Convert numpy array to torch tensor"""
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor

    def _predict_tta(self, tensor):
        """Test-time augmentation prediction"""
        with torch.no_grad():
            # Original
            output = self.model(tensor)

            # Horizontal flip
            tensor_flip_h = torch.flip(tensor, dims=[3])
            output_flip_h = self.model(tensor_flip_h)
            output_flip_h = torch.flip(output_flip_h, dims=[3])

            # Vertical flip
            tensor_flip_v = torch.flip(tensor, dims=[2])
            output_flip_v = self.model(tensor_flip_v)
            output_flip_v = torch.flip(output_flip_v, dims=[2])

            # Both flips
            tensor_flip_hv = torch.flip(tensor, dims=[2, 3])
            output_flip_hv = self.model(tensor_flip_hv)
            output_flip_hv = torch.flip(output_flip_hv, dims=[2, 3])

            # Average
            output_avg = (output + output_flip_h + output_flip_v + output_flip_hv) / 4
            mask = output_avg.argmax(dim=1).squeeze().cpu().numpy()

        return mask

    def visualize(self, image, mask, alpha=0.5):
        """
        Create overlay visualization

        Args:
            image: PIL Image or numpy array
            mask: segmentation mask
            alpha: overlay transparency

        Returns:
            overlay: numpy array with colored overlay
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        overlay = image.copy()

        for class_id, color in self.CLASS_COLORS.items():
            mask_bool = (mask == class_id)
            overlay[mask_bool] = (
                alpha * np.array(color) + (1 - alpha) * overlay[mask_bool]
            ).astype(np.uint8)

        return overlay.astype(np.uint8)


def load_image(path, size=448):
    """Load and preprocess image"""
    image = Image.open(path).convert('RGB')
    image = image.resize((size, size), Image.BILINEAR)
    return image


def visualize_results(image, mask, save_path=None):
    """Visualize segmentation results"""
    import matplotlib.pyplot as plt

    if isinstance(image, str):
        image = Image.open(image)
        image = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Mask
    cmap = plt.cm.colors.ListedColormap(['black', 'green', 'red'])
    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    # Overlay
    segmenter = FundusSegmenter()
    overlay = segmenter.visualize(image, mask)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# Convenience functions
def load_model(architecture='unetpp', encoder='efficientnet-b4', num_classes=3):
    """Load a new segmentation model"""
    return FundusSegmenter.create_model(architecture, encoder, num_classes)
