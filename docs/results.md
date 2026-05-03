# Ophthalmic Fundus Image Segmentation - Results

## Model Performance Summary

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Val Dice (no TTA)** | **0.9554** | ≥ 0.90 | ✅ **Achieved** |
| **Val Dice (4x TTA)** | **0.9398** | ≥ 0.90 | ✅ **Achieved** |
| **Full Dataset Dice (TTA)** | **0.9471** | - | Reference |
| **ROI Dice (TTA)** | **0.9282** | - | Reference |
| **Reflex Dice (TTA)** | **0.9660** | - | Reference |

## Dice Coefficient by Version

| Version | Architecture | Backbone | Input Size | Val Dice (TTA) |
|---------|--------------|----------|------------|-----------------|
| v10 | UNet | ResNet34 | 256 | 0.7283 |
| v11 | DeepLabV3+ | ResNet101 | 320 | 0.7754 |
| v12 | DeepLabV3+ | EfficientNet-B4 | 384 | 0.8498 |
| v13 | UNet++ | EfficientNet-B4 | 448 | 0.8790 |
| v15 | UNet++ | EfficientNet-B4 | 512 | 0.8175 (failed) |
| **v16** | **UNet++** | **EfficientNet-B4** | **448** | **0.9398** ✅ |

## Visualization Examples

Sample predictions on validation set:

```
examples/
├── sample_01_original.png
├── sample_01_mask.png
├── sample_01_overlay.png
├── sample_02_original.png
├── sample_02_mask.png
└── sample_02_overlay.png
```

## Training Curves

![Training History](training_history.png)

- **Training Loss**: Converged to ~0.28
- **Validation Loss**: Converged to ~0.29
- **Training Dice**: Converged to ~0.96
- **Validation Dice**: Converved to ~0.94

## Inference Speed

| Input Size | Batch Size | GPU | Time/Image | With TTA |
|-------------|------------|-----|-------------|----------|
| 448×448 | 4 | V100 | ~50ms | ~200ms |

## Usage

### Load Pretrained Model

```python
import torch
from ophthalmic_segmentation.model import create_model

# Load model
model = create_model('unetpp', 'efficientnet-b4', num_classes=3)
model.load_state_dict(torch.load('checkpoints/best_model_tta.pth'))
model.eval()

# Predict
# ... (see README.md for full example)
```

### Evaluate on Your Own Dataset

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model_tta.pth \
    --image-dir /path/to/images \
    --label-dir /path/to/labels \
    --output-dir results/
```

## Acknowledgments

- Dataset: Preprocessed fundus images (PIL version)
- Preprocessing: Cropped and aligned
- ROI & Reflex: Manually annotated

---
*For detailed training process, see [training_report.md](training_report.md)*
