# Eye Fundus Image Segmentation

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/ophthalmic-segmentation)](https://pypi.org/project/ophthalmic-segmentation/)
[![Python](https://img.shields.io/python/required version-toml-python-prj)](https://pypi.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model Dice](https://img.shields.io/badge/Val%20Dice-0.9554-brightgreen)](docs/results.md)

**A high-accuracy UNet++眼底图像分割模型 for multi-class retinal image segmentation.**

[English](README.md) | [中文](README_zh.md)

</div>

---

## ✨ Features

- **High Accuracy**: Val Dice = 0.9554 (TTA: 0.9398)
- **Multi-class Segmentation**: Background, ROI (Retinal Region), Valid Reflex
- **Pre-trained Models**: Ready-to-use checkpoints
- **Web Demo**: Interactive browser-based demo
- **Easy Training**: Single-file training script with comprehensive logging

## 📊 Performance

| Metric | Score |
|--------|-------|
| Val Dice (no TTA) | **0.9554** |
| Val Dice (4x TTA) | **0.9398** |
| ROI Dice | 0.9282 |
| Reflex Dice | 0.9660 |

## 🏗️ Architecture

```
UNet++ with EfficientNet-B4 encoder
├── Input: 448×448×3 RGB image
├── Encoder: EfficientNet-B4 (ImageNet pretrained)
├── Decoder: UNet++ dense skip connections
└── Output: 448×448×3 segmentation mask
```

## 🔧 Installation

```bash
pip install ophthalmic-segmentation
```

Or install from source:

```bash
git clone https://github.com/FeiFeiAlbert/ophthalmic-segmentation.git
cd ophthalmic-segmentation
pip install -e .
```

### Dependencies

```
torch >= 2.0
segmentation-models-pytorch
albumentations
opencv-python
pillow
matplotlib
```

## 🚀 Quick Start

### 1. Load Pre-trained Model

```python
import torch
import ophthalmic_segmentation as seg

# Load model
model = seg.load_model('unetpp_efficientnetb4', num_classes=3)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predict
image = seg.load_image('path/to/image.jpg', size=448)
mask = model.predict(image)
seg.visualize(image, mask)
```

### 2. Train Your Own Model

```python
from unet_train_v16 import train

# Train with default config
train(
    image_dir='data/images',
    label_dir='data/labels',
    save_dir='checkpoints'
)
```

## 🌐 Web Demo

Try it directly in your browser!

**Online Demo**: [Click to Open Demo](https://FeiFeiAlbert.github.io/ophthalmic-segmentation/demo/)

Or run locally:

```bash
cd demo
python -m http.server 8000
# Open http://localhost:8000 in browser
```

### Demo Features

- 📤 Upload your own fundus image
- 🔮 Real-time segmentation preview
- 📊 Side-by-side comparison (Original vs Masked)
- 🎨 Color-coded segmentation overlay

## 📂 Project Structure

```
ophthalmic-segmentation/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── demo/
│   ├── index.html          # Interactive demo
│   └── sample_image.jpg    # Demo image
├── ophthalmic_segmentation/
│   ├── __init__.py
│   ├── model.py            # Model definition
│   ├── data.py             # Dataset class
│   ├── losses.py           # Loss functions
│   └── utils.py            # Utility functions
├── scripts/
│   ├── train.py            # Training script
│   └── predict.py          # Inference script
├── checkpoints/
│   └── best_model.pth      # Pre-trained weights
├── docs/
│   ├── training_report.md   # Detailed training log
│   └── results.md          # Evaluation results
└── tests/
    └── test_model.py       # Unit tests
```

## 📖 Documentation

- [Training Report](docs/training_report.md) - Detailed training process and results
- [Results Analysis](docs/results.md) - Performance metrics and comparisons
- [API Reference](docs/api.md) - API documentation

## 🔬 Segmentation Classes

| Class | Color | Description |
|-------|-------|-------------|
| 0: Background | Black | Non-retinal regions |
| 1: ROI | Green | Retinal region of interest |
| 2: Valid Reflex | Red | Valid red reflex area |

## 📜 Citation

If this project helps your research, please cite:

```bibtex
@misc{ophthalmic-segmentation,
  title={Ophthalmic Fundus Image Segmentation with UNet++},
  author={Albert Long},
  year={2026},
  howpublished={\url{https://github.com/FeiFeiAlbert/ophthalmic-segmentation}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [UNet++](https://arxiv.org/abs/1912.05074)
