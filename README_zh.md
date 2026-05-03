# 眼底图像分割系统

<div align="center">

[![PyPI版本](https://img.shields.io/pypi/v/ophthalmic-segmentation)](https://pypi.org/project/ophthalmic-segmentation/)
[![Python版本](https://img.shields.io/python/required version-toml-python-prj)](https://pypi.org)
[![许可证](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![验证集Dice](https://img.shields.io/badge/Val%20Dice-0.9554-brightgreen)](docs/results.md)

**基于UNet++的高精度眼底图像多类别分割系统**

[English](README.md) | [中文](README_zh.md)

</div>

---

## ✨ 功能特性

- **高精度**: 验证集Dice系数 = 0.9554 (TTA: 0.9398)
- **多类别分割**: 支持背景、视网膜区域、有效反光点三类分割
- **预训练模型**: 开箱即用的模型权重
- **在线Demo**: 浏览器端交互式演示
- **简单易用**: 单文件训练脚本，附带完整日志

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 验证集 Dice (无TTA) | **0.9554** |
| 验证集 Dice (4x TTA) | **0.9398** |
| ROI Dice | 0.9282 |
| 反光点 Dice | 0.9660 |

## 🏗️ 模型架构

```
UNet++ with EfficientNet-B4 编码器
├── 输入: 448×448×3 RGB图像
├── 编码器: EfficientNet-B4 (ImageNet预训练)
├── 解码器: UNet++ 密集跳跃连接
└── 输出: 448×448×3 分割掩码
```

## 🔧 安装

```bash
pip install ophthalmic-segmentation
```

或从源码安装:

```bash
git clone https://github.com/FeiFeiAlbert/ophthalmic-segmentation.git
cd ophthalmic-segmentation
pip install -e .
```

### 依赖项

```
torch >= 2.0
segmentation-models-pytorch
albumentations
opencv-python
pillow
matplotlib
```

## 🚀 快速开始

### 1. 加载预训练模型

```python
import torch
import ophthalmic_segmentation as seg

# 加载模型
model = seg.load_model('unetpp_efficientnetb4', num_classes=3)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 预测
image = seg.load_image('path/to/image.jpg', size=448)
mask = model.predict(image)
seg.visualize(image, mask)
```

### 2. 训练自己的模型

```python
from ophthalmic_segmentation import train

# 使用默认配置训练
train(
    image_dir='data/images',
    label_dir='data/labels',
    save_dir='checkpoints'
)
```

## 🌐 在线演示

直接在浏览器中体验！

**在线Demo**: [点击打开演示](https://your-demo-url.com)

本地运行:

```bash
cd demo
python -m http.server 8000
# 在浏览器打开 http://localhost:8000
```

### 演示功能

- 📤 上传自己的眼底图像
- 🔮 实时分割预览
- 📊 原图与分割结果对比
- 🎨 彩色分割叠加显示

## 📂 项目结构

```
ophthalmic-segmentation/
├── README.md
├── README_zh.md
├── LICENSE
├── requirements.txt
├── setup.py
├── demo/
│   ├── index.html          # 交互式演示
│   └── sample_image.jpg     # 演示用图像
├── ophthalmic_segmentation/
│   ├── __init__.py
│   ├── model.py             # 模型定义
│   ├── data.py              # 数据集类
│   ├── losses.py            # 损失函数
│   └── utils.py             # 工具函数
├── scripts/
│   ├── train.py             # 训练脚本
│   └── predict.py           # 推理脚本
├── checkpoints/
│   └── best_model.pth       # 预训练权重
├── docs/
│   ├── training_report.md   # 详细训练报告
│   └── results.md           # 评估结果
└── tests/
    └── test_model.py        # 单元测试
```

## 📖 文档

- [训练报告](docs/training_report.md) - 详细的训练过程和结果
- [结果分析](docs/results.md) - 性能指标和对比
- [API参考](docs/api.md) - API文档

## 🔬 分割类别说明

| 类别 | 颜色 | 说明 |
|------|------|------|
| 0: 背景 | 黑色 | 非视网膜区域 |
| 1: ROI | 绿色 | 视网膜感兴趣区域 |
| 2: 有效反光点 | 红色 | 有效红反射区域 |

## 📜 引用

如果该项目对你的研究有帮助，请引用:

```bibtex
@misc{ophthalmic-segmentation,
  title={基于UNet++的眼底图像分割系统},
  author={你的名字},
  year={2026},
  howpublished={\url{https://github.com/YOUR_USERNAME/ophthalmic-segmentation}}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [UNet++](https://arxiv.org/abs/1912.05074)
