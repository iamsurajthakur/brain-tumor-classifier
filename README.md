# Multi Class Brain Tumor Classification using ResNet50

A deep learning image classifier that detects and categorizes brain tumors from MRI scans using a fine-tuned ResNet50 model trained with PyTorch.

---

## Overview

This project uses transfer learning on a pretrained ResNet50 model to classify brain MRI images into **4 categories**. The model is fine-tuned with data augmentation, cosine annealing scheduling, early stopping, and validation tracking to maximize generalization on unseen scans.

---

## Project Structure

```
├── data/
│   ├── Training/          # Training images (organized by class subfolder)
│   └── Testing/           # Test images (organized by class subfolder)
├── train.py               # Main training script
├── brain_tumor_model_best.pth   # Best model checkpoint (saved during training)
├── brain_tumor_model.pth        # Final model weights
├── classes.json           # Class label mapping
└── training_history.json  # Loss and accuracy logs per epoch
```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision

Install dependencies:

```bash
pip install torch torchvision
```

---

## Dataset

Images should be organized in `ImageFolder` format — each class in its own subfolder:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The training set is automatically split **75% train / 25% validation** at runtime.

---

## Model Architecture

| Component      | Detail                                      |
|----------------|---------------------------------------------|
| Base model     | ResNet50 (ImageNet pretrained)              |
| Frozen layers  | Layers 1 & 2                                |
| Fine-tuned     | Layer 3, Layer 4, and the final FC head     |
| Classifier     | `Dropout(0.4)` → `Linear(2048 → 4)`        |
| Loss function  | Cross-Entropy Loss                          |
| Optimizer      | Adam (layer-wise learning rates)            |
| Scheduler      | Cosine Annealing LR                         |

### Learning Rates

| Layer   | Learning Rate |
|---------|--------------|
| Layer 3 | `3e-5`       |
| Layer 4 | `5e-5`       |
| FC head | `3e-4`       |

---

## Training

Run the training script:

```bash
python train.py
```

**Training features:**
- **Data augmentation** on training split: horizontal flips, rotation, affine translation, color jitter
- **Early stopping** with a patience of 15 epochs
- **Best model checkpointing** — saves whenever validation accuracy improves
- **Cosine Annealing LR** scheduler over 30 epochs

Training logs are printed to the console in this format:

```
Epoch | Train Loss | Train Acc | Val Loss | Val Acc |       LR
    1 |     0.5231 |   78.34%  |   0.4812 |  81.20% | 0.000300
          New best val accuracy: 81.20% — model saved
```

---

## Outputs

After training, the following files are saved:

| File | Description |
|------|-------------|
| `brain_tumor_model_best.pth` | Weights at peak validation accuracy |
| `brain_tumor_model.pth` | Final epoch weights |
| `classes.json` | Class index → label mapping |
| `training_history.json` | Per-epoch train/val loss and accuracy |

---

## Evaluation

The best checkpoint is automatically loaded at the end of training and evaluated on the held-out test set:

```
Test Accuracy: XX.XX%
Best Val Accuracy seen during training: XX.XX%
```

---

## Inference (Example)

```python
import torch, json
from torchvision import models, transforms
from PIL import Image
from torch import nn

# Load class names
with open('classes.json') as f:
    classes = json.load(f)

# Rebuild model
model = models.resnet50()
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, 4))
model.load_state_dict(torch.load('brain_tumor_model_best.pth', map_location='cpu'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('your_mri_image.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

print(f"Predicted class: {classes[pred]}")
```

---

## 📌 Notes

- Training is done on **GPU** if available, otherwise falls back to CPU.
- The dataset split is random each run; set a seed for reproducibility if needed.
- Model was designed for a **4-class** dataset. Adjust `nn.Linear(2048, N)` for a different number of classes.
