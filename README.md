# CIFAR-10 Image Classifier

A custom Convolutional Neural Network (CNN) built with PyTorch to classify images from the CIFAR-10 dataset, achieving **78.1% test accuracy**.

## Project Overview

This project implements a deep learning solution for image classification on the CIFAR-10 dataset. The model was designed from scratch and successfully exceeds the target accuracy of 70%, demonstrating competitive performance with established benchmark models.

### Key Results
- **Test Accuracy:** 78.1%
- **Target Accuracy:** 70% ✅ (exceeded by 8.1%)
- **Training Time:** ~7 epochs with early stopping
- **Model Type:** Custom 4-layer CNN with batch normalization

## Dataset

**CIFAR-10** consists of 60,000 32×32 color images across 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Dataset Split:**
- Training: (90%)
- Validation: (10%)
- Test: 10,000 images

## 🏗️ Model Architecture

```
Input (32×32×3)
    ↓
Conv2d (3→64) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d (64→128) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d (128→256) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d (256→512) + BatchNorm + ReLU + MaxPool
    ↓
Flatten
    ↓
Linear (512×2×2 → 512) + ReLU + Dropout(0.5)
    ↓
Linear (512 → 10)
    ↓
Output (10 classes)
```

### Model Features
- **Batch Normalization:** Stabilizes training and improves convergence
- **Dropout (0.5):** Prevents overfitting
- **Data Augmentation:** Random horizontal flip, random crop with padding
- **Early Stopping:** Prevents overtraining (patience=5)

## 🛠️ Requirements

```bash
# Core Dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/horpeazy/cifar10-image-classifier.git
cd cifar10-image-classifier

# Install dependencies
pip install torch torchvision numpy matplotlib

# Run the notebook
jupyter notebook CIFAR-10_Image_Classifier-STARTER.ipynb
```

## Usage

### Training the Model

1. Open `CIFAR-10_Image_Classifier-STARTER.ipynb`
2. Run all cells sequentially
3. The dataset will be automatically downloaded to `./data/`
4. Training will begin with progress updates every 100 batches

### Making Predictions

```python
import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = Cifar10CNN()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Make prediction
image = Image.open('your_image.jpg')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f'Predicted class: {classes[predicted.item()]}')
```

## 📈 Training Progress

- **Epoch 0:** Training Loss: 1.377 | Validation Accuracy: 46.9%
- **Epoch 3:** Training Loss: 0.859 | Validation Accuracy: 69.3%
- **Epoch 6:** Training Loss: 0.641 | Validation Accuracy: 78.0%
- **Final Test Accuracy:** 78.1%

## 💡 Build vs Buy Recommendation

**Recommendation: BUILD**

Based on the evaluation results, I recommend building a custom image classification solution rather than purchasing an off-the-shelf product. Our custom CNN achieved 78.1% test accuracy, exceeding the 70% target by a significant margin and matching the performance of established benchmark models from 2010. The model trained efficiently in approximately 7 epochs with early stopping, demonstrating that we can develop production-ready solutions without requiring extensive computational resources or training time.

From a strategic perspective, building our own solution provides several critical advantages. We maintain full control over the model architecture and can customize it for specific use cases as requirements evolve. There are no vendor dependencies, API rate limits, or recurring subscription costs, and all data processing happens within our infrastructure, ensuring complete privacy and security. The successful development of this model also demonstrates that our team possesses the technical capability to design, train, and deploy deep learning solutions independently.

## Benchmarks

| Model | Year | Accuracy |
|-------|------|----------|
| **Our CNN** | **2024** | **78.1%** ✅ |
| Deep Belief Networks | 2010 | 78.9% |
| Maxout Networks | 2013 | 90.6% |
| Wide ResNets | 2016 | 96.0% |
| GPipe | 2018 | 99.0% |

## 🔧 Hardware Requirements

- **Minimum:** CPU-only (slower training)
- **Recommended:** GPU with 4GB+ VRAM
- **Tested on:** Apple M-series (MPS backend), NVIDIA CUDA GPUs

## 📝 Project Structure

```
CIFAR-10-Image-Classifier/
├── CIFAR-10_Image_Classifier-STARTER.ipynb  # Main notebook
├── checkpoint.pth                            # Saved model weights
├── data/                                     # Dataset (auto-downloaded)
├── .gitignore                                # Git ignore rules
└── README.md                                 # This file
```

## 🎓 Learning Outcomes

This project demonstrates:
- CNN architecture design for image classification
- Data augmentation techniques
- Training optimization with batch normalization and dropout
- Early stopping to prevent overfitting
- Model evaluation and comparison with benchmarks
- Production-ready deployment considerations

## 📄 License

MIT License - feel free to use this project for educational purposes.

## Author

**Hope**
- GitHub: [@horpeazy](https://github.com/horpeazy)

## Acknowledgments

- CIFAR-10 dataset: [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch framework: [PyTorch Team](https://pytorch.org/)

