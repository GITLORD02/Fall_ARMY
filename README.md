-----
model_name: Wheat Anomaly Detection Model
tags:
  - pytorch
  - resnet
  - agriculture
  - anomaly-detection
  - image-classification
  - wheat-disease-detection
  - pest-detection
  - agricultural-ai
license: apache-2.0
library_name: pytorch
datasets:
  - wheat-dataset   # Replace with the actual dataset name on Hugging Face if available
model_type: resnet50
preprocessing:
  - resize: 256
  - center_crop: 224
  - normalize: [0.485, 0.456, 0.406]
  - normalize_std: [0.229, 0.224, 0.225]
framework: pytorch
task: image-classification
pipeline_tag: image-classification

-------

# Maize Anomaly Detection Model

## Model Overview

This model is trained to detect anomalies in Maize crops, such as pest infections (e.g., Fall Armyworm), diseases, or nutrient deficiencies. The model is based on the **ResNet50** architecture and was fine-tuned on a dataset of wheat images.

## Model Details

- **Model Architecture**: ResNet50
- **Number of Classes**: 2 (Fall Armyworm, Healthy Wheat)
- **Input Shape**: 224x224 pixels, 3 channels (RGB)
- **Training Framework**: PyTorch
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 20
- **Batch Size**: 32

## Training

The model was fine-tuned using a balanced dataset with images of healthy wheat and wheat infected by fall armyworms. The training involved transferring knowledge from a pretrained ResNet50 model and adjusting the final classification layer for the binary classification task.

### Dataset

The model was trained on a dataset hosted on Hugging Face. You can access it here:

- **Dataset**: `your_huggingface_username/your_dataset_name`

## How to Use

To load and use this model in PyTorch, follow the steps below:

### 1. Load the Model

```python
import torch
import timm

# Load the pre-trained model (fine-tuned ResNet50 for wheat anomaly detection)
model = timm.create_model("resnet50", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("path_to_saved_model.pth"))
model.eval()

