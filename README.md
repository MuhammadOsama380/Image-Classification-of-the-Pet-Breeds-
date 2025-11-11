# Image Classification of Pet Breeds using PyTorch

**Author:** Muhammad Osama  
**Course:** Deep Learning with PyTorch (Fanshawe College)  
**Institution:** Fanshawe College  
**LinkedIn:** [muhammad-osama-872328202](https://www.linkedin.com/in/muhammad-osama-872328202)  
**GitHub:** [MuhammadOsama380](https://github.com/MuhammadOsama380)

---

## Project Overview
This project implements **image classification** using **deep learning models** to classify pet breeds (cats and dogs) from the **Oxford-IIIT Pet Dataset**.  
It explores both **custom-built CNNs** and **transfer learning** with pre-trained **ResNet models** (ResNet18, ResNet34, ResNet50).  

The project demonstrates a complete deep-learning workflow including data preprocessing, augmentation, model training, validation, and comparison across architectures.  
ResNet50 achieved the highest accuracy (≈92.4%), outperforming all other models.

---

## Problem Statement
To build a deep-learning model capable of **classifying 37 pet breeds** (cats and dogs) under real-world conditions such as varying **lighting, poses, and backgrounds**.  
The goal is to compare performance between:
- A **custom CNN** built from scratch, and  
- **Pre-trained ResNet** models using transfer learning.

---

## Dataset Overview
- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)  
- **Images:** ~7,400 (≈200 per class)  
- **Classes:** 37 (cats and dogs)  
- **Splits:**  
  - Training: 70%  
  - Validation: 15%  
  - Testing: 15%  

### Preprocessing & Augmentation
- Resize: shorter edge → 256 px  
- RandomCrop: 224×224  
- RandomHorizontalFlip (50%)  
- RandomRotation: ±15°  
- Normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]  

---

## Model Architectures

### Custom CNN
A simple yet efficient CNN built from scratch:
- 4 Convolutional Layers (32 → 256 filters)  
- BatchNorm + ReLU after each layer  
- MaxPooling after each block  
- AdaptiveAvgPool2d → Fully Connected Layer (128 units)  
- Dropout (p = 0.5)  
- Output layer (37-class Softmax)

**Test Accuracy:** ~30.46% (baseline model)

### ResNet18
- 18 layers, trained using transfer learning  
- All layers frozen except the final fully connected layer  
- Pretrained on ImageNet (≈1.2M images)  
- Excellent performance with low computational cost  

**Accuracy:** ~89.76%

### ResNet34
- 34 layers (deeper feature extraction than ResNet18)  
- Fine-tuned final layer for 37-class prediction  
- Balanced trade-off between speed and accuracy  

**Accuracy:** ~90.57%

### ResNet50
- 50-layer deep CNN using bottleneck residual blocks  
- Higher abstraction capacity for fine-grained features  
- Pretrained on ImageNet, retrained final FC layer  

**Accuracy:** ~92.38% — best performing model

---

## Training Methodology
- **Framework:** PyTorch  
- **Optimizer:** Adam (lr = 0.001)  
- **Loss Function:** CrossEntropyLoss  
- **Scheduler:** StepLR (step_size = 7, gamma = 0.1)  
- **Epochs:** 30 (Custom CNN), 20-23 (ResNet models)  
- **Early Stopping:** patience = 5  
- **Device:** GPU (CUDA)  

Validation and training loss were monitored at each epoch.  
Best weights were saved at the highest validation accuracy.

---

## Evaluation

| Model | Accuracy | Depth | Notes |
|--------|-----------|--------|--------|
| Custom CNN | 30.46% | 4 Conv Layers | Baseline |
| ResNet18 | 89.76% | 18 | Fast and stable |
| ResNet34 | 90.57% | 34 | Balanced |
| ResNet50 | 92.38% | 50 | Best performing |

### Metrics:
- Test Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-Score  

---

## Observations
- **Transfer learning** drastically improved performance over scratch-built CNNs.  
- **ResNet34** offers optimal efficiency for medium compute budgets.  
- **ResNet50** provides the highest accuracy but requires more memory and training time.  
- **StepLR scheduler** and **early stopping** effectively prevented overfitting.

---

## Future Improvements
- Fine-tune deeper layers instead of freezing all  
- Experiment with EfficientNet or Vision Transformers (ViT)  
- Use Grad-CAM for interpretability  
- Employ ensemble models for higher accuracy  
- Apply Bayesian or Optuna hyperparameter optimization  

---

## Repository Structure
```
PyTorch-Pet-Breed-Classification/
│
├── data/
│   └── (Oxford-IIIT Pet Dataset)
│
├── notebooks/
│   └── Project_Pytorch_Muhammad_Osama.ipynb
│
├── reports/
│   ├── Project_Report_Muhammad_Osama.pdf
│   ├── Project_Proposal_Muhammad_Osama.pdf
│
├── presentation/
│   └── Project_Presentation_Muhammad_Osama_Updated.pptx
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Tools & Libraries
- PyTorch / Torchvision  
- NumPy / Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  
- GPU (CUDA) Support

---

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Developed by **Muhammad Osama** as part of the **Deep Learning with PyTorch** course at **Fanshawe College**.  
If you found this project insightful, please ⭐ the repository or connect with me on [LinkedIn](https://www.linkedin.com/in/muhammad-osama-872328202).
