# Insect Genus Classification Using Transfer Learning and Hybrid Ensemble Models

This repository contains the source code developed as part of my MSc thesis. The work has also been extended into the research manuscript entitled:

**"Insect Genus Classification Using Transfer Learning and Hybrid Ensemble Models."**

The project investigates the use of deep learning and transfer learning for automated insect genus classification using image datasets collected from publicly available sources. A custom Convolutional Neural Network (CNN), four transfer learning models, and a hybrid ensemble model were developed and evaluated.

---

# Insect Genera

The models classify images into the following seven insect genera:

- Aedes
- Anopheles
- Culex
- Cimex
- Ctenocephalides
- Pediculus
- Triatoma

---

# Models Implemented

- Custom Convolutional Neural Network (CNN)
- InceptionV3
- DenseNet201
- EfficientNet-B5
- ConvNeXt-Large
- Hybrid Ensemble (ConvNeXt-Large + EfficientNet-B5)

---

# Dataset

The dataset was compiled from multiple publicly available repositories:

- Chula Mosquito Classification Dataset (Kaggle)
- GBIF
- Mendeley Data
- iNaturalist

The final dataset contains **12,969 images** from seven insect genera.

---

# Development Environment

- Python 3.10
- PyTorch
- Google Colab
- NVIDIA T4 / A100 GPU

---

# Repository Contents

```
├── notebooks/
├── models/
├── outputs/
├── README.md
├── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/zbumar17/insect-classification-deep-learning.git

cd insect-classification-deep-learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

The project requires:

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- opencv-python
- pillow

Install using:

```bash
pip install -r requirements.txt
```

---

# Usage

1. Download the datasets from the sources listed above.
2. Organize the dataset into training, validation, and testing folders.
3. Open the required notebook in Google Colab or Jupyter Notebook.
4. Train the desired model.
5. Evaluate the trained model.
6. Run the hybrid ensemble notebook to reproduce the ensemble results.

---

# Methodology

The workflow consists of:

1. Data collection
2. Data preprocessing
3. Data augmentation
4. Model training
5. Transfer learning
6. Hybrid ensemble learning
7. Performance evaluation

---

# Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall Curves

---

# Citation

If you use this repository in your research, please cite:

**Umar, Z. B., et al.**  
*Insect Genus Classification Using Transfer Learning and Hybrid Ensemble Models.*  
(Full citation will be updated after publication.)

---

# License

This repository is intended for academic and research purposes.
