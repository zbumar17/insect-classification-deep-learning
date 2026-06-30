# Insect Genus Classification Using Transfer Learning and Hybrid Ensemble Models

This repository contains the source code, experimental workflow, and reproducibility materials developed as part of an MSc thesis and extended into the research manuscript entitled:

**“Insect Genus Classification Using Transfer Learning and Hybrid Ensemble Models.”**

The project investigates the use of deep learning, transfer learning, and hybrid ensemble learning for automated insect genus classification using image datasets collected from publicly available third-party repositories. A custom Convolutional Neural Network (CNN), four transfer learning models, and a hybrid ensemble model were developed, trained, evaluated, and compared.



## 1. Project Overview

Accurate identification of medically and environmentally important insect genera is useful for entomological research, vector surveillance, public health monitoring, and automated image-based biodiversity analysis.

This project classifies insect images into seven genera using convolutional deep learning models. The study compares the performance of a custom CNN, individual transfer learning architectures, and a hybrid ensemble model that combines the outputs of the best-performing models.

The main goals of this repository are to:

* Provide reproducible source code for insect genus classification.
* Document the dataset sources and preparation workflow.
* Train and evaluate multiple deep learning architectures.
* Compare individual transfer learning models with a hybrid ensemble approach.
* Support reproducibility for the associated AI Application manuscript.



## 2. Insect Genera

The models classify images into the following seven insect genera:

1. **Aedes**
2. **Anopheles**
3. **Culex**
4. **Cimex**
5. **Ctenocephalides**
6. **Pediculus**
7. **Triatoma**



## 3. Models Implemented

The following models were implemented and evaluated:

1. **Custom Convolutional Neural Network (CNN)**
2. **InceptionV3**
3. **DenseNet201**
4. **EfficientNet-B5**
5. **ConvNeXt-Large**
6. **Hybrid Ensemble Model**

   * ConvNeXt-Large + EfficientNet-B5

The hybrid ensemble model combines predictions from ConvNeXt-Large and EfficientNet-B5 to improve classification robustness and comparative performance.



## 4. Dataset Information

The final dataset contains **12,969 images** from seven insect genera.

The images were compiled from publicly available third-party repositories, including mosquito image datasets and open biodiversity image sources.

### 4.1 Dataset Sources

The data used in this study are available from the following sources:

1. **Chula Mosquito Classification Dataset**

   * Source: Kaggle
   * URL: https://www.kaggle.com/datasets/cyberthorn/chula-mosquito-classification?select=Ae-aegypti

2. **GBIF**

   * Source: Global Biodiversity Information Facility
   * URL: https://www.gbif.org/species/4987991

3. **Mendeley Data – Dataset of Vector Mosquito Images**

   * Source: Mendeley Data
   * URL: https://data.mendeley.com/datasets/88s6fvgg2p/4
   * DOI: 10.17632/88s6fvgg2p.4

4. **iNaturalist – Pediculus**

   * Source: iNaturalist
   * URL: https://www.inaturalist.org/taxa/47437-Pediculus



## 5. Third-Party Data Acknowledgement and Licensing

The image data used in this project were obtained from publicly available third-party repositories. The authors do not claim ownership of the original images. Copyright, ownership, and licensing terms remain with the original dataset owners, repositories, and image contributors.

Users of this repository must verify and comply with the licensing terms of each dataset before reusing, redistributing, or publishing any image data.

For GBIF and iNaturalist images, license and attribution information should be checked at the individual occurrence or observation level because licenses may vary between image contributors.

The Mendeley Data mosquito image dataset is available as:

**Dataset of Vector Mosquito Images, Version 4**
DOI: **10.17632/88s6fvgg2p.4**

If any representative images are reproduced in a manuscript, report, presentation, or publication, users must ensure that the specific image license permits reproduction and that proper attribution is provided.



## 6. Dataset Structure

After downloading and preparing the data, organize the dataset using the following directory structure:


dataset/
├── train/
│   ├── Aedes/
│   ├── Anopheles/
│   ├── Culex/
│   ├── Cimex/
│   ├── Ctenocephalides/
│   ├── Pediculus/
│   └── Triatoma/
├── val/
│   ├── Aedes/
│   ├── Anopheles/
│   ├── Culex/
│   ├── Cimex/
│   ├── Ctenocephalides/
│   ├── Pediculus/
│   └── Triatoma/
└── test/
    ├── Aedes/
    ├── Anopheles/
    ├── Culex/
    ├── Cimex/
    ├── Ctenocephalides/
    ├── Pediculus/
    └── Triatoma/


Each genus folder should contain image files belonging to that class.

Recommended image formats:


.jpg
.jpeg
.png


## 7. Data Preparation

The general dataset preparation workflow consists of:

1. Downloading images from the listed public repositories.
2. Removing corrupted or unreadable images.
3. Sorting images into the seven genus-level classes.
4. Resizing images according to the input size required by each model.
5. Splitting the dataset into training, validation, and testing subsets.
6. Applying data augmentation to the training subset only.
7. Using the validation set for model selection and checkpoint monitoring.
8. Using the test set only for final performance evaluation.



## 8. Preprocessing and Augmentation

The preprocessing steps include:

* Image loading.
* RGB conversion where required.
* Resizing to the required model input dimensions.
* Normalization using model-compatible preprocessing.
* Conversion to tensor format.
* Label encoding for the seven insect genera.

Data augmentation may include:

* Random horizontal flipping.
* Random rotation.
* Random resized cropping.
* Color jittering.
* Random affine transformations.
* Normalization.

Augmentation is applied only to the training set to improve model generalization. Validation and testing images are not augmented, except for deterministic resizing and normalization.



## 9. Repository Contents


insect-classification-deep-learning/
insect-classification-deep-learning/
├── notebooks/
│   ├── ConvNeXt.ipynb
│   ├── Hybrid_conv_eff.ipynb
│   ├── Inception_tl.ipynb
│   ├── denseNet_tl.ipynb
│   └── efficientNet_tl.ipynb  
├── README.md
└── requirements.txt
Code Information
notebooks/

This folder contains the experimental notebooks used to train and evaluate the transfer learning and hybrid ensemble models.

The available notebooks are:

ConvNeXt.ipynb

This notebook implements the ConvNeXt-Large transfer learning model for seven-class insect genus classification. It includes dataset loading, preprocessing, model preparation, training, validation, and final evaluation.

efficientNet_tl.ipynb

This notebook implements the EfficientNet-B5 transfer learning model. It includes the full workflow for training and evaluating EfficientNet-B5 on the insect genus image dataset.

denseNet_tl.ipynb

This notebook implements the DenseNet201 transfer learning model. It includes preprocessing, model fine-tuning, performance evaluation, and generation of classification metrics.

Inception_tl.ipynb

This notebook implements the InceptionV3 transfer learning model. It trains and evaluates the model using the prepared insect genus dataset.

Hybrid_conv_eff.ipynb

This notebook implements the hybrid ensemble model using ConvNeXt-Large and EfficientNet-B5. It loads or uses the predictions from the trained ConvNeXt and EfficientNet models and combines them to produce the final ensemble classification results.

The hybrid ensemble is used to compare whether combining two high-performing transfer learning models improves classification performance compared with using individual models alone.


## 10. Code Information

### 10.1 `notebooks/`

This folder contains the experimental notebooks used to train and evaluate each model.

Suggested notebook purpose:

* `custom_cnn.ipynb`
  Trains and evaluates the custom CNN baseline model.

* `inceptionv3_tl.ipynb`
  Implements transfer learning using InceptionV3.

* `densenet201_tl.ipynb`
  Implements transfer learning using DenseNet201.

* `efficientnet_b5_tl.ipynb`
  Implements transfer learning using EfficientNet-B5.

* `convnext_large_tl.ipynb`
  Implements transfer learning using ConvNeXt-Large.

* `hybrid_convnext_efficientnet.ipynb`
  Implements the hybrid ensemble model using ConvNeXt-Large and EfficientNet-B5 predictions.



## 11. Development Environment

The experiments were developed and executed using the following environment:


Python 3.10
PyTorch
Torchvision
Google Colab
NVIDIA T4 GPU
NVIDIA A100 GPU


The notebooks can be executed using either:

* Google Colab, or
* A local Jupyter Notebook environment with GPU support.


## 12. Requirements

The project requires the following Python libraries:


torch
torchvision
numpy
matplotlib
scikit-learn
opencv-python
pillow


Install all dependencies using:


pip install -r requirements.txt


A recommended `requirements.txt` file is:


torch
torchvision
numpy
matplotlib
scikit-learn
opencv-python
pillow


For stricter reproducibility, users are encouraged to record the exact package versions used in their execution environment.



## 13. Installation

Clone the repository:


git clone https://github.com/zbumar17/insect-classification-deep-learning.git


Move into the project directory:


cd insect-classification-deep-learning


Install the required dependencies:


pip install -r requirements.txt




## 14. Usage Instructions

### Step 1: Download the datasets

Download the required image datasets from the public sources listed in the Dataset Information section.

### Step 2: Prepare the dataset folders

Organize the images into the following structure:


dataset/train/
dataset/val/
dataset/test/


Each split should contain seven class folders:


Aedes
Anopheles
Culex
Cimex
Ctenocephalides
Pediculus
Triatoma


### Step 3: Upload the dataset to the working environment

If using Google Colab, upload the dataset to Google Drive or directly to the Colab runtime.

Example Google Drive structure:


/content/drive/MyDrive/insect_dataset/
├── train/
├── val/
└── test/


### Step 4: Open the required notebook

Open the notebook corresponding to the model you want to train.

For example:


notebooks/densenet201_tl.ipynb
notebooks/efficientnet_b5_tl.ipynb
notebooks/convnext_large_tl.ipynb


### Step 5: Update dataset paths

Inside the notebook, update the dataset path variable to match your local or Google Drive path.

Example:


DATA_DIR = "/content/drive/MyDrive/insect_dataset"


### Step 6: Train the model

Run the notebook cells sequentially to:

1. Load the dataset.
2. Apply preprocessing.
3. Build the model.
4. Train the model.
5. Save the best checkpoint.
6. Evaluate the model.

### Step 7: Evaluate individual models

Each model notebook produces evaluation results such as:

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Confusion matrix.
* ROC-AUC curve.
* Precision-recall curve.

### Step 8: Run the hybrid ensemble notebook

After training ConvNeXt-Large and EfficientNet-B5, run:


notebooks/hybrid_convnext_efficientnet.ipynb


This notebook loads the trained models and combines their predictions to generate the hybrid ensemble results.



## 15. Methodology

The complete workflow consists of the following stages:

1. **Data Collection**
   Images were collected from publicly available repositories, including Kaggle, GBIF, Mendeley Data, and iNaturalist.

2. **Data Cleaning**
   Corrupted, unreadable, and irrelevant images were removed where applicable.

3. **Data Organization**
   Images were organized into seven genus-level categories.

4. **Data Splitting**
   The dataset was divided into training, validation, and testing subsets.

5. **Data Preprocessing**
   Images were resized, normalized, and converted into tensor format.

6. **Data Augmentation**
   Augmentation techniques were applied to the training data to improve model generalization.

7. **Model Development**
   A custom CNN and four transfer learning models were implemented.

8. **Transfer Learning**
   Pre-trained deep learning architectures were fine-tuned for seven-class insect genus classification.

9. **Hybrid Ensemble Learning**
   ConvNeXt-Large and EfficientNet-B5 were combined to create a hybrid ensemble classifier.

10. **Performance Evaluation**
    All models were evaluated using the same test set and the same assessment metrics.

11. **Comparative Analysis**
    The performance of the custom CNN, individual transfer learning models, and hybrid ensemble model was compared.



## 16. Evaluation Method

The models were evaluated using a held-out test set that was not used during training or validation.

The evaluation method included:

1. Training each model on the training set.
2. Monitoring performance using the validation set.
3. Saving the best-performing checkpoint where applicable.
4. Loading the best checkpoint for final testing.
5. Evaluating all models on the same test set.
6. Computing classification metrics for each model.
7. Comparing the results of all individual models.
8. Evaluating the hybrid ensemble model using the same test data.

The models evaluated include:

* Custom CNN.
* InceptionV3.
* DenseNet201.
* EfficientNet-B5.
* ConvNeXt-Large.
* Hybrid Ensemble Model.

The comparative analysis was performed to determine whether the hybrid ensemble improved classification performance compared with individual models.



## 17. Assessment Metrics

The following metrics were used to evaluate model performance:

### 17.1 Accuracy

Accuracy measures the proportion of correctly classified images among all test images.

It provides an overall measure of classification performance.

### 17.2 Precision

Precision measures the proportion of correctly predicted positive samples among all samples predicted as a given class.

Precision is useful when evaluating how reliable the model is when assigning an image to a specific insect genus.

### 17.3 Recall

Recall measures the proportion of correctly detected samples from all actual samples of a given class.

Recall is important when the objective is to reduce missed classifications for a genus.

### 17.4 F1-score

F1-score is the harmonic mean of precision and recall.

It is useful when evaluating performance across multiple classes, especially when class distributions are not perfectly balanced.

### 17.5 ROC-AUC

Receiver Operating Characteristic Area Under the Curve measures the ability of the model to distinguish between classes.

For multi-class classification, ROC-AUC can be computed using one-vs-rest class comparisons.

### 17.6 Precision-Recall Curves

Precision-recall curves show the relationship between precision and recall across different classification thresholds.

They are useful for understanding model behavior, especially when class imbalance is present.



## 18. Expected Outputs

After running the notebooks, the following outputs may be generated:


outputs/
├── accuracy_loss_curves/
├── confusion_matrices/
├── classification_reports/
├── roc_auc_curves/
├── precision_recall_curves/
└── model_comparison_results/


Typical output files include:


confusion_matrix_densenet201.png
classification_report_efficientnet_b5.txt
roc_auc_convnext_large.png
precision_recall_hybrid_ensemble.png
model_comparison_table.csv




## 19. Reproducibility Notes

To reproduce the experiments:

1. Use the same dataset sources listed in this README.
2. Organize the dataset using the required folder structure.
3. Use the same train, validation, and test split strategy.
4. Use the same preprocessing and augmentation steps.
5. Run each notebook from top to bottom.
6. Save model checkpoints after training.
7. Use the saved ConvNeXt-Large and EfficientNet-B5 checkpoints for the hybrid ensemble experiment.
8. Evaluate all models on the same test set.
9. Compare the output metrics.

Results may vary slightly depending on:

* GPU type.
* Random seed.
* PyTorch/CUDA version.
* Dataset split.
* Batch size.
* Training duration.
* Data augmentation randomness.

For best reproducibility, users should set random seeds in Python, NumPy, and PyTorch.

Example:


import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## 20. Citation

If you use this repository, dataset preparation workflow, or code in your research, please cite the associated manuscript:


Umar, Z. B., et al. Insect Genus Classification Using Transfer Learning and Hybrid Ensemble Models.


The full citation will be updated after publication.

Please also cite the original third-party datasets according to their required citation formats and license terms.



## 21. Dataset References

The datasets used in this study are available at:


Chula Mosquito Classification Dataset
https://www.kaggle.com/datasets/cyberthorn/chula-mosquito-classification?select=Ae-aegypti

GBIF
https://www.gbif.org/species/4987991

Mendeley Data – Dataset of Vector Mosquito Images
https://data.mendeley.com/datasets/88s6fvgg2p/4

iNaturalist – Pediculus
https://www.inaturalist.org/taxa/47437-Pediculus




## 22. License

This repository is provided for academic and research purposes.

The source code in this repository may be used, modified, and extended for non-commercial academic research, provided that appropriate credit is given to the authors.

The image datasets are not owned by the repository authors. Users must comply with the original licenses and terms of use of Kaggle, GBIF, Mendeley Data, iNaturalist, and the individual image contributors.



## 23. Contribution Guidelines

Contributions are welcome for improving reproducibility, documentation, code structure, and experimental clarity.

Suggested contribution areas include:

* Improving notebook organization.
* Adding exact package versions.
* Adding scripts for dataset splitting.
* Adding automated evaluation scripts.
* Improving visualization outputs.
* Adding clearer documentation for the hybrid ensemble method.

Before contributing, please ensure that no copyrighted or restricted third-party images are uploaded directly to this repository unless redistribution is explicitly permitted by the relevant license.



## 24. Contact

For questions related to this repository or the associated research manuscript, please contact the corresponding author or open an issue in the repository.


## 25. Disclaimer

This repository is intended for academic research and reproducibility purposes only. The authors are not responsible for misuse of third-party datasets or images. Users are responsible for verifying dataset licenses, permissions, and attribution requirements before using or redistributing any image data.
