# DEEP LEARNING FOR DETECTING MALICIOUS SOFTWARE

This repository contains the code, datasets, and documentation for the Final Year Project (FYP) titled Deep Learning for Detecting Malicious Software. The goal of this project is to leverage deep learning techniques including CNN, ResNet, Xception and Inception and transformer DeiT with Spatial Aggregation Vector Encoding (SAVE) to accurately classify malware images into benign or malicious categories.

---

## 📁 Project Structure

.
Source Code/DEEP LEARNING FOR DETECTING MALICIOUS SOFTWARE/
├── Source Code for Blended malware image/
│   ├── DeiT with SAVE.ipynb
│   ├── DeiT.ipynb
│   ├── Inception with SAVE.ipynb
│   ├── Inception.ipynb
│   ├── ResNet 50 with SAVE.ipynb
│   ├── ResNet.ipynb
│   ├── Xception with SAVE.ipynb
│   └── Xception.ipynb
│
└── Source Code for Malware as Images/
    ├── DeiT SAVE.ipynb
    ├── DeiT.ipynb
    ├── Inception with SAVE.ipynb
    ├── Inception.ipynb
    ├── ResNet with SAVE.ipynb
    ├── ResNet.ipynb
    ├── Xception with SAVE.ipynb
    └── Xception.ipynb


## 📂 `code/` Folder

This directory contains all experiment-related files, including model training and prediction generation notebooks. It is organized into three subfolders: `Dataset1`, `Dataset2`, and `Dataset3`.

### 📌 Contents

#### 1. **Blended Malware Image**

Location: Source Code/DEEP LEARNING FOR DETECTING MALICIOUS SOFTWARE/Source Code for Blended malware image/

Format: <Model>.ipynb and <Model> with SAVE.ipynb

Models Included:
DeiT & DeiT with SAVE
Inception & Inception with SAVE
ResNet 50 & ResNet 50 with SAVE
Xception & Xception with SAVE

Contents: Data loading via tf.data, model construction (backbone + SAVE layer), training with callbacks (best checkpoint), history plotting, validation & testing (classification report, confusion matrix), and final model export.

#### 2. **Malware as Images**

Location: Source Code/DEEP LEARNING FOR DETECTING MALICIOUS SOFTWARE/Source Code for Malware as Images/

Format: <Model>.ipynb and <Model> SAVE.ipynb

Models Included:
DeiT & DeiT SAVE
Inception & Inception with SAVE
ResNet & ResNet with SAVE
Xception & Xception with SAVE

Contents: Identical workflow as Blended notebooks, applied to the original Malware as Images dataset folder.


---

## 📂 `dataset/` Folder

This folder includes all datasets used for model training and evaluation. All datasets were sourced from Kaggle and are provided in `.csv` format for convenience.

### 📌 Datasets

1. **Blended Malware Image Dataset**  
   📎 https://www.kaggle.com/datasets/gauravpendharkar/blended-malware-image-dataset/data

2. **Malware as Images**  
   📎 https://www.kaggle.com/datasets/matthewfields/malware-as-images


---

## 💻 Technologies Used

- Languages & Tools: Python, Jupyter Notebook
- Frameworks: TensorFlow/Keras, PyTorch (for baseline comparisons)
- Libraries: NumPy, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib
- Hardware: NVIDIA GPU (CUDA)

---

## 📊 Task Overview

The primary objectives of this FYP are:

**1. Problem Statement: **Malware poses a significant security threat. Traditional signature-based detection methods struggle with new and obfuscated samples. Deep learning approaches can learn rich visual patterns from malware images, improving detection of novel variants.

**2. Goals:**
Convert raw malware binaries into grayscale images.
Implement and compare multiple architectures (ResNet50, InceptionV3, Xception, Custom CNN) enhanced with SAVE layers.
Integrate techniques such as SMOTE, data augmentation, batch normalization, L2 regularization, and dropout to address class imbalance and overfitting.
Evaluate models using classification metrics (accuracy, precision, recall, F1-score) and confusion matrices.

**3. Methodology:** Data preprocessing → Model design → Training and validation → Prediction generation → Performance analysis.

---


## 📜 License

This project is for academic and research purposes only. Datasets belong to their respective Kaggle  contributors.
