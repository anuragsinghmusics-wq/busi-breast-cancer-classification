# Breast Ultrasound Image Classification using Deep Learning

## Project Overview

Breast cancer is one of the leading causes of death among women worldwide. Early detection plays a crucial role in improving survival rates. Ultrasound imaging is widely used because it is non-invasive, safe, and cost-effective.

This project develops a **deep learning based image classification system** to classify breast ultrasound images into three categories:

- Benign
- Malignant
- Normal

The project uses the **BUSI (Breast Ultrasound Images) dataset** and implements a **Convolutional Neural Network (CNN)** using **PyTorch**.

---

# Dataset

Dataset: **BUSI Breast Ultrasound Dataset**

Class distribution:

| Class | Images |
|------|------|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |

The dataset is **imbalanced**, therefore several techniques are used to improve model performance.

Dataset split:

- Training: 70%
- Validation: 15%
- Testing: 15%

---

# Techniques Used

To address class imbalance and improve performance, the following techniques were implemented:

### 1. Baseline Model
The CNN model is trained using the original dataset without applying any imbalance handling techniques.

### 2. Oversampling
Oversampling increases the number of samples in minority classes by duplicating existing samples during training.

### 3. Data Augmentation
Data augmentation artificially increases the diversity of training images using transformations such as:

- Random horizontal flip
- Random rotation
- Random resized crop
- Color jitter

### 4. Focal Loss
Focal Loss focuses training on difficult misclassified samples and helps handle class imbalance.

Focal loss formula:
