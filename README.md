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
FL(pt) = -α (1 - pt)^γ log(pt)

Where:
- pt = predicted probability
- α = balancing factor
- γ = focusing parameter

---

# Model Architecture

A **pretrained CNN from torchvision** is used through **transfer learning**.

The final fully connected layer is modified to classify images into **three categories**:

- Benign
- Malignant
- Normal

---

# Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

---

# Results

| Model | Accuracy | Precision | Recall | F1 Score |
|------|------|------|------|------|
| Baseline | 0.7778 | 0.7705 | 0.7361 | 0.7501 |
| Oversampling | 0.7778 | 0.7997 | 0.7247 | 0.7538 |
| Augmentation | **0.8034** | 0.7896 | 0.7799 | **0.7838** |
| Focal Loss | 0.7436 | 0.7143 | 0.7156 | 0.7129 |

**Observation:**  
Data augmentation achieved the best overall performance.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# How to Run the Project

1 Clone the repository

```bash
git clone https://github.com/yourusername/breast-ultrasound-classification-cnn.git
