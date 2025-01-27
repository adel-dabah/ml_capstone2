# Capstone2 Project for MLZoomcamp 2024 

## Overview

### Understanding Brain Tumors
A brain tumor is an abnormal mass of cells in the brain. Due to the rigid structure of the skull, any growth in this confined space can lead to increased intracranial pressure, potentially causing brain damage or life-threatening complications. Tumors can be categorized as:
- **Benign (noncancerous)** 
- **Malignant (cancerous)**

### Why This Matters
Early detection and classification of brain tumors are crucial for selecting effective treatment options and improving patient outcomes. This project focuses on utilizing deep learning techniques to enhance diagnostic accuracy in brain tumor identification, classification, and localization.

---

## Methods

This project explores the use of **Convolutional Neural Networks (CNNs)** for:
- **Detection**: Identifying the presence of brain tumors.
- **Classification**: Categorizing tumors by type (glioma, meningioma, pituitary) and grade.

As part of MLzoomcamp, we developed a CNN model to achieve efficiency and accuracy by adjusting and tuning the model's parameters.

---

## Dataset Information

This project utilizes a dataset derived from three sources:
1. **Figshare**
2. **SARTAJ Dataset**
3. **Br35H Dataset**

### Key Details:
- The dataset contains **7,023 MRI images**, categorized into four classes:
  - Glioma
  - Meningioma
  - No tumor
  - Pituitary tumor
- Images for the "no tumor" class were sourced from the Br35H dataset.

#### Dataset Adjustments:
It was identified that the glioma class in the SARTAJ dataset had inconsistencies. To address this, images from the Figshare dataset replaced the problematic samples. This refinement ensures higher quality and reliability for training and evaluation purposes.

---
