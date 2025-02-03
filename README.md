# ml_capstone2
# Capstone2 Project for MLZoomcamp 2024 (Brain Tumors Classification)
![image](https://github.com/user-attachments/assets/632d114f-ede0-487c-a107-15f8ba2929f7)
![image](https://github.com/user-attachments/assets/a97d9b71-a717-4080-9089-9ddbfb522427)

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

---
## EDA 
![image](https://github.com/user-attachments/assets/86cb2b22-77cf-4981-9642-72c628a40e5d)


## Model and Training
### Transfer Learning (using Convolutional layers of EfficientNetB0 pre-trained model )
![image](https://github.com/user-attachments/assets/00c35524-04c6-4e09-ad42-ac913f70d4ca)

### Adjusting learning rate 
![image](https://github.com/user-attachments/assets/62a2fbdb-7823-4bca-b516-e5609956c3cf)

### Adjusting the dropout percentage 
![image](https://github.com/user-attachments/assets/a96572de-ca7b-4e5c-a492-6f0014928dd3)

### Tuning the size of inner layers 
![image](https://github.com/user-attachments/assets/e795157a-605f-47ec-a60a-c9aac086c05e)

### Adding checkpoints and increasing the size of the images input_size to 400 
![image](https://github.com/user-attachments/assets/63dec89b-f4e3-43be-9c98-034e4d97ca81)
![image](https://github.com/user-attachments/assets/bb936c74-0e6f-4644-8180-516b0356a453)

## Model Deployment 

## Dependencies 
The code has the following dependencies: 
tensorflow/ keras /matplotlib /numpy 

For Activating Environment: Run pipenv shell

## Dockerfile 
To build and run docker container : 

sudo docker build -t mri_model_d .

sudo docker run -it --rm -p 8080:8080  mri_model_d

python3 test_docker.py 

## AWS Cloud Deployment
The tensorflow lite model is depoyed using AWS lanbda service 
URL: https://f0dhzqffpk.execute-api.eu-north-1.amazonaws.com/test_stage/predict
to test the AWS deployement use : 

python3 test_AWS_deployement.py 