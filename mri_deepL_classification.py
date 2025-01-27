#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 


# In[2]:


#!/bin/bash
#!curl -L -o ./brain-tumor-mri-dataset.zip\
#  https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset


# In[3]:


import numpy as np 
import pandas as pd
import time
import matplotlib.pyplot as plt


# In[4]:


#cd ml_capstone2/


# In[5]:


#EDA


# In[6]:


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


# In[7]:


load_img("./dataset/Training/meningioma/Tr-me_0010.jpg")


# In[8]:


load_img("./dataset/Training/notumor/Tr-no_0010.jpg")


# In[9]:


#train a model 


# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# In[11]:


def make_model_large(input_size,learning_rate, inner_size,drop_rate):
    base_model1=EfficientNetB0(weights="imagenet",
                    include_top=False,
                    input_shape=(input_size,input_size,3)
                   )
    base_model1.trainable=False
    inputs1 =keras.Input(shape=(input_size,input_size,3))
    base1 =base_model1(inputs1,training=False)
    pooling=keras.layers.GlobalAvgPool2D()
    vectors1=pooling(base1)
    
    inner1 =keras.layers.Dense(inner_size,activation='relu')(vectors1) # inner layers # activation function relu for inner layers 
    drop1 =keras.layers.Dropout(drop_rate)(inner1)
    outputs1=keras.layers.Dense(4)(drop1) # ,activation='softmax'
    model1=keras.Model(inputs1,outputs1)
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)# optimize the wieghts of the dense layers 
    #meseuring the accuracy for the optimizer 
    loss= keras.losses.CategoricalCrossentropy(from_logits=True) #MSE for regression models (logist true makes it more stable)
    # in case of logists False you must add actimation fuction to the output1 layers activation='softmax'

    model1.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model1


# In[12]:


#adding checkpoints 


# In[13]:


checkpoint=keras.callbacks.ModelCheckpoint(
    'MRI_v1_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor ='val_accuracy',
    mode='max'
)


# In[ ]:


#CNN model with learning rate tuning 


# In[45]:


learning_rate_g = [0.001, 0.01, 0.1]
drop_rate=0.0
input_size=100
# Build image generator for training (takes preprocessing input function)
train_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
train_ds1 = train_gen1.flow_from_directory(directory='./dataset/Training/', # Train images directory
                                         target_size=(input_size,input_size), # resize images to train faster
                                         batch_size=32) # 32 images per batch
val_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
val_ds1 = val_gen1.flow_from_directory(directory='./dataset/Testing/', # Train images directory
                                         target_size=(input_size,input_size), # resize images to train faster
                                         batch_size=32) # 32 images per batch
hist=[]
for learning_rate in learning_rate_g:
    model3 = make_model_large(input_size=input_size,learning_rate=learning_rate,inner_size=5,drop_rate=drop_rate)
    history = model3.fit(
    train_ds1,
    epochs=10,
    validation_data=val_ds1,
    callbacks=[checkpoint]
    )
    hist.append((learning_rate, history)) 


# In[46]:


# Plot the accuracy for different learning rates
plt.figure(figsize=(12, 8))
for learning_rate, history in hist:
    plt.plot(history.history['accuracy'], label=f'Train Acc (lr={learning_rate})')
    plt.plot(history.history['val_accuracy'], label=f'Val Acc (lr={learning_rate})')

# Add plot details
plt.title("Accuracy vs Epochs for Different Learning Rates")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[ ]:


#CNN model with dropout tuning 


# In[50]:


learning_rate_g =  0.01
drop_rate_g =[0, 0.2, 0.4]
input_size=100
# Build image generator for training (takes preprocessing input function)
train_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
train_ds1 = train_gen1.flow_from_directory(directory='./dataset/Training/', # Train images directory
                                           target_size=(input_size,input_size), # resize images to train faster
                                           batch_size=32) # 32 images per batch
val_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
val_ds1 = val_gen1.flow_from_directory(directory='./dataset/Testing/', # Train images directory
                                         target_size=(input_size,input_size), # resize images to train faster
                                         batch_size=32) # 32 images per batch
hist_drop_rate=[]
for drop_rate in drop_rate_g:
    model3 = make_model_large(input_size=input_size,learning_rate=learning_rate,inner_size=5,drop_rate=drop_rate)
    history = model3.fit(
    train_ds1,
    epochs=10,
    validation_data=val_ds1,
    callbacks=[checkpoint]
    )
    hist_drop_rate.append((drop_rate, history)) 


# In[52]:


# Plot the accuracy for different drop rates
plt.figure(figsize=(12, 8))
for drop, history in hist_drop_rate:
    plt.plot(history.history['accuracy'], label=f'Train Acc (lr={drop})')
    plt.plot(history.history['val_accuracy'], label=f'Val Acc (lr={drop})')

# Add plot details
plt.title("Accuracy vs Epochs for Different Drop Rates")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[ ]:


#CNN model with inner_layer size tuning


# In[16]:


learning_rate =  0.01
drop_rate=0.2
input_size=100
# Build image generator for training (takes preprocessing input function)
train_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
train_ds1 = train_gen1.flow_from_directory(directory='./dataset/Training/', # Train images directory
                                           target_size=(input_size,input_size), # resize images to train faster
                                           batch_size=32) # 32 images per batch
val_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
val_ds1 = val_gen1.flow_from_directory(directory='./dataset/Testing/', # Train images directory
                                         target_size=(input_size,input_size), # resize images to train faster
                                         batch_size=32) # 32 images per batch
hist_inner_size=[]
for inner_size in [1,5,10]:
    model3 = make_model_large(input_size=input_size,learning_rate=learning_rate,inner_size=inner_size,drop_rate=drop_rate)
    history = model3.fit(
    train_ds1,
    epochs=10,
    validation_data=val_ds1,
    #callbacks=[checkpoint]
    )
    hist_inner_size.append((inner_size, history)) 


# In[ ]:





# In[17]:


# Plot the accuracy for different inner layer sizes 
plt.figure(figsize=(12, 8))
for size, history in hist_inner_size:
    plt.plot(history.history['accuracy'], label=f'Train Acc (lr={size})')
    plt.plot(history.history['val_accuracy'], label=f'Val Acc (lr={size})')

# Add plot details
plt.title("Accuracy vs Epochs for Different sizes of inner layers (top level of the ML model)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[ ]:


#Train final model 


# In[18]:


learning_rate =  0.01
drop_rate=0.2
input_size=400
# Build image generator for training (takes preprocessing input function)
train_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
train_ds1 = train_gen1.flow_from_directory(directory='./dataset/Training/', # Train images directory
                                           target_size=(input_size,input_size), # resize images to train faster
                                           batch_size=32) # 32 images per batch
val_gen1 = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the train dataset into the train generator
val_ds1 = val_gen1.flow_from_directory(directory='./dataset/Testing/', # Train images directory
                                         target_size=(input_size,input_size), # resize images to train faster
                                         batch_size=32) # 32 images per batch
inner_size = 16
model3 = make_model_large(input_size=input_size,learning_rate=learning_rate,inner_size=inner_size,drop_rate=drop_rate)
history = model3.fit(
    train_ds1,
    epochs=10,
    validation_data=val_ds1,
    callbacks=[checkpoint]
)


# In[19]:


plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test_accuracy')
plt.legend()

