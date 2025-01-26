#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf 


# In[7]:


#!/bin/bash
get_ipython().system('curl -L -o ./brain-tumor-mri-dataset.zip   https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset')


# In[22]:


import numpy as np 
import pandas as pd
import time
import matplotlib.pyplot as plt


# In[8]:


cd ml_capstone2/


# In[ ]:





# In[23]:


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


# In[24]:


img=load_img("./dataset/Training/meningioma/Tr-me_0010.jpg",target_size=(100,100))


# In[31]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# In[26]:


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


# In[33]:


checkpoint=keras.callbacks.ModelCheckpoint(
    'MRI_v1_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor ='val_accuracy',
    mode='max'
)


# In[34]:


learning_rate = 0.001
drop_rate=0.2
input_size=300
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

model3 = make_model_large(input_size=input_size,learning_rate=learning_rate,inner_size=20,drop_rate=drop_rate)
 
history = model3.fit(
    train_ds1,
    epochs=10,
    validation_data=val_ds1,
    callbacks=[checkpoint]
)


# In[36]:


plt.plot(history.history['accuracy'],label='train_accuracy')
plt.plot(history.history['val_accuracy'],label='test_accuracy')
plt.legend()


# In[ ]:





# In[ ]:




