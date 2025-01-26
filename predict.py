#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#load the saved model


# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


model=keras.models.load_model('MRI_v1_07_0.923.keras')


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


from tensorflow.keras.applications.efficientnet import preprocess_input


# In[5]:


input_size=300
# Build image generator for training (takes preprocessing input function)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the test dataset into the train generator
test_ds = test_gen.flow_from_directory(directory='./dataset/Testing/', # Train images directory
                                         target_size=(input_size,input_size), # resize images to train faster
                                         batch_size=32) # 32 images per batch


# In[6]:


#model.evaluate(test_ds)


# In[7]:


import numpy as np 


# In[8]:


from tensorflow.keras.preprocessing.image import load_img


# In[9]:


img =load_img("./dataset/Testing/notumor/Te-no_0029.jpg", target_size=(300,300))


# In[10]:


x=np.array(img)


# In[12]:


print(img)


# In[13]:


X=np.array([x])


# In[17]:


pred_x=model.predict(X)


# In[16]:


test_ds.class_indices


# In[21]:


pred=dict(zip(list(test_ds.class_indices.keys()), list(pred_x[0])))


# In[22]:


print(pred)


# In[ ]:




