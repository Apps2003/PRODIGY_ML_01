#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd #data read
from sklearn.linear_model import LinearRegression #funtionality
from sklearn import metrics #performance mesurement
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv('train.csv') #pandas function
print(train_data.shape) #rows & cols reading of data
train_data.head() #head as a first 5 records


# In[3]:


train_data.tail()


# In[4]:


type(train_data)


# In[5]:


train_data.shape


# In[6]:


train_data.info() #fulldescription
train_data.describe() #full description


# In[7]:


test_data = pd.read_csv('test.csv') #pandas function
print(test_data.shape) #rows & cols reading of data
test_data.head() #head as a first 5 records


# In[8]:


# Feature selection
X_train = train_data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']]  # Independent variables
Y_train = train_data['SalePrice']  # Dependent variable


# In[9]:


# Model Intialization
reg = LinearRegression()


# In[10]:


# Data Fitting
reg = reg.fit(X_train, Y_train) # training of a model


# In[11]:


X_test = test_data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']]  # Independent variables
#Y_test = test_data['SalePrice']  # Dependent variable


# In[12]:


from sklearn.impute import SimpleImputer

# Create an imputer
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like median or most_frequent

# Fit and transform the imputer on your data
X_test_imputed = imputer.fit_transform(X_test)



# In[13]:


# Predict using the imputed data
Y_pred = reg.predict(X_test_imputed)


# In[14]:


Y_pred


# In[ ]:




