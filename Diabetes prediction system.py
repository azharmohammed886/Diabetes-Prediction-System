#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[15]:


data = pd.read_csv("C:/Users/azhar/Downloads/Project 2 - Diabetes Data/Project 2 MeriSKILL/diabetes.csv")
data


# In[16]:


correlation =data.corr()
sns.heatmap(data.corr(), annot=True)


# In[18]:


data.isnull().sum()


# In[19]:


data.describe()


# In[8]:


import warnings
warnings.filterwarnings('ignore')


# In[9]:


X =data.drop('Outcome',axis=1)
Y =data['Outcome']
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.25,random_state=42)


# In[10]:


model=LogisticRegression()r
model.fit(X_train,Y_train)


# In[11]:


prediction = model.predict(X_test)


# In[12]:


print(prediction)


# In[13]:


accuracy = accuracy_score(prediction, Y_test)
print(accuracy*100)

