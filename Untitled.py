#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import tensorflow as tf


# In[12]:


import sklearn
from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[150,0]]))


# In[23]:


import sklearn 
from sklearn import tree
#features = [[60,"round"],[30,"square"],[40,"square"],[70,"round"]]
#labels = ["ball","book","book","ball"]
features = [[60,1],[50,0],[40,0],[70,1]] #0 = square , 1 = round
labels = [1,0,0,1] # 0 = book , 1 = ball
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[70,1]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




