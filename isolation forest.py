#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[12]:


dataset = pd.read_csv('C:/Users/Sai Thanishvi/Downloads/Credit card dataset/creditcard.csv')
dataset.describe()


# In[13]:


dataset = dataset.dropna()


# In[14]:


f = dataset[dataset['Class'] == 1]
v = dataset[dataset['Class'] == 0] 
o = len(f)/float(len(v))


# In[15]:


inp = dataset.drop('Class',axis = 1)
out = dataset['Class'] 


# In[ ]:


model = IsolationForest(max_samples = len(inp),contamination = o).fit(inp) 
out_pred = model.predict(inp)
out_pred[out_pred == 1] = 0 
out_pred[out_pred == -1] = 1 


# In[10]:


Number_of_errors = (out_pred != out).sum()
print("Number of errors occurred:",Number_of_errors)


# In[9]:


print("Accuracy using Isolation Forest:",accuracy_score(out_pred,out))
print("Classification Report for Isolation Forest:",classification_report(out_pred,out))


# In[ ]:




