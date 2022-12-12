#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


dataset=pd.read_csv('C:/Users/Sai Thanishvi/Downloads/Credit card dataset/creditcard.csv')
#shuffled data:
s_data = dataset.sample(frac=1) 
#data with one-hot encoding:
o_data = pd.get_dummies(s_data, columns=['Class']) 
# data after normalization:
n_data = (o_data - o_data.min()) / (o_data.max() - o_data.min()) 


# In[4]:


seed = 1337
data_inp = n_data.drop(['Class_0', 'Class_1'], axis=1)
data_out = n_data[['Class_0', 'Class_1']]
ar_inp, ar_out = np.asarray(data_inp.values, dtype='float32'), np.asarray(data_out.values, dtype='float32')
inp_train, inp_test, out_train, out_test = train_test_split(ar_inp, ar_out, test_size=0.30, random_state=42)
rout=np.argmax(out_test, axis=1)


# In[5]:


model = IsolationForest(random_state=seed)
model.fit(inp_train, out_train)


# In[6]:


def pred(inp):
    t = model.predict(inp)
    t = np.array([1 if out == -1 else 0 for out in t])
    return t


# In[7]:


t = pred(inp_test)


# In[15]:


print("Accuracy using Isolation Forest with one hot encoding:",accuracy_score(rout, t),"\n")
print("Classification Report for Isolation Forest with one hot encoding:\n",classification_report(rout, t))


# In[ ]:




