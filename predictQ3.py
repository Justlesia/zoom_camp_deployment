#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


# In[19]:


df = pd.read_csv('bank.csv', sep=';',decimal='.')


# In[22]:


y = df['y']


# In[23]:


features = ['job','duration', 'poutcome']
dicts = df[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)

model = LogisticRegression().fit(X, y)


# with open('model.bin', 'wb') as f_out: # 'wb' means write-binary
#     pickle.dump((dv, model), f_out)

# In[26]:


with open('model1.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    model = pickle.load(f_in)
## Note: never open a binary file you do not trust the source!

with open('dv.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    dv = pickle.load(f_in)
## Note: never open a binary file you do not trust the source!


client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)





