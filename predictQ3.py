#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


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





