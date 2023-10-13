#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import requests



with open('model1.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    model = pickle.load(f_in)
## Note: never open a binary file you do not trust the source!

with open('dv.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    dv = pickle.load(f_in)
## Note: never open a binary file you do not trust the source!



app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    y = y_pred >= 0.5

    result = {
        'get_card_probability': float(y_pred),
        'y': bool(y)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




