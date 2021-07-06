# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:37:30 2021

@author: Mate Rusz
y solution of a problem of the DataCamp course "Supervised Learning with scikit-learn"
https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn
"""

# Import necessary modules
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the Boston housing market dataset
boston = datasets.load_boston()
bostonDF = pd.DataFrame(data = boston.data, columns=boston.feature_names)
bostonDF['MEDV'] = boston.target 
print(bostonDF.head())

# Creating feature and target arrays
X = bostonDF.drop('MEDV', axis=1).values
y = bostonDF['MEDV'].values

# Predicting house value from only rooms ("RM")
X_rooms = bostonDF['RM'].values # alternatively X_rooms = X[:, 5]

# Add an additional dimension
X_rooms = X_rooms.reshape(-1, 1)
y = y.reshape(-1, 1)

# Use regression
reg = LinearRegression()


