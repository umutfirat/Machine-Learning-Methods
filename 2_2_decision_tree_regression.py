#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:12:12 2025

@author: umutfirat
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# GET DATA
diabetes = load_diabetes()

# PREPARE DATA

X = diabetes.data
y = diabetes.target


# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2408)


# REGRESSION
tree_reg = DecisionTreeRegressor(random_state=2408)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

# ERROR 
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")