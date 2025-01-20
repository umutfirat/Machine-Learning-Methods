#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:19:03 2025

@author: umutfirat
"""

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# CREATE A DATA SET

X = np.sort(5 * np.random.rand(80,1), axis=0)
y = np.sin(X).ravel()

# ADD NOISE
y[::5] += 0.5 * (0.5 - np.random.rand(16))


reg_1 = DecisionTreeRegressor(max_depth=2)
reg_2 = DecisionTreeRegressor(max_depth=15)

reg_1.fit(X,y)
reg_2.fit(X,y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]

y_pred1 = reg_1.predict(X_test)
y_pred2 = reg_2.predict(X_test)

plt.figure()
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X_test, y_pred1, color="blue", label = "depth: 2", linewidth= 2)
plt.plot(X_test, y_pred2, color="green", label = "depth: 15", linewidth= 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()