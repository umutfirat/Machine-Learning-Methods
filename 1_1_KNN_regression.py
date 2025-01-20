#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:52:08 2025

@author: umutfirat
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40,1), axis=0)
y = np.sin(X).ravel()

# Add Noise
y[::5] += 1 * (0.5 - np.random.rand(8))
T = np.linspace(0, 5, 500)[:, np.newaxis]

for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X,y).predict(T)
    
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="green", label="data")
    plt.plot(T, y_pred, color="blue", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))
    
plt.tight_layout()
plt.show()