#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:45:36 2025

@author: umutfirat
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
n_classes = len(iris.target_names)
plot_colors = "ryb"

# Tek bir figür ve 2x3 subplot yapısı oluşturuyoruz.
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Subplot index'ini açmak için eksenleri düzleştiriyoruz.
axes = axes.ravel()

for pairidx, pair in enumerate([[0,1], [0, 2], [0, 3], [1, 2], [1,3], [2,3]]):
    
    X = iris.data[:, pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(X, y)
    
    ax = axes[pairidx]  # Her bir eksen için subplot'ı seçiyoruz.
    plt.tight_layout(h_pad = 0.5, w_pad = 0.5, pad = 2.5)
    DecisionBoundaryDisplay.from_estimator(clf, X, 
                                            cmap = plt.cm.RdYlBu, 
                                            response_method="predict",
                                            xlabel=iris.feature_names[pair[0]],
                                            ylabel=iris.feature_names[pair[1]],
                                            ax=ax)  # Her eksende çizim yapılacak
    
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)  # y == i olmalı, y == 1 yerine
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], 
                    cmap=plt.cm.RdYlBu,
                    edgecolors="black")
    
    ax.set_title(f"{iris.feature_names[pair[0]]} vs {iris.feature_names[pair[1]]}")  # Başlık ekliyoruz
    
# Genel legend ekliyoruz.
plt.legend()
plt.show()
