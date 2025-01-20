#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:05:42 2025

@author: umutfirat
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# GET DATA
iris = load_iris()

# PREPARE DATA
X = iris.data
y = iris.target

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2408)

# TRAIN DATA WITH DT
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=422408
tree_clf.fit(X_train, y_train)

# TEST DATA
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# RESULT
print(f"Accuracy Rating: {accuracy * 100}\n")
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"confusion matrix: \n {conf_matrix}")

# DRAWING GRAPH
plt.figure(figsize= (15, 10))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


# IMPORTANCES
feature_importances = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")