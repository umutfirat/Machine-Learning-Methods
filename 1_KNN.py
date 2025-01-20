import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# GET DATA
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# PREPARE DATA
X = cancer.data
y = cancer.target

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2408)

# SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# TRAIN DATA WITH KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# TEST MODEL
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is: {accuracy * 100}%")
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix: \n{conf_matrix}")

"""
    KNN: Hyperparameter = K
    K: 1,2,3 ... N
    Accuracy: %A, %B, %C ...
"""

accuracy_values = []
k_values = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy_values.append(accuracy_score(y_test, y_pred) * 100)
    k_values.append(k)


plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle="-")
plt.title("Accuracy rating by K")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)