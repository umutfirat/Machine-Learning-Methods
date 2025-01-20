# Machine Learning Practice Repository

Welcome to my Machine Learning practice repository! This repository is dedicated to the projects and applications I have developed to enhance my skills in the field of Machine Learning.

## 1. K-Nearest Neighbors (KNN) Algorithm

This section contains the implementation of the KNN algorithm using the breast cancer dataset provided by scikit-learn. The workflow includes the following steps:

- **Get Data**: Load the breast cancer dataset using `scikit-learn` and prepare it in a `pandas` DataFrame.
- **Prepare Data**: Extract the features and target labels.
- **Split Data**: Divide the dataset into training and testing sets (70% training, 30% testing).
- **Scale Data**: Use `StandardScaler` to normalize the data for better performance.
- **Train Data**: Train the KNN algorithm with `n_neighbors=3`.
- **Test Data**: Evaluate the model's performance using accuracy score and confusion matrix.
- **Improve Model**: Perform hyperparameter tuning by varying the value of `K` (number of neighbors) and visualize the accuracy for different `K` values.

## Results

- **Initial Accuracy:** The model achieved an accuracy of approximately 95-97% with n_neighbors=3.

- **Hyperparameter Tuning:** The optimal value of K was determined by plotting accuracy values for K ranging from 1 to 20.