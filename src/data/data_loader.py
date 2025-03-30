import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    """
    Loads the Iris dataset and returns standardized features and labels.
    """
    iris = datasets.load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int64")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
