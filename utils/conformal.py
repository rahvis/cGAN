import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

# Inductive Conformal Prediction (ICP)
def compute_icp_conformity_scores(x):
    mean_val = np.mean(x, axis=0, keepdims=True)
    return np.linalg.norm(x - mean_val, axis=1, keepdims=True)

# Mondrian Conformal Prediction
def compute_mondrian_conformity_scores(x, labels):
    unique_classes = np.unique(labels)
    scores = np.zeros_like(labels, dtype=float)
    for cls in unique_classes:
        class_data = x[labels == cls]
        mean_val = np.mean(class_data, axis=0, keepdims=True)
        scores[labels == cls] = np.linalg.norm(class_data - mean_val, axis=1)
    return scores

# Cross-Conformal Prediction
def compute_cross_conformal_scores(x, labels, k=5):
    folds = np.array_split(np.arange(len(x)), k)
    all_scores = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        calib_idx = folds[i]
        mean_val = np.mean(x[train_idx], axis=0, keepdims=True)
        scores = np.linalg.norm(x[calib_idx] - mean_val, axis=1)
        all_scores.extend(scores)
    return np.array(all_scores)

# Venn-Abers Predictors
def compute_venn_abers_scores(x, labels):
    model = IsotonicRegression()
    model.fit(x, labels)
    return model.predict(x)

# Compute conformal intervals
# utils/conformal.py

def compute_conformal_intervals(synthetic_data, alpha):
    """
    Compute the conformal prediction intervals for the synthetic data at a given alpha.
    
    synthetic_data: numpy array or tensor of synthetic data points
    alpha: significance level for the conformal prediction interval (e.g., 0.05 for 95% confidence)
    
    Returns: A list of tuples representing the lower and upper bounds of the prediction intervals
    """
    # Placeholder logic: Calculate prediction intervals (you should adapt this based on your method)
    intervals = []
    for sample in synthetic_data:
        # Example logic (replace with your actual conformal method):
        lower_bound = np.percentile(sample, alpha * 100)  # lower bound at alpha percentile
        upper_bound = np.percentile(sample, (1 - alpha) * 100)  # upper bound at (1-alpha) percentile
        intervals.append((lower_bound, upper_bound))
    
        return intervals
