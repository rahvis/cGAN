import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import the required conformal interval computation function
from utils.conformal import compute_conformal_intervals

def evaluate_metrics(real_data, real_labels, synthetic_data, synthetic_labels, alpha_values):
    metrics = {}
    ks_scores = []
    wass_distances = []
    num_features = real_data.shape[1]
    
    # Calculate KS statistics and Wasserstein distances for each feature
    for i in range(num_features):
        real_feature = real_data[:, i]
        synth_feature = synthetic_data[:, i]
        ks_stat, _ = ks_2samp(real_feature, synth_feature)
        wd = wasserstein_distance(real_feature, synth_feature)
        ks_scores.append(ks_stat)
        wass_distances.append(wd)
        
    metrics['KS_mean'] = np.mean(ks_scores)
    metrics['Wasserstein_mean'] = np.mean(wass_distances)
    
    # Downstream accuracy
    clf = LogisticRegression(max_iter=200)
    clf.fit(synthetic_data, synthetic_labels)
    y_pred = clf.predict(real_data)
    acc = accuracy_score(real_labels, y_pred)
    metrics['Downstream_Accuracy'] = acc
    
    # Conformal metrics
    coverage_results = {}
    interval_widths = {}
    for alpha in alpha_values:
        intervals = compute_conformal_intervals(synthetic_data, alpha)  # Ensure this is imported
        coverage = np.mean([y in interval for y, interval in zip(real_labels, intervals)])
        width = np.mean([upper - lower for lower, upper in intervals])
        #coverage_results[f'Coverage_alpha_{alpha}'] = coverage
        interval_widths[f'Interval_Width_alpha_{alpha}'] = width
    
    #metrics.update(coverage_results)
    metrics.update(interval_widths)
    
    return metrics
