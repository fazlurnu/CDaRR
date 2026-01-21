from sampling_techniques.models.tanh_feature_logit import TanhFeatureLogit
from sampling_techniques.models.logistic_regression import logistic_regression

import json

# plot the points and decision boundary where p = 0.5
import numpy as np
import matplotlib.pyplot as plt

# read json file
with open("results/example/initial_samples_results_1.5_0.5_0.95_180.json", "r") as f:
    test_data = json.load(f)

lookahead_time_array = [d["x2_lookahead_time"] for d in test_data]
resofach_array = [d["x1_resofach"] for d in test_data]
safe_array = [d["sim_results"]["overall_ipr"] >= 0.999 for d in test_data]

acc, params = logistic_regression(
    lookahead_time_array,
    resofach_array,
    safe_array)

model = TanhFeatureLogit.from_dict(params)
print("Validation Accuracy:", acc)
print("Model Parameters:", model)

x1 = np.array(lookahead_time_array)
x2 = np.array(resofach_array)
ipr = np.array(safe_array)
# Color condition
colors = np.where(ipr < 0.999, 'tab:red', 'tab:green')

plt.figure(figsize=(8, 6))
plt.scatter(x2, x1, c=colors, s=40)
plt.xlabel("x1 (resofach)")
plt.ylabel("x2 (lookahead time)")
plt.title("Simulation Results Colored by IPR Threshold")

# Create p=0.5 line
x1_range = np.linspace(min(x1), max(x1), 100)
x2_range = np.linspace(min(x2), max(x2), 100)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T
# Predict probabilities on the grid
probs = model.predict_proba(X_grid).reshape(X1_grid.shape)
# Contour where probability = 0.5
contour = plt.contour(X2_grid, X1_grid, probs, levels=[0.5], colors='black', linewidths=2)
plt.clabel(contour, inline=True, fontsize=8)
plt.show()

