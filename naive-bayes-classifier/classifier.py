#!/usr/bin/env python3

import numpy as np
from sklearn.naive_bayes import CategoricalNB

# Sample dataset
#
# Features X:
#   Weather: 0=Sunny, 1=Overcast, 2=Rainy
#   Temperature: 0=Hot, 1=Mild, 2=Cool
#
# Labels y:
#   Play Tennis: 0=No, 1=Yes

# Input features (Weather, Temperature)
X = np.array([
    [0, 0],  # Sunny, Hot
    [0, 0],  # Sunny, Hot
    [1, 0],  # Overcast, Hot
    [2, 1],  # Rainy, Mild
    [2, 2],  # Rainy, Cool
    [2, 2],  # Rainy, Cool
    [1, 2],  # Overcast, Cool
    [0, 1],  # Sunny, Mild
    [0, 2],  # Sunny, Cool
    [2, 1],  # Rainy, Mild
    [0, 1],  # Sunny, Mild
    [1, 1],  # Overcast, Mild
    [1, 0],  # Overcast, Hot
    [2, 1],  # Rainy, Mild
])

# Labels (Play Tennis)
y = np.array([
    0,  # No
    0,  # No
    1,  # Yes
    1,  # Yes
    1,  # Yes
    0,  # No
    1,  # Yes
    0,  # No
    1,  # Yes
    1,  # Yes
    1,  # Yes
    1,  # Yes
    1,  # Yes
    0,  # No
])

# Instantiate the Categorical Naive Bayes classifier with Laplace smoothing
clf = CategoricalNB(alpha=1)

# Train classifier
clf.fit(X, y)

# Input sample for prediction (Sunny, Cool)
sample = np.array([[0, 2]])  # Weather=Sunny, Temperature=Cool

# Predict the class probabilities
proba = clf.predict_proba(sample)

# Display the predicted probabilities
print("Predicted probabilities for sample (Weather=Sunny, Temperature=Cool):")
print(f"No (Play Tennis=No): {proba[0][0]:.4f}")
print(f"Yes (Play Tennis=Yes): {proba[0][1]:.4f}")

# Display the posterior probabilities (log probabilities)
print("\nClass Log Prior Probabilities (log P(Class)):")
print(f"No: {clf.class_log_prior_[0]:.4f}")
print(f"Yes: {clf.class_log_prior_[1]:.4f}")

print("\nFeature Log Probabilities (log P(Feature|Class)):")

# Number of features
n_features = X.shape[1]

# Number of categories per feature
n_categories = clf.n_categories_

for class_index, class_label in enumerate(clf.classes_):
    class_name = 'No' if class_label == 0 else 'Yes'
    print(f"\nFor class '{class_name}' (Play Tennis={class_name}):")
    for feature_index in range(n_features):
        feature_name = 'Weather' if feature_index == 0 else 'Temperature'
        category_probs = []
        num_categories = n_categories[feature_index]
        for category_index in range(num_categories):
            log_prob = clf.feature_log_prob_[class_index][feature_index][category_index]
            category_probs.append(f"{log_prob:.4f}")
        print(f"{feature_name}: {', '.join(category_probs)}")
