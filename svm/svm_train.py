import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from svm_dataset import build_dataset
from svm_extract_features import pca_reduce  # comment this out if you skip PCA

# Load raw features + labels
X, y = build_dataset("../data/sub-sevir-train.zarr")
print("Label distribution:", np.bincount(y))

# Dimensionality reduction (optional)
X = pca_reduce(X, n_components=300)

# Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
clf = SVC(kernel="rbf", gamma="scale", class_weight="balanced")
clf.fit(X_tr, y_tr)

# Quick validation report
print(classification_report(y_val, clf.predict(X_val)))

# Save model
joblib.dump(clf, "svm_model.joblib")
