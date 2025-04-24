import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from rf_dataset import build_dataset
from rf_extract_features import pca_reduce  # comment this out if you skip PCA

# Load raw features + labels
X, y = build_dataset("../data/sub-sevir-train.zarr")
print("Label distribution:", np.bincount(y))

# Dimensionality reduction (optional)
X = pca_reduce(X, n_components=300)

# Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_tr, y_tr)

scores = clf.predict_proba(X_val)[:, 1]

# Apply custom threshold for storm detection
y_pred = (scores > 0.2).astype(int)  # try 0.15 or 0.25 later too

# Print evaluation
print(classification_report(y_val, y_pred))

# Save model
joblib.dump(clf, "rf_model.joblib")
