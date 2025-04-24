import joblib
from sklearn.metrics import classification_report, confusion_matrix
from rf_dataset import build_dataset
from rf_extract_features import pca_reduce  # keep in sync with training
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

X_test, y_test = build_dataset("../data/sub-sevir-test.zarr")
X_test = pca_reduce(X_test)                 # use the already-saved PCA

clf = joblib.load("rf_model.joblib")

# Use predicted probabilities for thresholding
scores = clf.predict_proba(X_test)[:, 1]

# Apply a custom threshold instead of 0.5
y_pred = (scores > 0.05).astype(int)  # try 0.3, 0.5, etc. for tuning
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Get precision-recall pairs and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, scores)
pr_auc = auc(recall, precision)

# Plot PR curve
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker='.', label=f'AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve – RF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_pr_curve.png", dpi=300)
plt.show()