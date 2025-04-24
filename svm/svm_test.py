import joblib
from sklearn.metrics import classification_report, confusion_matrix
from svm_dataset import build_dataset
from svm_extract_features import pca_reduce  # keep in sync with training
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

X_test, y_test = build_dataset("../data/sub-sevir-test.zarr")
X_test = pca_reduce(X_test)                 # use the already-saved PCA

clf = joblib.load("svm_model.joblib")
# Use decision_function to get raw scores
scores = clf.decision_function(X_test)

# Apply a custom threshold instead of 0
y_pred = (scores > -1.0).astype(int)  # try -0.5, -1.5 later to compare
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Use decision scores
scores = clf.decision_function(X_test)

# Get precision-recall pairs and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, scores)
pr_auc = auc(recall, precision)

# Plot
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker='.', label=f'AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve – SVM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("svm_pr_curve.png", dpi=300)
plt.show()
