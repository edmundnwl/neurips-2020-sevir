"""
svm_visualize.py
----------------
Visual-diagnostics for the SEVIR-SVM model.
• Confusion-matrix heat-map
• ROC curve with AUC
• 2-D PCA scatter (decision regions + test points)

Requires the model (`svm_model.joblib`) and, if you used PCA,
the transformer (`pca.joblib`) produced by svm_train.py.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.decomposition import PCA
from svm_dataset import build_dataset
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Load test data & model
# ----------------------------------------------------------------------
TEST_ZARR = "../data/sub-sevir-test.zarr"

print("[INFO] Loading test set …")
X_test, y_test = build_dataset(TEST_ZARR)

print("[INFO] Loading PCA (if it exists) …")
pca_path = Path("pca.joblib")
if pca_path.exists():
    pca: PCA = joblib.load(pca_path)
    X_plot = pca.transform(X_test)      # reduced to n_components
else:
    pca = None
    X_plot = X_test

print("[INFO] Loading trained SVM …")
clf = joblib.load("svm_model.joblib")
y_pred = clf.predict(X_plot)
y_proba = (
    clf.decision_function(X_plot) if hasattr(clf, "decision_function") else y_pred
)

print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------
# 2. Confusion-matrix
# ----------------------------------------------------------------------
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, cmap="Blues", colorbar=False
)
disp.ax_.set_title("Confusion matrix – SEVIR SVM")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=300)
plt.close()

# ----------------------------------------------------------------------
# 3. ROC curve (binary only)
# ----------------------------------------------------------------------
if len(np.unique(y_test)) == 2:
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC curve – SEVIR SVM")
    plt.tight_layout()
    plt.savefig("svm_roc_curve.png", dpi=300)
    plt.close()

# ----------------------------------------------------------------------
# 4. 2-D PCA decision regions (optional but fun)
# ----------------------------------------------------------------------
if pca is not None and pca.n_components_ >= 2:
    print("[INFO] Plotting 2-D PCA decision surface …")
    # Use first 2 dims for a scatter + mesh-grid
    X2 = X_plot[:, :2]

    # build mesh
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    mesh_pts = np.c_[xx.ravel(), yy.ravel()]
    # Pad back to full PCA dimension (fill zeros for higher comps)
    if pca.n_components_ > 2:
        zeros = np.zeros((mesh_pts.shape[0], pca.n_components_ - 2))
        mesh_pts_full = np.hstack([mesh_pts, zeros])
    else:
        mesh_pts_full = mesh_pts

    Z = clf.predict(mesh_pts_full).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(-0.5, 2), cmap="coolwarm")
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y_test, cmap="coolwarm", s=10, edgecolor="k")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Decision surface in first 2 PCA dims")
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()
    plt.savefig("svm_pca_decision.png", dpi=300)
    plt.close()

print("[DONE] Visuals saved:\n"
      "  • svm_confusion_matrix.png\n"
      "  • svm_roc_curve.png    (if binary)\n"
      "  • svm_pca_decision.png (if PCA≥2)")