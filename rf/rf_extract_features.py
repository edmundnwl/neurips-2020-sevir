from sklearn.decomposition import PCA
import joblib
import numpy as np

def pca_reduce(X: np.ndarray, n_components=300):
    pca = PCA(n_components=n_components, svd_solver='randomized')
    X_red = pca.fit_transform(X)
    joblib.dump(pca, "pca.joblib")
    return X_red