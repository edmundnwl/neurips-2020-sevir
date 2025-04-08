import xarray as xr
import numpy as np
import tensorflow as tf

# Load test data from Zarr
ds = xr.open_zarr('data/sub-sevir-test.zarr', consolidated=True)
features = ds['features'].values  # shape: (N, 12, 384, 384)

# If only one variable is included, reshape accordingly
vil_data = features[..., np.newaxis]  # shape: (N, 12, 384, 384, 1)

# Split into input (first 6) and output (last 6)
X = vil_data[:, :6]  # shape: (N, 6, 384, 384, 1)
Y = vil_data[:, 6:]  # shape: (N, 6, 384, 384, 1)

# Load pretrained Keras model
model = tf.keras.models.load_model('models/nowcast/mse_model.h5')

# Run inference
preds = model.predict(X[:10])  # test on first 10 samples

# Evaluate MSE
mse = np.mean((preds - Y[:10]) ** 2)
print("MSE on 10 test samples:", mse)
