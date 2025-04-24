# visualize_predictions.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn_sevirdataset import SEVIRDatasetZarr
from cnn_model import CNN3DNowcast
from torch.utils.data import DataLoader
import cv2

# Load dataset and model
test_set = SEVIRDatasetZarr('data/sub-sevir-test.zarr')
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

model = CNN3DNowcast()
model.load_state_dict(torch.load('cnn/cnn_nowcast.pth', map_location='cpu'))
model.eval()

# Pick a single batch
x, y_true = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(x)

# Remove batch and channel dimensions explicitly
x = x[0, 0].numpy()         # (13, H, W)
y_true = y_true[0, 0].numpy()   # (12, H, W)
y_pred = y_pred[0, 0].numpy()   # (12, H, W)

# Ensure each frame is square (H, W); transpose if needed
def fix_frame_shape(frame):
    if frame.shape[0] < frame.shape[1]:
        return frame.T
    return frame

x = np.array([fix_frame_shape(f) for f in x])
y_true = np.array([fix_frame_shape(f) for f in y_true])
y_pred = np.array([fix_frame_shape(f) for f in y_pred])

# Upscale frames for visualization
SCALE = 8  # Scale factor for visual clarity
x = np.array([cv2.resize(f, (f.shape[1]*SCALE, f.shape[0]*SCALE), interpolation=cv2.INTER_NEAREST) for f in x])
y_true = np.array([cv2.resize(f, (f.shape[1]*SCALE, f.shape[0]*SCALE), interpolation=cv2.INTER_NEAREST) for f in y_true])
y_pred = np.array([cv2.resize(f, (f.shape[1]*SCALE, f.shape[0]*SCALE), interpolation=cv2.INTER_NEAREST) for f in y_pred])

# Plot input, prediction, and ground truth
fig, axes = plt.subplots(3, 12, figsize=(18, 6))

for i in range(12):
    axes[0, i].imshow(x[min(i, 12-1)], cmap='viridis', vmin=0, vmax=1)
    axes[0, i].set_title(f"In {min(i+1,13)}")
    axes[0, i].axis('off')
    axes[0, i].set_aspect('equal')

    axes[1, i].imshow(y_pred[i], cmap='viridis', vmin=0, vmax=1)
    axes[1, i].set_title(f"Pred {i+1}")
    axes[1, i].axis('off')
    axes[1, i].set_aspect('equal')

    axes[2, i].imshow(y_true[i], cmap='viridis', vmin=0, vmax=1)
    axes[2, i].set_title(f"True {i+1}")
    axes[2, i].axis('off')
    axes[2, i].set_aspect('equal')

axes[0, 0].set_ylabel("Input")
axes[1, 0].set_ylabel("Predicted")
axes[2, 0].set_ylabel("Ground Truth")

plt.tight_layout()
plt.savefig('cnn/cnn_prediction_vs_truth.png', dpi=300)
plt.show()