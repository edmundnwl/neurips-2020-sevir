import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

class SEVIRDatasetZarr(Dataset):
    def __init__(self, zarr_path):
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.features = self.ds['features'].values  # shape: (N, 48, 48, 4)
        self.sequence_length = 25  # 13 input + 12 output
        self.vil_channel = 0  # VIL is assumed to be channel 0

    def __len__(self):
        return self.features.shape[0] - self.sequence_length

    def __getitem__(self, idx):
        # Get a sequence of 25 consecutive frames (VIL only)
        sequence = self.features[idx : idx + self.sequence_length]  # shape: (25, 48, 48, 4)
        vil_sequence = sequence[:, :, :, self.vil_channel]  # shape: (25, 48, 48)

        x = vil_sequence[:13]  # Input: (13, 48, 48)
        y = vil_sequence[13:]  # Output: (12, 48, 48)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)