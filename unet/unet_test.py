import torch
from torch.utils.data import DataLoader
from torch import nn
from unet_sevirdataset import SEVIRDatasetZarr
from unet_model import UNet
from tqdm import tqdm

# Load test dataset
TEST_PATH = 'data/sub-sevir-test.zarr'
test_set = SEVIRDatasetZarr(TEST_PATH)
test_loader = DataLoader(test_set, batch_size=2)

# Load trained model
model = UNet(in_channels=13, out_channels=12)
model.load_state_dict(torch.load("unet/unet_nowcast.pth", map_location="cpu"))
model.eval()

# Loss function
criterion = nn.MSELoss()

total_loss = 0
num_batches = 0

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="[Testing]"):
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        num_batches += 1

print(f"\nTest MSE: {total_loss / num_batches:.4f}")