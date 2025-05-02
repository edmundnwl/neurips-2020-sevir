import torch
from torch.utils.data import DataLoader
from torch import nn
from convGRU.sevir_dataset import SEVIRDatasetZarr
from convGRU.model_convGRU import ConvGRUNowcast
from tqdm import tqdm

# Load test dataset
TEST_PATH = 'data/sub-sevir-test.zarr'
test_set = SEVIRDatasetZarr(TEST_PATH)
test_loader = DataLoader(test_set, batch_size=2)

# Load trained model

model = ConvGRUNowcast(input_dim=13, hidden_dim=[64, 128], output_dim=12)
model.load_state_dict(torch.load('convGRU/convgru_nowcast.pth', map_location='cpu'))
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