import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sevir_dataset import SEVIRDatasetZarr
from cnn3d_model import CNN3DNowcast
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    train_set = torch.utils.data.Subset(SEVIRDatasetZarr('data/sub-sevir-train.zarr'), range(10000))
    val_set = torch.utils.data.Subset(SEVIRDatasetZarr('data/sub-sevir-val.zarr'), range(10000))
    # train_set = SEVIRDatasetZarr('data/sub-sevir-train.zarr')
    # val_set = SEVIRDatasetZarr('data/sub-sevir-val.zarr')

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN3DNowcast().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), 'cnn3d_nowcast.pth')

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
