import torch

ckpt = torch.load("models/nowcast/mse_model.h5", map_location="cpu")

print("Available keys:")
print(ckpt.keys())

if "model_state_dict" in ckpt:
    print("\nModel state_dict:")
    for k, v in ckpt["model_state_dict"].items():
        print(k, v.shape)
else:
    print("Not a standard checkpoint.")
