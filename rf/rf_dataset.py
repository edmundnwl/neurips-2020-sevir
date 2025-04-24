import zarr
import numpy as np

SEQUENCE_LEN = 25   # 13 input + 12 output
INPUT_FRAMES  = 13
VIL_CH        = 0   # channel index used by your friend
THRESHOLD     = 2.0  # VIL threshold in dBZ – change as you wish

def load_sevir_zarr(zarr_path: str) -> np.ndarray:
    features = zarr.open(f"{zarr_path}/features", mode="r")
    return features[:].astype(np.float32)

def build_dataset(zarr_path: str):
    feats = load_sevir_zarr(zarr_path)
    n_sequences = feats.shape[0] - SEQUENCE_LEN

    X_list, y_list = [], []
    for idx in range(n_sequences):
        block = feats[idx : idx + SEQUENCE_LEN]
        vil_block = block[..., VIL_CH]

        x_raw     = vil_block[:INPUT_FRAMES]
        y_future  = vil_block[INPUT_FRAMES:]

        X_list.append(x_raw.reshape(-1))
        stormy = (y_future.max() >= THRESHOLD)
        y_list.append(int(stormy))

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y

if __name__ == "__main__":
    X, y = build_dataset("../data/sub-sevir-train.zarr")
    print("Feature matrix:", X.shape, "Label vector:", y.shape)
