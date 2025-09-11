import torch
import numpy as np

from dataset import TimeSeriesDataset
from ts2vec import TS2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw.pkl or processed.pkl")
    parser.add_argument("--dataset_type", type=str, choices=["raw", "processed"], default="processed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--repr_dim", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    dataset = TimeSeriesDataset(pkl_path=args.data_path, mode=args.dataset_type, seq_length=args.seq_len)
    nb_features = dataset.dfs[0].shape[1]
    train_data = np.array([df.values for df in dataset.dfs], dtype=np.float32)
    print(train_data.shape)

    model = TS2Vec(
        input_dims=nb_features,
        device=device,
        output_dims=args.repr_dim,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        verbose=True
    )

    model.save("ts2vec_model.pt")
