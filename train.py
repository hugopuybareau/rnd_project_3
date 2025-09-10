import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import TimeSeriesDataset
from model import AutoEncoder



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw.pkl or processed.pkl")
    parser.add_argument("--dataset_type", type=str, choices=["raw", "processed"], default="processed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoder_hidden_size", type=int, default=64)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    train_loader, val_loader, nb_features = TimeSeriesDataset(
        pkl_path=args.data_path,
        mode=args.dataset_type,
        seq_len=args.seq_len,
        prediction_window=1
    ).get_loaders(batch_size=args.batch_size, train_split=0.9)

    model = AutoEncoder(
        input_size=nb_features,
        output_size=nb_features,
        encoder_hidden_size=args.encoder_hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        seq_len=args.seq_len,
        lr=args.lr
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
    )

    trainer.fit(model, train_loader, val_loader)

