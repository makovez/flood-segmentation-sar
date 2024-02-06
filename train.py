import argparse
from flood_mapping.trainer import Trainer

def main(data_path, epochs, batch_size, lr, patch_size):
    trainer = Trainer(data_folder=data_path, epoch_number=epochs, batch_size=batch_size, learning_rate=lr, patch_size=patch_size)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flood Mapping Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size of tiles (default: 128)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training (default: 100)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training (default: 2)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training (default: 0.0001)")
    args = parser.parse_args()

    main(args.data_path, args.epochs, args.batch_size, args.lr, args.patch_size)

# examples
    # python train.py --data_path data --epochs 50 --batch_size 8 --lr 0.001 --patch_size 64