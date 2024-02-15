import argparse
from flood_mapping.trainer import Trainer

def main(data_path, epochs=300, batch_size=8, lr=0.0001, patch_size=256, model_name='unet-voc.pt', backbone='resnet18', pretrained=False):
    trainer = Trainer(data_folder=data_path, epoch_number=epochs, batch_size=batch_size, 
                      learning_rate=lr, patch_size=patch_size, model_name=model_name, 
                      backbone=backbone, pretrained=pretrained)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flood Mapping Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size of tiles (default: 128)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training (default: 100)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training (default: 2)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training (default: 0.0001)")
    parser.add_argument("--model_name", type=str, default='unet-voc.pt', help="Name to save the model file (default: 'unet-voc.pt'")
    parser.add_argument("--backbone", type=str, default='resnet18', help="Choose backbone: [resnet18, resnet34, resnet50, etc..]")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (default: true)")
    args = parser.parse_args()

    main(args.data_path, args.epochs, args.batch_size, args.lr, args.patch_size, args.model_name, args.backbone, args.pretrained)
    # main('s12_fusion')

# examples
    # python train.py --data_path data --epochs 50 --batch_size 8 --lr 0.0001 --patch_size 256