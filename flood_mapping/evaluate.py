import os
import torch, numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import matplotlib.pyplot as plt

from flood_mapping.models import UNet
from flood_mapping.dataset import FloodDataset, FloodDatasetLoader
from flood_mapping import utils

class ModelEvaluate:
    def __init__(self, data_folder="data", output_folder='output', model_folder="model", model_name="unet-voc.pt",
                 clip_value=1.0, shuffle_data_loader=True, patch_size=128, batch_size=1):
        self.data_folder = data_folder
        self.model_folder = Path(model_folder)
        self.model_path = self.model_folder / model_name
        self.output_folder = Path(os.path.join(output_folder, utils.today()))
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.output_path = self.output_folder
        self.clip_value = clip_value
        self.shuffle_data_loader = shuffle_data_loader
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=2).to(self.device)  # Assuming binary classification
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(self.device)
        self.normalize_transform = self._create_transforms()
        self.flood_dataset = FloodDataset(self.data_folder, (patch_size, patch_size), train=False)

    def _create_dataset(self):
        return FloodDatasetLoader(self.flood_dataset.data, self.flood_dataset.labels)

    def _create_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: torch.clamp(x, 0, self.clip_value)),
            transforms.Normalize(0, self.clip_value)
        ])

    def evaluate(self):
        dataset = self._create_dataset()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        model = UNet(n_channels=2, labels=2)
        model.to(self.device)

        assert os.path.isfile(self.model_path) is True, f'Cant find model on path: {self.model_path}'

        model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        model.eval() # Set model to evaluation mode   

        iou_scores = []
        accuracy_scores = []
        tiles = []

        with tqdm(total=len(data_loader), desc=f"Predict test") as pbar:
            for i, batch in enumerate(data_loader):
                input, target = batch

                input = self.normalize_transform(input)
                input = input.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                # if input.shape[0] < 2:
                #     continue

                output = model(input)
                output_preds = torch.split(output.argmax(dim=1), 1, dim=0)

                for tile in output_preds:
                    tiles.append(tile.squeeze().cpu())

                # Calculate IOU and accuracy for each batch
                iou_batch = self.iou_metric(output.argmax(dim=1).squeeze(), target.squeeze())
                accuracy_batch = self.accuracy_metric(output.argmax(dim=1).squeeze(), target.squeeze())
                iou_scores.append(iou_batch.item())
                accuracy_scores.append(accuracy_batch.item())

                pbar.update(1)
                pbar.set_postfix({"IoU": iou_batch.item(), "Accuracy": accuracy_batch.item()})

        average_iou = sum(iou_scores) / len(iou_scores)
        average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

        print(f"Average IoU: {average_iou}, Average Accuracy: {average_accuracy}")
        return tiles

    def save_images(self, tiles):
        for (meta_image, n_tiles), sar_image, mask in zip(self.flood_dataset.meta_tiles, self.flood_dataset.sar_images, self.flood_dataset.masks):
            # Pop the first n_tiles from the tiles list
            meta_tiles = tiles[:n_tiles]
            tiles = tiles[n_tiles:]

            # Reconstruct image
            image = self.flood_dataset.image_tiler.reconstruct(meta_image, meta_tiles)

            # Post-processing
            output_array = np.where(image > 0, 255, 0)
            output_img = output_array.squeeze().astype(dtype=np.uint8)

            # Plot SAR image and output prediction side by side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(sar_image.squeeze())
            axes[0].set_title('SAR Image')
            axes[0].axis('off')
            axes[1].imshow(mask.squeeze())
            axes[1].set_title('MMFlood Mask')
            axes[1].axis('off')
            axes[2].imshow(output_img, cmap='gray')
            axes[2].set_title('Model Prediction')
            axes[2].axis('off')

            # Save the plot
            plt.savefig(self.output_path / f"{meta_image.image.event_name}.png")
            plt.close()



if __name__ == "__main__":
    model_evaluate = ModelEvaluate()
    model_evaluate.evaluate()
