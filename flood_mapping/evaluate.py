import os
import torch, numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# from flood_mapping.models import UNet, LightUNet
from flood_mapping.models.unet_resnet import Unet
from flood_mapping.dataset import FloodDataset, FloodDatasetLoader
from flood_mapping import utils
from PIL import Image

class ModelEvaluate:
    def __init__(self, model_path = 'model/unet-voc.pt', data_folder="data", output_folder='output',
                 clip_value=1.0, patch_size=128, batch_size=1, backbone='resnet18', pretrained=False):
        self.data_folder = data_folder
        self.model_path = model_path
        self.output_folder = Path(os.path.join(output_folder, utils.today()))
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.output_path = self.output_folder
        self.clip_value = clip_value
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.backbone = backbone
        self.pretrained = pretrained
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

        # model = UNet(n_channels=2, labels=2)
        model = Unet(backbone_name=self.backbone, pretrained=self.pretrained)
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
        self.save_json(average_iou, average_accuracy)
        return tiles

    def save_images(self, tiles):
        results = self.output_path / 'results'
        preds = self.output_path / 'preds'
        os.makedirs(results)
        os.makedirs(preds)
        for (meta_image, n_tiles), sar_image, mask in zip(self.flood_dataset.meta_tiles, self.flood_dataset.sar_images, self.flood_dataset.masks):
            
            # Pop the first n_tiles from the tiles list
            meta_tiles = tiles[:n_tiles]
            tiles = tiles[n_tiles:]

            # Reconstruct image
            image = self.flood_dataset.image_tiler.reconstruct(meta_image, meta_tiles)
            print(sar_image.shape, image.shape)
            # Post-processing
            output_array = np.where(image > 0, 255, 0)
            output_img = output_array.squeeze().astype(dtype=np.uint8)

            # Assuming 'output_img' is the image you want to save
            output_pil_img = Image.fromarray(output_img)
            output_pil_img.save(results / f"pred_{meta_image.image.event_name}.png")

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
            plt.savefig(preds / f"{meta_image.image.event_name}.png")
            plt.close()

    def save_json(self, iou, accuracy):
        data = {
            "data_folder":self.data_folder,
            "model_path":self.model_path,
            "clip_value":self.clip_value,
            "patch_size":self.patch_size,
            "batch_size":self.batch_size,
            "backbone":self.backbone,
            "pretrained":self.pretrained,
            "IoU":iou,
            "accuracy":accuracy
        }

        with open(self.output_path / 'results.json', 'w') as f:
            json.dump(data, f)





if __name__ == "__main__":
    model_evaluate = ModelEvaluate()
    model_evaluate.evaluate()
