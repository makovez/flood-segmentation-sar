from flood_mapping.models.unet_resnet import Unet
import os, torch, rasterio
from torch.utils.data import Dataset
from torchvision import transforms
from flood_mapping.dataset.tiling.ImageTiler import ImageTiler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path
from flood_mapping import utils

model_path = 'model/s12final-resnet50-unet-adam-29.pt'

def calculate_iou(prediction, target):
    intersection = np.logical_and(prediction, target).sum().item()
    union = np.logical_or(prediction, target).sum().item()
    iou = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return iou

def calculate_precision(prediction, target):
    true_positive = np.logical_and(prediction, target).sum().item()
    false_positive = np.logical_and(prediction, np.logical_not(target)).sum().item()
    precision = true_positive / (true_positive + false_positive + 1e-6)  # Adding epsilon to avoid division by zero
    return precision

def calculate_recall(prediction, target):
    true_positive = np.logical_and(prediction, target).sum().item()
    false_negative = np.logical_and(np.logical_not(prediction), target).sum().item()
    recall = true_positive / (true_positive + false_negative + 1e-6)  # Adding epsilon to avoid division by zero
    return recall

class FloodDatasetLoader(Dataset):
    def __init__(self, root_dir, patch_size=(256, 256)):
        self.root_dir = root_dir
        self.event_folders = os.listdir(root_dir)
        self.clip_value = 1
        self.patch_size = patch_size
        self.image_tiler = ImageTiler(patch_size)
        self.transform = self._create_transforms()

    def __len__(self):
        return len(self.event_folders)

    def _create_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: torch.clamp(x, 0, self.clip_value)),
            transforms.Normalize(0, self.clip_value)
        ])


    def __getitem__(self, index):
        event_folder = self.event_folders[index]
        pre_flood_path = os.path.join(self.root_dir, event_folder, 'pre_flood.tif')
        post_flood_path = os.path.join(self.root_dir, event_folder, 'post_flood.tif')
        mask_path = os.path.join(self.root_dir, event_folder, 'mask.tif')

        pre_flood_image = self._read_image(pre_flood_path)
        post_flood_image = self._read_image(post_flood_path)
        mask_image = self._read_image(mask_path)

        if self.transform:
            pre_flood_image = self.transform(pre_flood_image)
            post_flood_image = self.transform(post_flood_image)
            mask_image = self.transform(mask_image)

        return event_folder, pre_flood_image, post_flood_image, mask_image

    def deconstruct(self, image):
        pos, tiles = self.image_tiler.tile_image(image)
        return pos, tiles

    def reconstruct(self, tiles, pos, shape):
        image = self.image_tiler.reconstruct(pos, tiles, shape=shape, checksum=False)
        return image

    def _read_image(self, image_path):
        with rasterio.open(image_path) as src:
            arr = src.read()
            return torch.from_numpy(arr)


# Example usage:

# for pre_flood, post_flood, mask in dataset:
#     # Assuming you have a model to predict water in pre and post flood images
#     # You need to define your model and perform inference here
#     # Let's assume 'outputs' are the predicted flood extents

#     # Calculate IOU
#     outputs = torch.rand_like(mask)  # Random output for demonstration
#     iou = calculate_iou(outputs, mask)

#     print("IOU:", iou.item())


class Predictor:
    def __init__(self, model_path, data_path = 'emsr_flood', backbone='resnet50', output_folder='output') -> None:
        self.model_path = model_path
        self.backbone = backbone
        self.data_path = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = FloodDatasetLoader(data_path)
        self.model = self.load_model()
        self.output_folder = Path(os.path.join(output_folder, utils.today()))
        self.output_folder.mkdir(exist_ok=True, parents=True)


    def load_model(self):
        assert os.path.isfile(self.model_path) is True, f'Cant find model on path: {self.model_path}'

        model = Unet(backbone_name=self.backbone)
        model.to(self.device)

        model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        model.eval() # Set model to evaluation mod
        return model 


    def save_image(self, event_name, pred, mask, pre_flood_pred, post_flood_pred, pre_flood, post_flood, iou, precision, recall):
        # Create a new figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Plot each image on the corresponding subplot
        axs[0, 0].imshow(mask.squeeze(), cmap='gray')
        axs[0, 0].set_title('Mask')

        axs[0, 1].imshow(pred.squeeze(), cmap='gray')
        axs[0, 1].set_title('Change')

        axs[0, 2].imshow(pre_flood_pred.squeeze(), cmap='gray')
        axs[0, 2].set_title('Pre-Flood Prediction')

        axs[1, 0].imshow(utils.generate_preview(post_flood))
        axs[1, 0].set_title('SAR Image Post-Flood')

        axs[1, 1].imshow(utils.generate_preview(pre_flood))
        axs[1, 1].set_title('SAR Image Pre-Flood')

        axs[1, 2].imshow(post_flood_pred.squeeze(), cmap='gray')
        axs[1, 2].set_title('Post-Flood Prediction')

        fig.suptitle(f"{event_name[0]} - IOU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}", fontsize=14)

        # Save the figure
        plt.savefig(self.output_folder / f"{event_name[0]}.png")
        plt.close()


    def predict(self):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        
        iou_total, precision_total, recall_total = 0, 0, 0
        num_samples = len(data_loader)
        
        with tqdm(total=num_samples, desc=f"Predicting") as pbar:
            for event_name, pre_flood, post_flood, mask in data_loader:
                # Calculate IOU
                pre_flood_tiles, post_flood_tiles = [], []

                pre_pos, pre_tiles = self.dataset.deconstruct(pre_flood.squeeze())
                for tile in pre_tiles:
                    out = self.model(tile[None, :].to(self.device))
                    pre_flood_tiles.append(out.argmax(dim=1).cpu())
                
                post_pos, post_tiles = self.dataset.deconstruct(post_flood.squeeze())
                for tile in post_tiles:
                    out = self.model(tile[None, :].to(self.device))
                    post_flood_tiles.append(out.argmax(dim=1).cpu())
                
                pre_flood_pred = self.dataset.reconstruct(pre_flood_tiles, pre_pos, (1, *pre_flood.squeeze().shape[1:]))
                post_flood_pred = self.dataset.reconstruct(post_flood_tiles, post_pos, (1, *post_flood.squeeze().shape[1:]))

                change = post_flood_pred - pre_flood_pred
                change[change == -1] = 0
                iou = calculate_iou(change, mask.numpy())
                precision = calculate_precision(change, mask.numpy())
                recall = calculate_recall(change, mask.numpy())
                
                iou_total += iou
                precision_total += precision
                recall_total += recall


                self.save_image(event_name, change, mask.squeeze().numpy(), pre_flood_pred, post_flood_pred, pre_flood.squeeze().numpy(), post_flood.squeeze().numpy(), iou, precision, recall)
                pbar.update(1)
                pbar.set_postfix({"IoU": iou, "Precision": precision, "Recall":recall})

        # Compute averages
        average_iou = iou_total / num_samples
        average_precision = precision_total / num_samples
        average_recall = recall_total / num_samples

        print(f"Average IOU: {average_iou:.4f}, Average Precision: {average_precision:.4f}, Average Recall: {average_recall:.4f}")

    def save_json(self, iou, accuracy):
        data = {
            "data_folder":self.data_path,
            "model_path":self.model_path,
            "backbone":self.backbone,
            "IoU":iou,
            "accuracy":accuracy
        }

        with open(self.output_path / 'results.json', 'w') as f:
            json.dump(data, f)

p = Predictor('model/rn18-wce-pre.pt', data_path='emsr_flood', backbone='resnet18')
p.predict()