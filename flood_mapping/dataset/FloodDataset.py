import os
import rasterio
import numpy as np
from flood_mapping.dataset.tiling.ImageTiler import ImageTiler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from torchvision import transforms
import torch

class FloodDataset:
    def __init__(self, dataset_folder, patch_size=(512, 512), train=True):
        self.dataset_folder = dataset_folder
        self.image_tiler = ImageTiler(patch_size)
        self.train = train
        self.sar_images, self.masks, self.meta_tiles, self.data, self.labels = self._load_data()

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        dataset_type = 'train' if self.train else 'test'
        root_path = os.path.join(self.dataset_folder, dataset_type)

        sar_images, masks, meta_tiles, data, labels = [], [], [], [], []
        
        for event_name in tqdm(os.listdir(root_path), desc="Loading data"):
            preview_sar_path = os.path.join(root_path, event_name, 'preview_image', event_name + '.jpg')
            preview_mask_path = os.path.join(root_path, event_name, 'preview_image', 'mask.jpg')  
            flood_image_path = os.path.join(root_path, event_name, event_name + '.tif')
            mask_image_path = os.path.join(root_path, event_name, 'mask.tif')

            flood = self._read_image(flood_image_path)
            mask = self._read_image(mask_image_path)
            if mask.max() > 1:
                mask[mask != 6] = 0
                mask[mask == 6] = 1
            flood = np.nan_to_num(flood, nan=0)
            mask = np.nan_to_num(mask, nan=0)
            
            preview_flood = plt.imread(preview_sar_path)
            preview_mask = plt.imread(preview_mask_path)

            _, raw_tiles = self.image_tiler.tile_image(flood, event_name)
            meta_tile, raw_tiles_mask = self.image_tiler.tile_image(mask, event_name)

            data += raw_tiles
            labels += raw_tiles_mask
            meta_tiles.append((meta_tile, len(raw_tiles)))
            sar_images.append(preview_flood)
            masks.append(mask)

        return sar_images, masks, np.array(meta_tiles), np.array(data), np.array(labels)

    def train_test(self, test_size=0.4):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=test_size)
        return X_train, X_test, y_train, y_test
    
    def _read_image(self, image_path):
        with rasterio.open(image_path) as src:
            return src.read()
        
    def calculate_class_weights(self):
        # Reshape labels to flatten the spatial dimensions
        flat_labels = self.labels.reshape(-1)

        # Calculate class counts
        class_counts = np.bincount(flat_labels)

        # Total number of pixels
        total_pixels = np.sum(class_counts)

        # Class frequencies
        class_frequencies = class_counts / total_pixels

        # Inverse class frequencies as class weights
        class_weights = 1.0 / class_frequencies

        # Normalize class weights
        class_weights /= np.sum(class_weights)

        return class_weights

class SARDataAugmentation:
    def __init__(self):
        pass


    def __call__(self, sample):
        # Add SAR-specific augmentations
        sample = self.random_noise(sample)

        return sample

    def random_noise(self, image):
        # Add Gaussian noise
        noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Clip values to [0, 1]
        return noisy_image



class FloodDatasetLoader(Dataset):
    def __init__(self, data, labels, aug=False):
        self.data = data
        self.labels = labels
        self.aug = SARDataAugmentation() if aug else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        sample = self.apply_augmentations(sample)
        return sample, label
    
    def apply_augmentations(self, image):
        if self.aug:
            return self.aug(image)
        return image


# Example usage with augmentations:

# Additional augmentations specific to SAR data


# Example usage:
# train_dataset = FloodDataset('emsr', patch_size=(16, 16), train=True)
# test_dataset = FloodDataset('emsr', patch_size=(16, 16), train=False)
# train_loader = DataLoader(train_dataset, batch_size=your_batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=your_batch_size, shuffle=False)
