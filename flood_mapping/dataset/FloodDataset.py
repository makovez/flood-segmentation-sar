import os
import rasterio
import numpy as np
from flood_mapping.dataset.tiling.ImageTiler import ImageTiler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class FloodDataset:
    def __init__(self, dataset_folder, patch_size=(512, 512), train=True):
        self.dataset_folder = dataset_folder
        self.image_tiler = ImageTiler(patch_size)
        self.train = train
        self.data, self.labels = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

    def _load_data(self):
        dataset_type = 'train' if self.train else 'test'
        root_path = os.path.join(self.dataset_folder, dataset_type)

        data, labels = [], []
        for event in os.listdir(root_path):
            flood_image_path = os.path.join(root_path, event, event + '.tif')
            mask_image_path = os.path.join(root_path, event, 'mask.tif')

            flood = self._read_image(flood_image_path)
            mask = self._read_image(mask_image_path)

            tiled_image, raw_tiles = self.image_tiler.tile_image(flood)
            _, raw_tiles_mask = self.image_tiler.tile_image(mask)

            data += raw_tiles
            labels += raw_tiles_mask

        return np.array(data), np.array(labels)

    def train_test(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=test_size)
        return X_train, X_test, y_train, y_test
    
    def _read_image(self, image_path):
        with rasterio.open(image_path) as src:
            return src.read()

class FloodDatasetLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label


# Example usage:
# train_dataset = FloodDataset('emsr', patch_size=(16, 16), train=True)
# test_dataset = FloodDataset('emsr', patch_size=(16, 16), train=False)
# train_loader = DataLoader(train_dataset, batch_size=your_batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=your_batch_size, shuffle=False)
