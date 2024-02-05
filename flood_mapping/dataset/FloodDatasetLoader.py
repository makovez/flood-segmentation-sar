import os, glob
import rasterio, numpy as np
from rasterio.windows import Window
import matplotlib.pyplot as plt
from flood_mapping.dataset.tiling.ImageTiler import ImageTiler
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader


class FloodDataset:
    def __init__(self, dataset_folder, patch_size = (512, 512)):
        self.dataset_folder = dataset_folder
        self.image_tiler = ImageTiler(patch_size)
        _, self.X, self.y = self.get_flood_events()
        self.test, self.testX, self.testY = self.get_flood_events(train=False)

    def get_flood_events(self, train=True):
        dataset_type = 'train' if train else 'test'
        root_path = os.path.join(self.dataset_folder, dataset_type)

        tiles, X, y = [], [], []
        for event in os.listdir(root_path):
            flood_image = os.path.join(root_path, event, event + '.tif')
            mask_image = os.path.join(root_path, event, 'mask.tif')
            flood = self._read_image(flood_image)
            #flood = (flood[0] + flood[1])/2
            mask = self._read_image(mask_image)
            tiled_image, raw_tiles = self.image_tiler.tile_image(flood)
            _, raw_tiles_mask = self.image_tiler.tile_image(mask)
            X += raw_tiles
            y += raw_tiles_mask
            tiles.append((tiled_image, X, y))
        
        return tiles, np.array(X), np.array(y)
    

    def _read_image(self, image_path):
        with rasterio.open(image_path) as src:
            return src.read()
        


# Define your custom dataset class
class FloodDatasetLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve a sample and its corresponding label
        sample = self.data[index]
        label = self.labels[index]

        return sample, label 

        
# fdl = FloodDatasetLoader('emsr', (16, 16))
# data = fdl.load_dataset()
# fdl.visualize_tiles(data)
