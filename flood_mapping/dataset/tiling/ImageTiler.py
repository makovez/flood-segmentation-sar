import numpy as np
from pydantic import BaseModel
from typing import List, Tuple, Union

from hashlib import sha1

class Tile(BaseModel):
    x_start: int
    x_end: int
    y_start: int
    y_end: int

class Image(BaseModel):
    checksum: str
    event_name: str
    shape: Union[
        Tuple[int, int, int, int],
        Tuple[int, int, int],
        Tuple[int, int]
    ]

class TiledImage(BaseModel):
    tiles: List[Tile]
    image: Image
    patch_size: Tuple[int, int]

class ImageTiler:
    def __init__(self, patch_size=(512, 512)):
        self.patch_size = patch_size

    @staticmethod
    def _calculate_overlap(image_shape, tile_size=(512, 512)):

        width, height = image_shape[-2], image_shape[-1]
        rows = (width + tile_size[0] - 1) // tile_size[0]
        cols = (height + tile_size[1] - 1) // tile_size[1]
        overlap_rows = int(np.ceil((tile_size[0] * rows - width) / (rows - 1)))
        overlap_cols = int(np.ceil((tile_size[1] * cols - height) / (cols - 1)))
        return overlap_rows, overlap_cols, rows, cols

    def reconstruct(self, tiled_image: TiledImage, tiles: List[np.ndarray]):
        rec_image = np.zeros(tiled_image.image.shape)
        for i, tile in enumerate(tiled_image.tiles):
            if len(rec_image.shape) >= 4:
                rec_image[:, :, tile.x_start:tile.x_end, tile.y_start:tile.y_end] = tiles[i]    
            elif len(rec_image.shape) >= 3:
                rec_image[:, tile.x_start:tile.x_end, tile.y_start:tile.y_end] = tiles[i]
            elif len(rec_image.shape) >= 2:
                rec_image[tile.x_start:tile.x_end, tile.y_start:tile.y_end] = tiles[i]
        #assert sha1(rec_image).hexdigest() == tiled_image.image.checksum, "Reconstructed img do not match original"
        return rec_image

    def tile_image(self, image, event_name):
        overlap_rows, overlap_cols, rows, cols = self._calculate_overlap(image.shape, self.patch_size)
        tiles = []
        raw_tiles = []

        t = np.zeros(self.patch_size)
        for i in range(cols):
            for j in range(rows):
                x = j * (self.patch_size[0] - overlap_rows)
                y = i * (self.patch_size[1] - overlap_cols)

                if (j == rows - 1):  # adjust for last row
                    x = x - (x + self.patch_size[0] - image.shape[1])
                if (i == cols - 1):  # adjust for last col
                    y = y - (y + self.patch_size[1] - image.shape[2])

                if len(image.shape) >= 4:
                    t = image[:, :, x:x + self.patch_size[0], y:y + self.patch_size[1]]
                elif len(image.shape) >= 3:
                    t = image[:, x:x + self.patch_size[0], y:y + self.patch_size[1]]
                elif len(image.shape) >= 2:
                    t = image[x:x + self.patch_size[0], y:y + self.patch_size[1]]
                
                tile = Tile(
                    x_start=x, x_end=x + self.patch_size[0],
                    y_start=y, y_end=y + self.patch_size[1])

                tiles.append(tile)
                raw_tiles.append(t)

                # print(f"[{j}] x({x}-{x + self.patch_size[0]}), [{i}] y({y}-{y + self.patch_size[1]})\n")
        image = Image(event_name=event_name, checksum=sha1(image).hexdigest(), shape=image.shape)
        tiled_image = TiledImage(tiles=tiles, image=image, patch_size=self.patch_size)

        return tiled_image, raw_tiles

if __name__ == '__main__':
    imgtiler = ImageTiler((512,512))
    image = np.random.rand(2, 2717, 2333)
    tiled_image, raw_tiles = imgtiler.tile_image(image)
    image = imgtiler.reconstruct(tiled_image, raw_tiles)
