import cv2
import numpy as np
from torch.utils.data import Dataset

from data.utils import load_data, make_tiles, remap_mask_color, resize_to_short_edge


class TrainDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        resize_size=None,
        patch_size=None,
        train_id=None,
        duplicate_data=0,
        resize_short_edge=None,
    ):
        super().__init__()
        images, masks, image_file_names, mask_file_names = load_data(image_dir, mask_dir, train_id)

        self.image_file_names = image_file_names
        self.mask_file_names = mask_file_names

        if (resize_size is not None) and (resize_size[0] > 0) and (resize_size[1] > 0):
            images = [
                cv2.resize(
                    image,
                    (resize_size[0], resize_size[1]),
                    interpolation=cv2.INTER_LINEAR_EXACT,
                )
                for image in images
            ]
            masks = [
                cv2.resize(
                    mask,
                    (resize_size[0], resize_size[1]),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
                for mask in masks
            ]

        if (patch_size is not None) and (patch_size > 0):
            for i, _ in enumerate(images):
                if min(images[i].shape[0:2]) < patch_size:
                    images[i] = resize_to_short_edge(images[i], short_edge_length=patch_size)
                    masks[i] = resize_to_short_edge(masks[i], short_edge_length=patch_size)
            images, masks = self.split_tiles(images, masks, patch_size)
        elif (resize_short_edge is not None) and (resize_short_edge > 0):
            for i, _ in enumerate(images):
                if min(images[i].shape[0:2]) < resize_short_edge:
                    images[i] = resize_to_short_edge(images[i], short_edge_length=resize_short_edge)
                    masks[i] = resize_to_short_edge(masks[i], short_edge_length=resize_short_edge)

        if (duplicate_data > 0) and (len(images) < duplicate_data):
            images = images * (duplicate_data // len(images) + 1)
            masks = masks * (duplicate_data // len(masks) + 1)

        self.images = images
        self.masks = masks

    @staticmethod
    def split_tiles(images, masks, patch_size):
        new_images = []
        new_masks = []
        assert len(images) == len(masks)
        for i, _ in enumerate(images):
            image = images[i]
            mask = masks[i]

            image_tiles, _, _, _, _ = make_tiles(
                image.transpose(2, 0, 1),
                bsize=patch_size,
                augment=False,
                tile_overlap=0.5,
            )
            mask_tiles, _, _, _, _ = make_tiles(
                mask[np.newaxis, ...], bsize=patch_size, augment=False, tile_overlap=0.5
            )
            mask_tiles = mask_tiles[:, :, 0, :, :]
            for r in range(image_tiles.shape[0]):
                for c in range(image_tiles.shape[1]):
                    new_images.append(image_tiles[r, c].transpose(1, 2, 0).astype(np.uint8))
                    new_masks.append(remap_mask_color(mask_tiles[r, c]))

        return new_images, new_masks

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        resize_size=None,
    ):
        super().__init__()
        images, masks, image_file_names, mask_file_names = load_data(image_dir, mask_dir)
        if (resize_size is not None) and (resize_size[0] > 0) and (resize_size[1] > 0):
            images = [
                cv2.resize(
                    image,
                    (resize_size[0], resize_size[1]),
                    interpolation=cv2.INTER_LINEAR_EXACT,
                )
                for image in images
            ]
            masks = [
                cv2.resize(
                    mask,
                    (resize_size[0], resize_size[1]),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
                for mask in masks
            ]
        self.images = images
        self.masks = masks
        self.image_file_names = image_file_names
        self.mask_file_names = mask_file_names

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.images)
